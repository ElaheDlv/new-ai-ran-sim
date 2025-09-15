"""DQN-based PRB allocator xApp (with extensive inline documentation).

This xApp implements a light Deep Q-Network policy that shifts downlink PRBs
between slices in each cell. It closely follows the Tractor paper’s MDP:

- State (per cell): [#mMTC UEs, #URLLC UEs, #eMBB UEs, PRBs_mMTC, PRBs_URLLC].
  PRBs in eMBB are implicit: max_dl_prb - PRBs_mMTC - PRBs_URLLC.
- Actions: 0=keep, 1=mMTC→URLLC, 2=mMTC→eMBB, 3=URLLC→mMTC, 4=URLLC→eMBB,
           5=eMBB→mMTC, 6=eMBB→URLLC. Each action moves K PRBs (K is configurable).
- Reward: weighted sum of slice‑specific scores (eMBB queue drain proxy,
  URLLC delay proxy from buffer/rate, mMTC utilization/idle penalty).

Training support:
- Online training with epsilon‑greedy exploration, replay buffer, target network.
- Optional telemetry: TensorBoard and Weights & Biases for metrics.

All key methods include detailed comments to make the logic easy to follow.
"""

from .xapp_base import xAppBase

import os
import math
import random
from collections import deque, defaultdict
from datetime import datetime

import settings  # global configuration and constants

try:  # torch is optional; the xApp disables itself if not available
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    # Do not crash if torch is missing; xApp will disable itself.
    TORCH_AVAILABLE = False
    print("xapp_dqn_prb_allocator: torch not available; xApp will disable itself.")


# Short aliases for slice names (read from settings with fallbacks)
SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")   # Enhanced Mobile Broadband
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")  # Ultra Reliable Low Latency
SL_M = getattr(settings, "NETWORK_SLICE_MTC_NAME", "mMTC")     # Massive Machine Type


if TORCH_AVAILABLE:
    class _ReplayBuffer:
        """Minimal replay buffer for off‑policy training.

        Stores tuples (state, action, reward, next_state, done) in a ring buffer
        and supports random mini‑batch sampling.
        """
        def __init__(self, capacity: int = 50000):
            self.buf = deque(maxlen=int(capacity))  # fixed‑size circular buffer

        def push(self, s, a, r, ns, d):
            self.buf.append((s, a, r, ns, d))

        def sample(self, batch):
            import numpy as np

            batch = min(batch, len(self.buf))  # cap to current buffer size
            idx = np.random.choice(len(self.buf), batch, replace=False)  # unique indices
            s, a, r, ns, d = zip(*[self.buf[i] for i in idx])
            return (
                torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(ns, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32),
            )

        def __len__(self):
            return len(self.buf)


    class _DQN(nn.Module):
        """Small fully‑connected Q‑network.

        Architecture: 2 hidden layers (64 units each, ReLU) → Q‑values.
        """
        def __init__(self, in_dim: int, n_actions: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, n_actions),
            )

        def forward(self, x):
            return self.net(x)
else:
    # Placeholders to avoid NameError if referenced in disabled code paths
    _ReplayBuffer = None
    _DQN = None


class xAppDQNPRBAllocator(xAppBase):
    """DQN-based PRB allocator (Table 3 actions).

    Public methods:
    - start(): lifecycle hook when xApp is loaded
    - step(): called every simulation step; performs observe→learn→act
    - to_json(): returns metadata for introspection

    Private helpers:
    - _epsilon(): exploration schedule
    - _get_state(): builds the per‑cell state vector
    - _aggregate_slice_metrics(): aggregates KPIs needed for rewards
    - _slice_scores(): computes eMBB/URLLC/mMTC slice scores
    - _reward(): combines slice scores with weights
    - _act(): epsilon‑greedy action selection
    - _opt_step(): one optimization step on a replay mini‑batch
    - _apply_action(): applies PRB move to cell quotas
    - _log(): emits metrics to TB/W&B
    """

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = getattr(settings, "DQN_PRB_ENABLE", False)  # on/off toggle

        # Runtime / training knobs
        self.train_mode = getattr(settings, "DQN_PRB_TRAIN", True)  # train or inference only
        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))  # act every N steps
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))  # PRBs moved per action

        # Reward shaping/weights
        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.33))   # weight for eMBB slice score
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.34))  # weight for URLLC slice score
        self.w_m = float(getattr(settings, "DQN_WEIGHT_MMTC", 0.33))   # weight for mMTC slice score
        self.urlc_gamma_s = float(getattr(settings, "DQN_URLLC_GAMMA_S", 0.01))  # max tolerable delay (s)

        # DQN parameters
        self.gamma = float(getattr(settings, "DQN_PRB_GAMMA", 0.99))        # discount factor
        self.lr = float(getattr(settings, "DQN_PRB_LR", 1e-3))              # learning rate
        self.batch = int(getattr(settings, "DQN_PRB_BATCH", 64))             # mini‑batch size
        self.buffer_cap = int(getattr(settings, "DQN_PRB_BUFFER", 50000))    # replay capacity
        self.eps_start = float(getattr(settings, "DQN_PRB_EPSILON_START", 1.0))  # ε start
        self.eps_end = float(getattr(settings, "DQN_PRB_EPSILON_END", 0.1))      # ε end
        self.eps_decay = int(getattr(settings, "DQN_PRB_EPSILON_DECAY", 10000))  # ε decay steps
        self.model_path = getattr(settings, "DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")  # checkpoint

        # Internal state
        self._t = 0  # global decision counter (used for ε schedule and logging)
        self._per_cell_prev = {}  # cell_id -> {state, action} for previous decision
        self._last_loss = None    # last training loss (for TB/W&B)
        self._action_counts = defaultdict(int)  # histogram of actions taken

        # NN
        self._n_actions = 7  # size of the discrete action space
        self._state_dim = 5  # length of the state vector
        self._device = torch.device("cpu") if TORCH_AVAILABLE else None  # device for torch tensors
        if self.enabled and not TORCH_AVAILABLE:
            print(f"{self.xapp_id}: torch not available; disabling.")
            self.enabled = False

        # Telemetry: TensorBoard / W&B
        self._tb = None     # TensorBoard SummaryWriter (optional)
        self._wandb = None  # Weights & Biases run object (optional)
        if self.enabled:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self._q = _DQN(self._state_dim, self._n_actions).to(self._device)          # online network
            self._q_target = _DQN(self._state_dim, self._n_actions).to(self._device)   # target network
            self._q_target.load_state_dict(self._q.state_dict())
            self._opt = optim.Adam(self._q.parameters(), lr=self.lr)  # optimizer
            self._buf = _ReplayBuffer(self.buffer_cap)                 # replay buffer
            # Try to load existing model
            try:
                if os.path.exists(self.model_path):
                    self._q.load_state_dict(torch.load(self.model_path, map_location=self._device))
                    self._q_target.load_state_dict(self._q.state_dict())
                    print(f"{self.xapp_id}: loaded model from {self.model_path}")
            except Exception as e:
                print(f"{self.xapp_id}: failed loading model: {e}")

            # TensorBoard
            if getattr(settings, "DQN_TB_ENABLE", False):  # TensorBoard logging (optional)
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    base = getattr(settings, "DQN_TB_DIR", "backend/tb_logs")
                    run = datetime.now().strftime("%Y%m%d_%H%M%S")
                    logdir = os.path.join(base, f"dqn_prb_{run}")
                    os.makedirs(logdir, exist_ok=True)
                    # SummaryWriter expects 'log_dir', not 'logdir'
                    self._tb = SummaryWriter(log_dir=logdir)
                    print(f"{self.xapp_id}: TensorBoard logging to {logdir}")
                except Exception as e:
                    print(f"{self.xapp_id}: TensorBoard unavailable: {e}")
                    self._tb = None
            # W&B
            if getattr(settings, "DQN_WANDB_ENABLE", False):  # W&B logging (optional)
                try:
                    import wandb
                    cfg = {
                        "gamma": self.gamma,
                        "lr": self.lr,
                        "batch": self.batch,
                        "buffer": self.buffer_cap,
                        "epsilon_start": self.eps_start,
                        "epsilon_end": self.eps_end,
                        "epsilon_decay": self.eps_decay,
                        "period_steps": self.period_steps,
                        "move_step": self.move_step,
                    }
                    proj = getattr(settings, "DQN_WANDB_PROJECT", "ai-ran-dqn")
                    name = getattr(settings, "DQN_WANDB_RUNNAME", "") or None
                    self._wandb = wandb.init(project=proj, name=name, config=cfg)
                    print(f"{self.xapp_id}: W&B logging enabled (project={proj})")
                except Exception as e:
                    print(f"{self.xapp_id}: W&B unavailable: {e}")
                    self._wandb = None

    # ---------------- xApp lifecycle ----------------
    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        print(f"{self.xapp_id}: enabled (train={self.train_mode}, period={self.period_steps} steps)")

    def _epsilon(self):
        """Return current exploration epsilon based on linear decay schedule."""
        if self.eps_decay <= 0:
            return self.eps_end
        frac = min(1.0, self._t / float(self.eps_decay))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def _get_slice_counts(self, cell):
        """Count UEs per slice in a cell.

        Returns a dict {slice_name: count} for the three slices.
        """
        cnt = {SL_E: 0, SL_U: 0, SL_M: 0}
        for ue in cell.connected_ue_list.values():
            s = getattr(ue, "slice_type", None)
            if s in cnt:
                cnt[s] += 1
        return cnt

    def _get_state(self, cell):
        """Build normalized state vector for a cell.

        Normalization keeps inputs within roughly [0,1] to stabilize learning.
        """
        cnt = self._get_slice_counts(cell)
        prb_map = getattr(cell, "slice_dl_prb_quota", {}) or {}  # current PRB quotas per slice
        prb_m = float(prb_map.get(SL_M, 0))
        prb_u = float(prb_map.get(SL_U, 0))
        n_m = float(cnt.get(SL_M, 0))
        n_u = float(cnt.get(SL_U, 0))
        n_e = float(cnt.get(SL_E, 0))
        # Normalize by rough scales to keep inputs in sane ranges
        max_ue = max(1.0, float(getattr(settings, "UE_DEFAULT_MAX_COUNT", 50)))
        s = [
            n_m / max_ue,
            n_u / max_ue,
            n_e / max_ue,
            prb_m / float(max(1, cell.max_dl_prb)),
            prb_u / float(max(1, cell.max_dl_prb)),
        ]
        return s

    def _aggregate_slice_metrics(self, cell):
        """Return per-slice aggregates used to compute rewards.

        eMBB/URLLC: sum served DL Mbps and DL buffer bytes over UEs in slice.
        mMTC: sum requested PRBs and granted PRBs; include slice PRB quota.
        """
        agg = {
            SL_E: {"tx_mbps": 0.0, "buf_bytes": 0.0},
            SL_U: {"tx_mbps": 0.0, "buf_bytes": 0.0},
            SL_M: {"prb_req": 0.0, "prb_granted": 0.0, "slice_prb": float(cell.slice_dl_prb_quota.get(SL_M, 0))},
        }
        # Per-UE bitrate/buffer by slice
        for ue in cell.connected_ue_list.values():
            sl = getattr(ue, "slice_type", None)
            if sl == SL_E:
                agg[SL_E]["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
                agg[SL_E]["buf_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            elif sl == SL_U:
                agg[SL_U]["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
                agg[SL_U]["buf_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
        # mMTC PRB demand/grant
        req_map = getattr(cell, "dl_total_prb_demand", {}) or {}
        alloc_map = getattr(cell, "prb_ue_allocation_dict", {}) or {}
        for ue in cell.connected_ue_list.values():
            if getattr(ue, "slice_type", None) != SL_M:
                continue
            imsi = ue.ue_imsi
            agg[SL_M]["prb_req"] += float(req_map.get(imsi, 0) or 0.0)
            agg[SL_M]["prb_granted"] += float((alloc_map.get(imsi, {}) or {}).get("downlink", 0) or 0.0)
        return agg

    def _slice_scores(self, cell, T_s):
        """Compute per-slice scores in [0,1] based on current KPIs.

        Implements the Tractor paper’s formulas more literally:
        - eMBB: score = alpha * (beta + tx_bits - buf_bits), clipped to [0,1]
        - URLLC: score = (1/gamma) * max(0, gamma - (buf_bits / tx_bps)) with
                  the special case score=1 when both tx==0 and buf==0.
        - mMTC: utilization ratio with idle penalty (1/slice_prb).
        """
        kappa = 8.0  # bits per byte
        agg = self._aggregate_slice_metrics(cell)
        # eMBB: linear score with scaling + offset, then clipped to [0,1]
        tx_bits = max(0.0, agg[SL_E]["tx_mbps"]) * 1e6 * T_s  # Mbps -> bps * T
        buf_bits = max(0.0, agg[SL_E]["buf_bytes"]) * kappa   # bytes -> bits
        alpha = float(getattr(settings, "DQN_EMBB_ALPHA", 1.0))
        beta = float(getattr(settings, "DQN_EMBB_BETA", 0.0))
        embb_score = alpha * (beta + tx_bits - buf_bits)
        embb_score = max(0.0, min(1.0, float(embb_score)))

        # URLLC: queueing delay proxy
        tx_mbps_u = max(0.0, agg[SL_U]["tx_mbps"])    # Mbps
        buf_bytes_u = max(0.0, agg[SL_U]["buf_bytes"])  # bytes
        if tx_mbps_u <= 0.0 and buf_bytes_u <= 0.0:
            # Special case per paper: no RAN-induced delay
            urllc_score = 1.0
        elif tx_mbps_u <= 0.0 and buf_bytes_u > 0.0:
            # Non-zero queue and zero tx -> infinite delay -> score 0
            urllc_score = 0.0
        else:
            delay_s = (buf_bytes_u * kappa) / (tx_mbps_u * 1e6)
            gamma = max(1e-12, float(self.urlc_gamma_s))
            urllc_score = (1.0 / gamma) * max(0.0, gamma - delay_s)
            urllc_score = max(0.0, min(1.0, float(urllc_score)))

        # mMTC: utilization ratio / idle penalty
        prb_req = agg[SL_M]["prb_req"]
        prb_g = agg[SL_M]["prb_granted"]
        slice_prb = max(1.0, agg[SL_M]["slice_prb"])  # avoid div0
        if prb_req <= 0:
            mmtc_score = min(1.0, 1.0 / slice_prb)
        else:
            mmtc_score = min(1.0, prb_g / max(1.0, prb_req))
        return embb_score, urllc_score, mmtc_score

    def _reward(self, cell, T_s):
        """Combine slice scores using configured weights to a scalar reward."""
        e, u, m = self._slice_scores(cell, T_s)
        return float(self.w_e * e + self.w_u * u + self.w_m * m)

    def _act(self, state):
        """Epsilon‑greedy action selection (random with prob ε; otherwise argmax Q)."""
        eps = self._epsilon() if self.train_mode else 0.0
        if random.random() < eps or not TORCH_AVAILABLE:
            return random.randrange(self._n_actions)
        with torch.no_grad():
            x = torch.tensor([state], dtype=torch.float32)
            q = self._q(x)
            return int(torch.argmax(q, dim=1).item())

    def _opt_step(self):
        """One DQN optimization step over a replay mini‑batch.

        Returns the scalar loss value (float) when training occurs; otherwise None.
        """
        if not self.train_mode or not TORCH_AVAILABLE:
            return None
        if len(self._buf) < max(32, self.batch):
            return None
        s, a, r, ns, d = self._buf.sample(self.batch)
        q = self._q(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            qn = self._q_target(ns).max(1)[0]
            tgt = r + (1.0 - d) * self.gamma * qn
        loss = (q - tgt).pow(2).mean()
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        # Periodically refresh target network
        if self._t % 200 == 0:
            self._q_target.load_state_dict(self._q.state_dict())
            try:
                torch.save(self._q.state_dict(), self.model_path)
            except Exception:
                pass
        return float(loss.item())

    def _apply_action(self, cell, action: int):
        """Apply a discrete action as a PRB move between slice quotas for the cell."""
        # Map action to pair (src, dst)
        amap = {
            0: None,
            1: (SL_M, SL_U),
            2: (SL_M, SL_E),
            3: (SL_U, SL_M),
            4: (SL_U, SL_E),
            5: (SL_E, SL_M),
            6: (SL_E, SL_U),
        }
        mv = amap.get(action)
        if mv is None:
            return False
        src, dst = mv
        if hasattr(cell, "adjust_slice_quota_move_rb"):
            ok = cell.adjust_slice_quota_move_rb(src, dst, prb_step=self.move_step)
            if ok:
                self._action_counts[action] += 1
            return ok
        return False

    def _log(self, step_idx: int, cell_id: str, metrics: dict):
        """Emit metrics to TensorBoard and/or Weights & Biases."""
        # TensorBoard scalars per cell
        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(f"cell/{cell_id}/{k}", float(v), step_idx)
                except Exception:
                    pass
        # W&B
        if self._wandb is not None:
            try:
                import wandb
                self._wandb.log({f"cell/{cell_id}/{k}": v for k, v in metrics.items()}, step=step_idx)
            except Exception:
                pass

    def step(self):
        """Main loop: observe→(learn)→act at the configured decision period.

        - Early return on non‑decision steps to reduce overhead.
        - For each cell: compute new state s_t; if an (s_{t-1}, a_{t-1}) exists,
          compute reward r_t and push a transition (s_{t-1}, a_{t-1}, r_t, s_t).
        - Optionally optimize the network, then select and apply the next action.
        - Log per‑cell metrics and a periodic action histogram.
        """
        if not self.enabled:
            return
        sim_step = getattr(getattr(self.ric, "simulation_engine", None), "sim_step", 0)
        if sim_step % self.period_steps != 0:
            return
        self._t += 1
        T_s = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1)) * float(self.period_steps)

        for cell_id, cell in self.cell_list.items():
            # Compute state now (after environment step allocated PRBs)
            s_now = self._get_state(cell)

            # If we have a pending (s,a) from previous decision, compute reward and push transition
            prev = self._per_cell_prev.get(cell_id)
            if prev is not None:
                r = self._reward(cell, T_s)
                self._buf.push(prev["state"], prev["action"], r, s_now, 0.0)
                loss = self._opt_step()
                if loss is not None:
                    self._last_loss = loss
                # Per-slice components for visibility
                e, u, m = self._slice_scores(cell, T_s)
                # Quotas snapshot
                prb_map = getattr(cell, "slice_dl_prb_quota", {}) or {}
                prb_e = float(prb_map.get(SL_E, 0))
                prb_u = float(prb_map.get(SL_U, 0))
                prb_m = float(prb_map.get(SL_M, 0))
                # Emit logs
                self._log(self._t, cell_id, {
                    "reward": r,
                    "embb_score": e,
                    "urllc_score": u,
                    "mmtc_score": m,
                    "epsilon": self._epsilon() if self.train_mode else 0.0,
                    "loss": self._last_loss if self._last_loss is not None else 0.0,
                    "prb_eMBB": prb_e,
                    "prb_URLLC": prb_u,
                    "prb_mMTC": prb_m,
                    "action_prev": int(prev["action"]),
                })

            # Choose and apply new action for next period
            a = self._act(s_now)
            self._apply_action(cell, a)
            self._per_cell_prev[cell_id] = {"state": s_now, "action": a}

        # Log action histogram occasionally
        if self._tb is not None and self._t % 50 == 0:
            try:
                import numpy as np
                import torch as _torch
                counts = [self._action_counts.get(i, 0) for i in range(self._n_actions)]
                self._tb.add_histogram("actions/hist", _torch.tensor(counts, dtype=_torch.float32), self._t)
            except Exception:
                pass

    def __del__(self):
        """Best‑effort cleanup of telemetry handles at interpreter shutdown."""
        try:
            if self._tb is not None:
                self._tb.flush(); self._tb.close()
        except Exception:
            pass
        try:
            if self._wandb is not None:
                self._wandb.finish()
        except Exception:
            pass

    def to_json(self):
        """Expose a compact JSON for UI/knowledge endpoints."""
        j = super().to_json()
        j.update({
            "train_mode": self.train_mode,
            "period_steps": self.period_steps,
            "move_step": self.move_step,
            "enabled": self.enabled,
        })
        return j
