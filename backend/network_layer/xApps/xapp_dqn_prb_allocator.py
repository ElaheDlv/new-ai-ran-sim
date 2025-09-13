from .xapp_base import xAppBase

import os
import math
import random
from collections import deque, defaultdict

import settings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    # Do not crash if torch is missing; xApp will disable itself.
    TORCH_AVAILABLE = False


SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")
SL_M = getattr(settings, "NETWORK_SLICE_MTC_NAME", "mMTC")


if TORCH_AVAILABLE:
    class _ReplayBuffer:
        def __init__(self, capacity: int = 50000):
            self.buf = deque(maxlen=int(capacity))

        def push(self, s, a, r, ns, d):
            self.buf.append((s, a, r, ns, d))

        def sample(self, batch):
            import numpy as np

            batch = min(batch, len(self.buf))
            idx = np.random.choice(len(self.buf), batch, replace=False)
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

    - State (per cell): [#mMTC UEs, #URLLC UEs, #eMBB UEs, PRBs_mMTC, PRBs_URLLC]
      (PRBs in eMBB implied by max_dl_prb - others)
    - Actions: 0 keep; 1 mMTC→URLLC; 2 mMTC→eMBB; 3 URLLC→mMTC; 4 URLLC→eMBB; 5 eMBB→mMTC; 6 eMBB→URLLC
    - Reward: weighted sum of per-slice scores described in the Tractor paper.
    """

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = getattr(settings, "DQN_PRB_ENABLE", False)

        # Runtime / training knobs
        self.train_mode = getattr(settings, "DQN_PRB_TRAIN", True)
        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))

        # Reward shaping/weights
        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.33))
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.34))
        self.w_m = float(getattr(settings, "DQN_WEIGHT_MMTC", 0.33))
        self.urlc_gamma_s = float(getattr(settings, "DQN_URLLC_GAMMA_S", 0.01))

        # DQN parameters
        self.gamma = float(getattr(settings, "DQN_PRB_GAMMA", 0.99))
        self.lr = float(getattr(settings, "DQN_PRB_LR", 1e-3))
        self.batch = int(getattr(settings, "DQN_PRB_BATCH", 64))
        self.buffer_cap = int(getattr(settings, "DQN_PRB_BUFFER", 50000))
        self.eps_start = float(getattr(settings, "DQN_PRB_EPSILON_START", 1.0))
        self.eps_end = float(getattr(settings, "DQN_PRB_EPSILON_END", 0.1))
        self.eps_decay = int(getattr(settings, "DQN_PRB_EPSILON_DECAY", 10000))
        self.model_path = getattr(settings, "DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")

        # Internal state
        self._t = 0
        self._per_cell_prev = {}  # cell_id -> {state, action}

        # NN
        self._n_actions = 7
        self._state_dim = 5
        self._device = torch.device("cpu") if TORCH_AVAILABLE else None
        if self.enabled and not TORCH_AVAILABLE:
            print(f"{self.xapp_id}: torch not available; disabling.")
            self.enabled = False

        if self.enabled:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self._q = _DQN(self._state_dim, self._n_actions).to(self._device)
            self._q_target = _DQN(self._state_dim, self._n_actions).to(self._device)
            self._q_target.load_state_dict(self._q.state_dict())
            self._opt = optim.Adam(self._q.parameters(), lr=self.lr)
            self._buf = _ReplayBuffer(self.buffer_cap)
            # Try to load existing model
            try:
                if os.path.exists(self.model_path):
                    self._q.load_state_dict(torch.load(self.model_path, map_location=self._device))
                    self._q_target.load_state_dict(self._q.state_dict())
                    print(f"{self.xapp_id}: loaded model from {self.model_path}")
            except Exception as e:
                print(f"{self.xapp_id}: failed loading model: {e}")

    # ---------------- xApp lifecycle ----------------
    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        print(f"{self.xapp_id}: enabled (train={self.train_mode}, period={self.period_steps} steps)")

    def _epsilon(self):
        # Linear decay
        if self.eps_decay <= 0:
            return self.eps_end
        frac = min(1.0, self._t / float(self.eps_decay))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def _get_slice_counts(self, cell):
        cnt = {SL_E: 0, SL_U: 0, SL_M: 0}
        for ue in cell.connected_ue_list.values():
            s = getattr(ue, "slice_type", None)
            if s in cnt:
                cnt[s] += 1
        return cnt

    def _get_state(self, cell):
        cnt = self._get_slice_counts(cell)
        prb_map = getattr(cell, "slice_dl_prb_quota", {}) or {}
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
        """Return per-slice aggregates used to compute rewards."""
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
        """Compute per-slice scores in [0,1] based on current KPIs."""
        kappa = 8.0  # bits per byte
        agg = self._aggregate_slice_metrics(cell)
        # eMBB: drain queue (scaled to [0,1])
        # score_bits = beta + tx_brate*1e6*T - dl_buffer*8; map to [0,1] via sigmoid-like squash
        tx_bits = max(0.0, agg[SL_E]["tx_mbps"]) * 1e6 * T_s
        buf_bits = max(0.0, agg[SL_E]["buf_bytes"]) * kappa
        embb_raw = tx_bits - buf_bits
        # squash to [0,1] using tanh-like mapping with scale
        scale_bits = 1e7  # tune
        embb_score = 0.5 * (math.tanh(embb_raw / max(1.0, scale_bits)) + 1.0)

        # URLLC: queueing delay proxy
        tx_mbps_u = max(0.0, agg[SL_U]["tx_mbps"])  # Mbps
        buf_bytes_u = max(0.0, agg[SL_U]["buf_bytes"])  # bytes
        if tx_mbps_u <= 1e-9 and buf_bytes_u <= 1.0:
            urllc_score = 1.0
        else:
            delay_s = (buf_bytes_u * kappa) / max(1e-6, tx_mbps_u * 1e6)
            urllc_score = max(0.0, min(1.0, (self.urlc_gamma_s - min(self.urlc_gamma_s, delay_s)) / self.urlc_gamma_s))

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
        e, u, m = self._slice_scores(cell, T_s)
        return float(self.w_e * e + self.w_u * u + self.w_m * m)

    def _act(self, state):
        eps = self._epsilon() if self.train_mode else 0.0
        if random.random() < eps or not TORCH_AVAILABLE:
            return random.randrange(self._n_actions)
        with torch.no_grad():
            x = torch.tensor([state], dtype=torch.float32)
            q = self._q(x)
            return int(torch.argmax(q, dim=1).item())

    def _opt_step(self):
        if not self.train_mode or not TORCH_AVAILABLE:
            return
        if len(self._buf) < max(32, self.batch):
            return
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

    def _apply_action(self, cell, action: int):
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
            return cell.adjust_slice_quota_move_rb(src, dst, prb_step=self.move_step)
        return False

    def step(self):
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
                self._opt_step()

            # Choose and apply new action for next period
            a = self._act(s_now)
            self._apply_action(cell, a)
            self._per_cell_prev[cell_id] = {"state": s_now, "action": a}

    def to_json(self):
        j = super().to_json()
        j.update({
            "train_mode": self.train_mode,
            "period_steps": self.period_steps,
            "move_step": self.move_step,
            "enabled": self.enabled,
        })
        return j
