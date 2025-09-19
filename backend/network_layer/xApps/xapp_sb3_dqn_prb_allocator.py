"""Stable-Baselines3 DQN-based PRB allocator xApp.

This module mirrors the behaviour of ``xapp_dqn_prb_allocator`` but delegates the
value function approximation and optimisation loops to Stable-Baselines3's DQN
implementation. The goal is to offer a drop-in alternative that allows
cross-checking policies trained with SB3 against the custom PyTorch variant.

If Stable-Baselines3 (and its Gymnasium dependency) are not available at
runtime the xApp disables itself gracefully.
"""

import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import numpy as np

import settings

from .xapp_base import xAppBase

try:
    from gymnasium import Env, spaces
    from stable_baselines3 import DQN as SB3DQN
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False
    spaces = None  # type: ignore[assignment]

# Reuse slice labels from settings (keep fallbacks identical to the original xApp)
SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")
SL_M = getattr(settings, "NETWORK_SLICE_MTC_NAME", "mMTC")


def _linear_eps(start: float, end: float, decay_steps: int, step: int) -> float:
    """Utility: linearly interpolate epsilon for exploration."""
    if decay_steps <= 0:
        return end
    frac = min(1.0, step / float(decay_steps))
    return start + (end - start) * frac


if SB3_AVAILABLE:

    class _StaticPRBEnv(Env):
        """Minimal Gymnasium env used solely to initialise SB3's DQN structures."""

        metadata = {"render_modes": []}

        def __init__(self, obs_dim: int, n_actions: int):
            super().__init__()
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(n_actions)

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):  # type: ignore[override]
            if seed is not None:
                super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):  # type: ignore[override]
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = True
            truncated = False
            info: Dict = {}
            return obs, reward, terminated, truncated, info

else:

    class _StaticPRBEnv:  # type: ignore[too-few-public-methods]
        """Placeholder so type checkers know the symbol exists when SB3 is missing."""

        def __init__(self, *_, **__):
            raise RuntimeError("Stable-Baselines3 is required for xAppSB3DQNPRBAllocator")


class xAppSB3DQNPRBAllocator(xAppBase):
    """PRB allocator powered by Stable-Baselines3 DQN."""

    def __init__(self, ric=None):
        super().__init__(ric=ric)

        self.enabled = getattr(settings, "SB3_DQN_PRB_ENABLE", False)
        self.train_mode = getattr(settings, "DQN_PRB_TRAIN", True)
        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))

        # Reward weights mirror the custom DQN xApp
        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.33))
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.34))
        self.w_m = float(getattr(settings, "DQN_WEIGHT_MMTC", 0.33))
        self.urlc_gamma_s = float(getattr(settings, "DQN_URLLC_GAMMA_S", 0.01))

        # Hyperparameters shared with the PyTorch implementation for easy comparison
        self.gamma = float(getattr(settings, "DQN_PRB_GAMMA", 0.99))
        self.lr = float(getattr(settings, "DQN_PRB_LR", 1e-3))
        self.batch = int(getattr(settings, "DQN_PRB_BATCH", 64))
        self.buffer_cap = int(getattr(settings, "DQN_PRB_BUFFER", 50_000))
        self.eps_start = float(getattr(settings, "DQN_PRB_EPSILON_START", 1.0))
        self.eps_end = float(getattr(settings, "DQN_PRB_EPSILON_END", 0.1))
        self.eps_decay = int(getattr(settings, "DQN_PRB_EPSILON_DECAY", 10_000))

        default_model_path = getattr(settings, "DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")
        self.model_path = getattr(settings, "SB3_DQN_MODEL_PATH", default_model_path.replace(".pt", "_sb3.zip"))
        self.sb3_total_steps = int(getattr(settings, "SB3_DQN_TOTAL_STEPS", 100_000))
        self.sb3_target_update = int(getattr(settings, "SB3_DQN_TARGET_UPDATE", 1_000))
        self.sb3_log_interval = int(getattr(settings, "SB3_DQN_SAVE_INTERVAL", 5_000))

        self._t = 0
        self._per_cell_prev: Dict[str, Dict] = {}
        self._action_counts = defaultdict(int)

        self._state_dim = 5
        self._n_actions = 7

        self._sb3_env = None
        self._model: Optional[SB3DQN] = None
        self._tb = None
        self._wandb = None

        if not self.enabled:
            return

        if not SB3_AVAILABLE:
            print(f"{self.xapp_id}: Stable-Baselines3 not available; disabling xApp.")
            self.enabled = False
            return

        # Optional TensorBoard support (re-using the same flag names as the custom xApp)
        if getattr(settings, "DQN_TB_ENABLE", False):
            try:
                from torch.utils.tensorboard import SummaryWriter

                base = getattr(settings, "DQN_TB_DIR", "backend/tb_logs")
                run = datetime.now().strftime("%Y%m%d_%H%M%S")
                logdir = os.path.join(base, f"sb3_dqn_prb_{run}")
                os.makedirs(logdir, exist_ok=True)
                self._tb = SummaryWriter(log_dir=logdir)
                print(f"{self.xapp_id}: TensorBoard logging to {logdir}")
            except Exception as exc:
                print(f"{self.xapp_id}: TensorBoard unavailable ({exc}); continuing without it.")
                self._tb = None

        if getattr(settings, "DQN_WANDB_ENABLE", False):
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
                    "sb3_target_update": self.sb3_target_update,
                }
                proj = getattr(settings, "DQN_WANDB_PROJECT", "ai-ran-dqn")
                name = getattr(settings, "DQN_WANDB_RUNNAME", "") or None
                self._wandb = wandb.init(project=proj, name=name, config=cfg)
                print(f"{self.xapp_id}: W&B logging enabled (project={proj})")
            except Exception as exc:
                print(f"{self.xapp_id}: W&B unavailable ({exc}); continuing without it.")
                self._wandb = None

        # Build a trivial Gym env so SB3 can instantiate its policy/Q-networks
        def _make_env():
            return _StaticPRBEnv(self._state_dim, self._n_actions)

        self._sb3_env = DummyVecEnv([_make_env])

        policy_kwargs = {"net_arch": [64, 64]}
        self._model = SB3DQN(
            policy="MlpPolicy",
            env=self._sb3_env,
            learning_rate=self.lr,
            buffer_size=self.buffer_cap,
            learning_starts=0,
            batch_size=self.batch,
            gamma=self.gamma,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=max(1, self.sb3_target_update),
            exploration_fraction=1.0,  # we control epsilon manually
            exploration_initial_eps=self.eps_start,
            exploration_final_eps=self.eps_end,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=None,
        )

        # Try loading existing parameters
        try:
            if os.path.exists(self.model_path):
                self._model = SB3DQN.load(self.model_path, env=self._sb3_env)
                print(f"{self.xapp_id}: loaded SB3 model from {self.model_path}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to load SB3 model ({exc}); starting fresh.")

    # ---------------- Lifecycle ----------------
    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        mode = "train" if self.train_mode else "eval"
        print(f"{self.xapp_id}: enabled (mode={mode}, period={self.period_steps} steps)")

    # ---------------- Helper utilities (shared with custom DQN xApp) ----------------
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
        max_ue = max(1.0, float(getattr(settings, "UE_DEFAULT_MAX_COUNT", 50)))
        return [
            n_m / max_ue,
            n_u / max_ue,
            n_e / max_ue,
            prb_m / float(max(1, cell.max_dl_prb)),
            prb_u / float(max(1, cell.max_dl_prb)),
        ]

    def _aggregate_slice_metrics(self, cell):
        agg = {
            SL_E: {"tx_mbps": 0.0, "buf_bytes": 0.0},
            SL_U: {"tx_mbps": 0.0, "buf_bytes": 0.0},
            SL_M: {
                "prb_req": 0.0,
                "prb_granted": 0.0,
                "slice_prb": float(getattr(cell, "slice_dl_prb_quota", {}).get(SL_M, 0)),
            },
        }
        for ue in cell.connected_ue_list.values():
            sl = getattr(ue, "slice_type", None)
            if sl == SL_E:
                agg[SL_E]["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
                agg[SL_E]["buf_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            elif sl == SL_U:
                agg[SL_U]["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
                agg[SL_U]["buf_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
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
        kappa = 8.0
        agg = self._aggregate_slice_metrics(cell)

        tx_bits = max(0.0, agg[SL_E]["tx_mbps"]) * 1e6 * T_s
        buf_bits = max(0.0, agg[SL_E]["buf_bytes"]) * kappa
        alpha = float(getattr(settings, "DQN_EMBB_ALPHA", 1.0))
        beta = float(getattr(settings, "DQN_EMBB_BETA", 0.0))
        embb_score = alpha * (beta + tx_bits - buf_bits)
        embb_score = max(0.0, min(1.0, float(embb_score)))

        tx_mbps_u = max(0.0, agg[SL_U]["tx_mbps"])
        buf_bytes_u = max(0.0, agg[SL_U]["buf_bytes"])
        if tx_mbps_u <= 0.0 and buf_bytes_u <= 0.0:
            urllc_score = 1.0
        elif tx_mbps_u <= 0.0 and buf_bytes_u > 0.0:
            urllc_score = 0.0
        else:
            delay_s = (buf_bytes_u * kappa) / (tx_mbps_u * 1e6)
            gamma = max(1e-12, float(self.urlc_gamma_s))
            urllc_score = (1.0 / gamma) * max(0.0, gamma - delay_s)
            urllc_score = max(0.0, min(1.0, float(urllc_score)))

        prb_req = agg[SL_M]["prb_req"]
        prb_g = agg[SL_M]["prb_granted"]
        slice_prb = max(1.0, agg[SL_M]["slice_prb"])
        if prb_req <= 0:
            mmtc_score = min(1.0, 1.0 / slice_prb)
        else:
            mmtc_score = min(1.0, prb_g / max(1.0, prb_req))
        return embb_score, urllc_score, mmtc_score

    def _reward(self, cell, T_s):
        e, u, m = self._slice_scores(cell, T_s)
        return float(self.w_e * e + self.w_u * u + self.w_m * m)

    def _select_action(self, obs_vec: np.ndarray):
        if not self._model:
            return 0
        if self.train_mode:
            eps = _linear_eps(self.eps_start, self.eps_end, self.eps_decay, self._t)
            if random.random() < eps:
                return int(self._sb3_env.action_space.sample())  # type: ignore[union-attr]
        action, _ = self._model.predict(obs_vec, deterministic=True)
        return int(action)

    def _apply_action(self, cell, action: int):
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
        sim_step = getattr(getattr(self.ric, "simulation_engine", None), "sim_step", None)
        if mv is None:
            try:
                setattr(cell, "rl_last_action", {
                    "actor": "DQN",
                    "source": "SB3",
                    "code": int(action),
                    "label": "keep",
                    "moved": 0,
                    "src": None,
                    "dst": None,
                    "step": int(sim_step) if sim_step is not None else None,
                    "quota": dict(getattr(cell, "slice_dl_prb_quota", {}) or {}),
                })
            except Exception:
                pass
            return False
        src, dst = mv
        if hasattr(cell, "adjust_slice_quota_move_rb"):
            ok = cell.adjust_slice_quota_move_rb(src, dst, prb_step=self.move_step)
            if ok:
                self._action_counts[action] += 1
                try:
                    setattr(cell, "rl_last_action", {
                        "actor": "DQN",
                        "source": "SB3",
                        "code": int(action),
                        "label": f"{src}→{dst}",
                        "moved": int(self.move_step),
                        "src": src,
                        "dst": dst,
                        "step": int(sim_step) if sim_step is not None else None,
                        "quota": dict(getattr(cell, "slice_dl_prb_quota", {}) or {}),
                    })
                except Exception:
                    pass
            return ok
        return False

    # ---------------- Main control loop ----------------
    def step(self):
        if not self.enabled or not self._model:
            return

        sim_engine = getattr(self.ric, "simulation_engine", None)
        sim_step = getattr(sim_engine, "sim_step", 0)
        if sim_step % self.period_steps != 0:
            return

        dt = getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0)

        for cell in self.cell_list.values():
            obs = np.array(self._get_state(cell), dtype=np.float32)
            prev = self._per_cell_prev.get(cell.cell_id)

            if prev is not None and self.train_mode:
                reward = self._reward(cell, dt * self.period_steps)
                obs_prev = np.array(prev["obs"], dtype=np.float32)
                action_prev = int(prev["action"])

                obs_batch = obs_prev.reshape(1, -1)
                next_obs_batch = obs.reshape(1, -1)
                action_batch = np.array([[action_prev]], dtype=np.int64)
                reward_batch = np.array([reward], dtype=np.float32)
                done_batch = np.array([0.0], dtype=np.float32)

                try:
                    self._model.replay_buffer.add(
                        obs_batch,
                        next_obs_batch,
                        action_batch,
                        reward_batch,
                        done_batch,
                        infos=[{}],
                    )
                    if self._model.replay_buffer.size() >= max(32, self.batch):
                        self._model.train(gradient_steps=1, batch_size=self.batch)
                    progress = max(0.0, 1.0 - self._t / max(1, self.sb3_total_steps))
                    self._model._current_progress_remaining = progress
                    self._model._on_step()
                except Exception as exc:
                    print(f"{self.xapp_id}: SB3 training error: {exc}")

            action = self._select_action(obs)
            self._apply_action(cell, action)
            self._per_cell_prev[cell.cell_id] = {"obs": obs, "action": action}

            # Optional TB logging for visibility
            if self._tb is not None:
                try:
                    metrics = self._aggregate_slice_metrics(cell)
                    self._tb.add_scalar(f"cell/{cell.cell_id}/embb_buf_bytes", metrics[SL_E]["buf_bytes"], self._t)
                    self._tb.add_scalar(f"cell/{cell.cell_id}/urllc_buf_bytes", metrics[SL_U]["buf_bytes"], self._t)
                except Exception:
                    pass
            if self._wandb is not None:
                try:
                    import wandb

                    metrics = self._aggregate_slice_metrics(cell)
                    self._wandb.log(
                        {
                            f"cell/{cell.cell_id}/embb_buf_bytes": metrics[SL_E]["buf_bytes"],
                            f"cell/{cell.cell_id}/urllc_buf_bytes": metrics[SL_U]["buf_bytes"],
                        },
                        step=self._t,
                    )
                except Exception:
                    pass

        self._t += 1

        if (
            self.train_mode
            and self.sb3_log_interval > 0
            and self._t % self.sb3_log_interval == 0
            and self._model is not None
        ):
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self._model.save(self.model_path)
                print(f"{self.xapp_id}: saved SB3 model to {self.model_path}")
            except Exception as exc:
                print(f"{self.xapp_id}: failed to save SB3 model ({exc})")

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "sb3_enabled": SB3_AVAILABLE,
                "train_mode": self.train_mode,
                "buffer_size": self.buffer_cap,
                "period_steps": self.period_steps,
                "move_step": self.move_step,
            }
        )
        return data
