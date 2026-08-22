# ==========================================
# TwinEV v2 - Experiment 8
# SERVICE-CONSTRAINED TwinEV environment
# ==========================================
# Adds an explicit EV service (energy-delivery)
# constraint on top of the ORIGINAL TwinEVEnv.
# The original environment is NOT modified.
#
# Mechanism (smallest mathematically defensible):
#   energy_service_ratio = cumulative_delivered / cumulative_available
#     delivered  = sum of actual grid power delivered (kWh, hourly)
#     available  = sum of raw EV charging demand (grid_load, kWh)
#   - per-step penalty: -W * max(0, target - running_ratio)
#   - terminal bonus when the episode-level target is met
#   - running ratio appended to the observation so the
#     policy can condition on its service progress
#
# Grid safety is NOT removed: the PI controller still
# clamps output to [0, MAX_GRID_KW] and the original
# grid-reward penalty is retained unmodified.
# ==========================================

import numpy as np
from gymnasium import spaces

from twinev_env import TwinEVEnv


class ServiceConstrainedEnv(TwinEVEnv):

    def __init__(self, service_target=0.70,
                 service_penalty_weight=50.0,
                 terminal_bonus=10.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_target = float(service_target)
        self.service_penalty_weight = float(service_penalty_weight)
        self.terminal_bonus = float(terminal_bonus)

        # Observation: original 7 features + running service ratio
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

        self.ep_delivered = 0.0
        self.ep_available = 0.0

    # ------------------------------------------
    # RESET
    # ------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.ep_delivered = 0.0
        self.ep_available = 0.0
        idx = self.episode_start
        return self._get_obs(idx), {}

    # ------------------------------------------
    # STEP - accumulate available demand first
    # ------------------------------------------

    def step(self, action):
        idx = self.episode_start + self.current_step
        self.ep_available += float(self.grid_load[idx])
        return super().step(action)

    # ------------------------------------------
    # OBSERVATION - append running service ratio
    # ------------------------------------------

    def _get_obs(self, idx):
        base = super()._get_obs(idx)
        ratio = 0.0
        if self.ep_available > 0:
            ratio = float(np.clip(
                self.ep_delivered / self.ep_available, 0.0, 1.0))
        return np.concatenate([
            base, np.array([ratio], dtype=np.float32)
        ]).astype(np.float32)

    # ------------------------------------------
    # REWARD - original + service deficit term
    # ------------------------------------------

    def _compute_reward(self, actual_power, setpoint_kw, grid_load,
                        solar_gen, energy_price, queue_len,
                        lstm_forecast):
        self.ep_delivered += float(actual_power)

        reward, info = super()._compute_reward(
            actual_power=actual_power,
            setpoint_kw=setpoint_kw,
            grid_load=grid_load,
            solar_gen=solar_gen,
            energy_price=energy_price,
            queue_len=queue_len,
            lstm_forecast=lstm_forecast,
        )

        ratio = self.ep_delivered / (self.ep_available + 1e-9)
        deficit = max(0.0, self.service_target - ratio)
        r_service = -self.service_penalty_weight * deficit

        if self.current_step >= self.max_steps - 1:
            if ratio >= self.service_target:
                r_service += self.terminal_bonus
            else:
                r_service -= self.terminal_bonus * (
                    deficit / max(self.service_target, 1e-9))

        total = reward + r_service
        info["r_service"] = round(r_service, 4)
        info["service_ratio"] = round(float(ratio), 4)
        info["total_reward"] = round(float(total), 4)
        return float(total), info


if __name__ == "__main__":
    env = ServiceConstrainedEnv(mode="val", service_target=0.70)
    obs, _ = env.reset()
    print("Obs dim:", obs.shape, "ratio:", obs[-1])
    total = 0.0
    for _ in range(env.max_steps):
        obs, r, done, _, info = env.step(env.action_space.sample())
        total += r
        if done:
            break
    print("Episode reward:", round(total, 2),
          "final ratio:", info["service_ratio"])
    print("Service-constrained env OK.")
