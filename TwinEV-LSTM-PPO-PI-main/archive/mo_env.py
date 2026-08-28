# ==========================================
# TwinEV v2 - EXPERIMENT 10
# FORECAST-INFORMED MULTI-OBJECTIVE ENV
# ==========================================
# New, clearly documented multi-objective reward
# with normalized/bounded components.  Subclasses
# TwinEVEnv so the original environment, PI
# controller, action space and observation are
# unchanged except for the reward and (optionally)
# the forecast feature in the state.
#
# Reward components (each bounded, comparable):
#   r_service  = tanh(3 * served_fraction)          [0, 1)   diminishing returns
#   r_cost     = -price_norm * power_norm           [-1, 0]
#   r_solar    = min(power, solar)/SOLAR_MAX        [0, 1]
#   r_grid     = soft/hard overload penalty         [-2, 0]  (safety critical)
#   r_smooth   = -|P_t - P_{t-1}| / MAX_GRID_KW     [-1, 0]
#   r_fc       = -fc_pressure * power_norm          [-1, 0]  forecast-aware
#
# Weights: service 1.0, cost 1.0, solar 1.0, grid 2.0, smooth 1.0, fc 1.0
# ==========================================

import numpy as np
from gymnasium import spaces

from twinev_env import (
    TwinEVEnv,
    MAX_GRID_KW,
    OVERLOAD_KW,
    MAX_PRICE,
    SOLAR_MAX_KW,
)

MO_WEIGHTS = {
    "service": 1.5,
    "cost": 1.5,
    "solar": 1.0,
    "grid": 2.0,
    "smooth": 1.0,
    "forecast": 1.0,
}


class MOEnv(TwinEVEnv):
    """Multi-objective environment with optional components.

    use_forecast   : include the forecast feature in the observation AND
                     include the forecast-aware reward component
    use_cost       : include the electricity-cost penalty
    use_solar      : include the solar-utilization reward
    use_smoothness : include the power-smoothness penalty
    """

    def __init__(self, use_forecast=True, use_cost=True, use_solar=True,
                 use_smoothness=True, mo_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_forecast = bool(use_forecast)
        self.use_cost = bool(use_cost)
        self.use_solar = bool(use_solar)
        self.use_smoothness = bool(use_smoothness)
        self.MO_W = dict(MO_WEIGHTS) if mo_weights is None else dict(mo_weights)

        n = 7 if self.use_forecast else 6
        self.observation_space = spaces.Box(
            low=np.zeros(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32),
            dtype=np.float32,
        )

        self._prev_power = None

    # ------------------------------------------
    # RESET
    # ------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_power = None
        return self._get_obs(self.episode_start), {}

    # ------------------------------------------
    # STEP - capture previous actual power
    # ------------------------------------------

    def step(self, action):
        if self._prev_power is None:
            self._prev_power = self.actual_power
        return super().step(action)

    # ------------------------------------------
    # OBSERVATION - optionally drop forecast
    # ------------------------------------------

    def _get_obs(self, idx):
        full = super()._get_obs(idx)
        if self.use_forecast:
            return full.astype(np.float32)
        return np.concatenate([full[:4], full[5:]]).astype(np.float32)

    # ------------------------------------------
    # MULTI-OBJECTIVE REWARD
    # ------------------------------------------

    def _compute_reward(self, actual_power, setpoint_kw, grid_load,
                        solar_gen, energy_price, queue_len, lstm_forecast):
        W = self.MO_W
        p = float(actual_power)
        p_norm = p / MAX_GRID_KW

        # 1. SERVICE (diminishing returns on demand served)
        demand = max(float(grid_load), 1e-3)
        served = min(p, float(grid_load)) / demand          # [0,1]
        r_service = float(np.tanh(3.0 * served))            # [0,1)

        # 2. COST penalty
        price_norm = float(energy_price) / MAX_PRICE        # [0,1]
        r_cost = -price_norm * p_norm                       # [-1,0]

        # 3. SOLAR utilization reward
        r_solar = float(min(p, float(solar_gen)) / SOLAR_MAX_KW)  # [0,1]

        # 4. GRID SAFETY penalty (hard > soft; no drag below 125 kW)
        if p > MAX_GRID_KW:
            r_grid = -3.0
        elif p > 125.0:
            r_grid = -1.0 * (p - 125.0) / (MAX_GRID_KW - 125.0)   # [0,-1]
        else:
            r_grid = 0.0

        # 5. SMOOTHNESS penalty
        if self._prev_power is not None:
            r_smooth = -min(1.0, abs(p - self._prev_power) / MAX_GRID_KW)
        else:
            r_smooth = 0.0

        # 6. FORECAST-AWARE component (bonus when forecast low,
        #    penalty when forecast high -> genuine anticipation)
        fc_pressure = float(np.clip(
            (float(lstm_forecast) - 90.0) / (MAX_GRID_KW - 90.0),
            0.0, 1.0))                                       # [0,1]
        r_fc = p_norm * (0.5 - fc_pressure)                  # [-0.5,+0.5]

        total = (
            W["service"] * r_service
            + (W["cost"] * r_cost if self.use_cost else 0.0)
            + (W["solar"] * r_solar if self.use_solar else 0.0)
            + W["grid"] * r_grid
            + (W["smooth"] * r_smooth if self.use_smoothness else 0.0)
            + (W["forecast"] * r_fc if self.use_forecast else 0.0)
        )

        info = {
            "r_service": round(r_service, 4),
            "r_cost": round(r_cost, 4),
            "r_solar": round(r_solar, 4),
            "r_grid": round(r_grid, 4),
            "r_smooth": round(r_smooth, 4),
            "r_fc": round(r_fc, 4),
            "total_reward": round(float(total), 4),
            "actual_power": round(p, 2),
            "setpoint_kw": round(setpoint_kw, 2),
        }
        return float(total), info


# --------------------------------------------------
# ABLATION SUBCLASSES
# --------------------------------------------------

class MOFullEnv(MOEnv):
    """service + cost + solar + grid + smoothness + forecast"""

    def __init__(self, *args, **kwargs):
        super().__init__(use_forecast=True, use_cost=True,
                         use_solar=True, use_smoothness=True,
                         *args, **kwargs)


class MONoForecastEnv(MOEnv):
    """service + cost + solar + grid + smoothness  (no forecast)"""

    def __init__(self, *args, **kwargs):
        super().__init__(use_forecast=False, use_cost=True,
                         use_solar=True, use_smoothness=True,
                         *args, **kwargs)


class MONoCostEnv(MOEnv):
    """service + solar + grid + smoothness + forecast  (no cost)"""

    def __init__(self, *args, **kwargs):
        super().__init__(use_forecast=True, use_cost=False,
                         use_solar=True, use_smoothness=True,
                         *args, **kwargs)


class MONoSolarEnv(MOEnv):
    """service + cost + grid + smoothness + forecast  (no solar)"""

    def __init__(self, *args, **kwargs):
        super().__init__(use_forecast=True, use_cost=True,
                         use_solar=False, use_smoothness=True,
                         *args, **kwargs)


class MONoSmoothnessEnv(MOEnv):
    """service + cost + solar + grid + forecast  (no smoothness)"""

    def __init__(self, *args, **kwargs):
        super().__init__(use_forecast=True, use_cost=True,
                         use_solar=True, use_smoothness=False,
                         *args, **kwargs)


# --------------------------------------------------
# SANITY CHECK
# --------------------------------------------------

if __name__ == "__main__":
    env = MOFullEnv(mode="val")
    obs, _ = env.reset()
    print("Obs dim:", obs.shape)
    totals = {"r_service": [], "r_cost": [], "r_solar": [],
              "r_grid": [], "r_smooth": [], "r_fc": []}
    for _ in range(168):
        obs, r, done, _, info = env.step(env.action_space.sample())
        for k in totals:
            totals[k].append(info[k])
        if done:
            break
    print("\nComponent ranges over one random episode:")
    for k, v in totals.items():
        print(f"  {k:10s} mean={np.mean(v):+.3f} min={np.min(v):+.3f} "
              f"max={np.max(v):+.3f}")
    print("\nMO env OK.")
