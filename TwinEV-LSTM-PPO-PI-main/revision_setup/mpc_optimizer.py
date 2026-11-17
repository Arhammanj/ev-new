"""Receding-horizon MPC optimizer for ACNChargingEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog


class ForecastLookupError(KeyError):
    pass


@dataclass
class MPCResult:
    action_kw: np.ndarray
    success: bool
    status: int
    message: str
    objective: float
    horizon_steps: int
    reserve_kw: np.ndarray
    future_demand_kwh: np.ndarray


class StepwiseForecastLookup:
    """Timestamp lookup for a six-step future-demand artifact."""

    def __init__(
        self,
        csv_path: str | Path,
        horizon: int = 6,
        clip_negative: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        frame = pd.read_csv(self.csv_path)
        required = {"timestamp"} | {f"predicted_h{i}" for i in range(1, horizon + 1)}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        if timestamps.duplicated().any():
            raise ValueError(f"{self.csv_path} contains duplicate timestamps")
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"{self.csv_path} timestamps are not sorted")
        values = frame[[f"predicted_h{i}" for i in range(1, horizon + 1)]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{self.csv_path} contains non-finite forecasts")
        if clip_negative:
            values = np.maximum(values, 0.0)
        self.horizon = horizon
        self._by_time = {
            pd.Timestamp(timestamp): values[index]
            for index, timestamp in enumerate(timestamps)
        }
        self.first_valid_time = pd.Timestamp(timestamps.min())
        self.last_valid_time = pd.Timestamp(timestamps.max())

    def __call__(self, timestamp: pd.Timestamp) -> np.ndarray:
        key = pd.Timestamp(timestamp)
        values = self._by_time.get(key)
        if values is None:
            raise ForecastLookupError(
                f"no stepwise forecast for {key}; valid range is "
                f"[{self.first_valid_time}, {self.last_valid_time}]"
            )
        return values.copy()


def solve_mpc_action(
    present: np.ndarray,
    remaining_kwh: np.ndarray,
    departure_step: np.ndarray,
    step_idx: int,
    dt_hours: float,
    per_ev_max_kw: float,
    capacity_kw: float,
    future_demand_kwh: np.ndarray,
    horizon_hours: int = 6,
    episode_len: Optional[int] = None,
    capacity_margin_kw: float = 1e-4,
    unmet_penalty: float = 1e6,
) -> MPCResult:
    present = np.asarray(present, dtype=bool)
    remaining = np.asarray(remaining_kwh, dtype=float)
    deadlines = np.asarray(departure_step, dtype=int)
    if present.shape != remaining.shape or present.shape != deadlines.shape:
        raise ValueError("present, remaining_kwh, and departure_step must have equal shape")
    if not np.isfinite(remaining).all() or np.any(remaining < 0):
        raise ValueError("remaining_kwh must be finite and non-negative")
    if dt_hours <= 0 or per_ev_max_kw <= 0 or capacity_kw <= 0:
        raise ValueError("dt_hours, per_ev_max_kw, and capacity_kw must be positive")
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be at least 1")
    if capacity_margin_kw < 0 or capacity_margin_kw >= capacity_kw:
        raise ValueError("capacity_margin_kw must be in [0, capacity_kw)")

    if episode_len is None:
        horizon = int(horizon_hours)
    else:
        horizon = max(1, min(int(horizon_hours), int(episode_len) - int(step_idx)))
    needed = max(0, horizon - 1)
    future_demand = np.asarray(future_demand_kwh, dtype=float)
    if future_demand.size < needed:
        raise ValueError(f"need at least {needed} future-demand values, got {future_demand.size}")
    if not np.isfinite(future_demand[:needed]).all():
        raise ValueError("future demand contains non-finite values")
    future_demand = np.maximum(future_demand[:needed], 0.0)

    n_slots = len(present)
    usable_capacity = float(capacity_kw - capacity_margin_kw)
    reserve = np.zeros(horizon, dtype=float)
    if needed:
        reserve[1:] = np.clip(future_demand / dt_hours, 0.0, usable_capacity)
    max_kw = np.where(
        present,
        np.minimum(float(per_ev_max_kw), remaining / dt_hours),
        0.0,
    )

    slack_slots = [
        slot for slot in range(n_slots)
        if present[slot] and 0 <= int(deadlines[slot]) - int(step_idx) < horizon
    ]
    slack_index = {slot: n_slots * horizon + index for index, slot in enumerate(slack_slots)}
    n_vars = n_slots * horizon + len(slack_slots)

    objective = np.zeros(n_vars, dtype=float)
    for k in range(horizon):
        objective[k * n_slots:(k + 1) * n_slots] = -dt_hours - 1e-6 / (k + 1)
    if slack_slots:
        objective[n_slots * horizon:] = unmet_penalty

    bounds = []
    for k in range(horizon):
        for slot in range(n_slots):
            active = bool(present[slot] and step_idx + k <= deadlines[slot])
            bounds.append((0.0, float(max_kw[slot]) if active else 0.0))
    for slot in slack_slots:
        bounds.append((0.0, float(remaining[slot])))

    a_ub = []
    b_ub = []
    for k in range(horizon):
        row = np.zeros(n_vars, dtype=float)
        row[k * n_slots:(k + 1) * n_slots] = 1.0
        a_ub.append(row)
        b_ub.append(max(0.0, usable_capacity - reserve[k]))

    for slot in range(n_slots):
        for k in range(horizon):
            row = np.zeros(n_vars, dtype=float)
            for j in range(k + 1):
                row[j * n_slots + slot] = dt_hours
            a_ub.append(row)
            b_ub.append(float(remaining[slot]))

    for slot in slack_slots:
        deadline = int(deadlines[slot]) - int(step_idx)
        row = np.zeros(n_vars, dtype=float)
        for j in range(deadline + 1):
            row[j * n_slots + slot] = -dt_hours
        row[slack_index[slot]] = -1.0
        a_ub.append(row)
        b_ub.append(-float(remaining[slot]))

    result = linprog(
        c=objective,
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return MPCResult(
            action_kw=np.zeros(n_slots, dtype=np.float32),
            success=False,
            status=int(result.status),
            message=str(result.message),
            objective=float(result.fun) if np.isfinite(result.fun) else float("nan"),
            horizon_steps=horizon,
            reserve_kw=reserve,
            future_demand_kwh=future_demand,
        )

    action = np.asarray(result.x[:n_slots], dtype=float)
    action = np.clip(action, 0.0, max_kw)
    return MPCResult(
        action_kw=action.astype(np.float32),
        success=True,
        status=int(result.status),
        message=str(result.message),
        objective=float(result.fun),
        horizon_steps=horizon,
        reserve_kw=reserve,
        future_demand_kwh=future_demand,
    )


def mpc_action_for_env(
    env,
    forecast: StepwiseForecastLookup,
    horizon_hours: int = 6,
    capacity_kw: float = 50.0,
) -> MPCResult:
    inner = env.inner
    step_idx = int(inner._step_idx)
    horizon = max(1, min(int(horizon_hours), int(env.episode_len) - step_idx))
    needed = max(0, horizon - 1)
    forecast_values = forecast(inner.current_time()) if needed else np.empty(0, dtype=float)
    return solve_mpc_action(
        present=inner.true_presence(),
        remaining_kwh=inner.true_remaining_kwh(),
        departure_step=inner._departure_step,
        step_idx=step_idx,
        dt_hours=inner.dt_hours,
        per_ev_max_kw=inner.per_ev_max_kw,
        capacity_kw=capacity_kw,
        future_demand_kwh=forecast_values[:needed],
        horizon_hours=horizon,
        episode_len=env.episode_len,
    )
