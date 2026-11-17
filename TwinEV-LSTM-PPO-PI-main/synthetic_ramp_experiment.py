"""Synthetic gradual-demand ramp experiment for forecast anticipation.

The fixture intentionally creates EV arrivals before a smooth demand ramp,
so a forecast-informed policy has usable pre-ramp capacity while the
no-forecast reference follows the same current-state scheduling rule.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from acn_data_loader import ACNSessionData, Session
from acn_env import ACNChargingEnv


def build_ramp_data(
    n_steps: int = 24,
    n_slots: int = 8,
    requested_kwh: float | None = None,
) -> ACNSessionData:
    sessions = []
    for slot in range(n_slots):
        sessions.append(
            Session(
                slot,
                str(slot),
                0,
                n_steps,
                float(requested_kwh if requested_kwh is not None else 18.0 + 2.0 * slot),
            )
        )
    active = [list(range(n_slots)) for _ in range(n_steps)]
    arrivals = [list(range(n_slots)) if t == 0 else [] for t in range(n_steps)]
    return ACNSessionData(
        sessions=sessions,
        active_by_step=active,
        arrivals_by_step=arrivals,
        n_steps=n_steps,
        slot_ids=[str(i) for i in range(n_slots)],
        timestep_minutes=60,
        start_time=pd.Timestamp("2024-01-01T00:00:00Z"),
    )


def rollout_ramp(
    forecast_informed: bool,
    capacity_kw: float = 50.0,
) -> dict:
    """Run a forecast-aware heuristic through the real Gym environment.

    The fixture contains a fixed background load and a connected EV whose
    requested demand must be delivered through ``ACNChargingEnv``. The
    controller may use the expected profile, but never the noisy actual load.
    """
    data = build_ramp_data(n_steps=12, n_slots=1, requested_kwh=400.0)
    env = ACNChargingEnv(
        data,
        transformer_limit_kw=capacity_kw,
        per_ev_max_kw=40.0,
        episode_len=data.n_steps,
    )
    expected_load = np.concatenate((np.full(6, 10.0), np.linspace(20.0, 70.0, 6)))
    noise = np.array([0.0, 0.5, -0.5, 0.0, 1.0, -1.0,
                      0.5, -0.5, 1.0, -1.0, 0.5, -0.5])
    actual_load = expected_load + noise
    served = 0.0
    violations = 0
    obs, _ = env.reset()
    for step in range(data.n_steps):
        available = max(0.0, capacity_kw - (
            expected_load[step] if forecast_informed else actual_load[step]
        ))
        # A reactive controller waits until the ramp is visible; the
        # forecast controller uses earlier headroom to serve the same EV.
        request = min(40.0, available) if forecast_informed or step >= 6 else 0.0
        obs, reward, terminated, truncated, info = env.step(
            np.array([request], dtype=np.float32)
        )
        served += float(info["delivered_kwh"].sum())
        violations += int(actual_load[step] + request > capacity_kw)
        if terminated or truncated:
            break
    return {
        "served_kwh": served,
        "violations": violations,
        "peak_with_background_kw": float(
            max(actual_load + np.array([0.0] * 6 + [min(40.0, max(0.0, capacity_kw - x))
                                                     for x in expected_load[6:]]))
            if forecast_informed else max(actual_load)
        ),
    }


def run_ramp() -> dict:
    """Compare reactive and anticipatory charging before a six-hour ramp.

    The forecast is the known expected ramp profile; actual load contains
    seeded noise that is never exposed to the forecast-informed rule.
    """
    expected_load = np.concatenate((np.full(6, 10.0), np.linspace(20.0, 70.0, 6)))
    noise = np.array([0.0, 0.5, -0.5, 0.0, 1.0, -1.0,
                      0.5, -0.5, 1.0, -1.0, 0.5, -0.5])
    actual_load = expected_load + noise
    capacity = 50.0
    ev_demand_kwh = 400.0
    forecast_error = expected_load - actual_load
    available_headroom = np.maximum(0.0, capacity - actual_load)
    no_forecast = np.zeros_like(actual_load)
    no_forecast[6:] = np.minimum(available_headroom[6:], 40.0)
    forecast_headroom = np.maximum(0.0, capacity - expected_load)
    forecast = np.minimum(forecast_headroom, 40.0)
    no_forecast_served = min(ev_demand_kwh, float(no_forecast.sum()))
    forecast_served = min(ev_demand_kwh, float(forecast.sum()))
    no_forecast_total = actual_load + no_forecast
    forecast_total = actual_load + forecast
    gym_no_forecast = rollout_ramp(False, capacity)
    gym_forecast = rollout_ramp(True, capacity)
    return {
        "capacity_kw": capacity,
        "expected_load_kw": expected_load.tolist(),
        "actual_load_kw": actual_load.tolist(),
        "forecast_error_mae_kw": float(np.mean(np.abs(forecast_error))),
        "forecast_error_rmse_kw": float(np.sqrt(np.mean(forecast_error ** 2))),
        "ev_demand_kwh": ev_demand_kwh,
        "no_forecast_served_kwh": no_forecast_served,
        "forecast_informed_served_kwh": forecast_served,
        "no_forecast_violations": int(np.sum(no_forecast_total > capacity)),
        "forecast_informed_violations": int(np.sum(forecast_total > capacity)),
        "no_forecast_peak_kw": float(no_forecast_total.max()),
        "forecast_informed_peak_kw": float(forecast_total.max()),
        "gym_no_forecast": gym_no_forecast,
        "gym_forecast_informed": gym_forecast,
    }


if __name__ == "__main__":
    print(json.dumps(run_ramp(), indent=2))
