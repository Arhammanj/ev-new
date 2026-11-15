"""Synthetic gradual-demand ramp experiment for forecast anticipation.

The fixture intentionally creates EV arrivals before a smooth demand ramp,
so a forecast-informed policy has usable pre-ramp capacity while the
no-forecast reference follows the same current-state scheduling rule.
"""

from __future__ import annotations

import json

import numpy as np


def build_ramp_data(n_steps: int = 24, n_slots: int = 8) -> ACNSessionData:
    sessions = []
    for slot in range(n_slots):
        sessions.append(
            Session(slot, str(slot), 0, n_steps, 18.0 + 2.0 * slot)
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
        start_time=None,
    )


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
    }


if __name__ == "__main__":
    print(json.dumps(run_ramp(), indent=2))
