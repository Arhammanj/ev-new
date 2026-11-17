"""Diagnostic-only resolution and per-EV charging-rate sensitivity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acn_data_loader import load_acn_sessions
from acn_env import ACNChargingEnv


def rollout(data_path: str, timestep_minutes: int, per_ev_max_kw: float) -> dict:
    data = load_acn_sessions(data_path, timestep_minutes=timestep_minutes)
    env = ACNChargingEnv(
        data,
        transformer_limit_kw=50.0,
        per_ev_max_kw=per_ev_max_kw,
        episode_len=data.n_steps,
    )
    env.reset()
    achievable, presence, remaining, departure = [], [], [], []
    for _ in range(data.n_steps):
        present = env.true_presence().copy()
        remaining_kwh = env.true_remaining_kwh().copy()
        draw = np.where(
            present,
            np.minimum(per_ev_max_kw, remaining_kwh / env.dt_hours),
            0.0,
        )
        achievable.append(float(draw.sum()))
        presence.append(present)
        remaining.append(remaining_kwh)
        departure.append(env._departure_step.copy())
        _, _, terminated, truncated, _ = env.step(draw.astype(np.float32))
        if terminated or truncated:
            break
    return {
        "achievable_kw": np.asarray(achievable),
        "presence": np.asarray(presence),
        "remaining": np.asarray(remaining),
        "departure_step": np.asarray(departure),
        "dt_hours": float(env.dt_hours),
        "timestep_minutes": timestep_minutes,
        "per_ev_max_kw": per_ev_max_kw,
    }


def score(rollout_data: dict, capacity_kw: float = 50.0) -> pd.DataFrame:
    risk_hours = np.flatnonzero(rollout_data["achievable_kw"] > capacity_kw)
    rows = []
    for horizon_hours in range(1, 11):
        horizon_steps = int(round(horizon_hours * 60 / rollout_data["timestep_minutes"]))
        values = []
        for hour in risk_hours:
            responsible = rollout_data["presence"][hour] & (
                rollout_data["remaining"][hour] > 0
            )
            surge = np.minimum(
                rollout_data["per_ev_max_kw"],
                rollout_data["remaining"][hour] / rollout_data["dt_hours"],
            )
            surge_energy = float((surge * responsible).sum() * rollout_data["dt_hours"])
            anchor = int(hour) - horizon_steps
            if anchor < 0 or surge_energy <= 0:
                values.append(0.0)
                continue
            eligible = (
                rollout_data["presence"][anchor]
                & responsible
                & (
                    rollout_data["departure_step"][anchor] - anchor
                    >= horizon_steps
                )
            )
            anchor_draw = np.minimum(
                rollout_data["per_ev_max_kw"],
                rollout_data["remaining"][anchor] / rollout_data["dt_hours"],
            ) * rollout_data["presence"][anchor]
            slack_energy = max(
                0.0,
                capacity_kw - float(anchor_draw.sum()),
            ) * rollout_data["dt_hours"]
            available_energy = float(
                (rollout_data["remaining"][anchor] * eligible).sum()
            )
            values.append(min(1.0, min(slack_energy, available_energy) / surge_energy))
        values = np.asarray(values, dtype=float)
        rows.append(
            {
                "timestep_minutes": rollout_data["timestep_minutes"],
                "per_ev_max_kw": rollout_data["per_ev_max_kw"],
                "capacity_kw": capacity_kw,
                "horizon_hours": horizon_hours,
                "risk_hours": len(risk_hours),
                "energy_X_mean": float(np.mean(values)) if len(values) else np.nan,
                "energy_X_median": float(np.median(values)) if len(values) else np.nan,
                "exploitable_hours_percent": 100.0 * float(np.mean(values > 0.1))
                if len(values)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "resolution_rate_sensitivity.csv")
    parser.add_argument("--timesteps-minutes", nargs="+", type=int, default=[60, 15])
    parser.add_argument("--rates-kw", nargs="+", type=float, default=[3.3, 6.6])
    args = parser.parse_args()
    rows = []
    for timestep_minutes in args.timesteps_minutes:
        if 60 % timestep_minutes != 0:
            raise ValueError("Each timestep must divide one hour")
        for rate in args.rates_kw:
            rows.append(score(rollout(args.data_path, timestep_minutes, rate)))
    result = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(
        "\nThis is a diagnostic-only sensitivity analysis; no PPO training was "
        "rerun at 15-minute resolution."
    )


if __name__ == "__main__":
    main()
