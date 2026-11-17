"""Energy-based lead-time score using responsible-EV pairs."""

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
from diagnose_forecast import greedy_rollout
from exp_2x2_run import shared_setup
from train_controller import ACNWeeklyEpisodeEnv, PER_EV_MAX_KW


HORIZONS = tuple(range(1, 11))


def full_year_rollout(data_path: str, capacity_kw: float) -> dict:
    data = load_acn_sessions(data_path, timestep_minutes=60)
    env = ACNChargingEnv(data, transformer_limit_kw=capacity_kw, episode_len=data.n_steps)
    if not hasattr(env, "_departure_step"):
        raise RuntimeError("Environment does not expose departure state required for this calculation")
    env.reset()
    achievable, presence, remaining, departure = [], [], [], []
    for _ in range(data.n_steps):
        present = env.true_presence().copy()
        remaining_kwh = env.true_remaining_kwh().copy()
        draw = np.where(
            present,
            np.minimum(PER_EV_MAX_KW, remaining_kwh / env.dt_hours),
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
    }


def test_week_rollouts(capacity_kw: float) -> list[dict]:
    setup = shared_setup()
    weeks = []
    for start in setup["test_starts"]:
        env = ACNWeeklyEpisodeEnv(
            setup["full_data"],
            setup["slot_roster"],
            setup["test_range"],
            transformer_limit_kw=capacity_kw,
            seed=999,
        )
        env.reset(start_step=start)
        if not hasattr(env.inner, "_departure_step"):
            raise RuntimeError("Environment does not expose departure state required for this calculation")
        achievable, presence, remaining, departure = [], [], [], []
        for _ in range(env.episode_len):
            present = env.inner.true_presence().copy()
            remaining_kwh = env.inner.true_remaining_kwh().copy()
            draw = np.where(
                present,
                np.minimum(PER_EV_MAX_KW, remaining_kwh / env.inner.dt_hours),
                0.0,
            )
            achievable.append(float(draw.sum()))
            presence.append(present)
            remaining.append(remaining_kwh)
            departure.append(env.inner._departure_step.copy())
            _, _, terminated, truncated, _ = env.step(draw.astype(np.float32))
            if terminated or truncated:
                break
        weeks.append(
            {
                "achievable_kw": np.asarray(achievable),
                "presence": np.asarray(presence),
                "remaining": np.asarray(remaining),
                "departure_step": np.asarray(departure),
                "dt_hours": float(env.inner.dt_hours),
            }
        )
    return weeks


def calculate(
    weeks: list[dict],
    capacity_kw: float,
    binary_joint: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    events = []
    for week in weeks:
        for risk_hour in np.flatnonzero(week["achievable_kw"] > capacity_kw):
            events.append((week, int(risk_hour)))

    rows = []
    for horizon in HORIZONS:
        scores = []
        for week, risk_hour in events:
            presence = week["presence"]
            remaining = week["remaining"]
            departure = week["departure_step"]
            dt_hours = week["dt_hours"]
            responsible = presence[risk_hour] & (remaining[risk_hour] > 0)
            surge_power = np.minimum(PER_EV_MAX_KW, remaining[risk_hour] / dt_hours)
            surge_energy = float((surge_power * dt_hours * responsible).sum())
            anchor = risk_hour - horizon
            if anchor < 0 or surge_energy <= 0:
                scores.append(0.0)
                continue

            connected = presence[anchor] & responsible
            departure_eligible = departure[anchor] - anchor >= horizon
            eligible = connected & departure_eligible
            anchor_draw = np.minimum(
                PER_EV_MAX_KW, remaining[anchor] / dt_hours
            ) * presence[anchor]
            slack_kw = max(0.0, capacity_kw - float(anchor_draw.sum()))
            available_energy = float((remaining[anchor] * eligible).sum())
            shiftable_energy = min(slack_kw * dt_hours, available_energy)
            scores.append(min(1.0, shiftable_energy / surge_energy))

        values = np.asarray(scores, dtype=float)
        binary = binary_joint.loc[
            (binary_joint["scope"] == scope)
            & (binary_joint["capacity_kw"] == capacity_kw)
            & (binary_joint["horizon_hours"] == horizon),
            "joint_pair_percent",
        ]
        if len(binary) != 1:
            raise ValueError(f"Missing unique binary joint rate for {scope}, {capacity_kw:g} kW, {horizon}h")
        rows.append(
            {
                "scope": scope,
                "capacity_kw": capacity_kw,
                "horizon_hours": horizon,
                "risk_hours": len(events),
                "energy_X_mean": float(np.mean(values)),
                "energy_X_median": float(np.median(values)),
                "energy_X_p10": float(np.quantile(values, 0.10)),
                "energy_X_p90": float(np.quantile(values, 0.90)),
                "energy_X_positive_0_1_percent": 100.0 * float(np.mean(values > 0.1)),
                "binary_joint_pair_percent": float(binary.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument(
        "--binary-csv",
        type=Path,
        default=Path(__file__).parent / "outputs" / "lead_time_distribution.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "energy_based_x.csv",
    )
    args = parser.parse_args()
    if args.capacity_kw <= 0:
        raise ValueError("--capacity-kw must be positive")
    if not args.binary_csv.exists():
        raise FileNotFoundError(
            f"{args.binary_csv} is required; run lead_time_distribution.py first"
        )

    binary_joint = pd.read_csv(args.binary_csv)
    required = {"scope", "capacity_kw", "horizon_hours", "joint_pair_percent"}
    missing = required - set(binary_joint.columns)
    if missing:
        raise ValueError(f"Binary distribution CSV is missing columns: {sorted(missing)}")

    full = full_year_rollout(args.data_path, args.capacity_kw)
    test = test_week_rollouts(args.capacity_kw)
    results = pd.concat(
        [
            calculate([full], args.capacity_kw, binary_joint, "full_year"),
            calculate(test, args.capacity_kw, binary_joint, "test_weeks"),
        ],
        ignore_index=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
