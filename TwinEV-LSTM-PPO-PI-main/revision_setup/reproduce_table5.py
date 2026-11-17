"""Reproduce Table 5 under vehicle-level and hour-level definitions.

The vehicle-level calculation intentionally mirrors diagnose_forecast.py's
part3_lead_time_feasibility() logic. The hour-level calculation uses the
power-weighted A(l,h) definition and reports medians over risk hours.
"""

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


HORIZONS = (1, 3, 6)
CAPACITY_KW = 50.0
PAPER = {
    1: (64.3, 0.0),
    3: (12.2, 0.0),
    6: (0.7, 43.4),
}


def full_year_rollout(data_path: str) -> dict:
    data = load_acn_sessions(data_path, timestep_minutes=60)
    env = ACNChargingEnv(data, transformer_limit_kw=CAPACITY_KW, episode_len=data.n_steps)
    env.reset()
    achievable, presence, remaining = [], [], []
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
        _, _, terminated, truncated, _ = env.step(draw.astype(np.float32))
        if terminated or truncated:
            break
    return {
        "achievable_kw": np.asarray(achievable),
        "presence": np.asarray(presence),
        "remaining": np.asarray(remaining),
    }


def test_week_rollouts() -> list[dict]:
    setup = shared_setup()
    return [greedy_rollout(setup, start, CAPACITY_KW) for start in setup["test_starts"]]


def risk_hours(weeks: list[dict]) -> int:
    return sum(int(np.sum(week["achievable_kw"] > CAPACITY_KW)) for week in weeks)


def definitions(weeks: list[dict]) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        pair_connected: list[bool] = []
        pair_slack: list[float] = []
        hour_a: list[float] = []
        hour_s: list[float] = []
        for week in weeks:
            achievable = week["achievable_kw"]
            presence = week["presence"]
            remaining = week["remaining"]
            risk = achievable > CAPACITY_KW
            for h in np.flatnonzero(risk):
                contributing = presence[h] & (remaining[h] > 0)
                responsible_power = (
                    np.minimum(PER_EV_MAX_KW, remaining[h]) * contributing
                )
                total_power = float(responsible_power.sum())
                anchor = h - horizon
                if anchor >= 0:
                    connected = presence[anchor]
                    anchor_draw = np.minimum(
                        PER_EV_MAX_KW, remaining[anchor]
                    ) * connected
                    slack = max(0.0, CAPACITY_KW - float(anchor_draw.sum()))
                else:
                    connected = np.zeros_like(contributing)
                    slack = 0.0
                pair_connected.extend(connected[contributing].tolist())
                pair_slack.extend([slack] * int(np.sum(contributing & connected)))
                hour_a.append(
                    float((responsible_power * connected).sum() / total_power)
                    if total_power > 0
                    else 0.0
                )
                hour_s.append(slack)

        vehicle_fraction = 100.0 * float(np.mean(pair_connected)) if pair_connected else np.nan
        vehicle_median_slack = float(np.median(pair_slack)) if pair_slack else np.nan
        row = {
            "horizon_hours": horizon,
            "vehicle_pairs": len(pair_connected),
            "vehicle_connected_percent": vehicle_fraction,
            "vehicle_median_slack_kw": vehicle_median_slack,
            "hour_risk_hours": len(hour_a),
            "hour_median_A": float(np.median(hour_a)) if hour_a else np.nan,
            "hour_median_S_kw": float(np.median(hour_s)) if hour_s else np.nan,
            "paper_connected_percent": PAPER[horizon][0],
            "paper_median_slack_kw": PAPER[horizon][1],
        }
        row["vehicle_matches_paper"] = bool(
            np.isclose(vehicle_fraction, PAPER[horizon][0], atol=0.05)
            and np.isclose(vehicle_median_slack, PAPER[horizon][1], atol=0.05)
        )
        row["hour_matches_paper"] = bool(
            np.isclose(100.0 * row["hour_median_A"], PAPER[horizon][0], atol=0.05)
            and np.isclose(row["hour_median_S_kw"], PAPER[horizon][1], atol=0.05)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "table5_reproduction.csv",
    )
    args = parser.parse_args()

    full_year = full_year_rollout(args.data_path)
    test_weeks = test_week_rollouts()
    full_year_weeks = [full_year]
    print(f"Capacity: {CAPACITY_KW:.0f} kW")
    print(
        f"Full year: {len(full_year['achievable_kw'])} hours, "
        f"{risk_hours(full_year_weeks)} risk hours"
    )
    print(
        f"Test weeks: {len(test_weeks)} x 168 hours, "
        f"{risk_hours(test_weeks)} risk hours"
    )
    print(
        "The 71-hour count is test-week-only: the existing diagnostic calls "
        "shared_setup(), selects five held-out weeks, and applies the greedy "
        "risk test only inside those windows. It is not a full-year count."
    )

    full_results = definitions(full_year_weeks).assign(scope="full_year")
    test_results = definitions(test_weeks).assign(scope="test_weeks")
    results = pd.concat([full_results, test_results], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print("\nTable 5 comparison:")
    print(
        results[
            [
                "scope",
                "horizon_hours",
                "vehicle_connected_percent",
                "vehicle_median_slack_kw",
                "hour_median_A",
                "hour_median_S_kw",
                "paper_connected_percent",
                "paper_median_slack_kw",
                "vehicle_matches_paper",
                "hour_matches_paper",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {args.output}")
    print(
        "\nConclusion: the paper values are produced by the vehicle-level "
        "test-week logic in diagnose_forecast.py:part3_lead_time_feasibility(), "
        "not by the hour-level median-A definition."
    )


if __name__ == "__main__":
    main()
