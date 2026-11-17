"""Lead-time distributions using the vehicle-level Table 5 definition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acn_env import ACNChargingEnv
from acn_data_loader import load_acn_sessions
from diagnose_forecast import greedy_rollout
from exp_2x2_run import shared_setup
from train_controller import PER_EV_MAX_KW


HORIZONS = tuple(range(1, 11))


def full_year_rollout(data_path: str, capacity_kw: float) -> dict:
    data = load_acn_sessions(data_path, timestep_minutes=60)
    env = ACNChargingEnv(data, transformer_limit_kw=capacity_kw, episode_len=data.n_steps)
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


def test_week_rollouts(capacity_kw: float) -> list[dict]:
    setup = shared_setup()
    return [greedy_rollout(setup, start, capacity_kw) for start in setup["test_starts"]]


def risk_events(weeks: list[dict], capacity_kw: float) -> list[tuple[dict, int]]:
    events = []
    for week in weeks:
        for risk_hour in np.flatnonzero(week["achievable_kw"] > capacity_kw):
            events.append((week, int(risk_hour)))
    return events


def pair_observations(
    events: list[tuple[dict, int]],
    horizon: int,
    capacity_kw: float,
) -> tuple[list[bool], list[float], list[float], list[bool]]:
    connected_pairs: list[bool] = []
    connected_slacks: list[float] = []
    hour_a: list[float] = []
    hour_joint: list[bool] = []
    for week, risk_hour in events:
        presence = week["presence"]
        remaining = week["remaining"]
        contributing = presence[risk_hour] & (remaining[risk_hour] > 0)
        responsible_power = np.minimum(PER_EV_MAX_KW, remaining[risk_hour]) * contributing
        total_power = float(responsible_power.sum())
        anchor = risk_hour - horizon
        if anchor >= 0:
            connected = presence[anchor]
            anchor_draw = np.minimum(PER_EV_MAX_KW, remaining[anchor]) * connected
            slack = max(0.0, capacity_kw - float(anchor_draw.sum()))
        else:
            connected = np.zeros_like(contributing)
            slack = 0.0
        pair_connected = connected[contributing].astype(bool)
        connected_pairs.extend(pair_connected.tolist())
        connected_slacks.extend([slack] * int(np.sum(pair_connected)))
        hour_a.append(
            float((responsible_power * connected).sum() / total_power)
            if total_power > 0
            else 0.0
        )
        hour_joint.append(bool(np.any(pair_connected) and slack > 0))
    return connected_pairs, connected_slacks, hour_a, hour_joint


def distribution_table(
    scopes: dict[str, list[tuple[dict, int]]],
    capacity_kw: float,
) -> pd.DataFrame:
    rows = []
    for scope, events in scopes.items():
        for horizon in HORIZONS:
            connected, slacks, hour_a, hour_joint = pair_observations(
                events, horizon, capacity_kw
            )
            slack_values = np.asarray(slacks, dtype=float)
            rows.append(
                {
                    "scope": scope,
                    "capacity_kw": capacity_kw,
                    "horizon_hours": horizon,
                    "risk_hours": len(events),
                    "responsible_ev_pairs": len(connected),
                    "connected_pair_percent": 100.0 * np.mean(connected),
                    "joint_pair_percent": 100.0
                    * (np.sum(slack_values > 0) / len(connected))
                    if connected
                    else np.nan,
                    "risk_hour_joint_percent": 100.0 * np.mean(hour_joint),
                    "slack_p10_kw": float(np.quantile(slack_values, 0.10))
                    if len(slack_values)
                    else np.nan,
                    "slack_p50_kw": float(np.quantile(slack_values, 0.50))
                    if len(slack_values)
                    else np.nan,
                    "slack_p90_kw": float(np.quantile(slack_values, 0.90))
                    if len(slack_values)
                    else np.nan,
                    "slack_mean_connected_pairs_kw": float(np.mean(slack_values))
                    if len(slack_values)
                    else np.nan,
                    "hour_power_weighted_A_mean": float(np.mean(hour_a)),
                    "hour_power_weighted_A_median": float(np.median(hour_a)),
                }
            )
    return pd.DataFrame(rows)


def saturated_periods(week: dict, capacity_kw: float) -> list[tuple[int, int]]:
    saturated = week["achievable_kw"] >= capacity_kw
    padded = np.concatenate(([False], saturated, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def onset_table(
    scopes: dict[str, list[tuple[dict, int]]],
    capacity_kw: float,
) -> pd.DataFrame:
    rows = []
    for scope, events in scopes.items():
        period_records = []
        onset_events = []
        for week, risk_hour in events:
            containing = [
                (start, end)
                for start, end in saturated_periods(week, capacity_kw)
                if start <= risk_hour < end
            ]
            if not containing:
                continue
            start, end = containing[0]
            period_records.append((scope, start, end, end - start))
            onset_events.append((week, risk_hour, start, end))
        unique_periods = {(start, end, length) for _, start, end, length in period_records}
        for horizon in HORIZONS:
            connected, slacks = [], []
            for week, risk_hour, onset, _ in onset_events:
                contributing = week["presence"][risk_hour] & (
                    week["remaining"][risk_hour] > 0
                )
                anchor = onset - horizon
                if anchor >= 0:
                    anchor_connected = week["presence"][anchor][contributing]
                    anchor_draw = np.minimum(
                        PER_EV_MAX_KW, week["remaining"][anchor]
                    ) * week["presence"][anchor]
                    slack = max(0.0, capacity_kw - float(anchor_draw.sum()))
                else:
                    anchor_connected = np.zeros(int(np.sum(contributing)), dtype=bool)
                    slack = 0.0
                connected.extend(anchor_connected.tolist())
                slacks.extend([slack] * int(np.sum(anchor_connected)))
            rows.append(
                {
                    "scope": scope,
                    "capacity_kw": capacity_kw,
                    "horizon_hours": horizon,
                    "saturated_period_count": len(unique_periods),
                    "saturated_period_length_median_hours": float(
                        np.median([item[2] for item in unique_periods])
                    )
                    if unique_periods
                    else np.nan,
                    "saturated_period_length_p90_hours": float(
                        np.quantile([item[2] for item in unique_periods], 0.90)
                    )
                    if unique_periods
                    else np.nan,
                    "onset_responsible_ev_pairs": len(connected),
                    "onset_connected_pair_percent": 100.0 * np.mean(connected)
                    if connected
                    else np.nan,
                    "onset_median_slack_connected_pairs_kw": float(np.median(slacks))
                    if slacks
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_plot(
    scopes: dict[str, list[tuple[dict, int]]],
    capacity_kw: float,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, len(scopes), figsize=(12, 5), squeeze=False)
    for axis, (scope, events) in zip(axes[0], scopes.items()):
        for horizon in HORIZONS:
            a_values, s_values = [], []
            for week, risk_hour in events:
                connected, slacks, hour_a, _ = pair_observations(
                    [(week, risk_hour)], horizon, capacity_kw
                )
                a_values.append(hour_a[0])
                if risk_hour - horizon >= 0:
                    anchor = risk_hour - horizon
                    draw = np.minimum(PER_EV_MAX_KW, week["remaining"][anchor]) * week[
                        "presence"
                    ][anchor]
                    s_values.append(max(0.0, capacity_kw - float(draw.sum())))
                else:
                    s_values.append(0.0)
            axis.scatter(
                a_values,
                s_values,
                s=18,
                alpha=0.45,
                label=f"{horizon} h",
            )
        axis.set_title(scope.replace("_", " ").title())
        axis.set_xlabel("Power-weighted A(l,h)")
        axis.set_ylabel("Slack S(l,h) [kW]")
        axis.grid(alpha=0.25)
    axes[0, -1].legend(title="Lead time", fontsize=8)
    fig.suptitle(f"Lead-time A versus slack, C={capacity_kw:g} kW")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    if args.capacity_kw <= 0:
        raise ValueError("--capacity-kw must be positive")

    full = full_year_rollout(args.data_path, args.capacity_kw)
    test = test_week_rollouts(args.capacity_kw)
    scopes = {
        "full_year": risk_events([full], args.capacity_kw),
        "test_weeks": risk_events(test, args.capacity_kw),
    }
    distribution = distribution_table(scopes, args.capacity_kw)
    onset = onset_table(scopes, args.capacity_kw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(args.output_dir / "lead_time_distribution.csv", index=False)
    onset.to_csv(args.output_dir / "saturation_onset.csv", index=False)
    make_plot(
        scopes,
        args.capacity_kw,
        args.output_dir / "lead_time_A_vs_S.png",
    )

    print(distribution.to_string(index=False))
    print("\n" + onset.to_string(index=False))
    print("\nJoint-rate summary:")
    for scope, group in distribution.groupby("scope"):
        rates = group["joint_pair_percent"].to_numpy()
        print(
            f"  {scope}: joint pair rate range "
            f"{np.nanmin(rates):.2f}%–{np.nanmax(rates):.2f}%"
        )
        if np.nanmax(rates) < 1.0:
            print(
                "    Joint occurrence is near zero: connected EVs and positive "
                "slack rarely co-occur under this pair definition."
            )
        else:
            print(
                "    Joint occurrence is not near zero at every horizon; the "
                "claim that the conditions never co-occur is too strong."
            )
    print(f"\nSaved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
