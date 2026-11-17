"""Save lead-time A/S arrays from the existing greedy physical diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_2x2_run import shared_setup
from train_controller import ACNWeeklyEpisodeEnv, PER_EV_MAX_KW


def greedy_rollout_with_departures(setup, start_step: int) -> dict:
    env = ACNWeeklyEpisodeEnv(
        setup["full_data"],
        setup["slot_roster"],
        setup["test_range"],
        transformer_limit_kw=50.0,
        seed=999,
    )
    env.reset(start_step=start_step)
    timestamps, achievable, violation, presence, remaining, time_left = [], [], [], [], [], []
    for _ in range(env.episode_len):
        present = env.inner.true_presence().copy()
        remaining_kwh = env.inner.true_remaining_kwh().copy()
        draw = np.where(
            present,
            np.minimum(PER_EV_MAX_KW, remaining_kwh / env.inner.dt_hours),
            0.0,
        )
        timestamps.append(env.inner.current_time())
        achievable.append(float(draw.sum()))
        violation.append(max(0.0, float(draw.sum()) - 50.0))
        presence.append(present)
        remaining.append(remaining_kwh)
        time_left.append(
            np.where(
                present,
                np.maximum(env.inner._departure_step - env.inner._step_idx, 0)
                * env.inner.dt_hours,
                0.0,
            )
        )
        _, _, terminated, truncated, _ = env.step(draw.astype(np.float32))
        if terminated or truncated:
            break
    return {
        "timestamps": timestamps,
        "achievable_kw": np.asarray(achievable),
        "violation_kw": np.asarray(violation),
        "presence": np.asarray(presence),
        "remaining": np.asarray(remaining),
        "time_to_departure": np.asarray(time_left),
    }


def compute_arrays(weeks: list[dict], capacity_kw: float, horizons: np.ndarray, dt_hours: float):
    events = []
    for week_index, week in enumerate(weeks):
        risk = week["achievable_kw"] > capacity_kw
        for h in np.flatnonzero(risk):
            contributing = week["presence"][h] & (week["remaining"][h] > 0)
            responsible_power = np.minimum(
                6.6, week["remaining"][h] / dt_hours
            ) * contributing
            total_responsible = float(responsible_power.sum())
            a_values, s_values, energy_values, departure_values = [], [], [], []
            for lead in horizons:
                anchor = h - int(round(lead / dt_hours))
                connected = (
                    week["presence"][anchor] if anchor >= 0 else np.zeros_like(contributing)
                )
                anchor_remaining = (
                    week["remaining"][anchor] if anchor >= 0 else np.zeros_like(week["remaining"][h])
                )
                anchor_draw = (
                    np.minimum(6.6, anchor_remaining / dt_hours) * connected
                )
                a = (
                    float((responsible_power * connected).sum() / total_responsible)
                    if total_responsible > 0
                    else 0.0
                )
                s = max(0.0, capacity_kw - float(anchor_draw.sum()))
                a_values.append(a)
                s_values.append(s)
                energy_values.append(float((anchor_remaining * connected).sum()))
                departure_values.append(
                    float((week["time_to_departure"][anchor] * connected).sum())
                    if anchor >= 0 else 0.0
                )
            events.append(
                (week_index, int(h), a_values, s_values, energy_values, departure_values)
            )
    if not events:
        raise ValueError("No violation-risk hours found")
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    if args.capacity_kw <= 0 or any(h <= 0 for h in args.horizons):
        raise ValueError("capacity and horizons must be positive")

    setup = shared_setup()
    weeks = [greedy_rollout_with_departures(setup, start) for start in setup["test_starts"]]
    dt_hours = 1.0
    events = compute_arrays(
        weeks, args.capacity_kw, np.asarray(sorted(set(args.horizons))), dt_hours
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"lead_time_arrays_C{args.capacity_kw:g}.npz"
    h_index = np.asarray([event[1] for event in events], dtype=np.int64)
    week_index = np.asarray([event[0] for event in events], dtype=np.int64)
    np.savez(
        output,
        h_index=h_index,
        l_values=np.asarray(sorted(set(args.horizons)), dtype=np.int64),
        A=np.asarray([event[2] for event in events], dtype=float),
        S=np.asarray([event[3] for event in events], dtype=float),
        remaining_energy=np.asarray([event[4] for event in events], dtype=float),
        time_to_departure=np.asarray([event[5] for event in events], dtype=float),
        week_index=week_index,
    )
    print(f"Saved {output} with {len(events)} risk hours and {len(args.horizons)} horizons")
    paper_table5 = {1: (64.3, 0.0), 3: (12.2, 0.0), 6: (0.7, 43.4)}
    print("\nTable 5 sanity check (array-level event aggregation):")
    for horizon in (1, 3, 6):
        if horizon not in args.horizons:
            continue
        column = list(np.asarray(sorted(set(args.horizons)))).index(horizon)
        connected_fraction = 100.0 * float(np.mean(np.asarray([event[2][column] for event in events]) > 0))
        positive_slack = np.asarray([event[3][column] for event in events])
        median_slack = float(np.median(positive_slack[positive_slack > 0])) if np.any(positive_slack > 0) else 0.0
        expected_fraction, expected_slack = paper_table5[horizon]
        print(
            f"  {horizon}h: computed={connected_fraction:.1f}% / {median_slack:.1f} kW; "
            f"paper={expected_fraction:.1f}% / {expected_slack:.1f} kW"
        )
    print(
        "  Note: these arrays aggregate all responsible EV power per risk hour. "
        "The paper values are pair-level connected-EV/slack statistics, so exact "
        "agreement requires retaining one row per risk-hour/EV pair."
    )


if __name__ == "__main__":
    main()
