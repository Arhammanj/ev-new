"""Energy-based X versus transformer capacity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from energy_based_x import full_year_rollout, test_week_rollouts


HORIZONS = tuple(range(1, 11))


def score(weeks: list[dict], capacity_kw: float, scope: str) -> list[dict]:
    events = [
        (week, int(hour))
        for week in weeks
        for hour in np.flatnonzero(week["achievable_kw"] > capacity_kw)
    ]
    rows = []
    for horizon in HORIZONS:
        values = []
        for week, hour in events:
            responsible = week["presence"][hour] & (week["remaining"][hour] > 0)
            surge = np.minimum(6.6, week["remaining"][hour] / week["dt_hours"])
            surge_energy = float((surge * responsible).sum() * week["dt_hours"])
            anchor = hour - horizon
            if anchor < 0 or surge_energy <= 0:
                values.append(0.0)
                continue
            connected = week["presence"][anchor] & responsible
            eligible = connected & (
                week["departure_step"][anchor] - anchor >= horizon
            )
            anchor_draw = np.minimum(
                6.6, week["remaining"][anchor] / week["dt_hours"]
            ) * week["presence"][anchor]
            slack_energy = max(
                0.0,
                capacity_kw - float(anchor_draw.sum()),
            ) * week["dt_hours"]
            available_energy = float((week["remaining"][anchor] * eligible).sum())
            values.append(min(1.0, min(slack_energy, available_energy) / surge_energy))
        values = np.asarray(values, dtype=float)
        rows.append(
            {
                "scope": scope,
                "capacity_kw": capacity_kw,
                "horizon_hours": horizon,
                "risk_hours": len(events),
                "energy_X_mean": float(np.mean(values)) if len(values) else np.nan,
                "energy_X_median": float(np.median(values)) if len(values) else np.nan,
                "exploitable_hours_percent": 100.0 * float(np.mean(values > 0.1))
                if len(values)
                else np.nan,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities-kw", nargs="+", type=float, default=[40, 50, 60, 75, 100])
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    if any(capacity <= 0 for capacity in args.capacities_kw):
        raise ValueError("capacities must be positive")

    rows = []
    for capacity in args.capacities_kw:
        full = full_year_rollout(args.data_path, capacity)
        test = test_week_rollouts(capacity)
        rows.extend(score([full], capacity, "full_year"))
        rows.extend(score(test, capacity, "test_weeks"))
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "capacity_x_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for scope, group in result.groupby("scope"):
        for horizon in HORIZONS:
            subset = group[group["horizon_hours"] == horizon]
            axes[0].plot(
                subset["capacity_kw"],
                subset["energy_X_mean"],
                marker="o",
                alpha=0.55,
                label=f"{scope}, {horizon}h",
            )
        axes[1].plot(
            group.groupby("capacity_kw")["exploitable_hours_percent"].mean().index,
            group.groupby("capacity_kw")["exploitable_hours_percent"].mean().values,
            marker="o",
            label=scope,
        )
    axes[0].set_ylabel("Mean energy-based X")
    axes[1].set_ylabel("Exploitable risk hours (>0.1), %")
    for axis in axes:
        axis.set_xlabel("Transformer capacity [kW]")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend()
    fig.suptitle("Energy-based lead-time usefulness versus capacity")
    fig.tight_layout()
    fig.savefig(args.output_dir / "capacity_x_sweep.png", dpi=200)
    plt.close(fig)
    print(result.to_string(index=False))
    print(f"\nSaved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
