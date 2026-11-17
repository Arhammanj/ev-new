"""Compare real ACN rollups at hourly and 15-minute resolution.

This is a data audit, not a synthetic experiment. The loader maps the same
cached sessions onto each grid, and the reported greedy pressure is computed
before any controller action or transformer clipping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acn_data_loader import load_acn_sessions


def summarize(data, capacity_kw: float) -> dict:
    achievable = np.zeros(data.n_steps, dtype=float)
    for step, indices in enumerate(data.active_by_step):
        achievable[step] = sum(
            min(6.6, data.sessions[idx].requested_kwh /
                (data.timestep_minutes / 60.0))
            for idx in indices
        )
    return {
        "timestep_minutes": data.timestep_minutes,
        "sessions": len(data.sessions),
        "slots": len(data.slot_ids),
        "steps": data.n_steps,
        "start_time": str(data.start_time),
        "end_time": str(
            data.start_time + pd.Timedelta(
                minutes=data.n_steps * data.timestep_minutes
            )
        ),
        "greedy_peak_kw": float(achievable.max()),
        "greedy_mean_kw": float(achievable.mean()),
        "fraction_above_capacity": float(np.mean(achievable > capacity_kw)),
        "p95_kw": float(np.quantile(achievable, 0.95)),
        "_achievable": achievable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="acn_caltech_2019.csv")
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--output-dir", default="paper_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    series = {}
    for minutes in (60, 15):
        data = load_acn_sessions(args.data_path, timestep_minutes=minutes)
        result = summarize(data, args.capacity_kw)
        series[minutes] = result.pop("_achievable")
        summaries.append(result)

    table = pd.DataFrame(summaries)
    table.to_csv(output_dir / "resolution_comparison.csv", index=False)
    with open(output_dir / "resolution_comparison.json", "w") as handle:
        json.dump(summaries, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for minutes, values in series.items():
        axes[0].plot(
            np.arange(len(values)) * minutes / 60.0,
            values,
            linewidth=0.7,
            alpha=0.8,
            label=f"{minutes}-min",
        )
        axes[1].hist(values, bins=40, alpha=0.45, label=f"{minutes}-min")
    axes[0].axhline(args.capacity_kw, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Elapsed time (hours)", ylabel="Greedy achievable kW")
    axes[1].axvline(args.capacity_kw, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Greedy achievable kW", ylabel="Grid steps")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "resolution_comparison.png", dpi=300)
    fig.savefig(output_dir / "resolution_comparison.pdf", bbox_inches="tight")
    print(table.to_string(index=False))
    print(f"Saved resolution artifacts under {output_dir}")


if __name__ == "__main__":
    main()
