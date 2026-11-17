"""Exact lead-time criterion sweep over transformer capacities.

This is a separate analysis entry point. It preserves the original
cross-dataset outputs and evaluates the unchanged median criterion at a
common 60-minute resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_dataset_lead_time import (
    DATASETS,
    HORIZONS_HOURS,
    criterion_for_dataset,
    load_grid,
    raw_verification,
    statistical_diagnostics,
)


DEFAULT_CAPACITIES = (25.0, 50.0, 75.0, 100.0, 125.0, 150.0)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "capacity_sweep_outputs"


def run_sweep(
    capacities: tuple[float, ...],
    timestep_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = []
    events = []
    for dataset, path in DATASETS.items():
        if dataset == "Office001":
            _, unsupported = criterion_for_dataset(
                dataset, type("UnsupportedGrid", (), {"n_steps": 0})(),
                horizons=HORIZONS_HOURS,
            )
            for capacity in capacities:
                row = unsupported.copy()
                row["capacity_kw"] = capacity
                scores.append(row)
            continue

        data = load_grid(path, timestep_minutes)
        for capacity in capacities:
            event_df, score_df = criterion_for_dataset(
                dataset,
                data,
                capacity_kw=capacity,
                horizons=HORIZONS_HOURS,
            )
            event_df = event_df.copy()
            score_df = score_df.copy()
            event_df["capacity_kw"] = capacity
            score_df["capacity_kw"] = capacity
            events.append(event_df)
            scores.append(score_df)

    scores_df = pd.concat(scores, ignore_index=True)
    events_df = pd.concat(events, ignore_index=True)
    diagnostics_df, persistence_df = statistical_diagnostics(events_df)
    return scores_df, events_df, diagnostics_df


def save_figures(
    scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "paper_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    supported_scores = scores[scores["criterion_status"].str.startswith("computed")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for capacity, group in supported_scores.groupby("capacity_kw"):
        ax.plot(
            group["horizon_hours"],
            group["criterion_x"],
            marker="o",
            label=f"{capacity:g} kW",
        )
    ax.set(
        xlabel="Lead time (hours)",
        ylabel="Median lead-time score X(l,C)",
        title="Exact lead-time criterion across transformer capacities",
    )
    ax.set_xticks(list(HORIZONS_HOURS))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "capacity_lead_time_scores.png", dpi=300)
    fig.savefig(figure_dir / "capacity_lead_time_scores.pdf")
    plt.close(fig)

    pivot = supported_scores.pivot_table(
        index="capacity_kw", columns="horizon_hours", values="criterion_x"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    ax.set(
        xlabel="Lead time (hours)",
        ylabel="Transformer capacity (kW)",
        title="Median lead-time score heatmap",
        xticks=np.arange(len(pivot.columns)),
        xticklabels=pivot.columns,
        yticks=np.arange(len(pivot.index)),
        yticklabels=[f"{value:g}" for value in pivot.index],
    )
    fig.colorbar(image, ax=ax, label="X(l,C)")
    fig.tight_layout()
    fig.savefig(figure_dir / "capacity_lead_time_heatmap.png", dpi=300)
    fig.savefig(figure_dir / "capacity_lead_time_heatmap.pdf")
    plt.close(fig)

    if diagnostics.empty:
        return
    one_hour = diagnostics[diagnostics["horizon_hours"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for dataset, group in one_hour.groupby("dataset"):
        group = group.sort_values("capacity_kw")
        axes[0].plot(group["capacity_kw"], group["risk_hours_evaluated"], marker="o", label=dataset)
        axes[1].plot(group["capacity_kw"], group["pct_S_positive"], marker="o", label=dataset)
        axes[2].plot(group["capacity_kw"], group["pct_both_positive"], marker="o", label=dataset)
    for ax, title in zip(
        axes,
        ("Risk hours", "S > 0 (%)", "A > 0 and S > 0 (%)"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Transformer capacity (kW)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Risk hours (%)")
    axes[2].set_ylabel("Risk hours (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "capacity_flexibility_relationship.png", dpi=300)
    fig.savefig(figure_dir / "capacity_flexibility_relationship.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capacities-kw",
        nargs="+",
        type=float,
        default=list(DEFAULT_CAPACITIES),
    )
    parser.add_argument("--timestep-minutes", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(capacity <= 0 for capacity in args.capacities_kw):
        raise SystemExit("All transformer capacities must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, events, diagnostics = run_sweep(
        tuple(args.capacities_kw),
        args.timestep_minutes,
    )
    scores.to_csv(output_dir / "capacity_sweep_scores.csv", index=False)
    events.to_csv(output_dir / "capacity_sweep_event_records.csv", index=False)
    diagnostics.to_csv(output_dir / "capacity_sweep_statistics.csv", index=False)
    scores.pivot_table(
        index=["capacity_kw", "horizon_hours"],
        columns="dataset",
        values="criterion_x",
    ).to_csv(output_dir / "capacity_sweep_comparison.csv")
    verification = pd.DataFrame(
        [raw_verification(name, path) for name, path in DATASETS.items()]
    )
    verification.to_csv(output_dir / "dataset_verification.csv", index=False)
    configuration = {
        "capacities_kw": list(args.capacities_kw),
        "timestep_minutes": args.timestep_minutes,
        "horizons_hours": list(HORIZONS_HOURS),
        "criterion": "median_h[A(l,h) * I(S(l,h) > 0)]",
        "demand_definition": "sum(min(6.6, requested_kWh / timestep_hours))",
        "risk_definition": "greedy achievable draw > capacity",
        "office001": "unsupported without requested_kWh; not scored using kWhDelivered",
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    save_figures(scores, diagnostics, output_dir)
    print(scores.to_string(index=False))
    print(f"Saved capacity sweep outputs under {output_dir}")


if __name__ == "__main__":
    main()
