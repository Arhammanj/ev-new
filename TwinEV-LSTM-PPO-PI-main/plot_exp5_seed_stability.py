"""Plot the legacy FULL TwinEV Experiment 5 seed-stability results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import set_style

INPUT_CSV = Path("results/statistics/twinev_seed_stability.csv")
OUTPUT_DIR = Path("results/plots")
PAPER_DIR = Path("paper_figures")

PANELS = [
    ("energy_delivered_kwh", "Delivered energy (kWh)", "(a)", "#1f77b4"),
    ("peak_power_kw", "Peak power (kW)", "(b)", "#d62728"),
    ("eval_total_reward", "Evaluation reward", "(c)", "#2ca02c"),
    ("power_variance", "Power variance (kW$^2$)", "(d)", "#9467bd"),
]


def main():
    set_style()
    plt.rcParams.update({
        "figure.figsize": (7.16, 4.8),
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    df = pd.read_csv(INPUT_CSV)
    seeds = df["seed"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))
    for ax, (column, label, tag, color) in zip(axes.flat, PANELS):
        values = df[column].to_numpy()
        mean = values.mean()
        std = values.std(ddof=1)
        cv = 100.0 * std / abs(mean)

        ax.plot(seeds, values, "o", color=color, markersize=6, zorder=3)
        ax.axhline(mean, color=color, linewidth=1.4, zorder=2)
        ax.fill_between(
            [seeds.min() - 0.4, seeds.max() + 0.4],
            mean - std,
            mean + std,
            color=color,
            alpha=0.15,
            zorder=1,
        )
        ax.set_xlim(seeds.min() - 0.4, seeds.max() + 0.4)
        ax.set_xticks(seeds)
        ax.set_xlabel("PPO seed")
        ax.set_ylabel(label)
        ax.set_title(
            f"{tag} {label}\n"
            f"mean {mean:,.2f} ± {std:,.2f}  (CV {cv:.1f}%)",
            loc="left",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Legacy Experiment 5: FULL TwinEV seed stability (5 PPO seeds)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.005,
        "All seeds: 0 hard (>150 kW) and 0 soft (>130 kW) overloads. "
        "The legacy FULL TwinEV environment hard-clips at 150 kW, so the "
        "zero-overload count is by construction, not a learned-safety result.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUTPUT_DIR / "twinev_seed_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "twinev_seed_stability.pdf", bbox_inches="tight")
    fig.savefig(
        PAPER_DIR / "legacy_exp5_twinev_seed_stability.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        PAPER_DIR / "legacy_exp5_twinev_seed_stability.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    for path in [
        OUTPUT_DIR / "twinev_seed_stability.png",
        OUTPUT_DIR / "twinev_seed_stability.pdf",
        PAPER_DIR / "legacy_exp5_twinev_seed_stability.png",
        PAPER_DIR / "legacy_exp5_twinev_seed_stability.pdf",
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
