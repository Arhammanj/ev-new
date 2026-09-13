# ==========================================
# Publication-quality figures from existing results
# ==========================================
# Generates three figures for the paper:
#   Figure 1: Lead-time finding (dual-axis)
#   Figure 2: Main trade-off (scatter plot)
#   Figure 3: Training curves (SKIPPED if reward logs not available)
#
# All figures use consistent styling, color mapping, and IEEE column width.
# Saves as both PDF (vector) and PNG (300 dpi) to paper_figures/.
# ==========================================

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import existing plot style
from plot_style import set_style

# ==========================================
# CONFIGURATION
# ==========================================

OUTPUT_DIR = "paper_figures"
FIGURE_SIZE_INCH = (3.5, 2.4)  # IEEE single-column width

# Fixed, consistent color mapping across all conditions
COLOR_MAP = {
    "cap_only": "#1f77b4",      # blue
    "no_forecast": "#ff7f0e",    # orange
    "forecast": "#2ca02c",       # green
}

# Hardcoded verified values - do not compute
FIGURE_1_DATA = {
    "lead_hours": [1, 3, 6],
    "pct_connected": [64.3, 12.2, 0.7],
    "median_slack_kw": [0.0, 0.0, 43.4],
}

FIGURE_2_DATA = {
    "cap_only": {
        "delivered_kwh": 2161.6,
        "violation_pct": 0.00,
        "delivered_err": 0.0,
        "violation_err": 0.0,
    },
    "no_forecast": {
        "delivered_kwh": 1884.3,
        "violation_pct": 3.29,
        "delivered_err": 34.9,
        "violation_err": 0.94,
    },
    "forecast": {
        "delivered_kwh": 1899.0,
        "violation_pct": 3.00,
        "delivered_err": 24.3,
        "violation_err": 0.53,
    },
}

# Figure 3: Forecaster accuracy (WAPE %)
FIGURE_3_DATA = {
    "models": ["BiLSTM", "LightGBM", "Seasonal-naive 168h", "Seasonal-naive 24h", "Persistence"],
    "wape": [23.80, 67.11, 67.83, 71.13, 132.74],
    "highlight_idx": 0,  # BiLSTM is first (top bar, shortest)
}

# Figure 4: Forecast alignment vs. horizon
FIGURE_4_DATA = {
    "horizons": [1, 3, 6],
    "corr_achievable": [0.40, 0.72, 0.90],
    "corr_violation": [0.33, 0.63, 0.81],
}


def setup_figure_style():
    """Configure matplotlib for publication-quality output."""
    set_style()
    
    # Override to IEEE single-column dimensions
    plt.rcParams.update({
        "figure.figsize": FIGURE_SIZE_INCH,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, name: str):
    """Save figure as both PDF and PNG at 300 dpi."""
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


def figure_1_lead_time():
    """
    Dual-axis chart: lead time vs. EV presence and system slack.
    
    X-axis: lead time (1h, 3h, 6h)
    Left y-axis: % of surge-causing EVs already present (bars)
    Right y-axis: median system slack in kW (line with markers)
    
    Demonstrates the key finding: vehicles and spare capacity never
    co-occur, making anticipation physically impossible.
    Direct labeling (no legend) to avoid overlap.
    """
    print("\n[Figure 1] Lead-time finding...")
    
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE_INCH)
    
    data = FIGURE_1_DATA
    x_pos = np.arange(len(data["lead_hours"]))
    x_labels = [f"{h}h" for h in data["lead_hours"]]
    
    # Left y-axis: % already connected (bars)
    bars = ax1.bar(
        x_pos,
        data["pct_connected"],
        color=COLOR_MAP["no_forecast"],
        alpha=0.7,
        width=0.5,
    )
    ax1.set_ylabel("% of surge-causing EVs\nalready present", fontsize=10, fontweight="normal")
    ax1.set_xlabel("Lead time before violation-risk hour", fontsize=10, fontweight="normal")
    ax1.set_ylim(0, 120)  # Extra headroom for labels at top
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Add bar % labels ABOVE each bar, centered
    for bar, val in zip(bars, data["pct_connected"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=COLOR_MAP["no_forecast"],
        )
    
    # Direct label for bar series (near 1h bar)
    ax1.text(
        0.08,
        0.95,
        "% already connected",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR_MAP["no_forecast"],
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    
    # Right y-axis: median system slack (line with markers)
    ax2 = ax1.twinx()
    line = ax2.plot(
        x_pos,
        data["median_slack_kw"],
        color=COLOR_MAP["forecast"],
        marker="o",
        linewidth=2.0,
        markersize=6,
        zorder=5,
    )
    ax2.set_ylabel("Median system slack (kW)", fontsize=10, fontweight="normal")
    ax2.set_ylim(-5, 50)
    
    # Add slack kW labels BELOW-LEFT of each marker
    slack_labels = [
        (0, -4.5),   # 1h: below-left
        (1, -4.5),   # 3h: below-left
        (2, -5),     # 6h: below-left (further down to avoid edge)
    ]
    for i, (x, y) in enumerate(zip(x_pos, data["median_slack_kw"])):
        label_x, offset_y = slack_labels[i]
        ax2.text(
            x - 0.08,
            y + offset_y,
            f"{y:.1f} kW",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=COLOR_MAP["forecast"],
        )
    
    # Direct label for line series (between 3h and 6h, above line)
    ax2.text(
        1.6,
        35,
        "median slack (kW)",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR_MAP["forecast"],
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    
    fig.suptitle(
        "No exploitable lead time: vehicles and spare capacity\nnever co-occur",
        fontsize=11,
        fontweight="normal",
        y=0.98,
    )
    
    fig.tight_layout()
    save_figure(fig, "figure_1_lead_time")
    plt.close(fig)


def figure_2_main_tradeoff():
    """
    Scatter plot: violation rate vs. delivered energy.
    
    X-axis: delivered energy (kWh)
    Y-axis: violation rate (%)
    
    Three points in consistent condition colors:
    - cap_only: (2161.6, 0.00) deterministic, no error bars
    - no_forecast: (1884.3, 3.29) with error bars
    - forecast: (1899.0, 3.00) with error bars
    
    Visual story: cap_only dominates in the top-right (high energy,
    zero violations); learned controllers overlap and are dominated.
    Explicit label placement with xytext to avoid all collisions.
    """
    print("\n[Figure 2] Main trade-off...")
    
    fig, ax = plt.subplots(figsize=(3.8, 2.4))  # Slightly wider for label room
    
    data = FIGURE_2_DATA
    
    # Plot each condition
    for condition, color in COLOR_MAP.items():
        d = data[condition]
        xerr = d["delivered_err"] if d["delivered_err"] > 0 else None
        yerr = d["violation_err"] if d["violation_err"] > 0 else None
        
        ax.errorbar(
            d["delivered_kwh"],
            d["violation_pct"],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=7,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            alpha=0.8,
            zorder=5,
        )
    
    # EXPLICIT label placement with xytext offsets (no automatic placement)
    # Cap Only - ABOVE the blue point
    ax.annotate(
        "Cap Only\n(2162, 0.00%)",
        xy=(2161.6, 0.00),
        xytext=(2161.6, 1.2),
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5),
        arrowprops=None,  # No arrow - label floats above
    )
    
    # No Forecast - LEFT of the orange point
    ax.annotate(
        "No Forecast\n(1884, 3.29%)",
        xy=(1884.3, 3.29),
        xytext=(1800, 3.5),
        ha="center",
        va="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5),
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", color="gray", lw=0.5),
    )
    
    # Forecast - ABOVE the green point
    ax.annotate(
        "Forecast\n(1899, 3.00%)",
        xy=(1899.0, 3.00),
        xytext=(1899, 4.0),
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5),
        arrowprops=None,  # No arrow
    )
    
    # Formatting
    ax.set_xlabel("Delivered energy (kWh)", fontsize=10, fontweight="normal")
    ax.set_ylabel("Violation rate (%)", fontsize=10, fontweight="normal")
    ax.set_xlim(1700, 2250)
    ax.set_ylim(-0.5, 5.5)
    ax.grid(True, alpha=0.3)
    
    # Add annotation in lower-left empty region (x=1720, y=1)
    ax.text(
        1730,
        0.8,
        "Cap-only dominates:\nhigh energy,\nzero violations.",
        ha="left",
        va="center",
        fontsize=8,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.7, edgecolor="gray", linewidth=0.5),
    )
    
    fig.suptitle(
        "Main trade-off: energy vs. constraint satisfaction",
        fontsize=11,
        fontweight="normal",
        y=0.98,
    )
    
    fig.tight_layout()
    save_figure(fig, "figure_2_tradeoff")
    plt.close(fig)


def figure_3_forecast_wape():
    """
    Horizontal bar chart: WAPE (%) for five forecaster models.
    
    Models ordered best-to-worst (shortest bar at top).
    BiLSTM highlighted in accent color (green).
    Others in neutral gray.
    
    Demonstrates that BiLSTM is substantially more accurate than
    classical baselines.
    """
    print("\n[Figure 3] Forecaster accuracy...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCH)
    
    data = FIGURE_3_DATA
    y_pos = np.arange(len(data["models"]))
    
    # Color each bar: BiLSTM (highlight index) in green, others in gray
    colors = ["#888888"] * len(data["models"])
    colors[data["highlight_idx"]] = COLOR_MAP["forecast"]  # green for BiLSTM
    
    bars = ax.barh(y_pos, data["wape"], color=colors, alpha=0.8)
    
    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["models"], fontsize=9)
    ax.set_xlabel("WAPE (%)", fontsize=10, fontweight="normal")
    ax.set_xlim(0, 145)
    ax.invert_yaxis()  # Top bar = best (shortest)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add value labels on each bar (right-aligned, outside bar)
    for i, (bar, val) in enumerate(zip(bars, data["wape"])):
        ax.text(
            val + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=colors[i],
        )
    
    fig.suptitle(
        "BiLSTM forecast is substantially more accurate\nthan baselines",
        fontsize=11,
        fontweight="normal",
        y=0.98,
    )
    
    fig.tight_layout()
    save_figure(fig, "figure_3_forecast_wape")
    plt.close(fig)


def figure_4_alignment():
    """
    Line plot: Forecast alignment (correlation) vs. horizon.
    
    Two series:
    - Correlation with future achievable draw (green)
    - Correlation with violation occurrence (orange)
    
    Both rising left-to-right, showing that longer horizons have
    stronger alignment with transformer pressure dynamics.
    """
    print("\n[Figure 4] Forecast alignment...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCH)
    
    data = FIGURE_4_DATA
    x_pos = np.array(data["horizons"])
    
    # Plot both series
    line1 = ax.plot(
        x_pos,
        data["corr_achievable"],
        color=COLOR_MAP["forecast"],  # green
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="corr. w/ future achievable draw",
        zorder=5,
    )
    
    line2 = ax.plot(
        x_pos,
        data["corr_violation"],
        color=COLOR_MAP["no_forecast"],  # orange
        marker="s",
        linewidth=2.0,
        markersize=6,
        label="corr. w/ violation occurrence",
        zorder=5,
    )
    
    # Formatting
    ax.set_xlabel("Forecast horizon (hours)", fontsize=10, fontweight="normal")
    ax.set_ylabel("Correlation coefficient", fontsize=10, fontweight="normal")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{h}h" for h in x_pos])
    ax.set_ylim(0.25, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on each point (offset above/below alternately)
    for i, (x, y_a, y_v) in enumerate(zip(x_pos, data["corr_achievable"], data["corr_violation"])):
        # Achievable draw correlation (green circles) - label above
        ax.text(
            x,
            y_a + 0.03,
            f"{y_a:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=COLOR_MAP["forecast"],
        )
        # Violation occurrence correlation (orange squares) - label below
        ax.text(
            x,
            y_v - 0.03,
            f"{y_v:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=COLOR_MAP["no_forecast"],
        )
    
    # Legend (clean placement, doesn't overlap lines)
    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.95,
        edgecolor="gray",
    )
    
    fig.suptitle(
        "Forecast alignment with transformer pressure\nincreases with horizon",
        fontsize=11,
        fontweight="normal",
        y=0.98,
    )
    
    fig.tight_layout()
    save_figure(fig, "figure_4_alignment")
    plt.close(fig)


HORIZON_COMPARISON_CSV = "acn_horizon_comparison.csv"


def figure_5_lead_time_controller_performance():
    """
    Dual-axis chart: actual trained-controller evidence for the lead-time
    argument, across every forecast horizon that was trained end-to-end
    (see report_horizon_comparison.py).

    X-axis: forecast horizon (1h, 3h, 6h)
    Left y-axis: forecast WAPE (%) - lower is a more accurate forecast
    Right y-axis: forecast-vs-no_forecast delivered-kWh delta (line) -
        positive means the forecast condition delivers more energy than
        the no-forecast baseline at that horizon

    SKIPPED if acn_horizon_comparison.csv doesn't exist yet (i.e. the
    per-horizon PPO training/eval sweep hasn't been run).
    """
    print("\n[Figure 5] Lead time vs. forecast accuracy vs. controller performance...")

    if not os.path.exists(HORIZON_COMPARISON_CSV):
        print(f"  SKIPPED: {HORIZON_COMPARISON_CSV} not found.")
        print("  (Run report_horizon_comparison.py after training/evaluating "
              "every horizon to generate it.)")
        return False

    df = pd.read_csv(HORIZON_COMPARISON_CSV).sort_values("horizon_h")
    x_pos = np.arange(len(df))
    x_labels = [f"{int(h)}h" for h in df["horizon_h"]]

    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE_INCH)

    bars = ax1.bar(
        x_pos, df["forecast_wape"], color=COLOR_MAP["no_forecast"], alpha=0.7, width=0.5,
    )
    ax1.set_ylabel("Forecast WAPE (%)", fontsize=10, fontweight="normal")
    ax1.set_xlabel("Forecast horizon (exploitable lead time)", fontsize=10, fontweight="normal")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, df["forecast_wape"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color=COLOR_MAP["no_forecast"],
        )

    ax2 = ax1.twinx()
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax2.plot(
        x_pos, df["kwh_diff"], color=COLOR_MAP["forecast"], marker="o",
        linewidth=2.0, markersize=6, zorder=5,
    )
    ax2.set_ylabel("Δ delivered kWh\n(forecast - no_forecast)", fontsize=8, fontweight="normal")

    for x, y in zip(x_pos, df["kwh_diff"]):
        ax2.text(
            x, y, f"{y:+.1f}", ha="center",
            va="bottom" if y >= 0 else "top", fontsize=8.5, fontweight="bold",
            color=COLOR_MAP["forecast"],
        )

    fig.suptitle(
        "Forecast accuracy and controller benefit\nvs. lead time",
        fontsize=10, fontweight="normal", y=1.06,
    )

    fig.tight_layout()
    save_figure(fig, "figure_5_lead_time_controller_performance")
    plt.close(fig)
    return True


CAPACITY_COMPARISON_CSV = "acn_capacity_comparison.csv"


def figure_6_capacity_sweep():
    """
    Line plot: violation rate and delivered kWh vs. transformer capacity,
    one series per condition (cap_only, no_forecast, forecast), across
    every capacity that was trained/evaluated (see
    report_capacity_comparison.py). Companion to Figure 2's single-capacity
    (50kW) snapshot - this shows how all three conditions move as the
    constraint is relaxed (40kW, tightest) to loosened (100kW).

    SKIPPED if acn_capacity_comparison.csv doesn't exist yet (i.e. the
    per-capacity PPO training/eval sweep hasn't been run).
    """
    print("\n[Figure 6] Capacity sweep...")

    if not os.path.exists(CAPACITY_COMPARISON_CSV):
        print(f"  SKIPPED: {CAPACITY_COMPARISON_CSV} not found.")
        print("  (Run report_capacity_comparison.py after training/evaluating "
              "every capacity to generate it.)")
        return False

    df = pd.read_csv(CAPACITY_COMPARISON_CSV).sort_values("capacity_kw")
    x = df["capacity_kw"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_SIZE_INCH[0] * 2, FIGURE_SIZE_INCH[1]))

    series = [
        ("cap_only", "cap_only_viol_pct", "cap_only_kwh", "s"),
        ("no_forecast", "no_forecast_viol_mean_pct", "no_forecast_kwh_mean", "o"),
        ("forecast", "forecast_viol_mean_pct", "forecast_kwh_mean", "^"),
    ]
    for condition, viol_col, kwh_col, marker in series:
        ax1.plot(x, df[viol_col], color=COLOR_MAP[condition], marker=marker,
                 linewidth=1.8, markersize=5, label=condition)
        ax2.plot(x, df[kwh_col], color=COLOR_MAP[condition], marker=marker,
                 linewidth=1.8, markersize=5, label=condition)

    ax1.set_xlabel("Transformer capacity (kW)", fontsize=9)
    ax1.set_ylabel("Violation rate (%)", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, framealpha=0.95, edgecolor="gray")

    ax2.set_xlabel("Transformer capacity (kW)", fontsize=9)
    ax2.set_ylabel("Delivered kWh (per test week)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, framealpha=0.95, edgecolor="gray")

    fig.suptitle("Controller performance vs. transformer capacity", fontsize=11, y=1.03)
    fig.tight_layout()
    save_figure(fig, "figure_6_capacity_sweep")
    plt.close(fig)
    return True


def check_reward_logs_exist() -> bool:
    """
    Check if per-seed training reward logs are available.
    
    Returns True if logs found, False otherwise.
    """
    # Typical stable-baselines3 Monitor output
    candidates = [
        "acn_exp_2x2_checkpoints/no_forecast_seed1.monitor.csv",
        "acn_exp_2x2_checkpoints/forecast_seed1.monitor.csv",
        "train_logs/2x2_rewards.json",
        "acn_exp_2x2_training_logs/",
    ]
    
    for cand in candidates:
        if os.path.exists(cand):
            return True
    
    return False


def figure_3_training_curves():
    """
    Training curves: episode reward over timesteps.
    
    Line plot with shaded ±std bands for each learned condition
    (no_forecast, forecast) across 5 seeds.
    
    SKIPPED if reward logs are not available on disk.
    """
    print("\n[Figure 3] Training curves...")
    
    if not check_reward_logs_exist():
        print("  SKIPPED: Training reward logs not found on disk.")
        print("  (To include Figure 3, add per-seed reward logging to training.)")
        return False
    
    print("  Reward logs found. Figure would be generated here.")
    print("  (Implementation pending if logs are located.)")
    return False


def main():
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 70)
    
    setup_figure_style()
    ensure_output_dir()
    
    # Generate figures
    figure_1_lead_time()
    figure_2_main_tradeoff()
    figure_3_forecast_wape()
    figure_4_alignment()
    fig5_generated = figure_5_lead_time_controller_performance()
    fig6_generated = figure_6_capacity_sweep()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"[OK] Figure 1 (Lead-time):       Generated")
    print(f"[OK] Figure 2 (Trade-off):       Generated")
    print(f"[OK] Figure 3 (Forecast WAPE):   Generated")
    print(f"[OK] Figure 4 (Alignment):       Generated")
    print(f"{'[OK]' if fig5_generated else '[--]'} Figure 5 (Horizon sweep):    "
          f"{'Generated' if fig5_generated else 'Skipped (no horizon sweep results yet)'}")
    print(f"{'[OK]' if fig6_generated else '[--]'} Figure 6 (Capacity sweep):   "
          f"{'Generated' if fig6_generated else 'Skipped (no capacity sweep results yet)'}")
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("  - PDF (vector, publication-ready)")
    print("  - PNG (300 dpi, for preview/slides)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
