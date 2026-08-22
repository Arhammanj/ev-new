# ==========================================
# TwinEV v2 - Experiment 8 plots
# SERVICE-CONSTRAINED TwinEV
# ==========================================
# Generates the 7 requested plots under
# results/plots/ using the CSVs produced by
# exp8_service_constrained.py.
# ==========================================

import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from plot_style import set_style
from plot_utils import save_figure

set_style()

OUT = "results/plots"
SUMMARY = "results/service_constrained/service_constrained_summary.csv"
CEILING = 77.73  # 150 kW * 1286 h / 248165.66 kWh


def load():
    df = pd.read_csv(SUMMARY)
    df["label"] = df["model"].str.replace("_", " ").str.title()
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()

    zero_hard = df["zero_hard_overload"].astype(bool).values
    met = df["service_target_met"].fillna(False).astype(bool).values
    greens = ["#2e8b57", "#2e8b57", "#2e8b57", "#2e8b57", "#2e8b57"]
    m = ["o", "s", "^", "D", "v", "*", "P", "X", "h"]

    # --------------------------------------------------
    # 1. Energy vs overload (hard + soft)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    x = df["energy_delivered_mwh"].values
    y = df["hard_overloads"].values + df["soft_overloads"].values
    sc = ax.scatter(x, y, s=140, c=zero_hard, cmap="RdYlGn", vmin=0, vmax=1,
                    edgecolors="k", zorder=3)
    for i, r in df.iterrows():
        ax.annotate(r["label"], (r["energy_delivered_mwh"],
                                 y[i]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Energy delivered (MWh)")
    ax.set_ylabel("Overload events (hard + soft)")
    ax.set_title("Service Energy vs Grid Overloads")
    ax.grid(True, alpha=0.3)
    save_figure(OUT, "service_energy_vs_overload.png")

    # --------------------------------------------------
    # 2. Service target vs actual
    # --------------------------------------------------
    serv = df[df["type"] == "rl"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = serv["label"].tolist()
    actual = serv["energy_delivery_pct"].values
    target = serv["service_target_pct"].fillna(0).values
    idx = np.arange(len(serv))
    ax.bar(idx - 0.2, actual, 0.4, label="Actual service ratio (%)",
           color="#4682b4")
    ax.bar(idx + 0.2, target, 0.4, label="Service target (%)",
           color="#ffa07a")
    ax.axhline(CEILING, color="red", linestyle="--",
               label=f"Physical ceiling ({CEILING:.1f}%)")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Percent of available EV energy")
    ax.set_ylim(0, 110)
    ax.set_title("Service Target vs Achieved")
    ax.legend()
    save_figure(OUT, "service_target_vs_actual.png")

    # --------------------------------------------------
    # 3. Energy comparison
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(df["label"], df["energy_delivered_mwh"],
                  color=[("#2e8b57" if z else "#d62728")
                         for z in zero_hard])
    ax.axhline(df["energy_delivered_mwh"].iloc[2],
               color="gray", linestyle="--",
               label="Greedy 130 kW energy")
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    ax.set_ylabel("Energy delivered (MWh)")
    ax.set_title("Energy Delivery Comparison")
    ax.legend()
    save_figure(OUT, "service_energy_comparison.png")

    # --------------------------------------------------
    # 4. Peak power comparison
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df["label"], df["peak_power_kw"],
           color=[("#2e8b57" if z else "#d62728") for z in zero_hard])
    ax.axhline(150, color="red", linestyle=":", label="Hard limit (150 kW)")
    ax.axhline(130, color="orange", linestyle=":", label="Soft limit (130 kW)")
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    ax.set_ylabel("Peak grid power (kW)")
    ax.set_title("Peak Power Comparison")
    ax.legend()
    save_figure(OUT, "service_peak_comparison.png")

    # --------------------------------------------------
    # 5. Cost comparison
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df["label"], df["charging_cost"] / 1000.0,
           color=[("#2e8b57" if z else "#d62728") for z in zero_hard])
    for i, r in df.iterrows():
        ax.annotate(f"{r['charging_cost']:,.0f}",
                    (i, r['charging_cost'] / 1000.0),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    ax.set_ylabel("Charging cost (thousand EUR)")
    ax.set_title("Charging Cost Comparison")
    save_figure(OUT, "service_cost_comparison.png")

    # --------------------------------------------------
    # 6. Variance comparison
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df["label"], df["power_variance"],
           color=[("#2e8b57" if z else "#d62728") for z in zero_hard])
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    ax.set_ylabel("Power variance (kW^2)")
    ax.set_title("Power Variance Comparison")
    save_figure(OUT, "service_variance_comparison.png")

    # --------------------------------------------------
    # 7. MOST IMPORTANT - safety-service trade-off
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 7))
    models = ["Greedy 130 kW", "Original TwinEV",
              "twinev_service70", "twinev_service80",
              "twinev_service90", "twinev_service95"]
    sub = df[df["model"].isin(models)].copy()
    sub = sub.sort_values("energy_delivery_pct")
    colors = []
    for _, r in sub.iterrows():
        if r["zero_hard_overload"]:
            colors.append("#2e8b57" if r["service_target_met"] else "#1f77b4")
        else:
            colors.append("#d62728")

    for pos, (_, r) in enumerate(sub.iterrows()):
        mrk = "o"
        edge = "k"
        if r["zero_hard_overload"]:
            edge = "#2e8b57" if r["service_target_met"] else "#1f77b4"
        ax.scatter(r["energy_delivery_pct"], r["hard_overloads"],
                   s=220, c=colors[pos], marker=mrk, edgecolors=edge,
                   linewidths=2, zorder=3)
        tgt = (f"  ({r['label']})"
               if pd.isna(r["service_target_pct"]) else
               f"  {r['label']}\n  target {r['service_target_pct']:.0f}%"
               + ("  MET" if r["service_target_met"] else "  not met"))
        ax.annotate(tgt, (r["energy_delivery_pct"], r["hard_overloads"]),
                    textcoords="offset points", xytext=(12, 8), fontsize=9,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=3,
                                                foreground="white")])

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(CEILING, color="gray", linestyle="--",
               label=f"Physical ceiling ({CEILING:.1f}%)")
    ax.text(98, 20, "SAFE zone\n(hard overloads = 0)",
            color="#2e8b57", fontsize=11, ha="right", va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#2e8b57"))
    ax.set_xlim(35, 105)
    ax.set_ylim(-60, 1350)
    ax.set_xlabel("Energy service ratio (%)")
    ax.set_ylabel("Hard overload events")
    ax.set_title("Safety-Service Trade-Off (TwinEV v2)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    save_figure(OUT, "service_tradeoff.png")

    print("\nAll 7 service-constrained plots saved.")


if __name__ == "__main__":
    main()
