# ==========================================
# TwinEV v2 - publication-quality plots
# ==========================================
# Reads the experiment CSVs/profiles and
# writes every figure listed in the paper's
# results/plots directory.
# ==========================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import set_style

set_style()

PLOTS = "results/plots"
os.makedirs(PLOTS, exist_ok=True)

C1, C2 = "#2c7be5", "#e74a3b"
SOFT, HARD = 130.0, 150.0

MODEL_LABELS = {
    "A_ppo_no_forecast": "PPO + PI",
    "B_full_twinev": "BiLSTM + PPO + PI",
    "C_bilstm_ppo_no_pi": "BiLSTM + PPO",
    "D_bilstm_ppo_pi": "BiLSTM + PPO + PI",
}


def save(name):
    path = os.path.join(PLOTS, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def limit_lines(ax, label_fmt=False):
    ax.axhline(HARD, color=C2, linestyle=":", linewidth=1.2,
               label="Hard limit 150 kW")
    ax.axhline(SOFT, color="#f39c12", linestyle="--", linewidth=1.2,
               label="Soft limit 130 kW")


# ==========================================
# EXPERIMENT 1 - FORECAST ABLATION
# ==========================================

def exp1():
    df = pd.read_csv("results/ablation/forecast_ablation.csv")
    df = df.copy()
    df["label"] = df["model"].map(MODEL_LABELS)
    x = df["label"]
    order = x.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Experiment 1 - Forecast Ablation (same PPO+PI, forecast on/off)",
                 fontsize=15)

    ax = axes[0, 0]
    ax.bar(x, df["peak_power_kw"], color=[C1, C2])
    limit_lines(ax)
    ax.set_ylabel("Peak power (kW)")
    ax.set_title("Peak power")

    ax = axes[0, 1]
    w = 0.35
    ax.bar(np.arange(2) - w / 2, df["hard_overloads"], w, label="Hard")
    ax.bar(np.arange(2) + w / 2, df["soft_overloads"], w, label="Soft",
           color="#f39c12")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(x)
    ax.legend(); ax.set_title("Overload events"); ax.set_ylabel("Events")

    ax = axes[0, 2]
    ax.bar(x, df["energy_delivered_kwh"], color=[C1, C2])
    for i, v in enumerate(df["energy_delivered_kwh"]):
        ax.text(i, v, f"{v/1e3:.1f} MWh", ha="center", va="bottom")
    ax.set_ylabel("Energy delivered (kWh)")
    ax.set_title("Energy delivered")

    ax = axes[1, 0]
    ax.bar(x, df["charging_cost"], color=[C1, C2])
    ax.set_ylabel("Charging cost (EUR)")
    ax.set_title("Charging cost")

    ax = axes[1, 1]
    ax.bar(x, df["power_variance"], color=[C1, C2])
    ax.set_ylabel("Power variance")
    ax.set_title("Power variance")

    ax = axes[1, 2]
    ax.bar(x, df["total_reward"], color=[C1, C2])
    ax.set_ylabel("Total reward")
    ax.set_title("Total reward")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save("forecast_ablation_summary.png")

    for col, fname, ylabel in [
        ("peak_power_kw", "forecast_ablation_peak.png", "Peak power (kW)"),
        ("energy_delivered_kwh", "forecast_ablation_energy.png", "Energy (kWh)"),
        ("charging_cost", "forecast_ablation_cost.png", "Cost (EUR)"),
        ("power_variance", "forecast_ablation_variance.png", "Variance"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, df[col], color=[C1, C2])
        if col == "peak_power_kw":
            limit_lines(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Forecast ablation - {ylabel}")
        fig.tight_layout()
        save(fname)

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.35
    ax.bar(np.arange(2) - w / 2, df["hard_overloads"], w,
           label="Hard", color=C2)
    ax.bar(np.arange(2) + w / 2, df["soft_overloads"], w,
           label="Soft", color="#f39c12")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(x)
    ax.set_ylabel("Overload events")
    ax.set_title("Forecast ablation - overload events")
    ax.legend()
    fig.tight_layout()
    save("forecast_ablation_overload.png")


# ==========================================
# EXPERIMENT 2 - PI CONTROL ABLATION
# ==========================================

def exp2():
    df = pd.read_csv("results/ablation/pi_ablation.csv").copy()
    df["label"] = df["model"].map(MODEL_LABELS)
    x = df["label"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Experiment 2 - PI Control Ablation (BiLSTM+PPO, PI on/off)",
                 fontsize=15)

    ax = axes[0, 0]
    w = 0.35
    ax.bar(np.arange(2) - w / 2, df["mean_ramp_kw"], w, label="Mean ramp")
    ax.bar(np.arange(2) + w / 2, df["max_ramp_kw"], w, label="Max ramp",
           color="#f39c12")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(x)
    ax.legend(); ax.set_title("Power ramps"); ax.set_ylabel("Ramp (kW/h)")

    ax = axes[0, 1]
    ax.bar(x, df["power_variance"], color=[C1, C2])
    ax.set_title("Power variance"); ax.set_ylabel("Variance")

    ax = axes[0, 2]
    ax.bar(x, df["peak_power_kw"], color=[C1, C2])
    limit_lines(ax)
    ax.set_title("Peak power"); ax.set_ylabel("Peak (kW)")

    ax = axes[1, 0]
    w = 0.35
    ax.bar(np.arange(2) - w / 2, df["mean_tracking_error_kw"], w,
           label="Mean tracking err")
    ax.bar(np.arange(2) + w / 2, df["max_tracking_error_kw"], w,
           label="Max tracking err", color="#f39c12")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(x)
    ax.legend(); ax.set_title("Tracking error"); ax.set_ylabel("|err| (kW)")

    ax = axes[1, 1]
    ax.bar(x, df["energy_delivered_kwh"], color=[C1, C2])
    for i, v in enumerate(df["energy_delivered_kwh"]):
        ax.text(i, v, f"{v/1e3:.1f} MWh", ha="center", va="bottom")
    ax.set_title("Energy delivered"); ax.set_ylabel("Energy (kWh)")

    ax = axes[1, 2]
    ax.bar(x, df["total_reward"], color=[C1, C2])
    ax.set_title("Total reward"); ax.set_ylabel("Reward")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save("pi_ablation_summary.png")

    for col, fname, ylabel, grouped in [
        ("peak_power_kw", "pi_ablation_peak.png", "Peak power (kW)", False),
        ("power_variance", "pi_ablation_variance.png", "Variance", False),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, df[col], color=[C1, C2])
        if grouped:
            limit_lines(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"PI ablation - {ylabel}")
        fig.tight_layout()
        save(fname)

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.35
    ax.bar(np.arange(2) - w / 2, df["mean_ramp_kw"], w, label="Mean ramp")
    ax.bar(np.arange(2) + w / 2, df["max_ramp_kw"], w, label="Max ramp",
           color="#f39c12")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(x)
    ax.set_ylabel("Ramp (kW/h)")
    ax.set_title("PI ablation - power ramps (smoothness)")
    ax.legend()
    fig.tight_layout()
    save("pi_ablation_ramp.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.35
    ax.bar(np.arange(2) - w / 2, df["mean_tracking_error_kw"], w,
           label="Mean tracking err")
    ax.bar(np.arange(2) + w / 2, df["max_tracking_error_kw"], w,
           label="Max tracking err", color="#f39c12")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(x)
    ax.set_ylabel("|error| (kW)")
    ax.set_title("PI ablation - tracking error")
    ax.legend()
    fig.tight_layout()
    save("pi_ablation_tracking.png")


# ==========================================
# EXPERIMENT 3 - COMPLETE CONTROLLER COMPARISON
# ==========================================

def exp3():
    df = pd.read_csv("results/comparison/all_models.csv").copy()
    order = ["Raw Demand", "FCFS", "Round Robin", "Greedy",
             "PPO + PI", "BiLSTM + PPO", "BiLSTM + PPO + PI"]
    df["label"] = pd.Categorical(df["model"], categories=order,
                                 ordered=True)
    df = df.sort_values("label")
    x = df["label"].astype(str).tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Experiment 3 - Complete Controller Comparison (test window)",
                 fontsize=15)

    ax = axes[0, 0]
    colors = ["#95a5a6"] * 4 + [C1, "#8e44ad", C2]
    bars = ax.bar(x, df["peak_power_kw"], color=colors)
    limit_lines(ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Peak power"); ax.set_ylabel("Peak (kW)")

    ax = axes[0, 1]
    w = 0.35
    xx = np.arange(len(df))
    ax.bar(xx - w / 2, df["hard_overloads"], w, label="Hard", color=C2)
    ax.bar(xx + w / 2, df["soft_overloads"], w, label="Soft", color="#f39c12")
    ax.set_xticks(xx); ax.set_xticklabels(x, rotation=35)
    ax.legend(); ax.set_title("Overload events"); ax.set_ylabel("Events")

    ax = axes[0, 2]
    ax.bar(x, df["energy_delivered_kwh"], color=colors)
    for i, v in enumerate(df["energy_delivered_kwh"]):
        ax.text(i, v, f"{v/1e3:.0f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Energy delivered (MWh)"); ax.set_ylabel("Energy (kWh)")

    ax = axes[1, 0]
    ax.bar(x, df["charging_cost"], color=colors)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Charging cost"); ax.set_ylabel("Cost (EUR)")

    ax = axes[1, 1]
    ax.bar(x, df["power_variance"], color=colors)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Power variance"); ax.set_ylabel("Variance")

    ax = axes[1, 2]
    w = 0.35
    ax.bar(xx - w / 2, df["mean_ramp_kw"], w, label="Mean ramp")
    ax.bar(xx + w / 2, df["max_ramp_kw"], w, label="Max ramp",
           color="#f39c12")
    ax.set_xticks(xx); ax.set_xticklabels(x, rotation=35)
    ax.legend(); ax.set_title("Power ramps"); ax.set_ylabel("Ramp (kW/h)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save("all_models_summary.png")

    plots = [
        ("peak_power_kw", "all_models_peak.png", "Peak power (kW)", True),
        ("energy_delivered_kwh", "all_models_energy.png", "Energy (kWh)", False),
        ("charging_cost", "all_models_cost.png", "Cost (EUR)", False),
        ("power_variance", "all_models_variance.png", "Variance", False),
    ]
    for col, fname, ylabel, lims in plots:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x, df[col], color=colors)
        if lims:
            limit_lines(ax)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Controller comparison - {ylabel}")
        fig.tight_layout()
        save(fname)

    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.35
    ax.bar(xx - w / 2, df["hard_overloads"], w, label="Hard", color=C2)
    ax.bar(xx + w / 2, df["soft_overloads"], w, label="Soft", color="#f39c12")
    ax.set_xticks(xx); ax.set_xticklabels(x, rotation=35)
    ax.legend(); ax.set_ylabel("Overload events")
    ax.set_title("Controller comparison - overload events")
    fig.tight_layout()
    save("all_models_overload.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.35
    ax.bar(xx - w / 2, df["mean_ramp_kw"], w, label="Mean ramp")
    ax.bar(xx + w / 2, df["max_ramp_kw"], w, label="Max ramp",
           color="#f39c12")
    ax.set_xticks(xx); ax.set_xticklabels(x, rotation=35)
    ax.legend(); ax.set_ylabel("Ramp (kW/h)")
    ax.set_title("Controller comparison - power ramps")
    fig.tight_layout()
    save("all_models_ramp.png")


# ==========================================
# EXPERIMENT 4 - REWARD WEIGHT SENSITIVITY
# ==========================================

def exp4():
    df = pd.read_csv("results/reward_sensitivity/reward_sensitivity.csv").copy()
    df = df.sort_values("model")
    labels = df["model"].tolist()
    xx = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Experiment 4 - Reward Weight Sensitivity (R0..R3)", fontsize=15)

    ax = axes[0, 0]
    ax.bar(xx - 0.18, df["energy_delivery_pct"], 0.36, label="Energy delivery %",
           color=C1)
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel("Energy delivery (%)")
    ax2 = ax.twinx()
    ax2.plot(xx, df["soft_overloads"], "o-", color=C2, label="Soft overloads")
    ax2.set_ylabel("Soft overloads")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("Energy delivery vs overloads")

    ax = axes[0, 1]
    ax.scatter(df["energy_delivered_kwh"], df["charging_cost"],
               c=[C1, "#f39c12", "#8e44ad", C2], s=90)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (df["energy_delivered_kwh"][i],
                          df["charging_cost"][i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Energy delivered (kWh)")
    ax.set_ylabel("Charging cost (EUR)")
    ax.set_title("Cost vs energy")

    ax = axes[1, 0]
    ax.scatter(df["energy_delivered_kwh"], df["peak_power_kw"],
               c=[C1, "#f39c12", "#8e44ad", C2], s=90)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (df["energy_delivered_kwh"][i],
                          df["peak_power_kw"][i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.axhline(HARD, color=C2, linestyle=":", label="Hard limit")
    ax.axhline(SOFT, color="#f39c12", linestyle="--", label="Soft limit")
    ax.set_xlabel("Energy delivered (kWh)")
    ax.set_ylabel("Peak power (kW)")
    ax.set_title("Peak vs energy")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(xx, df["total_reward"], color=[C1, "#f39c12", "#8e44ad", C2])
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel("Total reward")
    ax.set_title("Total reward per config")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save("reward_tradeoff.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xx - 0.18, df["energy_delivery_pct"], 0.36, label="Energy delivery %",
           color=C1)
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel("Energy delivery (%)")
    ax2 = ax.twinx()
    ax2.plot(xx, df["soft_overloads"], "o-", color=C2, label="Soft overloads")
    ax2.set_ylabel("Soft overloads")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("Energy delivery vs overloads")
    fig.tight_layout()
    save("reward_energy_vs_overload.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["energy_delivered_kwh"], df["charging_cost"],
               c=[C1, "#f39c12", "#8e44ad", C2], s=90)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (df["energy_delivered_kwh"][i],
                          df["charging_cost"][i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Energy delivered (kWh)")
    ax.set_ylabel("Charging cost (EUR)")
    ax.set_title("Cost vs energy")
    fig.tight_layout()
    save("reward_cost_vs_energy.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["energy_delivered_kwh"], df["peak_power_kw"],
               c=[C1, "#f39c12", "#8e44ad", C2], s=90)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (df["energy_delivered_kwh"][i],
                          df["peak_power_kw"][i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.axhline(HARD, color=C2, linestyle=":")
    ax.axhline(SOFT, color="#f39c12", linestyle="--")
    ax.set_xlabel("Energy delivered (kWh)")
    ax.set_ylabel("Peak power (kW)")
    ax.set_title("Peak vs energy")
    fig.tight_layout()
    save("reward_peak_vs_energy.png")


# ==========================================
# EXPERIMENT 5 - SEED STABILITY
# ==========================================

def exp5():
    df = pd.read_csv("results/statistics/twinev_seed_stability.csv")
    seeds = df["seed"].astype(str).tolist()
    xx = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Experiment 5 - Random Seed Stability (FULL TwinEV, seeds 1-5)",
                 fontsize=15)

    ax = axes[0, 0]
    ax.bar(seeds, df["energy_delivered_kwh"], color=C1)
    ax.set_ylabel("Energy delivered (kWh)")
    ax.set_title("Energy delivered")
    m, s = df["energy_delivered_kwh"].mean(), df["energy_delivered_kwh"].std(ddof=1)
    ax.axhline(m, color=C2, linestyle="--", label=f"mean {m/1e3:.1f} MWh")
    ax.axhspan(m - s, m + s, alpha=0.15, color=C2)
    ax.legend()

    ax = axes[0, 1]
    ax.bar(seeds, df["eval_total_reward"], color=C1)
    ax.set_ylabel("Evaluation reward")
    ax.set_title("Evaluation total reward")

    ax = axes[1, 0]
    ax.bar(seeds, df["peak_power_kw"], color=C1)
    limit_lines(ax)
    ax.set_ylabel("Peak power (kW)")
    ax.set_title("Peak power")

    ax = axes[1, 1]
    ax.bar(seeds, df["power_variance"], color=C1)
    ax.set_ylabel("Power variance")
    ax.set_title("Power variance")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save("twinev_seed_stability.png")


# ==========================================
# EXPERIMENT 6 - FORECASTING COMPARISON
# ==========================================

def exp6():
    df = pd.read_csv("results/forecasting/forecast_model_comparison.csv")
    w = pd.read_csv("results/forecasting/forecast_test_window.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Experiment 6 - Forecasting Comparison (next-hour load, test window)",
                 fontsize=15)

    ax = axes[0]
    xx = np.arange(len(df))
    ax.bar(xx - 0.25, df["RMSE"], 0.25, label="RMSE", color=C1)
    ax.bar(xx, df["MAE"], 0.25, label="MAE", color="#f39c12")
    ax.bar(xx + 0.25, df["MAPE"], 0.25, label="MAPE", color="#8e44ad")
    ax.set_xticks(xx); ax.set_xticklabels(df["Model"])
    for i in range(len(df)):
        ax.text(i - 0.25, df["RMSE"][i], f"{df['RMSE'][i]:.1f}", ha="center",
                va="bottom", fontsize=8)
        ax.text(i, df["MAE"][i], f"{df['MAE'][i]:.1f}", ha="center",
                va="bottom", fontsize=8)
        ax.text(i + 0.25, df["MAPE"][i], f"{df['MAPE'][i]:.1f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylabel("Error")
    ax.set_title("Forecast accuracy")
    ax.legend()

    ax = axes[1]
    n = 168
    ax.plot(w["actual"][:n], label="Actual", color="black", linewidth=1.6)
    ax.plot(w["arima"][:n], label="ARIMA", linewidth=1.1, alpha=0.9)
    ax.plot(w["lstm"][:n], label="LSTM", linewidth=1.1)
    ax.plot(w["bilstm"][:n], label="BiLSTM", linewidth=1.1, color=C2)
    ax.set_xlabel("Hour within test window")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Forecast vs actual (first 168 h)")
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save("forecast_model_comparison.png")


# ==========================================
# EXPERIMENT 7 - STRESS TESTING
# ==========================================

def exp7():
    df = pd.read_csv("results/stress_tests/stress_test_results.csv").copy()
    x = df["model"].tolist()
    xx = np.arange(len(df))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Experiment 7 - Stress Testing (FULL TwinEV, no retraining)",
                 fontsize=15)

    ax = axes[0, 0]
    ax.bar(x, df["peak_power_kw"], color=C1)
    limit_lines(ax)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Peak power"); ax.set_ylabel("Peak (kW)")

    ax = axes[0, 1]
    w = 0.35
    ax.bar(xx - w / 2, df["hard_overloads"], w, label="Hard", color=C2)
    ax.bar(xx + w / 2, df["soft_overloads"], w, label="Soft", color="#f39c12")
    ax.set_xticks(xx); ax.set_xticklabels(x, rotation=25)
    ax.legend(); ax.set_title("Overload events"); ax.set_ylabel("Events")

    ax = axes[0, 2]
    ax.bar(x, df["energy_delivered_kwh"], color=C1)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Energy delivered"); ax.set_ylabel("Energy (kWh)")

    ax = axes[1, 0]
    ax.bar(x, df["charging_cost"], color=C1)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Charging cost"); ax.set_ylabel("Cost (EUR)")

    ax = axes[1, 1]
    ax.bar(x, df["power_variance"], color=C1)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Power variance"); ax.set_ylabel("Variance")

    ax = axes[1, 2]
    ax.bar(x, df["total_reward"], color=C1)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Total reward"); ax.set_ylabel("Reward")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save("stress_test_comparison.png")


if __name__ == "__main__":
    exp1()
    exp2()
    exp3()
    exp4()
    exp5()
    exp6()
    exp7()
    print("\nAll plots generated.")
