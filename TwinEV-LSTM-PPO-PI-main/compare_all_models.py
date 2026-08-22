# ==========================================
# TwinEV v2
# Complete Controller Comparison
# Compares on the identical aligned test window:
# 1. Raw Demand   2. FCFS   3. Round Robin
# 4. Greedy       5. PPO + PI (no forecast)
# 6. BiLSTM + PPO (no PI)   7. FULL TwinEV
# ==========================================

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from exp_utils import (
    NoForecastEnv,
    NoPIEnv,
    TwinEVEnv,
    ensure_dir,
    load_aligned_data,
    model_metrics_row,
    run_policy_sequential,
    TEST_START,
    TEST_END,
)

MODEL_DIR = "results/models"

ensure_dir("results/comparison")
ensure_dir("results/figures")

print("Loading test data...")
state = load_aligned_data()
raw_test = state["power"][TEST_START:TEST_END]
price_test = state["price"][TEST_START:TEST_END]
solar_test = state["solar"][TEST_START:TEST_END]
print("Test hours:", len(raw_test))

fcfs = np.array(raw_test)
rr = np.full_like(raw_test, np.clip(np.mean(raw_test), 0, 150.0))
greedy = np.minimum(raw_test, 130.0)

print("Loading RL policies...")
ppo_pi = PPO.load(f"{MODEL_DIR}/exp1_modelA_noforecast")
blstm_ppo = PPO.load(f"{MODEL_DIR}/exp2_modelC_no_pi")
full = PPO.load(f"{MODEL_DIR}/exp1_modelB_full")

print("Evaluating all controllers...")
out_ppo_pi = run_policy_sequential(ppo_pi, NoForecastEnv)
out_blstm = run_policy_sequential(blstm_ppo, NoPIEnv)
out_full = run_policy_sequential(full, TwinEVEnv)

profiles = {
    "Raw Demand": raw_test,
    "FCFS": fcfs,
    "Round Robin": rr,
    "Greedy": greedy,
    "PPO + PI": out_ppo_pi["power"],
    "BiLSTM + PPO": out_blstm["power"],
    "BiLSTM + PPO + PI": out_full["power"],
}

rewards = {
    "Raw Demand": None,
    "FCFS": None,
    "Round Robin": None,
    "Greedy": None,
    "PPO + PI": out_ppo_pi["reward"],
    "BiLSTM + PPO": out_blstm["reward"],
    "BiLSTM + PPO + PI": out_full["reward"],
}
setpoints = {
    "PPO + PI": out_ppo_pi["setpoint"],
    "BiLSTM + PPO": out_blstm["setpoint"],
    "BiLSTM + PPO + PI": out_full["setpoint"],
}

rows = []
for name, p in profiles.items():
    rows.append(model_metrics_row(
        name, power=p, setpoint=setpoints.get(name),
        reward=rewards[name], solar=solar_test,
        price=price_test, raw=raw_test))

comparison = pd.DataFrame(rows)
comparison.to_csv("results/comparison/final_controller_comparison.csv",
                  index=False)

keep = ["model", "peak_power_kw", "energy_delivered_kwh",
        "mean_power_kw", "power_variance", "charging_cost",
        "hard_overloads", "soft_overloads", "total_reward"]
print("\n================================")
print("FINAL CONTROLLER COMPARISON")
print("================================")
print(comparison[keep].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(comparison["model"], comparison["peak_power_kw"])
ax.axhline(130, color="orange", linestyle=":", label="Soft limit (130 kW)")
ax.set_ylabel("Peak Power (kW)")
ax.set_title("Peak Power Comparison")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/peak_power_comparison.png", dpi=300)
plt.close(fig)

print("\nSaved results/comparison/final_controller_comparison.csv")
print("Saved results/figures/peak_power_comparison.png")
