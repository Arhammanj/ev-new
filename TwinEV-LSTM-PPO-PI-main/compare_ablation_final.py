# ==========================================
# TwinEV v2 - Final Ablation Table
# Fixed: aligned sequential test harness +
# trained experiment models (no ad-hoc runs).
# Adds peak reduction % and charging cost.
# Controllers:
#   No Control | PPO + PI (no forecast)
#   BiLSTM + PPO (no PI) | FULL TwinEV
# ==========================================

import os

import numpy as np
import pandas as pd

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

state = load_aligned_data()
raw_test = state["power"][TEST_START:TEST_END]
price_test = state["price"][TEST_START:TEST_END]
solar_test = state["solar"][TEST_START:TEST_END]

print("Loading trained policies...")
ppo_pi = PPO.load(f"{MODEL_DIR}/exp1_modelA_noforecast")
blstm_ppo = PPO.load(f"{MODEL_DIR}/exp2_modelC_no_pi")
full = PPO.load(f"{MODEL_DIR}/exp1_modelB_full")

out_ppo_pi = run_policy_sequential(ppo_pi, NoForecastEnv)
out_blstm = run_policy_sequential(blstm_ppo, NoPIEnv)
out_full = run_policy_sequential(full, TwinEVEnv)

results = [
    model_metrics_row("No Control", power=raw_test, setpoint=raw_test,
                      solar=solar_test, price=price_test, raw=raw_test,
                      extra={"Forecast": "No", "RL": "No", "PI": "No"}),
    model_metrics_row("PPO + PI", power=out_ppo_pi["power"],
                      setpoint=out_ppo_pi["setpoint"],
                      reward=out_ppo_pi["reward"],
                      solar=solar_test, price=price_test, raw=raw_test,
                      extra={"Forecast": "No", "RL": "Yes", "PI": "Yes"}),
    model_metrics_row("BiLSTM + PPO", power=out_blstm["power"],
                      setpoint=out_blstm["setpoint"],
                      reward=out_blstm["reward"],
                      solar=solar_test, price=price_test, raw=raw_test,
                      extra={"Forecast": "Yes", "RL": "Yes", "PI": "No"}),
    model_metrics_row("BiLSTM + PPO + PI", power=out_full["power"],
                      setpoint=out_full["setpoint"],
                      reward=out_full["reward"],
                      solar=solar_test, price=price_test, raw=raw_test,
                      extra={"Forecast": "Yes", "RL": "Yes", "PI": "Yes"}),
]

df = pd.DataFrame(results)
df.to_csv("results/comparison/final_ablation_table.csv", index=False)

keep = ["model", "Forecast", "RL", "PI",
        "peak_power_kw", "peak_reduction_pct",
        "energy_delivered_kwh", "charging_cost",
        "hard_overloads", "soft_overloads"]
final = df[keep]
final.columns = ["Method", "Forecast", "RL", "PI",
                 "Peak Power (kW)", "Peak Reduction (%)",
                 "Energy Delivered (kWh)", "Cost",
                 "Hard Overloads", "Soft Overloads"]

print("\nFINAL ABLATION TABLE")
print(final.to_string(index=False))
print("\nSaved results/comparison/final_ablation_table.csv")
