# ==========================================
# TwinEV v2 - Legacy Ablation Table
# (superseded logic fixed: uses the aligned
#  sequential test harness and the trained
#  experiment models instead of ad-hoc runs)
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
    OVERLOAD_KW,
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

table = pd.DataFrame(results)
cols = ["model", "Forecast", "RL", "PI",
        "peak_power_kw", "energy_delivered_kwh",
        "hard_overloads", "soft_overloads"]
table = table[cols]
table.columns = ["Method", "Forecast", "RL", "PI",
                 "Peak kW", "Energy Delivered (kWh)",
                 "Hard Overloads", f"Soft Overloads (> {OVERLOAD_KW:.0f} kW)"]

print("\nFINAL ABLATION TABLE")
print(table.to_string(index=False))

table.to_csv("results/comparison/ablation_results.csv", index=False)
print("\nSaved results/comparison/ablation_results.csv")
