# ==========================================
# TwinEV v2 - EXPERIMENT 3
# Complete Controller Comparison
# ==========================================
# Same unseen test window for every controller:
#   1. Raw demand      2. FCFS      3. Round Robin
#   4. Greedy          5. PPO + PI  6. BiLSTM + PPO
#   7. BiLSTM + PPO + PI (FULL TwinEV)
# ==========================================

import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from baseline import FCFSBaseline, RoundRobinBaseline, GreedyBaseline
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


def main():
    ensure_dir("results/comparison")
    ensure_dir("results/profiles")

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]

    print("Loading baselines...")
    fcfs = np.array(FCFSBaseline().predict(raw_test))
    rr = np.array(RoundRobinBaseline(max_power=150).predict(raw_test))
    greedy = np.array(GreedyBaseline(limit=130).predict(raw_test))

    print("Loading RL policies...")
    ppo_pi = PPO.load(f"{MODEL_DIR}/exp1_modelA_noforecast")   # PPO + PI, no forecast
    blstm_ppo = PPO.load(f"{MODEL_DIR}/exp2_modelC_no_pi")     # BiLSTM + PPO, no PI
    full = PPO.load(f"{MODEL_DIR}/exp1_modelB_full")           # FULL TwinEV

    print("Evaluating all controllers over the same test window...")
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
    for name, p in profiles.items():
        pd.DataFrame({"power": p}).to_csv(
            f"results/profiles/exp3_{name.replace(' ', '_').replace('+', 'p')}.csv",
            index=False)

    rows = []
    # Baselines have no reward / setpoint / tracking
    for name, p in profiles.items():
        if name in ("Raw Demand", "FCFS", "Round Robin", "Greedy"):
            rows.append(model_metrics_row(
                name,
                power=p,
                setpoint=p,
                reward=None,
                solar=solar_test,
                price=price_test,
                raw=raw_test,
                extra={"type": "baseline"},
            ))
        elif name == "PPO + PI":
            rows.append(model_metrics_row(
                name,
                power=p,
                setpoint=out_ppo_pi["setpoint"],
                reward=out_ppo_pi["reward"],
                solar=solar_test,
                price=price_test,
                raw=raw_test,
                extra={"type": "rl", "forecast_in_state": "no", "pi_layer": "yes"},
            ))
        elif name == "BiLSTM + PPO":
            rows.append(model_metrics_row(
                name,
                power=p,
                setpoint=out_blstm["setpoint"],
                reward=out_blstm["reward"],
                solar=solar_test,
                price=price_test,
                raw=raw_test,
                extra={"type": "rl", "forecast_in_state": "yes", "pi_layer": "no"},
            ))
        elif name == "BiLSTM + PPO + PI":
            rows.append(model_metrics_row(
                name,
                power=p,
                setpoint=out_full["setpoint"],
                reward=out_full["reward"],
                solar=solar_test,
                price=price_test,
                raw=raw_test,
                extra={"type": "rl", "forecast_in_state": "yes", "pi_layer": "yes"},
            ))

    df = pd.DataFrame(rows)
    df.to_csv("results/comparison/all_models.csv", index=False)
    print("\nSaved results/comparison/all_models.csv")

    # Publication-ready table (subset of columns)
    table = df[[
        "model", "type",
        "hard_overloads", "soft_overloads",
        "peak_power_kw", "peak_reduction_pct",
        "mean_power_kw", "power_variance",
        "energy_delivered_kwh", "energy_delivery_pct",
        "charging_cost", "solar_utilization_pct",
        "mean_tracking_error_kw", "mean_ramp_kw", "max_ramp_kw",
        "total_reward",
    ]]
    table.to_csv("results/comparison/main_results_table.csv", index=False)
    print("Saved results/comparison/main_results_table.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
