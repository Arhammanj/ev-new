# ==========================================
# TwinEV v2 - EXPERIMENT 2
# PI Control Ablation
# ==========================================
# MODEL C : BiLSTM + PPO (no PI layer)
# MODEL D : BiLSTM + PPO + PI  (full TwinEV)
# Everything identical except the PI layer.
# ==========================================

import argparse
import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from exp_utils import (
    NoPIEnv,
    TwinEVEnv,
    ensure_dir,
    load_aligned_data,
    model_metrics_row,
    run_policy_sequential,
    TEST_START,
    TEST_END,
)
from train_agent import train_agent

MODEL_DIR = "results/models"
SEED = 1
TOTAL_TIMESTEPS = 500_000


def main(skip_train=False):
    ensure_dir(MODEL_DIR)
    ensure_dir("results/ablation")
    ensure_dir("results/profiles")

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]

    model_c_path = f"{MODEL_DIR}/exp2_modelC_no_pi"
    model_d_path = f"{MODEL_DIR}/exp1_modelB_full"   # reuse full TwinEV

    if skip_train and os.path.exists(model_c_path + ".zip"):
        model_c = PPO.load(model_c_path)
    else:
        print(f"\n===== Training Model C (BiLSTM+PPO, NO PI), seed {SEED} =====")
        model_c, _ = train_agent(
            NoPIEnv, model_c_path, reward_weights=None,
            seed=SEED, total_timesteps=TOTAL_TIMESTEPS)

    print(f"\n===== Loading Model D (full TwinEV) from {model_d_path} =====")
    model_d = PPO.load(model_d_path)

    out_c = run_policy_sequential(model_c, NoPIEnv)
    out_d = run_policy_sequential(model_d, TwinEVEnv)

    pd.DataFrame({"power": out_c["power"]}).to_csv(
        "results/profiles/exp2_modelC_no_pi.csv", index=False)

    results = [
        model_metrics_row(
            "C_bilstm_ppo_no_pi",
            power=out_c["power"],
            setpoint=out_c["setpoint"],
            reward=out_c["reward"],
            solar=solar_test,
            price=price_test,
            raw=raw_test,
            extra={"pi_layer": "no"},
        ),
        model_metrics_row(
            "D_bilstm_ppo_pi",
            power=out_d["power"],
            setpoint=out_d["setpoint"],
            reward=out_d["reward"],
            solar=solar_test,
            price=price_test,
            raw=raw_test,
            extra={"pi_layer": "yes"},
        ),
    ]

    df = pd.DataFrame(results)
    df.to_csv("results/ablation/pi_ablation.csv", index=False)
    print("\nSaved results/ablation/pi_ablation.csv")
    print(df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    main(skip_train=args.skip_train)
