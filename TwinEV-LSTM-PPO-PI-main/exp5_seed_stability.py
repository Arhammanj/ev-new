# ==========================================
# TwinEV v2 - EXPERIMENT 5
# Random Seed Stability (FULL TwinEV)
# ==========================================
# Train the identical BiLSTM+PPO+PI model with
# 5 independent seeds (1..5), evaluate all on
# the same test set, report mean/std/min/max.
# ==========================================

import argparse
import json
import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from exp_utils import (
    TwinEVEnv,
    ensure_dir,
    load_aligned_data,
    run_policy_sequential,
    TEST_START,
    TEST_END,
)
from train_agent import train_agent

MODEL_DIR = "results/models"
SEEDS = [1, 2, 3, 4, 5]
TOTAL_TIMESTEPS = 500_000


def main(skip_train=False):
    ensure_dir(MODEL_DIR)
    ensure_dir("results/statistics")
    ensure_dir("results/profiles")

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]

    try:
        with open("results/models/training_rewards.json") as f:
            train_rewards = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        train_rewards = {}

    records = []
    for seed in SEEDS:
        final_train_reward = None
        if seed == 1:
            model_path = f"{MODEL_DIR}/exp1_modelB_full"   # already trained
        else:
            model_path = f"{MODEL_DIR}/exp5_seed{seed}"
        if skip_train and os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path)
        else:
            print(f"\n===== Training FULL TwinEV seed {seed} =====")
            model, final_train_reward = train_agent(
                TwinEVEnv, model_path, reward_weights=None,
                seed=seed, total_timesteps=TOTAL_TIMESTEPS)

        if final_train_reward is None:
            final_train_reward = float(
                train_rewards.get(os.path.basename(model_path), np.nan))

        out = run_policy_sequential(model, TwinEVEnv)
        pd.DataFrame({"power": out["power"]}).to_csv(
            f"results/profiles/exp5_seed{seed}.csv", index=False)

        p = out["power"]
        rec = {
            "seed": seed,
            "final_training_reward": final_train_reward,
            "eval_total_reward": float(np.sum(out["reward"])),
            "hard_overloads": int(np.sum(p > 150)),
            "soft_overloads": int(np.sum(p > 130)),
            "peak_power_kw": float(np.max(p)),
            "energy_delivered_kwh": float(np.sum(p)),
            "charging_cost": float(np.sum(p * price_test)),
            "power_variance": float(np.var(p)),
        }
        records.append(rec)
        print(rec)

    df = pd.DataFrame(records)
    df.to_csv("results/statistics/twinev_seed_stability.csv", index=False)

    summary = {}
    for col in ["final_training_reward", "eval_total_reward",
                "hard_overloads", "soft_overloads", "peak_power_kw",
                "energy_delivered_kwh", "charging_cost", "power_variance"]:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std(ddof=1)
        summary[f"{col}_min"] = df[col].min()
        summary[f"{col}_max"] = df[col].max()
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("results/statistics/twinev_seed_stability_summary.csv",
                      index=False)
    print("\nSaved results/statistics/twinev_seed_stability.csv")
    print(summary_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    main(skip_train=args.skip_train)
