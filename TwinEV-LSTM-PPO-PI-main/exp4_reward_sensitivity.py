# ==========================================
# TwinEV v2 - EXPERIMENT 4
# Reward Weight Sensitivity
# ==========================================
# Same reward terms; only the weights change.
# R0 = current baseline, R1 = lower safety
# pressure, R2 = balanced, R3 = service-
# oriented. Look-ahead bonus weight reported
# separately and kept constant.
# ==========================================

import argparse
import json
import os

import pandas as pd

from stable_baselines3 import PPO

from exp_utils import (
    TwinEVEnv,
    ensure_dir,
    load_aligned_data,
    model_metrics_row,
    run_policy_sequential,
    TEST_START,
    TEST_END,
)
from train_agent import train_agent, train_reward_weights_report

MODEL_DIR = "results/models"
SEED = 1
TOTAL_TIMESTEPS = 500_000


def weights_from_config(cfg):
    return {k: v for k, v in cfg.items()
            if k not in ("description", "lookahead_bonus_weight")}


def main(skip_train=False):
    ensure_dir(MODEL_DIR)
    ensure_dir("results/reward_sensitivity")
    ensure_dir("results/profiles")

    with open("results/reward_sensitivity/reward_configs.json") as f:
        configs = json.load(f)

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]

    print("Exact reward configurations used for training:")
    rows_meta = []
    results = []
    for key, cfg in configs.items():
        weights = weights_from_config(cfg)
        lookahead = cfg["lookahead_bonus_weight"]
        meta = train_reward_weights_report(weights, lookahead_weight=lookahead)
        meta["config_id"] = key
        meta["description"] = cfg["description"]
        rows_meta.append(meta)
        print(f"  {key}: {json.dumps(meta)}")

    pd.DataFrame(rows_meta).to_csv(
        "results/reward_sensitivity/reward_configs_trained.csv", index=False)

    for key in configs:
        weights = weights_from_config(configs[key])
        if key == "R0":
            # R0 uses the same default weights and seed as the full
            # TwinEV model trained for Experiments 1/2/3.
            model_path = f"{MODEL_DIR}/exp1_modelB_full"
        else:
            model_path = f"{MODEL_DIR}/exp4_{key}"
        if skip_train and os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path)
        else:
            print(f"\n===== Training {key} (seed {SEED}, {TOTAL_TIMESTEPS} steps) =====")
            model, _ = train_agent(
                TwinEVEnv, model_path, reward_weights=weights,
                seed=SEED, total_timesteps=TOTAL_TIMESTEPS)

        out = run_policy_sequential(model, TwinEVEnv)
        pd.DataFrame({"power": out["power"]}).to_csv(
            f"results/profiles/exp4_{key}.csv", index=False)

        row = model_metrics_row(
            key,
            power=out["power"],
            setpoint=out["setpoint"],
            reward=out["reward"],
            solar=solar_test,
            price=price_test,
            raw=raw_test,
            extra={"config": key},
        )
        results.append(row)
        print(f"{key}: peak {row['peak_power_kw']} kW | hard {row['hard_overloads']} | "
              f"soft {row['soft_overloads']} | energy {row['energy_delivered_kwh']} kWh | "
              f"delivery {row['energy_delivery_pct']}% | total_reward {row['total_reward']}")

    df = pd.DataFrame(results)
    df.to_csv("results/reward_sensitivity/reward_sensitivity.csv", index=False)
    print("\nSaved results/reward_sensitivity/reward_sensitivity.csv")
    print(df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    main(skip_train=args.skip_train)
