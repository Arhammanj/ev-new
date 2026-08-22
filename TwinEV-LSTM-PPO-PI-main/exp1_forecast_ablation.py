# ==========================================
# TwinEV v2 - EXPERIMENT 1
# Forecast Ablation: BiLSTM forecast vs none
# ==========================================
# MODEL A : PPO + PI  (observation WITHOUT forecast)
# MODEL B : BiLSTM + PPO + PI  (full TwinEV)
# Same data/split/hyperparameters/action space/
# reward/PI/training timesteps/evaluation.
# Only the forecast feature differs.
# ==========================================

import argparse
import json
import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from exp_utils import (
    NoForecastEnv,
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


def main(skip_train=False):
    ensure_dir(MODEL_DIR)
    ensure_dir("results/ablation")
    ensure_dir("results/profiles")

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]

    with open("results/reward_sensitivity/reward_configs.json") as f:
        configs = json.load(f)
    weights = {k: v for k, v in configs["R0"].items() if k != "description" and k != "lookahead_bonus_weight"}
    print("Reward config R0 (baseline):", json.dumps(train_reward_weights_report(weights)))

    models = {
        "A_ppo_no_forecast": {"env_cls": NoForecastEnv, "path": f"{MODEL_DIR}/exp1_modelA_noforecast"},
        "B_full_twinev": {"env_cls": TwinEVEnv, "path": f"{MODEL_DIR}/exp1_modelB_full"},
    }

    results = []
    for name, cfg in models.items():
        model_path = cfg["path"]
        if skip_train and os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path)
        else:
            print(f"\n===== Training {name} (seed {SEED}, {TOTAL_TIMESTEPS} steps) =====")
            model, _ = train_agent(
                cfg["env_cls"], model_path, reward_weights=None,
                seed=SEED, total_timesteps=TOTAL_TIMESTEPS)

        out = run_policy_sequential(model, cfg["env_cls"])
        pd.DataFrame({"power": out["power"]}).to_csv(
            f"results/profiles/{name}.csv", index=False)

        row = model_metrics_row(
            name,
            power=out["power"],
            setpoint=out["setpoint"],
            reward=out["reward"],
            solar=solar_test,
            price=price_test,
            raw=raw_test,
            extra={"forecast_in_state": "yes" if name.startswith("B") else "no"},
        )
        results.append(row)
        print(f"{name}: peak {row['peak_power_kw']} kW | hard {row['hard_overloads']} | "
              f"soft {row['soft_overloads']} | energy {row['energy_delivered_kwh']} kWh | "
              f"total_reward {row['total_reward']}")

    df = pd.DataFrame(results)
    df.to_csv("results/ablation/forecast_ablation.csv", index=False)
    print("\nSaved results/ablation/forecast_ablation.csv")
    print(df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    main(skip_train=args.skip_train)
