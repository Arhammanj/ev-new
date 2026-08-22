# ==========================================
# TwinEV v2 - EXPERIMENT 8
# SERVICE-CONSTRAINED TwinEV
# ==========================================
# Question: Can TwinEV increase EV charging
# service while maintaining zero hard overloads?
#
# Trains 4 separate PPO models with an explicit
# service target (70/80/90/95 %) added as a
# constraint term, keeping the original TwinEV
# architecture, PI controller, data and test
# split.  Existing models/results are NOT
# modified.
# ==========================================

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from exp_utils import (
    TwinEVEnv,
    ensure_dir,
    load_aligned_data,
    model_metrics_row,
    run_policy_sequential,
    TEST_START,
    TEST_END,
)
from service_env import ServiceConstrainedEnv
from train_agent import PPO_KWARGS, N_ENVS

OUT_DIR = "results/service_constrained"
MODEL_DIR = f"{OUT_DIR}/models"
PROFILE_DIR = "results/profiles"
TOTAL_TIMESTEPS = 500_000
SEED = 1

SERVICE_TARGETS = {
    "twinev_service70": 0.70,
    "twinev_service80": 0.80,
    "twinev_service90": 0.90,
    "twinev_service95": 0.95,
}


class EpisodeRewardRecorder(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def make_service_env(service_target, mode):
    def _init():
        return Monitor(ServiceConstrainedEnv(
            mode=mode,
            service_target=service_target,
            service_penalty_weight=50.0,
            terminal_bonus=10.0,
        ))
    return _init


def train_service_agent(name, service_target, seed=SEED,
                        total_timesteps=TOTAL_TIMESTEPS):
    save_path = f"{MODEL_DIR}/{name}"
    train_env = DummyVecEnv(
        [make_service_env(service_target, "train")] * N_ENVS)
    train_env.seed(seed)
    eval_env = DummyVecEnv(
        [make_service_env(service_target, "val")])

    recorder = EpisodeRewardRecorder()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        seed=seed,
        **PPO_KWARGS,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[recorder, eval_callback],
                progress_bar=False)
    model.save(save_path)
    final_reward = (float(np.mean(recorder.episode_rewards[-20:]))
                    if recorder.episode_rewards else np.nan)
    return model, final_reward


def evaluate_controller(policy, env_cls, state, name, service_target):
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]

    out = run_policy_sequential(policy, env_cls)
    p = out["power"]
    pd.DataFrame({"power": p}).to_csv(
        f"{PROFILE_DIR}/exp8_{name}.csv", index=False)

    row = model_metrics_row(
        name,
        power=p,
        setpoint=out["setpoint"],
        reward=out["reward"],
        solar=solar_test,
        price=price_test,
        raw=raw_test,
        extra={"type": "rl"},
    )
    service_ratio_pct = row["energy_delivery_pct"]
    met = bool(np.isfinite(service_ratio_pct) and
               service_ratio_pct >= 100.0 * service_target - 1e-9) \
        if service_target is not None else np.nan
    row["service_target_pct"] = (100.0 * service_target
                                 if service_target is not None else np.nan)
    row["service_target_met"] = met
    row["zero_hard_overload"] = bool(row["hard_overloads"] == 0)
    row["energy_delivered_mwh"] = round(row["energy_delivered_kwh"] / 1000.0, 3)
    return row


def main(skip_train=False, only=None):
    ensure_dir(OUT_DIR)
    ensure_dir(MODEL_DIR)
    ensure_dir(PROFILE_DIR)

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]

    try:
        with open(f"{OUT_DIR}/training_log.json") as f:
            train_log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        train_log = {}

    # ------------------------------------------
    # TRAIN SERVICE MODELS
    # ------------------------------------------
    for name, target in SERVICE_TARGETS.items():
        if only is not None and name != only:
            continue
        model_path = f"{MODEL_DIR}/{name}.zip"
        if skip_train and os.path.exists(model_path):
            print(f"[skip-train] {name} exists")
            model = PPO.load(model_path)
            final_reward = float(train_log.get(name, {}).get(
                "final_training_reward", np.nan))
        else:
            print(f"\n===== Training {name} (target {int(target*100)}%) =====")
            t0 = time.time()
            model, final_reward = train_service_agent(name, target)
            elapsed = time.time() - t0
            print(f"  {name} trained in {elapsed/60:.2f} min | "
                  f"final mean episode reward {final_reward:.2f}")
            train_log[name] = {
                "service_target": target,
                "final_training_reward": final_reward,
                "training_time_min": round(elapsed / 60, 2),
                "total_timesteps": TOTAL_TIMESTEPS,
                "seed": SEED,
            }
            with open(f"{OUT_DIR}/training_log.json", "w") as f:
                json.dump(train_log, f, indent=2)

    # ------------------------------------------
    # EVALUATE SERVICE MODELS
    # ------------------------------------------
    rows = []
    for name, target in SERVICE_TARGETS.items():
        if only is not None and name != only:
            continue
        model = PPO.load(f"{MODEL_DIR}/{name}.zip")
        row = evaluate_controller(model, ServiceConstrainedEnv,
                                  state, name, target)
        rows.append(row)
        print(f"\n[{name}] target {int(target*100)}% -> "
              f"service {row['energy_delivery_pct']:.2f}% | "
              f"hard {row['hard_overloads']} soft {row['soft_overloads']} | "
              f"peak {row['peak_power_kw']} | met={row['service_target_met']}")

    service_df = pd.DataFrame(rows)
    service_df.to_csv(f"{OUT_DIR}/service_constrained_results.csv", index=False)
    print("\nSaved results/service_constrained/service_constrained_results.csv")

    # ------------------------------------------
    # REBUILD COMPARISON (baselines + original TwinEV)
    # ------------------------------------------
    greedy = np.minimum(raw_test, 130.0)
    rr = np.full_like(raw_test, np.clip(np.mean(raw_test), 0, 150.0))
    fcfs = np.array(raw_test)
    full = PPO.load("results/models/exp1_modelB_full")
    out_full = run_policy_sequential(full, TwinEVEnv)

    base_rows = []
    for bname, bp in [("FCFS", fcfs), ("Round Robin", rr),
                      ("Greedy 130 kW", greedy)]:
        r = model_metrics_row(bname, power=bp, setpoint=bp, reward=None,
                              solar=solar_test, price=price_test,
                              raw=raw_test, extra={"type": "baseline"})
        r["service_target_pct"] = np.nan
        r["service_target_met"] = np.nan
        r["zero_hard_overload"] = bool(r["hard_overloads"] == 0)
        r["energy_delivered_mwh"] = round(r["energy_delivered_kwh"] / 1000.0, 3)
        base_rows.append(r)

    twin_row = model_metrics_row(
        "Original TwinEV",
        power=out_full["power"],
        setpoint=out_full["setpoint"],
        reward=out_full["reward"],
        solar=solar_test, price=price_test, raw=raw_test,
        extra={"type": "rl"},
    )
    twin_row["service_target_pct"] = np.nan
    twin_row["service_target_met"] = np.nan
    twin_row["zero_hard_overload"] = bool(twin_row["hard_overloads"] == 0)
    twin_row["energy_delivered_mwh"] = round(twin_row["energy_delivered_kwh"] / 1000.0, 3)

    all_rows = base_rows + [twin_row] + rows
    comp = pd.DataFrame(all_rows)

    keep = [
        "model", "type", "service_target_pct", "energy_delivered_mwh",
        "energy_delivery_pct", "hard_overloads", "soft_overloads",
        "peak_power_kw", "peak_reduction_pct", "charging_cost",
        "solar_utilization_pct", "power_variance", "mean_ramp_kw",
        "max_ramp_kw", "mean_tracking_error_kw", "total_reward",
        "service_target_met", "zero_hard_overload",
    ]
    comp = comp[keep]
    comp.to_csv(f"{OUT_DIR}/service_constrained_summary.csv", index=False)
    print("\nSaved results/service_constrained/service_constrained_summary.csv")
    print(comp.to_string(index=False))

    # ------------------------------------------
    # FEASIBILITY CHECK
    # ------------------------------------------
    ceiling = 100.0 * 150.0 * len(raw_test) / np.sum(raw_test)
    print(f"\nPhysical service ceiling with 150 kW cap: {ceiling:.2f}%")
    print(f"(constant 150 kW output over the test window)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--only", default=None,
                        help="train/eval a single model name only")
    args = parser.parse_args()
    main(skip_train=args.skip_train, only=args.only)
