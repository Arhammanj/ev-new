# ==========================================
# TwinEV v2 - EXPERIMENT 10
# FORECAST-INFORMED MULTI-OBJECTIVE PPO
# TRAINING
# ==========================================
# Trains MO_FULL (3 seeds) and the 4 ablations.
# Identical PPO config / budget as baseline:
# 500k timesteps, 4 envs, net[256,256,128],
# deterministic seeds.
# ==========================================

import argparse
import json
import os
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mo_env import (
    MOFullEnv,
    MONoForecastEnv,
    MONoCostEnv,
    MONoSolarEnv,
    MONoSmoothnessEnv,
)
from train_agent import PPO_KWARGS, N_ENVS

OUT_DIR = "results/mo_control"
MODEL_DIR = f"{OUT_DIR}/models"
TOTAL_TIMESTEPS = 500_000

MODELS = [
    ("MO_FULL", MOFullEnv, 1),
    ("MO_FULL_s2", MOFullEnv, 2),
    ("MO_FULL_s3", MOFullEnv, 3),
    ("MO_NO_FORECAST", MONoForecastEnv, 1),
    ("MO_NO_COST", MONoCostEnv, 1),
    ("MO_NO_SOLAR", MONoSolarEnv, 1),
    ("MO_NO_SMOOTHNESS", MONoSmoothnessEnv, 1),
]


class EpisodeRewardRecorder(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def make_env(env_cls, mode):
    def _init():
        return Monitor(env_cls(mode=mode))
    return _init


def train_mo(name, env_cls, seed, total_timesteps=TOTAL_TIMESTEPS):
    save_path = f"{MODEL_DIR}/{name}"
    train_env = DummyVecEnv([make_env(env_cls, "train")] * N_ENVS)
    train_env.seed(seed)
    eval_env = DummyVecEnv([make_env(env_cls, "val")])

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
    return final_reward


def main(only=None, skip_train=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        with open(f"{OUT_DIR}/training_log.json") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = {}

    for name, env_cls, seed in MODELS:
        if only is not None and name != only:
            continue
        if skip_train and os.path.exists(f"{MODEL_DIR}/{name}.zip"):
            print(f"[skip-train] {name}")
            continue
        print(f"\n===== Training {name} (seed {seed}) =====")
        t0 = time.time()
        final_reward = train_mo(name, env_cls, seed)
        elapsed = time.time() - t0
        print(f"  {name} done in {elapsed/60:.2f} min | "
              f"final episode reward {final_reward:.2f}")
        log[name] = {
            "seed": seed,
            "final_training_reward": final_reward,
            "training_time_min": round(elapsed / 60, 2),
            "total_timesteps": TOTAL_TIMESTEPS,
        }
        with open(f"{OUT_DIR}/training_log.json", "w") as f:
            json.dump(log, f, indent=2)

    print("\nTraining phase complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    main(only=args.only, skip_train=args.skip_train)
