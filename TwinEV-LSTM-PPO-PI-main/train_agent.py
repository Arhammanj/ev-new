# ==========================================
# TwinEV v2 - shared PPO training helper
# ==========================================
# Reuses the exact PPO hyperparameters and
# network architecture from train_rl.py so
# that every ablation/experiment model is
# trained under identical settings.
# ==========================================

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from twinev_env import TwinEVEnv

# Hyperparameters (identical to train_rl.py)
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
)

N_ENVS = 4


class EpisodeRewardRecorder(BaseCallback):
    """Collects completed-episode rewards from the Monitor wrapper."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def make_env(env_cls, reward_weights=None, mode="train"):
    def _init():
        env = env_cls(mode=mode, reward_weights=reward_weights)
        return Monitor(env)
    return _init


def train_agent(env_cls, save_path, reward_weights=None, seed=0,
                total_timesteps=500_000, n_envs=N_ENVS):
    """Train PPO with the TwinEV hyperparameters and return
    (model, final_training_reward_mean)."""
    train_env = DummyVecEnv(
        [make_env(env_cls, reward_weights, "train")] * n_envs)
    train_env.seed(seed)
    eval_env = DummyVecEnv(
        [make_env(env_cls, reward_weights, "val")])

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

    model.learn(
        total_timesteps=total_timesteps,
        callback=[recorder, eval_callback],
        progress_bar=False,
    )
    model.save(save_path)

    final_train_reward = float(np.nan)
    if recorder.episode_rewards:
        final_train_reward = float(np.mean(
            recorder.episode_rewards[-20:]))

    return model, final_train_reward


def train_reward_weights_report(weights, lookahead_weight=0.3):
    """Human-readable description of a reward configuration."""
    return {
        "grid": weights.get("grid"),
        "solar": weights.get("solar"),
        "queue": weights.get("queue"),
        "price": weights.get("price"),
        "pi_error": weights.get("pi_error"),
        "lookahead_bonus_weight": lookahead_weight,
    }
