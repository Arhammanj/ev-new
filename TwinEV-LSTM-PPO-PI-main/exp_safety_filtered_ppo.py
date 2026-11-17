"""Compare ordinary penalty PPO with hard safety-filtered PPO."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from exp_2x2_run import S3_WEIGHTS, shared_setup
from train_controller import (
    ACNWeeklyEpisodeEnv,
    evaluate_policy_on_weeks,
)


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


def train_and_evaluate(
    setup,
    hard_safety_filter: bool,
    seed: int,
    timesteps: int,
    checkpoint_dir: str | None = None,
):
    train_env = Monitor(
        ACNWeeklyEpisodeEnv(
            setup["full_data"],
            setup["slot_roster"],
            setup["train_range_plain"],
            reward_weights=S3_WEIGHTS,
            symmetric_action=True,
            hard_safety_filter=hard_safety_filter,
            seed=seed,
        )
    )
    model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, **PPO_KWARGS)
    model.learn(total_timesteps=timesteps)
    checkpoint_path = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        name = "safety_filtered_ppo" if hard_safety_filter else "penalty_ppo"
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}_seed{seed}")
        model.save(checkpoint_path)
    eval_env = ACNWeeklyEpisodeEnv(
        setup["full_data"],
        setup["slot_roster"],
        setup["test_range"],
        reward_weights=S3_WEIGHTS,
        symmetric_action=True,
        hard_safety_filter=hard_safety_filter,
        seed=999 + seed,
    )
    results = evaluate_policy_on_weeks(
        eval_env,
        setup["test_starts"],
        lambda env, obs: model.predict(obs, deterministic=True)[0],
    )
    results["training_safety_filter_calls"] = train_env.env.safety_filter_calls
    results["training_safety_filter_interventions"] = (
        train_env.env.safety_filter_interventions
    )
    results["evaluation_safety_filter_calls"] = eval_env.safety_filter_calls
    if checkpoint_path is not None:
        results["checkpoint_path"] = f"{checkpoint_path}.zip"
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Run and aggregate multiple seeds; overrides --seed.",
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output", default="acn_safety_filtered_comparison.json")
    parser.add_argument(
        "--checkpoint-dir", default="acn_safety_filtered_checkpoints",
        help="Directory for loadable PPO .zip files; pass an empty string to disable.",
    )
    args = parser.parse_args()
    setup = shared_setup()
    checkpoint_dir = args.checkpoint_dir or None
    seeds = args.seeds if args.seeds is not None else [args.seed]
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = {
            "penalty_ppo": train_and_evaluate(
                setup, False, seed, args.timesteps, checkpoint_dir
            ),
            "safety_filtered_ppo": train_and_evaluate(
                setup, True, seed, args.timesteps, checkpoint_dir
            ),
        }

    def summarize(controller: str) -> dict:
        rows = [per_seed[str(seed)][controller] for seed in seeds]
        metrics = ("violation_rate", "unmet_fraction", "delivered_kwh")
        return {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "std": float(np.std([r[metric] for r in rows])),
            }
            for metric in metrics
        }

    results = {
        "seeds": seeds,
        "timesteps": args.timesteps,
        "per_seed": per_seed,
        "summary": {
            "penalty_ppo": summarize("penalty_ppo"),
            "safety_filtered_ppo": summarize("safety_filtered_ppo"),
        },
    }
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
