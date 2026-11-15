"""Compare ordinary penalty PPO with hard safety-filtered PPO."""

from __future__ import annotations

import argparse

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


def train_and_evaluate(setup, hard_safety_filter: bool, seed: int, timesteps: int):
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
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output", default="acn_safety_filtered_comparison.json")
    args = parser.parse_args()
    setup = shared_setup()
    results = {
        "penalty_ppo": train_and_evaluate(setup, False, args.seed, args.timesteps),
        "safety_filtered_ppo": train_and_evaluate(setup, True, args.seed, args.timesteps),
    }
    import json
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
