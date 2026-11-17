"""Train PPO and evaluate validation-week learning curves every 10k steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_2x2_run import S3_WEIGHTS, make_env, shared_setup
from train_controller import split_boundaries
from sweep_reward_weights import PPO_KWARGS


def get_validation_starts(setup: dict, condition: str, count: int) -> tuple[tuple[int, int], list[int]]:
    train_end, val_end = split_boundaries(setup["full_data"].n_steps)
    if condition == "forecast":
        if setup["fc"] is None:
            raise ValueError("Forecast condition requires a forecast file")
        first = int(round((setup["fc"].first_valid_time - setup["full_data"].start_time) / pd.Timedelta(hours=1)))
        allowed = (max(train_end, first), val_end - setup["episode_len"])
    else:
        allowed = (train_end, val_end - setup["episode_len"])
    if allowed[1] <= allowed[0]:
        raise ValueError(f"Validation range is empty: {allowed}")
    starts = sorted(set(np.linspace(allowed[0], allowed[1] - 1, count).astype(int).tolist()))
    return allowed, starts


def evaluate_validation(env, start_steps, policy) -> dict:
    rewards, violation_rates, unmet_fractions, delivered_kwhs = [], [], [], []
    for start_step in start_steps:
        obs, _ = env.reset(start_step=start_step)
        total_requested = float(sum(s.requested_kwh for s in env.inner.data.sessions))
        total_reward = 0.0
        delivered = 0.0
        unmet = 0.0
        violations = 0
        for _ in range(env.episode_len):
            action = policy(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            delivered += float(info["delivered_kwh"].sum())
            unmet += float(info["unmet_kwh_this_step"])
            violations += int(info["violation_kw"] > 0)
            if terminated or truncated:
                break
        rewards.append(total_reward)
        violation_rates.append(violations / env.episode_len)
        unmet_fractions.append(unmet / total_requested if total_requested > 0 else 0.0)
        delivered_kwhs.append(delivered)
    return {
        "reward": float(np.mean(rewards)),
        "violation_rate": float(np.mean(violation_rates)),
        "unmet_fraction": float(np.mean(unmet_fractions)),
        "delivered_kwh": float(np.mean(delivered_kwhs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["no_forecast", "forecast"], default="no_forecast")
    parser.add_argument("--forecast-csv", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--interval", type=int, default=10_000)
    parser.add_argument("--validation-weeks", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "learning_curves")
    args = parser.parse_args()
    if args.timesteps % args.interval != 0:
        raise ValueError("--timesteps must be divisible by --interval")

    setup = shared_setup(
        use_forecast=args.condition == "forecast",
        forecast_csv=str(args.forecast_csv) if args.forecast_csv else None,
    )
    train_range = setup["train_range_forecast"] if args.condition == "forecast" else setup["train_range_plain"]
    validation_range, validation_start_steps = get_validation_starts(
        setup, args.condition, args.validation_weeks
    )
    rows = []
    for seed in args.seeds:
        train_env = Monitor(make_env(setup, args.condition, seed, train_range))
        eval_env = make_env(setup, args.condition, 10_000 + seed, validation_range)
        model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, **PPO_KWARGS)
        for completed in range(args.interval, args.timesteps + 1, args.interval):
            model.learn(total_timesteps=args.interval, reset_num_timesteps=False)

            def policy(current_env, obs):
                action, _ = model.predict(obs, deterministic=True)
                return action

            metrics = evaluate_validation(eval_env, validation_start_steps, policy)
            rows.append(
                {
                    "seed": seed,
                    "condition": args.condition,
                    "timesteps": completed,
                    "validation_starts": ";".join(map(str, validation_start_steps)),
                    "validation_reward_mean": metrics["reward"],
                    "validation_violation_rate": metrics["violation_rate"],
                    "validation_unmet_fraction": metrics["unmet_fraction"],
                    "validation_delivered_kwh": metrics["delivered_kwh"],
                }
            )
            print(rows[-1])
        output = args.output_dir / f"learning_curve_{args.condition}_seed{seed}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).query("seed == @seed").to_csv(output, index=False)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
