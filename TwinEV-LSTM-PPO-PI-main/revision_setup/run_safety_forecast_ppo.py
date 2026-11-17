"""Run isolated hard-safety PPO with and without the forecast feature."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_2x2_run import S3_WEIGHTS, shared_setup
from train_controller import ACNWeeklyEpisodeEnv, evaluate_policy_on_weeks
from sweep_reward_weights import PPO_KWARGS

REQUIRED_TEST_STARTS = [7429, 7713, 7997, 8281, 8565]
RESTRICTED_TRAIN_START = 1881
DEFAULT_FORECAST_CSV = Path(__file__).resolve().parent / "forecast_oof.csv"


def run_one(
    setup,
    forecast: bool,
    seed: int,
    timesteps: int,
    capacity: float,
    train_range: tuple[int, int],
) -> dict:
    forecast_lookup = setup["fc"] if forecast else None
    train_env = Monitor(ACNWeeklyEpisodeEnv(
        setup["full_data"], setup["slot_roster"], train_range,
        transformer_limit_kw=capacity, reward_weights=S3_WEIGHTS,
        symmetric_action=True, hard_safety_filter=True,
        forecast=forecast_lookup, seed=seed,
    ))
    model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, **PPO_KWARGS)
    model.learn(total_timesteps=timesteps)
    eval_env = ACNWeeklyEpisodeEnv(
        setup["full_data"], setup["slot_roster"], setup["test_range"],
        transformer_limit_kw=capacity, reward_weights=S3_WEIGHTS,
        symmetric_action=True, hard_safety_filter=True,
        forecast=forecast_lookup, seed=1000 + seed,
    )
    result = evaluate_policy_on_weeks(
        eval_env, setup["test_starts"],
        lambda env, obs: model.predict(obs, deterministic=True)[0],
    )
    result.update({
        "condition": "forecast" if forecast else "no_forecast",
        "seed": seed,
        "capacity_kw": capacity,
        "timesteps": timesteps,
        "training_filter_interventions": train_env.env.safety_filter_interventions,
        "evaluation_filter_interventions": eval_env.safety_filter_interventions,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-csv", default=str(DEFAULT_FORECAST_CSV))
    parser.add_argument("--capacity", type=float, default=50.0)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "safety_forecast_ppo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit JSON path; defaults to one file per seed in --output-dir.",
    )
    args = parser.parse_args()
    setup = shared_setup(forecast_csv=args.forecast_csv, use_forecast=True)
    train_range = (RESTRICTED_TRAIN_START, setup["train_range_plain"][1])
    setup["test_starts"] = list(REQUIRED_TEST_STARTS)
    for start in setup["test_starts"]:
        if not (setup["test_range"][0] <= start < setup["test_range"][1]):
            raise ValueError(f"test start {start} outside {setup['test_range']}")

    rows = []
    for forecast in (False, True):
        started = time.perf_counter()
        row = run_one(setup, forecast, args.seed, args.timesteps, args.capacity, train_range)
        elapsed = time.perf_counter() - started
        rows.append(row)
        print(
            f"{row['condition']} seed={args.seed}: {args.timesteps} timesteps "
            f"in {elapsed:.1f} s"
        )

    output = args.output or (
        args.output_dir / f"safety_forecast_ppo_seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2))
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
