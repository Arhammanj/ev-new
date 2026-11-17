"""Train and evaluate one isolated PPO forecast/no-forecast experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_2x2_run import S3_WEIGHTS, shared_setup
from sweep_reward_weights import PPO_KWARGS
from train_controller import ACNWeeklyEpisodeEnv, evaluate_policy_on_weeks

REQUIRED_TEST_STARTS = [7429, 7713, 7997, 8281, 8565]
RESTRICTED_TRAIN_START = 1881


class AggregateDrawObservationEnv(ACNWeeklyEpisodeEnv):
    """ACNWeeklyEpisodeEnv with aggregate draw and slack appended to the observation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        base_dim = self.observation_space.shape[0]
        max_draw_ratio = (
            len(self.slot_roster) * self.per_ev_max_kw / self.transformer_limit_kw
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=max(3.0, max_draw_ratio),
            shape=(base_dim + 2,),
            dtype=np.float32,
        )

    def _aggregate_draw_kw(self) -> float:
        present = self.inner.true_presence()
        remaining_kwh = self.inner.true_remaining_kwh()
        max_draw_kw = np.where(
            present,
            np.minimum(self.inner.per_ev_max_kw, remaining_kwh / self.inner.dt_hours),
            0.0,
        )
        return float(max_draw_kw.sum())

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        obs = super()._augment(obs)
        capacity_kw = float(self.transformer_limit_kw)
        draw_kw = self._aggregate_draw_kw()
        slack_kw = max(0.0, capacity_kw - draw_kw)
        return np.concatenate([
            obs,
            np.array([draw_kw / capacity_kw, slack_kw / capacity_kw], dtype=np.float32),
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["forecast", "no_forecast", "no_forecast_restricted", "agg_obs"],
        required=True,
    )
    parser.add_argument("--forecast-csv")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--capacity", type=float, default=50.0)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup = shared_setup(
        horizon=args.horizon,
        forecast_csv=args.forecast_csv,
        use_forecast=args.condition == "forecast",
    )
    forecast = setup["fc"] if args.condition == "forecast" else None
    if args.condition in ("no_forecast_restricted", "agg_obs"):
        train_range = (RESTRICTED_TRAIN_START, setup["train_range_plain"][1])
    else:
        train_range = setup["train_range_forecast"] if forecast else setup["train_range_plain"]
    env_cls = AggregateDrawObservationEnv if args.condition == "agg_obs" else ACNWeeklyEpisodeEnv
    setup["test_starts"] = list(REQUIRED_TEST_STARTS)
    assert setup["test_starts"] == REQUIRED_TEST_STARTS, setup["test_starts"]
    for start in setup["test_starts"]:
        assert setup["test_range"][0] <= start < setup["test_range"][1], (
            start, setup["test_range"]
        )
    train_env = Monitor(env_cls(
        setup["full_data"], setup["slot_roster"], train_range,
        transformer_limit_kw=args.capacity, reward_weights=S3_WEIGHTS,
        symmetric_action=True, forecast=forecast, seed=args.seed,
    ))
    model = PPO("MlpPolicy", train_env, seed=args.seed, verbose=0, **PPO_KWARGS)
    model.learn(total_timesteps=args.timesteps)
    eval_env = env_cls(
        setup["full_data"], setup["slot_roster"], setup["test_range"],
        transformer_limit_kw=args.capacity, reward_weights=S3_WEIGHTS,
        symmetric_action=True, forecast=forecast, seed=10_000 + args.seed,
    )
    result = evaluate_policy_on_weeks(
        eval_env, setup["test_starts"],
        lambda env, obs: model.predict(obs, deterministic=True)[0],
    )
    result.update({
        "condition": args.condition, "horizon": args.horizon,
        "capacity_kw": args.capacity, "seed": args.seed,
        "timesteps": args.timesteps,
        "test_starts": setup["test_starts"],
        "forecast_csv": args.forecast_csv,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
