# ==========================================
# Full 2x2 experiment runner - {no_forecast, forecast} x {cap_only, safe-RL}
# ==========================================
# The paper's core result. Reward weights locked at S3 (energy=1,
# unmet=2, violation=6) per sweep_reward_weights_symmetric.py's
# recommendation; symmetric_action=True throughout (see
# train_controller.py's symmetric_to_kw()). cap_only + forecast is not
# a meaningful cell (a hard cap ignores forecasts) - only three real
# conditions: cap_only (non-learning), safe-RL no_forecast, safe-RL
# forecast, each safe-RL condition run over 5 seeds.
#
# Runs one (condition, seed) cell per invocation, in TIMESTEP CHUNKS
# with a checkpoint/sidecar file so a single long training run can be
# split across multiple process invocations (each comfortably under a
# few minutes) without losing progress - re-run the exact same command
# to continue from the last checkpoint; once total_timesteps is
# reached it evaluates and appends one row to the results CSV.
#
# TEST WEEKS: computed from a single, FORECAST-SAFE test range shared
# by every condition (cap_only, no_forecast, forecast) - the BiLSTM
# forecast (acn_target_a_bilstm_full.csv) does not cover the full raw
# test region (its last valid hour is a few steps short of the raw
# test range's end), so the shared test range is clipped to what the
# forecast can support. This costs a handful of hours off the trailing
# edge of the test region but guarantees ALL conditions are scored on
# the literal same held-out weeks, which is the point of the
# comparison.
# ==========================================

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from acn_data_loader import load_acn_sessions
from train_controller import (
    DATA_PATH, EPISODE_LEN, ACNWeeklyEpisodeEnv, ForecastLookup,
    evaluate_policy_on_weeks, split_boundaries,
)
from sweep_reward_weights import PPO_KWARGS

S3_WEIGHTS = {"energy": 1.0, "unmet": 2.0, "violation": 6.0}
N_TEST_WEEKS = 5
CHECKPOINT_DIR = "acn_exp_2x2_checkpoints"
RESULTS_DIR = "acn_exp_2x2_results"


def shared_setup():
    full_data = load_acn_sessions(DATA_PATH, timestep_minutes=60)
    slot_roster = full_data.slot_ids
    n_steps = full_data.n_steps
    train_end, val_end = split_boundaries(n_steps)

    fc = ForecastLookup()
    step_delta = pd.Timedelta(hours=1)
    fc_first_idx = int(round((fc.first_valid_time - full_data.start_time) / step_delta))
    fc_last_idx = int(round((fc.last_valid_time - full_data.start_time) / step_delta))

    # Test range shared by EVERY condition - clipped to what the
    # forecast can support so no condition ever sees a different set
    # of held-out weeks.
    test_lo = val_end
    test_hi = min(n_steps - EPISODE_LEN, fc_last_idx - EPISODE_LEN + 1)
    test_starts = sorted(set(np.linspace(test_lo, test_hi - 1, N_TEST_WEEKS).astype(int).tolist()))

    train_range_plain = (0, train_end - EPISODE_LEN)
    train_range_forecast = (max(0, fc_first_idx), train_end - EPISODE_LEN)

    return {
        "full_data": full_data, "slot_roster": slot_roster, "fc": fc,
        "train_range_plain": train_range_plain, "train_range_forecast": train_range_forecast,
        "test_range": (test_lo, test_hi), "test_starts": test_starts,
    }


def make_env(setup, condition: str, seed: int, allowed_start_range):
    forecast = setup["fc"] if condition == "forecast" else None
    return ACNWeeklyEpisodeEnv(
        setup["full_data"], setup["slot_roster"], allowed_start_range,
        reward_weights=S3_WEIGHTS, seed=seed, symmetric_action=True, forecast=forecast,
    )


def append_result_row(row: dict) -> None:
    # One file per (condition, seed) rather than a shared appended CSV -
    # parallel runs finishing at the same time would otherwise race on
    # a single file and risk an interleaved/corrupted write.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{row['condition']}_seed{row['seed']}.json")
    with open(path, "w") as f:
        json.dump(row, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["no_forecast", "forecast"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--chunk-timesteps", type=int, default=150_000)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.condition}_seed{args.seed}.zip")
    sidecar_path = os.path.join(CHECKPOINT_DIR, f"{args.condition}_seed{args.seed}.json")

    print(f"Loading full-year ACN data + forecast lookup...")
    setup = shared_setup()
    print(f"Shared test weeks (forecast-safe, same for every condition): {setup['test_starts']}")

    train_range = (setup["train_range_forecast"] if args.condition == "forecast"
                   else setup["train_range_plain"])
    train_env = Monitor(make_env(setup, args.condition, args.seed, train_range))

    done_timesteps = 0
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as f:
            done_timesteps = json.load(f)["done_timesteps"]
        model = PPO.load(ckpt_path, env=train_env)
        print(f"Resumed {args.condition} seed={args.seed} from checkpoint: "
              f"{done_timesteps}/{args.total_timesteps} timesteps already done")
    else:
        model = PPO("MlpPolicy", train_env, seed=args.seed, verbose=0, **PPO_KWARGS)
        print(f"Fresh model: {args.condition} seed={args.seed}")

    remaining = args.total_timesteps - done_timesteps
    if remaining > 0:
        this_chunk = min(args.chunk_timesteps, remaining)
        print(f"Training chunk: {this_chunk} timesteps "
              f"({done_timesteps} -> {done_timesteps + this_chunk} of {args.total_timesteps})")
        model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)
        done_timesteps += this_chunk
        model.save(ckpt_path)
        with open(sidecar_path, "w") as f:
            json.dump({"done_timesteps": done_timesteps}, f)
        print(f"Checkpoint saved: {done_timesteps}/{args.total_timesteps} timesteps done")

    if done_timesteps < args.total_timesteps:
        print(f"NOT YET COMPLETE - re-run this exact command to continue "
              f"({done_timesteps}/{args.total_timesteps} done)")
        return

    print(f"Training complete ({done_timesteps} timesteps). Evaluating on held-out test weeks...")
    eval_env = make_env(setup, args.condition, seed=999 + args.seed,
                        allowed_start_range=setup["test_range"])

    def ppo_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    result = evaluate_policy_on_weeks(eval_env, setup["test_starts"], ppo_fn)
    row = {
        "condition": args.condition, "seed": args.seed,
        "violation_rate": result["violation_rate"],
        "unmet_fraction": result["unmet_fraction"],
        "delivered_kwh": result["delivered_kwh"],
    }
    append_result_row(row)
    print(f"\n===== RESULT: {args.condition} seed={args.seed} =====")
    print(f"  violation_rate={100*result['violation_rate']:.2f}%  "
          f"unmet_fraction={100*result['unmet_fraction']:.2f}%  "
          f"delivered_kwh={result['delivered_kwh']:.1f}")
    print(f"Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
