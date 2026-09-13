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
BASE_HORIZON = 6   # the original/paper horizon - keeps today's unsuffixed dirs/filenames
BASE_CAPACITY_KW = 50.0   # the original/paper transformer limit - ditto
CHECKPOINT_DIR = "acn_exp_2x2_checkpoints"
RESULTS_DIR = "acn_exp_2x2_results"


def horizon_forecast_csv(horizon: int) -> str:
    return ("acn_target_a_bilstm_full.csv" if horizon == BASE_HORIZON
            else f"acn_target_a_h{horizon}_bilstm_full.csv")


def horizon_dirs(horizon: int) -> tuple[str, str]:
    """Backwards-compatible alias for run_dirs() at the base capacity."""
    return run_dirs(horizon, BASE_CAPACITY_KW)


def run_dirs(horizon: int, capacity_kw: float = BASE_CAPACITY_KW) -> tuple[str, str]:
    """(checkpoint_dir, results_dir) for this (horizon, capacity_kw) - the
    base combination (horizon==BASE_HORIZON AND capacity_kw==BASE_CAPACITY_KW)
    reuses the original unsuffixed dirs so already-computed 6h/50kW results/
    checkpoints stay valid without rerunning anything. Any other combination
    gets a suffix per axis that differs from its base value, so e.g. the
    horizon sweep's existing _h1/_h3 dirs (capacity left at base) are
    unaffected by this generalization."""
    suffix = ""
    if horizon != BASE_HORIZON:
        suffix += f"_h{horizon}"
    if capacity_kw != BASE_CAPACITY_KW:
        suffix += f"_cap{int(capacity_kw)}"
    return f"{CHECKPOINT_DIR}{suffix}", f"{RESULTS_DIR}{suffix}"


def shared_setup(horizon: int = BASE_HORIZON):
    full_data = load_acn_sessions(DATA_PATH, timestep_minutes=60)
    slot_roster = full_data.slot_ids
    n_steps = full_data.n_steps
    train_end, val_end = split_boundaries(n_steps)

    fc = ForecastLookup(csv_path=horizon_forecast_csv(horizon))
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


def make_env(setup, condition: str, seed: int, allowed_start_range,
             capacity_kw: float = BASE_CAPACITY_KW):
    forecast = setup["fc"] if condition == "forecast" else None
    return ACNWeeklyEpisodeEnv(
        setup["full_data"], setup["slot_roster"], allowed_start_range,
        transformer_limit_kw=capacity_kw,
        reward_weights=S3_WEIGHTS, seed=seed, symmetric_action=True, forecast=forecast,
    )


def append_result_row(row: dict, results_dir: str = RESULTS_DIR) -> None:
    # One file per (condition, seed) rather than a shared appended CSV -
    # parallel runs finishing at the same time would otherwise race on
    # a single file and risk an interleaved/corrupted write.
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{row['condition']}_seed{row['seed']}.json")
    with open(path, "w") as f:
        json.dump(row, f)


def checkpoint_done_timesteps(checkpoint_dir: str, condition: str, seed: int) -> int:
    """Timesteps already completed for (condition, seed) in checkpoint_dir,
    per its sidecar JSON - 0 if no checkpoint exists there yet."""
    sidecar_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}.json")
    if not os.path.exists(sidecar_path):
        return 0
    with open(sidecar_path) as f:
        return json.load(f)["done_timesteps"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["no_forecast", "forecast"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon", type=int, choices=[1, 3, 6], default=BASE_HORIZON,
                        help="Forecast lead time in hours (default: 6, the original "
                             "horizon). no_forecast's POLICY doesn't depend on horizon "
                             "at all, so at horizon != 6 (and capacity at its base value) "
                             "this reuses the already-trained base no_forecast checkpoint "
                             "if one is complete, skipping straight to evaluation on this "
                             "horizon's test weeks.")
    parser.add_argument("--capacity-kw", type=float, default=BASE_CAPACITY_KW,
                        help="Transformer capacity limit in kW (default: 50.0, the "
                             "original/paper limit). Unlike horizon, no_forecast's policy "
                             "DOES depend on capacity (it changes the environment's actual "
                             "constraint/reward), so both conditions always train fresh at "
                             "any non-base capacity - no checkpoint reuse.")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--chunk-timesteps", type=int, default=150_000)
    args = parser.parse_args()

    checkpoint_dir, results_dir = run_dirs(args.horizon, args.capacity_kw)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{args.condition}_seed{args.seed}.zip")
    sidecar_path = os.path.join(checkpoint_dir, f"{args.condition}_seed{args.seed}.json")

    print(f"Loading full-year ACN data + forecast lookup (horizon={args.horizon}h, "
          f"capacity={args.capacity_kw}kW)...")
    setup = shared_setup(args.horizon)
    print(f"Shared test weeks (forecast-safe, same for every condition): {setup['test_starts']}")

    # no_forecast's policy is horizon-independent (it never sees the
    # forecast feature) - reuse the base (horizon=6, capacity=BASE_CAPACITY_KW)
    # checkpoint instead of retraining if it's already complete, and just
    # re-evaluate on this horizon's (possibly shifted) test weeks. Gated on
    # capacity too: capacity DOES change the policy (different constraint/
    # reward), so this reuse only applies when capacity is unchanged from base.
    reused_base_checkpoint = False
    if (args.condition == "no_forecast" and args.horizon != BASE_HORIZON
            and args.capacity_kw == BASE_CAPACITY_KW):
        base_done = checkpoint_done_timesteps(CHECKPOINT_DIR, args.condition, args.seed)
        if base_done >= args.total_timesteps:
            base_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.condition}_seed{args.seed}.zip")
            print(f"Reusing complete base (horizon={BASE_HORIZON}) no_forecast checkpoint "
                  f"({base_done} timesteps) - policy doesn't depend on horizon, skipping training")
            model = PPO.load(base_ckpt_path)
            done_timesteps = args.total_timesteps
            reused_base_checkpoint = True

    if not reused_base_checkpoint:
        train_range = (setup["train_range_forecast"] if args.condition == "forecast"
                       else setup["train_range_plain"])
        train_env = Monitor(make_env(setup, args.condition, args.seed, train_range,
                                     capacity_kw=args.capacity_kw))

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
                        allowed_start_range=setup["test_range"], capacity_kw=args.capacity_kw)

    def ppo_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    result = evaluate_policy_on_weeks(eval_env, setup["test_starts"], ppo_fn)
    row = {
        "condition": args.condition, "seed": args.seed, "horizon": args.horizon,
        "capacity_kw": args.capacity_kw,
        "violation_rate": result["violation_rate"],
        "unmet_fraction": result["unmet_fraction"],
        "delivered_kwh": result["delivered_kwh"],
    }
    append_result_row(row, results_dir)
    print(f"\n===== RESULT: {args.condition} seed={args.seed} horizon={args.horizon}h =====")
    print(f"  violation_rate={100*result['violation_rate']:.2f}%  "
          f"unmet_fraction={100*result['unmet_fraction']:.2f}%  "
          f"delivered_kwh={result['delivered_kwh']:.1f}")
    print(f"Saved to {results_dir}/")


if __name__ == "__main__":
    main()
