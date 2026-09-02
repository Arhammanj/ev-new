# ==========================================
# Reward-weight sweep on the WORKING symmetric-action controller
# ==========================================
# All of sweep_reward_weights.py's results were run on a collapsed
# do-nothing policy (pre action-space fix) and are void. Now that
# symmetric_action=True (see train_controller.py's symmetric_to_kw())
# fixes the collapse - PPO delivers 1955.6 kWh at 5.48% violations
# instead of near-zero delivery - this re-runs the sweep to find
# reward weights that push violations below random (4.76%) while
# keeping delivery high, holding energy=1.0 fixed.
#
# Short runs only: 50k timesteps, single seed, no_forecast. This
# LOCKS the reward weights before any full 2x2 (forecast x PI)
# ablation, which is a separate later task.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from acn_data_loader import load_acn_sessions
from train_controller import (
    DATA_PATH, EPISODE_LEN, ACNWeeklyEpisodeEnv, EpisodeRewardCallback,
    cap_only_action, evaluate_policy_on_weeks, split_boundaries, kw_to_symmetric,
)
from sweep_reward_weights import PPO_KWARGS

SHORT_TIMESTEPS = 50_000
SEED = 1
N_TEST_WEEKS = 5

CONFIGS = {
    "S0 (unmet=1, viol=1)": {"energy": 1.0, "unmet": 1.0, "violation": 1.0},
    "S1 (unmet=1, viol=3)": {"energy": 1.0, "unmet": 1.0, "violation": 3.0},
    "S2 (unmet=1, viol=6)": {"energy": 1.0, "unmet": 1.0, "violation": 6.0},
    "S3 (unmet=2, viol=6)": {"energy": 1.0, "unmet": 2.0, "violation": 6.0},
    "S4 (unmet=2, viol=10)": {"energy": 1.0, "unmet": 2.0, "violation": 10.0},
}

RANDOM_VIOL_RATE = 0.0476   # cited reference (identical physical dynamics regardless of weights)
TARGET_VIOL_RATE = 0.02     # "ideally toward 0-2%"
TARGET_DELIVERED_KWH = 1800.0


def train_and_eval(weights, timesteps, seed, full_data, slot_roster, train_range,
                    eval_env, test_starts):
    train_env = Monitor(ACNWeeklyEpisodeEnv(
        full_data, slot_roster, train_range, reward_weights=weights, seed=seed,
        symmetric_action=True))
    model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, **PPO_KWARGS)
    cb = EpisodeRewardCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    n_check = max(1, len(cb.episode_rewards) // 5)
    early = float(np.mean(cb.episode_rewards[:n_check])) if cb.episode_rewards else float("nan")
    late = float(np.mean(cb.episode_rewards[-n_check:])) if cb.episode_rewards else float("nan")

    def ppo_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    result = evaluate_policy_on_weeks(eval_env, test_starts, ppo_fn)
    result["reward_early"] = early
    result["reward_late"] = late
    result["n_episodes"] = len(cb.episode_rewards)
    return result


def main() -> None:
    print("Loading full-year ACN data...")
    full_data = load_acn_sessions(DATA_PATH, timestep_minutes=60)
    slot_roster = full_data.slot_ids
    n_steps = full_data.n_steps
    train_end, val_end = split_boundaries(n_steps)
    train_range = (0, train_end - EPISODE_LEN)
    test_range = (val_end, n_steps - EPISODE_LEN)

    lo, hi = test_range
    test_starts = sorted(set(np.linspace(lo, hi - 1, N_TEST_WEEKS).astype(int).tolist()))
    print(f"Held-out test weeks (same as validate_symmetric_action.py): {test_starts}")

    eval_env = ACNWeeklyEpisodeEnv(full_data, slot_roster, test_range,
                                   reward_weights={"energy": 1.0}, seed=999,
                                   symmetric_action=True)

    # ---- Baselines (physical dynamics don't depend on reward_weights
    # or on the action-space parameterization - random re-derives its
    # own numbers here since env.action_space.sample() differs in
    # distribution shape (uniform on [-1,1] mapped affinely, still
    # uniform on [0, per_ev_max_kw]) but should match the cited 4.76%/
    # 30.60%/1865.2 reference within noise; cap_only round-trips
    # through kw_to_symmetric() and should match its reference exactly.
    print("\nEvaluating baselines (random, cap_only) under symmetric_action=True...")
    random_result = evaluate_policy_on_weeks(
        eval_env, test_starts, lambda env, obs: env.action_space.sample())
    cap_only_result = evaluate_policy_on_weeks(
        eval_env, test_starts,
        lambda env, obs: kw_to_symmetric(cap_only_action(env.inner), env.per_ev_max_kw))

    rows = []
    for name, r in [("random", random_result), ("cap_only", cap_only_result)]:
        rows.append({"Config": name, "ViolRate%": round(100 * r["violation_rate"], 2),
                     "UnmetFrac%": round(100 * r["unmet_fraction"], 2),
                     "Delivered_kWh": round(r["delivered_kwh"], 1),
                     "RewardEarly": None, "RewardLate": None})

    # ---- Sweep S0-S4 ----
    for name, weights in CONFIGS.items():
        print(f"\n===== {name}: reward_weights={weights}, {SHORT_TIMESTEPS} timesteps =====")
        r = train_and_eval(weights, SHORT_TIMESTEPS, SEED, full_data, slot_roster,
                           train_range, eval_env, test_starts)
        print(f"  violation_rate={100*r['violation_rate']:.2f}%  "
              f"unmet_fraction={100*r['unmet_fraction']:.2f}%  "
              f"delivered_kwh={r['delivered_kwh']:.1f}  "
              f"reward {r['reward_early']:.1f} -> {r['reward_late']:.1f} "
              f"({r['n_episodes']} episodes)")
        rows.append({"Config": name, "ViolRate%": round(100 * r["violation_rate"], 2),
                     "UnmetFrac%": round(100 * r["unmet_fraction"], 2),
                     "Delivered_kWh": round(r["delivered_kwh"], 1),
                     "RewardEarly": round(r["reward_early"], 1),
                     "RewardLate": round(r["reward_late"], 1)})

    table = pd.DataFrame(rows)
    print("\n===== SWEEP RESULTS (S0-S4, symmetric_action=True, 50k timesteps) =====")
    print(table.to_string(index=False))

    # ---- Trade-off / recommendation ----
    sweep_only = table[table["Config"].isin(CONFIGS.keys())].copy()
    cap_only_row = table[table["Config"] == "cap_only"].iloc[0]

    print("\n===== TRADE-OFF (violations vs delivery, S0-S4) =====")
    for _, row in sweep_only.iterrows():
        print(f"  {row['Config']:24s} viol={row['ViolRate%']:5.2f}%  "
              f"delivered={row['Delivered_kWh']:8.1f} kWh")

    candidates = sweep_only[
        (sweep_only["ViolRate%"] <= 100 * TARGET_VIOL_RATE)
        & (sweep_only["Delivered_kWh"] >= TARGET_DELIVERED_KWH)
    ]
    below_random = sweep_only[sweep_only["ViolRate%"] < 100 * RANDOM_VIOL_RATE]

    print(f"\nConfigs with violations <= {100*TARGET_VIOL_RATE:.0f}% AND "
          f"delivered >= {TARGET_DELIVERED_KWH:.0f} kWh: "
          f"{list(candidates['Config']) if len(candidates) else 'NONE'}")
    print(f"Configs with violations below random ({100*RANDOM_VIOL_RATE:.2f}%): "
          f"{list(below_random['Config']) if len(below_random) else 'NONE'}")

    if len(candidates):
        best = candidates.sort_values("Delivered_kWh", ascending=False).iloc[0]
    elif len(below_random):
        best = below_random.sort_values("Delivered_kWh", ascending=False).iloc[0]
    else:
        best = sweep_only.sort_values("ViolRate%").iloc[0]
    print(f"\n-> RECOMMENDED weight setting: {best['Config']} "
          f"(viol={best['ViolRate%']:.2f}%, delivered={best['Delivered_kWh']:.1f} kWh)")

    beats_cap = sweep_only[
        (sweep_only["Delivered_kWh"] > cap_only_row["Delivered_kWh"])
        | (sweep_only["ViolRate%"] < cap_only_row["ViolRate%"])
    ]
    print(f"\ncap_only reference: viol={cap_only_row['ViolRate%']:.2f}%  "
          f"delivered={cap_only_row['Delivered_kWh']:.1f} kWh")
    print(f"Any config beats cap_only on EITHER axis (more energy OR fewer violations)? "
          f"{'YES: ' + str(list(beats_cap['Config'])) if len(beats_cap) else 'NO'}")

    table.to_csv("acn_controller_reward_sweep_symmetric.csv", index=False)
    print("\nSaved acn_controller_reward_sweep_symmetric.csv")
    print("\nThis sweep LOCKS the reward weights for symmetric_action=True. "
          "The full 2x2 (forecast x PI) ablation is a separate, later task.")


if __name__ == "__main__":
    main()
