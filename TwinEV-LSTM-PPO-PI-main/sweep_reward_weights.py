# ==========================================
# Short reward-weight sweep - diagnose the do-nothing collapse
# ==========================================
# The no_forecast PPO (energy=1, unmet=1, violation=1) collapsed to a
# do-nothing policy: 86.5 kWh delivered vs cap_only's 2173.8, 97%
# unmet, 0% violations - it avoided the violation penalty by barely
# charging. This is diagnosis, not the final experiment: same 50k
# timesteps, single seed, no_forecast variant, same held-out test
# weeks as train_controller.py's validation run.
#
# random/cap_only baselines are IDENTICAL across every weight config -
# reward_weights only scales the logged reward scalar, never the
# physical dynamics (delivered/violation/unmet) - so they're computed
# once, not once per config.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from acn_data_loader import load_acn_sessions
from train_controller import (
    DATA_PATH, EPISODE_LEN, ACNWeeklyEpisodeEnv, EpisodeRewardCallback,
    cap_only_action, evaluate_policy_on_weeks, split_boundaries,
)

PPO_KWARGS = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
)

CONFIGS = {
    "A (unmet=3, viol=1)": {"energy": 1.0, "unmet": 3.0, "violation": 1.0},
    "B (unmet=5, viol=1)": {"energy": 1.0, "unmet": 5.0, "violation": 1.0},
    "C (unmet=5, viol=2)": {"energy": 1.0, "unmet": 5.0, "violation": 2.0},
    "D (unmet=10, viol=2)": {"energy": 1.0, "unmet": 10.0, "violation": 2.0},
}

SHORT_TIMESTEPS = 50_000
LONG_TIMESTEPS = 150_000
SEED = 1
N_TEST_WEEKS = 5

TARGET_DELIVERED_KWH = 2173.8 * 0.5   # "at least half of cap_only's ~2174 kWh"
TARGET_VIOL_RATE = 0.05               # "ideally under a few %"


def train_and_eval(weights, timesteps, seed, full_data, slot_roster, train_range,
                   eval_env, test_starts):
    train_env = Monitor(ACNWeeklyEpisodeEnv(
        full_data, slot_roster, train_range, reward_weights=weights, seed=seed))
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
    print(f"Held-out test weeks (same as train_controller.py's validation): {test_starts}")

    eval_env = ACNWeeklyEpisodeEnv(full_data, slot_roster, test_range,
                                   reward_weights={"energy": 1.0}, seed=999)

    # ---- Baselines (computed ONCE - independent of reward_weights) ----
    print("\nEvaluating baselines (random, cap_only)...")
    random_result = evaluate_policy_on_weeks(
        eval_env, test_starts, lambda env, obs: env.action_space.sample())
    cap_only_result = evaluate_policy_on_weeks(
        eval_env, test_starts, lambda env, obs: cap_only_action(env.inner))

    rows = []
    for name, r in [("random", random_result), ("cap_only", cap_only_result)]:
        rows.append({"Config": name, "ViolRate%": round(100 * r["violation_rate"], 2),
                     "UnmetFrac%": round(100 * r["unmet_fraction"], 2),
                     "Delivered_kWh": round(r["delivered_kwh"], 1),
                     "RewardEarly": None, "RewardLate": None})

    # ---- Sweep A-D ----
    best_config_name, best_delivered = None, -1.0
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
        if r["delivered_kwh"] > best_delivered:
            best_delivered = r["delivered_kwh"]
            best_config_name = name

    table = pd.DataFrame(rows)
    print("\n===== SWEEP RESULTS (A-D, 50k timesteps) =====")
    print(table.to_string(index=False))

    # ---- Direction check ----
    sweep_only = table[table["Config"].isin(CONFIGS.keys())]
    unmet_weights = [CONFIGS[c]["unmet"] for c in sweep_only["Config"]]
    delivered_vals = sweep_only["Delivered_kWh"].values
    corr = float(np.corrcoef(unmet_weights, delivered_vals)[0, 1]) if len(set(unmet_weights)) > 1 else float("nan")
    print(f"\nDelivered kWh vs unmet weight correlation across A-D: {corr:.3f} "
          f"({'trending UP with more unmet weight' if corr > 0.3 else 'NOT clearly trending up'})")

    any_success = any(
        (r["Delivered_kWh"] >= TARGET_DELIVERED_KWH and r["ViolRate%"] <= 100 * TARGET_VIOL_RATE)
        for r in rows if r["Config"] in CONFIGS
    )
    print(f"\nAny config reach >= {TARGET_DELIVERED_KWH:.0f} kWh delivered AND "
          f"<= {100*TARGET_VIOL_RATE:.0f}% violations? {'YES' if any_success else 'NO'}")

    # ---- Optional longer run on the best-of-4 ----
    if not any_success:
        print(f"\nNone of A-D reached the target. Best delivered_kWh was '{best_config_name}' "
              f"({best_delivered:.1f} kWh) - running it longer ({LONG_TIMESTEPS} timesteps) "
              "to check if this is partly a training-budget issue...")
        r_long = train_and_eval(CONFIGS[best_config_name], LONG_TIMESTEPS, SEED, full_data,
                                slot_roster, train_range, eval_env, test_starts)
        print(f"\n===== {best_config_name} @ {LONG_TIMESTEPS} timesteps =====")
        print(f"  violation_rate={100*r_long['violation_rate']:.2f}%  "
              f"unmet_fraction={100*r_long['unmet_fraction']:.2f}%  "
              f"delivered_kwh={r_long['delivered_kwh']:.1f}  "
              f"reward {r_long['reward_early']:.1f} -> {r_long['reward_late']:.1f} "
              f"({r_long['n_episodes']} episodes)")
        rows.append({"Config": f"{best_config_name} @ 150k", "ViolRate%": round(100 * r_long["violation_rate"], 2),
                     "UnmetFrac%": round(100 * r_long["unmet_fraction"], 2),
                     "Delivered_kWh": round(r_long["delivered_kwh"], 1),
                     "RewardEarly": round(r_long["reward_early"], 1),
                     "RewardLate": round(r_long["reward_late"], 1)})
        table = pd.DataFrame(rows)

    print("\n===== FINAL TABLE =====")
    print(table.to_string(index=False))
    table.to_csv("acn_controller_reward_sweep.csv", index=False)
    print("\nSaved acn_controller_reward_sweep.csv")


if __name__ == "__main__":
    main()
