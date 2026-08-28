# ==========================================
# Energy-reward vs violation-penalty sweep (+ soft violation ramp)
# ==========================================
# Step 1 (instrument_reward_terms.py) found the per-step imbalance
# hypothesis NOT confirmed: during busy hours, all-max charging earns
# +76.18 energy reward vs only -26.18 violation penalty at the
# original weights (net +50/step) - so a purely per-step-rational
# policy should already prefer charging. This sweep tests raising the
# energy weight anyway (as instructed), plus a softened (quadratic)
# violation ramp at the best energy weight found.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from acn_data_loader import load_acn_sessions
from train_controller import (
    DATA_PATH, EPISODE_LEN, ACNWeeklyEpisodeEnv, evaluate_policy_on_weeks,
    cap_only_action, split_boundaries,
)
from sweep_reward_weights import PPO_KWARGS, train_and_eval, SEED

CONFIGS = {
    "E1 (energy=5,  viol=1)":   {"energy": 5.0,  "unmet": 1.0, "violation": 1.0},
    "E2 (energy=10, viol=1)":   {"energy": 10.0, "unmet": 1.0, "violation": 1.0},
    "E3 (energy=10, viol=0.5)": {"energy": 10.0, "unmet": 1.0, "violation": 0.5},
    "E4 (energy=20, viol=1)":   {"energy": 20.0, "unmet": 1.0, "violation": 1.0},
}

SHORT_TIMESTEPS = 50_000
N_TEST_WEEKS = 5
TARGET_DELIVERED_KWH = 1000.0
TARGET_VIOL_RATE_PCT = 5.0


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
    print(f"Held-out test weeks: {test_starts}")

    eval_env = ACNWeeklyEpisodeEnv(full_data, slot_roster, test_range,
                                   reward_weights={"energy": 1.0}, seed=999)

    print("\nEvaluating baselines (random, cap_only)...")
    random_result = evaluate_policy_on_weeks(
        eval_env, test_starts, lambda env, obs: env.action_space.sample())
    cap_only_result = evaluate_policy_on_weeks(
        eval_env, test_starts, lambda env, obs: cap_only_action(env.inner))

    rows = []
    for name, r in [("random", random_result), ("cap_only", cap_only_result)]:
        rows.append({"Config": name, "ViolRate%": round(100 * r["violation_rate"], 2),
                     "UnmetFrac%": round(100 * r["unmet_fraction"], 2),
                     "Delivered_kWh": round(r["delivered_kwh"], 1)})

    best_name, best_weights, best_delivered = None, None, -1.0
    for name, weights in CONFIGS.items():
        print(f"\n===== {name}: reward_weights={weights}, {SHORT_TIMESTEPS} timesteps =====")
        r = train_and_eval(weights, SHORT_TIMESTEPS, SEED, full_data, slot_roster,
                           train_range, eval_env, test_starts)
        print(f"  violation_rate={100*r['violation_rate']:.2f}%  "
              f"unmet_fraction={100*r['unmet_fraction']:.2f}%  "
              f"delivered_kwh={r['delivered_kwh']:.1f}  "
              f"reward {r['reward_early']:.1f} -> {r['reward_late']:.1f}")
        rows.append({"Config": name, "ViolRate%": round(100 * r["violation_rate"], 2),
                     "UnmetFrac%": round(100 * r["unmet_fraction"], 2),
                     "Delivered_kWh": round(r["delivered_kwh"], 1)})
        if r["delivered_kwh"] > best_delivered:
            best_delivered = r["delivered_kwh"]
            best_name, best_weights = name, weights

    print(f"\nBest of E1-E4 by delivered kWh: {best_name} ({best_delivered:.1f} kWh) - "
          f"running the soft (quadratic) violation-penalty variant at these same weights.")

    soft_env_train = ACNWeeklyEpisodeEnv(full_data, slot_roster, train_range,
                                        reward_weights=best_weights, seed=SEED,
                                        violation_penalty_kind="quadratic",
                                        violation_softness_kw=25.0)
    from stable_baselines3 import PPO
    from train_controller import EpisodeRewardCallback

    train_env = Monitor(soft_env_train)
    model = PPO("MlpPolicy", train_env, seed=SEED, verbose=0, **PPO_KWARGS)
    cb = EpisodeRewardCallback()
    model.learn(total_timesteps=SHORT_TIMESTEPS, callback=cb)
    n_check = max(1, len(cb.episode_rewards) // 5)
    early = float(np.mean(cb.episode_rewards[:n_check])) if cb.episode_rewards else float("nan")
    late = float(np.mean(cb.episode_rewards[-n_check:])) if cb.episode_rewards else float("nan")

    soft_eval_env = ACNWeeklyEpisodeEnv(full_data, slot_roster, test_range,
                                       reward_weights=best_weights, seed=999,
                                       violation_penalty_kind="quadratic",
                                       violation_softness_kw=25.0)

    def ppo_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    r_soft = evaluate_policy_on_weeks(soft_eval_env, test_starts, ppo_fn)
    print(f"\n===== {best_name} + SOFT (quadratic) violation penalty =====")
    print(f"  violation_rate={100*r_soft['violation_rate']:.2f}%  "
          f"unmet_fraction={100*r_soft['unmet_fraction']:.2f}%  "
          f"delivered_kwh={r_soft['delivered_kwh']:.1f}  "
          f"reward {early:.1f} -> {late:.1f}")
    rows.append({"Config": f"{best_name} + soft-quadratic",
                "ViolRate%": round(100 * r_soft["violation_rate"], 2),
                "UnmetFrac%": round(100 * r_soft["unmet_fraction"], 2),
                "Delivered_kWh": round(r_soft["delivered_kwh"], 1)})

    table = pd.DataFrame(rows)
    print("\n===== FULL TABLE =====")
    print(table.to_string(index=False))
    table.to_csv("acn_controller_energy_sweep.csv", index=False)
    print("\nSaved acn_controller_energy_sweep.csv")

    non_degenerate = table[
        (table["Delivered_kWh"] >= TARGET_DELIVERED_KWH) &
        (table["ViolRate%"] <= TARGET_VIOL_RATE_PCT) &
        (~table["Config"].isin(["random", "cap_only"]))
    ]
    print(f"\nConfig(s) reaching >= {TARGET_DELIVERED_KWH:.0f} kWh at <= "
          f"{TARGET_VIOL_RATE_PCT:.0f}% violations: "
          f"{non_degenerate['Config'].tolist() if len(non_degenerate) else 'NONE'}")


if __name__ == "__main__":
    main()
