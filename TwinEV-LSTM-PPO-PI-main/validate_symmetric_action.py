# ==========================================
# Validate the symmetric [-1, 1] action-space fix (see symmetric_to_kw()
# in train_controller.py for the mapping and the root-cause writeup).
# ==========================================
# SHORT run only (50k timesteps, single seed) - confirms the rescale
# gets PPO off the zero-charging collapse WITHOUT the action-floor
# patch's forced minimum. Does NOT re-tune reward weights and does NOT
# run the full 2x2 (forecast x PI) ablation - that is a separate,
# later task if this fix is adopted.
# ==========================================

from __future__ import annotations

import numpy as np

from acn_data_loader import load_acn_sessions
from train_controller import (
    DATA_PATH, EPISODE_LEN, ACNWeeklyEpisodeEnv, cap_only_action,
    evaluate_policy_on_weeks, split_boundaries, symmetric_to_kw, kw_to_symmetric,
)
from sweep_reward_weights import PPO_KWARGS
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ORIGINAL_WEIGHTS = {"energy": 1.0, "unmet": 1.0, "violation": 1.0}
SHORT_TIMESTEPS = 50_000
SEED = 1
N_TEST_WEEKS = 5
PER_EV_MAX_KW = 6.6

# Known reference numbers from prior runs (cited, not recomputed here):
#   cap_only (non-learning baseline): 2173.8 kWh / 0% violations / 18.11% unmet
#   action-floor patch (1.0 kW floor, 50k steps, seed 1): 919 kWh delivered
CAP_ONLY_REFERENCE = {"delivered_kwh": 2173.8, "violation_rate": 0.0, "unmet_fraction": 0.1811}
ACTION_FLOOR_REFERENCE_KWH = 919.0


def cap_only_symmetric_fn(env: ACNWeeklyEpisodeEnv, obs):
    """cap_only_action() returns physical kW; invert the mapping so it
    round-trips correctly through a symmetric-action-space env.step()."""
    kw = cap_only_action(env.inner)
    return kw_to_symmetric(kw, env.per_ev_max_kw)


def random_fn(env: ACNWeeklyEpisodeEnv, obs):
    return env.action_space.sample()


def collect_action_stats(env: ACNWeeklyEpisodeEnv, test_starts, action_fn) -> dict:
    """Raw (external, [-1,1]) and mapped-kW action stats over present
    slots only - absent-slot actions are zeroed by the inner env
    regardless of what the policy outputs."""
    raw_actions, kw_actions = [], []
    for start_step in test_starts:
        obs, _ = env.reset(start_step=start_step)
        for _ in range(env.episode_len):
            action = action_fn(env, obs)
            present = env.inner.true_presence()
            raw = np.asarray(action)[present]
            raw_actions.extend(raw.tolist())
            kw_actions.extend(symmetric_to_kw(raw, env.per_ev_max_kw).tolist())
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    raw_arr = np.array(raw_actions)
    kw_arr = np.array(kw_actions)
    return {
        "n": len(raw_arr),
        "mean_raw": float(raw_arr.mean()) if len(raw_arr) else float("nan"),
        "std_raw": float(raw_arr.std()) if len(raw_arr) else float("nan"),
        "mean_kw": float(kw_arr.mean()) if len(kw_arr) else float("nan"),
        "std_kw": float(kw_arr.std()) if len(kw_arr) else float("nan"),
        "frac_near_lower": float(np.mean(raw_arr <= -0.9)) if len(raw_arr) else float("nan"),
        "frac_near_upper": float(np.mean(raw_arr >= 0.9)) if len(raw_arr) else float("nan"),
    }


def print_action_stats(label: str, stats: dict) -> None:
    print(f"\n--- {label} (n={stats['n']} present-slot actions) ---")
    print(f"  raw [-1,1]:  mean={stats['mean_raw']:.3f}  std={stats['std_raw']:.3f}")
    print(f"  mapped kW:   mean={stats['mean_kw']:.3f} kW  std={stats['std_kw']:.3f} kW "
          f"(range [0, {PER_EV_MAX_KW}] kW, midpoint {PER_EV_MAX_KW/2:.2f} kW)")
    print(f"  fraction raw <= -0.9 (near lower bound / near 0 kW):  {100*stats['frac_near_lower']:.1f}%")
    print(f"  fraction raw >=  0.9 (near upper bound / near max kW): {100*stats['frac_near_upper']:.1f}%")


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
    print(f"Held-out test weeks (start steps): {test_starts}")
    print(f"Reward weights (unchanged, NOT re-tuned): {ORIGINAL_WEIGHTS}")

    train_env = Monitor(ACNWeeklyEpisodeEnv(
        full_data, slot_roster, train_range, reward_weights=ORIGINAL_WEIGHTS,
        seed=SEED, symmetric_action=True))
    eval_env = ACNWeeklyEpisodeEnv(
        full_data, slot_roster, test_range, reward_weights=ORIGINAL_WEIGHTS,
        seed=999, symmetric_action=True)

    print(f"\naction_space (external, symmetric): {train_env.action_space}")
    assert train_env.action_space.low[0] == -1.0 and train_env.action_space.high[0] == 1.0

    print("\n" + "=" * 60)
    print(f"TRAINING: symmetric-action PPO, {SHORT_TIMESTEPS} timesteps, seed {SEED}")
    print("=" * 60)
    model = PPO("MlpPolicy", train_env, seed=SEED, verbose=0, **PPO_KWARGS)
    model.learn(total_timesteps=SHORT_TIMESTEPS)

    def ppo_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    print("\n" + "=" * 60)
    print("TRAINED POLICY ACTION DISTRIBUTION")
    print("=" * 60)
    trained_stats = collect_action_stats(eval_env, test_starts, ppo_fn)
    print_action_stats("TRAINED symmetric-action PPO (50k)", trained_stats)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST-WEEK PERFORMANCE")
    print("=" * 60)
    results = {
        "random": evaluate_policy_on_weeks(eval_env, test_starts, random_fn),
        "cap_only (recomputed)": evaluate_policy_on_weeks(eval_env, test_starts, cap_only_symmetric_fn),
        "ppo_symmetric (50k)": evaluate_policy_on_weeks(eval_env, test_starts, ppo_fn),
    }

    print(f"\n{'Policy':24s} {'ViolRate':>10s} {'UnmetFrac':>10s} {'Delivered(kWh)':>15s}")
    for pname, r in results.items():
        print(f"{pname:24s} {100 * r['violation_rate']:9.2f}% "
              f"{100 * r['unmet_fraction']:9.2f}% {r['delivered_kwh']:15.1f}")
    print(f"{'cap_only (cited ref.)':24s} {100 * CAP_ONLY_REFERENCE['violation_rate']:9.2f}% "
          f"{100 * CAP_ONLY_REFERENCE['unmet_fraction']:9.2f}% "
          f"{CAP_ONLY_REFERENCE['delivered_kwh']:15.1f}")
    print(f"{'action-floor (cited ref.)':24s} {'n/a':>10s} {'n/a':>10s} "
          f"{ACTION_FLOOR_REFERENCE_KWH:15.1f}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    sym_delivered = results["ppo_symmetric (50k)"]["delivered_kwh"]
    sym_viol = results["ppo_symmetric (50k)"]["violation_rate"]
    beats_floor = sym_delivered > ACTION_FLOOR_REFERENCE_KWH
    reaches_target = sym_delivered >= 1000.0
    low_violation = sym_viol < 0.20  # same ballpark as cap_only/floor references
    print(f"Delivered kWh: {sym_delivered:.1f} "
          f"(floor patch: {ACTION_FLOOR_REFERENCE_KWH:.1f}, cap_only: {CAP_ONLY_REFERENCE['delivered_kwh']:.1f})")
    print(f"Violation rate: {100*sym_viol:.2f}%")
    print(f"Beats action-floor patch: {'YES' if beats_floor else 'NO'}")
    print(f"Reaches >=1000 kWh target: {'YES' if reaches_target else 'NO'}")
    if reaches_target and low_violation:
        print("\n-> RECOMMENDATION: adopt the symmetric [-1,1] action-space rescale as "
              "THE fix for the collapse; retire the action_floor_kw patch.")
    elif beats_floor:
        print("\n-> Rescale outperforms the floor patch but did not clear the 1000 kWh bar - "
              "report honestly, do not oversell.")
    else:
        print("\n-> Rescale underperformed the floor patch in this short run - report honestly, "
              "do not adopt yet.")


if __name__ == "__main__":
    main()
