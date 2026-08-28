# ==========================================
# Instrument the trained policy's ACTIONS to find why it collapses
# ==========================================
# Diagnosis only - no reward tuning, no full training. Two independent
# reward sweeps (unmet, energy/violation) both collapsed to near-zero
# charging regardless of weights, and per-step reward-term
# instrumentation already ruled out reward-content imbalance (energy
# reward beats violation penalty 3:1 during surges). This checks
# whether it's a training-dynamics / action-distribution problem
# instead: (1) trained vs untrained action distributions, (2) whether
# an action floor prevents the collapse, (3) whether the Gaussian
# policy's pre-clip mean/std saturate/collapse over training, and
# (4) whether a simple non-learned heuristic proves the env/reward
# already reward a decent policy well.
# ==========================================

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback

from acn_data_loader import load_acn_sessions
from train_controller import (
    DATA_PATH, EPISODE_LEN, ACNWeeklyEpisodeEnv, cap_only_action,
    evaluate_policy_on_weeks, split_boundaries,
)
from sweep_reward_weights import PPO_KWARGS

ORIGINAL_WEIGHTS = {"energy": 1.0, "unmet": 1.0, "violation": 1.0}
SHORT_TIMESTEPS = 50_000
SEED = 1
N_TEST_WEEKS = 5
NEAR_ZERO_KW = 0.33   # 5% of per_ev_max_kw=6.6
NEAR_MAX_FRAC = 0.95  # >= 95% of per_ev_max_kw


# ==========================================
# Part 4: sanity-reference heuristic - fixed moderate rate per present
# EV, clipped to the transformer limit (same clipping structure as
# cap_only_action, but the PRE-clip target is a flat rate, not each
# EV's own remaining-demand-based max).
# ==========================================

def fixed_rate_action(env, rate_kw: float = 2.0) -> np.ndarray:
    present = env.true_presence()
    request = np.where(present, rate_kw, 0.0).astype(np.float64)
    total = request.sum()
    if total > env.transformer_limit_kw and total > 0:
        request = request * ((env.transformer_limit_kw - 1e-4) / total)
    return request.astype(np.float32)


# ==========================================
# Part 1: raw action distribution over present slots only (absent-slot
# actions are zeroed by the env regardless of what the policy outputs,
# so they're not informative about whether the policy has learned to
# withhold charging FROM EVs THAT NEED IT).
# ==========================================

def collect_action_stats(env: ACNWeeklyEpisodeEnv, test_starts, action_fn) -> dict:
    present_actions = []
    for start_step in test_starts:
        obs, _ = env.reset(start_step=start_step)
        for _ in range(env.episode_len):
            action = action_fn(env, obs)
            present = env.inner.true_presence()
            present_actions.extend(np.asarray(action)[present].tolist())
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    arr = np.array(present_actions)
    return {
        "n": len(arr),
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std()) if len(arr) else float("nan"),
        "p10": float(np.percentile(arr, 10)) if len(arr) else float("nan"),
        "p50": float(np.percentile(arr, 50)) if len(arr) else float("nan"),
        "p90": float(np.percentile(arr, 90)) if len(arr) else float("nan"),
        "frac_near_zero": float(np.mean(arr <= NEAR_ZERO_KW)) if len(arr) else float("nan"),
        "frac_near_max": float(np.mean(arr >= NEAR_MAX_FRAC * 6.6)) if len(arr) else float("nan"),
    }


def print_action_stats(label, stats):
    print(f"\n--- {label} (n={stats['n']} present-slot actions) ---")
    print(f"  mean={stats['mean']:.3f} kW  std={stats['std']:.3f} kW")
    print(f"  p10={stats['p10']:.3f}  p50={stats['p50']:.3f}  p90={stats['p90']:.3f} kW")
    print(f"  fraction <= {NEAR_ZERO_KW} kW (near lower bound): {100*stats['frac_near_zero']:.1f}%")
    print(f"  fraction >= {NEAR_MAX_FRAC*6.6:.2f} kW (near upper bound): {100*stats['frac_near_max']:.1f}%")


# ==========================================
# Part 3: policy saturation callback - tracks the Gaussian policy's
# PRE-CLIP mean/std on a fixed probe batch of real observations, at
# fixed intervals during training. SB3 PPO's continuous-action policy
# uses an unbounded DiagGaussianDistribution (not tanh-squashed - that's
# SAC); actions are clipped to the Box AFTER sampling. This reads the
# distribution directly, before that clip.
# ==========================================

class PolicySaturationCallback(BaseCallback):
    def __init__(self, probe_obs: np.ndarray, check_freq: int = 5000):
        super().__init__()
        self.probe_obs = probe_obs
        self.check_freq = check_freq
        self.history: list[dict] = []
        self._last_check = -1

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check >= self.check_freq:
            self._last_check = self.num_timesteps
            obs_tensor = obs_as_tensor(self.probe_obs[None, :], self.model.policy.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
            mean = dist.distribution.mean.cpu().numpy().reshape(-1)
            std = dist.distribution.stddev.cpu().numpy().reshape(-1)
            self.history.append({
                "timesteps": self.num_timesteps,
                "mean_of_means": float(mean.mean()),
                "mean_of_stds": float(std.mean()),
                "frac_mean_leq_0": float(np.mean(mean <= 0.0)),
            })
        return True


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
                                   reward_weights=ORIGINAL_WEIGHTS, seed=999)

    # ============================================================
    # Part 4: sanity reference heuristics
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 4 - sanity reference: does a simple non-learned policy")
    print("deliver well in this env/reward?")
    print("=" * 60)
    cap_result = evaluate_policy_on_weeks(
        eval_env, test_starts, lambda env, obs: cap_only_action(env.inner))
    fixed_result = evaluate_policy_on_weeks(
        eval_env, test_starts, lambda env, obs: fixed_rate_action(env.inner, rate_kw=2.0))
    print(f"cap_only          : viol={100*cap_result['violation_rate']:.2f}%  "
          f"unmet={100*cap_result['unmet_fraction']:.2f}%  "
          f"delivered={cap_result['delivered_kwh']:.1f} kWh")
    print(f"fixed_rate(2.0 kW): viol={100*fixed_result['violation_rate']:.2f}%  "
          f"unmet={100*fixed_result['unmet_fraction']:.2f}%  "
          f"delivered={fixed_result['delivered_kwh']:.1f} kWh")
    both_good = (cap_result["delivered_kwh"] > 1900 and cap_result["violation_rate"] == 0.0 and
                fixed_result["delivered_kwh"] > 1900 and fixed_result["violation_rate"] == 0.0)
    print(f"\nBoth simple heuristics deliver ~2000 kWh at 0% violations: "
          f"{'YES - env/reward reward a decent policy correctly' if both_good else 'NO - see numbers above'}")

    # ============================================================
    # Part 1 (untrained) + Part 3 setup: build a fresh, never-trained
    # PPO model on the training env for the probe/comparison.
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1 - action distribution: UNTRAINED policy")
    print("=" * 60)
    train_env_raw = ACNWeeklyEpisodeEnv(full_data, slot_roster, train_range,
                                        reward_weights=ORIGINAL_WEIGHTS, seed=SEED)
    train_env = Monitor(train_env_raw)
    model = PPO("MlpPolicy", train_env, seed=SEED, verbose=0, **PPO_KWARGS)

    def untrained_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    untrained_stats = collect_action_stats(eval_env, test_starts, untrained_fn)
    print_action_stats("UNTRAINED policy", untrained_stats)

    # probe observation for the saturation callback: a real reset obs
    # from a busy-ish training week
    probe_obs, _ = train_env_raw.reset(start_step=train_range[0])

    # ============================================================
    # Part 3: train while tracking policy mean/std on the probe obs
    # ============================================================
    print("\n" + "=" * 60)
    print(f"PART 3 - training {SHORT_TIMESTEPS} timesteps at original weights, "
          "tracking policy saturation...")
    print("=" * 60)
    sat_cb = PolicySaturationCallback(probe_obs, check_freq=2500)
    model.learn(total_timesteps=SHORT_TIMESTEPS, callback=sat_cb)

    print(f"\n{'timesteps':>10s} {'mean_of_means':>14s} {'mean_of_stds':>13s} {'frac_mean<=0':>13s}")
    for h in sat_cb.history:
        print(f"{h['timesteps']:10d} {h['mean_of_means']:14.4f} {h['mean_of_stds']:13.4f} "
              f"{100*h['frac_mean_leq_0']:12.1f}%")

    # ============================================================
    # Part 1 (trained)
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1 - action distribution: TRAINED (50k) policy")
    print("=" * 60)

    def trained_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    trained_stats = collect_action_stats(eval_env, test_starts, trained_fn)
    print_action_stats("TRAINED (50k) policy", trained_stats)
    trained_perf = evaluate_policy_on_weeks(eval_env, test_starts, trained_fn)
    print(f"\nTrained policy performance: viol={100*trained_perf['violation_rate']:.2f}%  "
          f"unmet={100*trained_perf['unmet_fraction']:.2f}%  "
          f"delivered={trained_perf['delivered_kwh']:.1f} kWh")

    # ============================================================
    # Part 2: action floor test
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2 - action-space floor test (1.0 kW minimum per present EV)")
    print("=" * 60)
    FLOOR_KW = 1.0
    train_env_floor = Monitor(ACNWeeklyEpisodeEnv(
        full_data, slot_roster, train_range, reward_weights=ORIGINAL_WEIGHTS,
        seed=SEED, action_floor_kw=FLOOR_KW))
    model_floor = PPO("MlpPolicy", train_env_floor, seed=SEED, verbose=0, **PPO_KWARGS)
    model_floor.learn(total_timesteps=SHORT_TIMESTEPS)

    eval_env_floor = ACNWeeklyEpisodeEnv(full_data, slot_roster, test_range,
                                        reward_weights=ORIGINAL_WEIGHTS, seed=999,
                                        action_floor_kw=FLOOR_KW)

    def floor_fn(env, obs):
        action, _ = model_floor.predict(obs, deterministic=True)
        return action

    floor_perf = evaluate_policy_on_weeks(eval_env_floor, test_starts, floor_fn)
    print(f"WITHOUT floor (from Part 1/3): viol={100*trained_perf['violation_rate']:.2f}%  "
          f"unmet={100*trained_perf['unmet_fraction']:.2f}%  "
          f"delivered={trained_perf['delivered_kwh']:.1f} kWh")
    print(f"WITH {FLOOR_KW} kW floor:          viol={100*floor_perf['violation_rate']:.2f}%  "
          f"unmet={100*floor_perf['unmet_fraction']:.2f}%  "
          f"delivered={floor_perf['delivered_kwh']:.1f} kWh")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Part 4 (sanity): simple heuristics deliver well? "
          f"{'YES' if both_good else 'NO'} -> env/reward are "
          f"{'NOT' if both_good else ''} the problem")
    collapse = (trained_stats["frac_near_zero"] > untrained_stats["frac_near_zero"] + 0.1)
    print(f"Part 1 (action dist): trained policy collapsed toward 0 vs untrained? "
          f"{'YES' if collapse else 'NO'} "
          f"(near-zero frac: untrained={100*untrained_stats['frac_near_zero']:.1f}% -> "
          f"trained={100*trained_stats['frac_near_zero']:.1f}%)")
    std_trend = sat_cb.history[-1]["mean_of_stds"] - sat_cb.history[0]["mean_of_stds"] if len(sat_cb.history) > 1 else float("nan")
    print(f"Part 3 (saturation): policy std change over training: {std_trend:+.4f} "
          f"({'COLLAPSING (exploration dying)' if std_trend < -0.1 else 'not clearly collapsing'})")
    floor_helped = floor_perf["delivered_kwh"] > trained_perf["delivered_kwh"] * 2
    print(f"Part 2 (floor): floor meaningfully increased delivered kWh? "
          f"{'YES' if floor_helped else 'NO'} "
          f"({trained_perf['delivered_kwh']:.1f} -> {floor_perf['delivered_kwh']:.1f} kWh)")


if __name__ == "__main__":
    main()
