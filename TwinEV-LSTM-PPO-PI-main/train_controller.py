# ==========================================
# Track B - safe-RL charging controller on ACNChargingEnv
# ==========================================
# PPO via the existing stable-baselines3 stack (no new safe-RL
# library - safety is via the reward's violation-penalty term, which
# is ACNChargingEnv's intended mechanism). Two observation variants
# (no_forecast / forecast) and a non-learning cap-only baseline.
#
# THIS FILE ONLY BUILDS THE SETUP AND RUNS A SHORT VALIDATION TRAIN -
# see __main__ at the bottom. Full multi-seed training is a separate,
# later task.
# ==========================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from acn_data_loader import ACNSessionData, load_acn_sessions, slice_session_data
from acn_env import ACNChargingEnv

DATA_PATH = "acn_caltech_2019.csv"
EPISODE_LEN = 168             # one week, hourly
TRANSFORMER_LIMIT_KW = 50.0    # matches acn_env.py's default (capacity-constrained scenario)
PER_EV_MAX_KW = 6.6

# ------------------------------------------------------------------
# CHRONOLOGICAL SPLIT - identical boundaries used throughout this
# project's forecasting work (build_demand_forecaster.py,
# test_alternative_targets.py): train = first 70%, val/buffer = next
# 15% (unused here - kept as a gap so no training episode can straddle
# into the test region), test = last 15%.
# ------------------------------------------------------------------


def split_boundaries(n_steps: int) -> tuple[int, int]:
    train_end = int(n_steps * 0.70)
    val_end = int(n_steps * 0.85)
    return train_end, val_end


# ==========================================
# Forecast feature (Target A: 6h-ahead demand, real BiLSTM predictions)
# ==========================================

class ForecastLookup:
    """Looks up the REAL (already-trained, already-wrong-where-it's-
    wrong) BiLSTM 6h-ahead-demand prediction for a given wall-clock
    hour, from generate_forecast_feature.py's saved output. Never the
    true future value - the whole point is the controller sees a
    forecast WITH its real errors, exactly what a deployed system
    would have."""

    def __init__(self, csv_path: str = "acn_target_a_bilstm_full.csv"):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # NOTE: index must be a tz-aware DatetimeIndex, not
        # df["timestamp"].values (.values on a tz-aware column drops
        # the tz, becoming naive datetime64[ns] - exact-match lookup
        # against a tz-aware Timestamp key then silently fails).
        self._by_time = pd.Series(
            df["predicted"].values, index=pd.DatetimeIndex(df["timestamp"])
        )
        self.first_valid_time = pd.Timestamp(df["timestamp"].min())
        self.last_valid_time = pd.Timestamp(df["timestamp"].max())
        # normalization constant for the extra observation feature -
        # p99-ish of the predicted range, consistent in spirit with
        # ACNChargingEnv's own max_demand_norm_kwh clipping approach
        self.norm_scale = float(np.percentile(df["predicted"].clip(lower=0), 99)) or 1.0

    def __call__(self, timestamp: pd.Timestamp) -> float:
        val = self._by_time.get(timestamp)
        if val is None:
            raise KeyError(
                f"no BiLSTM forecast available for {timestamp} - it falls "
                f"outside [{self.first_valid_time}, {self.last_valid_time}]; "
                "restrict episode sampling to this range for the forecast variant"
            )
        return float(np.clip(val / self.norm_scale, -1.0, 3.0))


# ==========================================
# Weekly-episode wrapper: fixed obs/action space across random
# one-week windows drawn from a restricted (train- or test-only)
# region of the full year, with an optional forecast feature appended.
# ==========================================

class ACNWeeklyEpisodeEnv(gym.Env):
    """Gymnasium env that, on each reset(), samples a fresh one-week
    (or `episode_len`-hour) window from `allowed_start_range` and
    builds a new ACNChargingEnv for it. `slot_roster` is the FULL
    dataset's slot list (not just the window's own sessions) so
    n_slots - and thus observation_space/action_space - never changes
    between episodes, which SB3 requires.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        full_data: ACNSessionData,
        slot_roster: list[str],
        allowed_start_range: tuple[int, int],
        transformer_limit_kw: float = TRANSFORMER_LIMIT_KW,
        per_ev_max_kw: float = PER_EV_MAX_KW,
        reward_weights: Optional[dict] = None,
        episode_len: int = EPISODE_LEN,
        forecast: Optional[ForecastLookup] = None,
        seed: Optional[int] = None,
        violation_penalty_kind: str = "linear",
        violation_softness_kw: float = 25.0,
        action_floor_kw: float = 0.0,
    ) -> None:
        super().__init__()
        lo, hi = allowed_start_range
        if hi <= lo:
            raise ValueError(f"empty allowed_start_range {allowed_start_range}")
        if not (0.0 <= action_floor_kw < per_ev_max_kw):
            raise ValueError("action_floor_kw must be in [0, per_ev_max_kw)")
        self.full_data = full_data
        self.slot_roster = slot_roster
        self.allowed_start_range = (lo, hi)
        self.transformer_limit_kw = transformer_limit_kw
        self.per_ev_max_kw = per_ev_max_kw
        self.reward_weights = reward_weights
        self.episode_len = episode_len
        self.forecast = forecast
        self.violation_penalty_kind = violation_penalty_kind
        self.violation_softness_kw = violation_softness_kw
        self.action_floor_kw = action_floor_kw
        self._rng = np.random.default_rng(seed)

        n_slots = len(slot_roster)
        base_dim = n_slots * 3 + 4
        obs_dim = base_dim + (1 if forecast is not None else 0)
        self.observation_space = spaces.Box(low=-1.0, high=3.0, shape=(obs_dim,), dtype=np.float32)
        # action_floor_kw > 0 forces a minimum non-zero request for
        # EVERY slot (matching the original synthetic env's [0.2, 1.0]-
        # scaled bounds, which had an explicit "prevents zero-charging
        # hack" comment) - SB3 clips sampled actions to this Box AFTER
        # sampling from the policy's Gaussian, so this floor is enforced
        # regardless of what the policy's raw output is. The inner env
        # zeros this out for slots with no EV present regardless, so the
        # floor only ever affects slots that actually have a connected EV.
        self.action_space = spaces.Box(
            low=action_floor_kw, high=per_ev_max_kw, shape=(n_slots,), dtype=np.float32,
        )

        self.inner: Optional[ACNChargingEnv] = None

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        if self.forecast is None:
            return obs
        f = self.forecast(self.inner.current_time())
        return np.concatenate([obs, np.array([f], dtype=np.float32)])

    def reset(self, *, seed=None, options=None, start_step: Optional[int] = None):
        super().reset(seed=seed)
        lo, hi = self.allowed_start_range
        if start_step is None:
            start_step = int(self._rng.integers(lo, hi))
        if not (lo <= start_step < hi):
            raise ValueError(f"start_step {start_step} outside allowed range [{lo},{hi})")

        window = slice_session_data(
            self.full_data, start_step, self.episode_len, slot_ids=self.slot_roster,
        )
        self.inner = ACNChargingEnv(
            window,
            transformer_limit_kw=self.transformer_limit_kw,
            per_ev_max_kw=self.per_ev_max_kw,
            reward_weights=self.reward_weights,
            episode_len=self.episode_len,
            violation_penalty_kind=self.violation_penalty_kind,
            violation_softness_kw=self.violation_softness_kw,
        )
        obs, info = self.inner.reset()
        info = {**info, "start_step": start_step}
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        return self._augment(obs), reward, terminated, truncated, info


# ==========================================
# Cap-only baseline ("fake safety" reference): greedily charge every
# connected EV up to its true remaining demand and per-EV rate limit,
# then hard-clip the TOTAL to the transformer limit by proportional
# scale-down. Uses the env's true (un-normalized) state, not the
# lossy observation - a real site controller would have real metering.
# ==========================================

def cap_only_action(env: ACNChargingEnv) -> np.ndarray:
    present = env.true_presence()
    remaining_kwh = env.true_remaining_kwh()
    max_deliverable_kw = np.where(
        present, np.minimum(env.per_ev_max_kw, remaining_kwh / env.dt_hours), 0.0
    )
    total = max_deliverable_kw.sum()
    # Small safety margin: re-summing float32 action values inside
    # env.step() (54 slots, arbitrary float32 action dtype after SB3/
    # gym round-trips) reliably reintroduces ~1e-6 kW of rounding noise
    # relative to this float64 computation. Scaling to EXACTLY the
    # limit left this baseline registering "violations" of ~1e-6 kW -
    # physically meaningless, but this baseline is specifically the
    # "hard, structural cap" reference and should have zero tolerance
    # for even a phantom float artifact.
    target = env.transformer_limit_kw - 1e-4
    if total > target and total > 0:
        max_deliverable_kw = max_deliverable_kw * (target / total)
    return max_deliverable_kw.astype(np.float32)


# ==========================================
# Episode reward tracking callback (for "does reward increase?")
# ==========================================

class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
                self.episode_timesteps.append(self.num_timesteps)
        return True


# ==========================================
# Rollout evaluation (no training) - used for random / cap-only / PPO
# ==========================================

def evaluate_policy_on_weeks(
    env: ACNWeeklyEpisodeEnv, start_steps: list[int], action_fn,
) -> dict:
    """`action_fn(env, obs)` -> action. Reports the same diagnostics
    as smoke_test_env.py, averaged over the given start weeks."""
    violation_rates, unmet_fractions, delivered_kwhs = [], [], []
    for start_step in start_steps:
        obs, _ = env.reset(start_step=start_step)
        total_requested = float(sum(s.requested_kwh for s in env.inner.data.sessions))
        total_delivered = 0.0
        total_unmet = 0.0
        n_violating = 0
        for _ in range(env.episode_len):
            action = action_fn(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_delivered += float(info["delivered_kwh"].sum())
            total_unmet += info["unmet_kwh_this_step"]
            if info["violation_kw"] > 0:
                n_violating += 1
            if terminated or truncated:
                break
        violation_rates.append(n_violating / env.episode_len)
        unmet_fractions.append(total_unmet / total_requested if total_requested > 0 else 0.0)
        delivered_kwhs.append(total_delivered)

    return {
        "violation_rate": float(np.mean(violation_rates)),
        "unmet_fraction": float(np.mean(unmet_fractions)),
        "delivered_kwh": float(np.mean(delivered_kwhs)),
    }


# ==========================================
# Short validation run (NOT full training - see task scope)
# ==========================================

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--violation-weight", type=float, default=1.0,
                        help="reward weight on the per-kW transformer violation penalty "
                             "(parameterized for a later sweep; default 1.0)")
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="SHORT validation budget - NOT full training")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-test-weeks", type=int, default=5)
    args = parser.parse_args()

    print("Loading full-year ACN data...")
    full_data = load_acn_sessions(DATA_PATH, timestep_minutes=60)
    slot_roster = full_data.slot_ids
    n_steps = full_data.n_steps
    train_end, val_end = split_boundaries(n_steps)
    print(f"Chronological split (N={n_steps}): train=[0,{train_end}) "
          f"val/buffer=[{train_end},{val_end}) (unused) test=[{val_end},{n_steps})")
    print(f"  train dates: {full_data.start_time} to "
          f"{full_data.start_time + pd.Timedelta(hours=train_end)}")
    print(f"  test  dates: {full_data.start_time + pd.Timedelta(hours=val_end)} to "
          f"{full_data.start_time + pd.Timedelta(hours=n_steps)}")

    train_range = (0, train_end - EPISODE_LEN)
    test_range = (val_end, n_steps - EPISODE_LEN)

    reward_weights = {"energy": args.energy_weight, "violation": args.violation_weight}
    print(f"\nreward_weights = {reward_weights} (unmet stays at ACNChargingEnv's default)")

    # ---- Build both variants ----
    print("\nBuilding no_forecast env...")
    probe_env = ACNWeeklyEpisodeEnv(full_data, slot_roster, train_range,
                                     reward_weights=reward_weights, seed=args.seed)
    obs, info = probe_env.reset()
    print(f"  no_forecast observation shape: {obs.shape}, "
          f"action shape: {probe_env.action_space.shape}")

    print("Building forecast env (build/sanity-check only - NOT trained this pass)...")
    forecast_lookup = ForecastLookup()
    step_delta = pd.Timedelta(hours=1)
    fc_first_idx = int(round((forecast_lookup.first_valid_time - full_data.start_time) / step_delta))
    fc_last_idx = int(round((forecast_lookup.last_valid_time - full_data.start_time) / step_delta))
    # an episode starting at `start_step` touches hours
    # [start_step, start_step + EPISODE_LEN - 1], so the latest safe
    # start is fc_last_idx - EPISODE_LEN + 1, not fc_last_idx itself.
    fc_test_range = (
        max(test_range[0], fc_first_idx),
        min(test_range[1], fc_last_idx - EPISODE_LEN + 1),
    )
    fc_env = ACNWeeklyEpisodeEnv(full_data, slot_roster, fc_test_range,
                                 reward_weights=reward_weights, forecast=forecast_lookup,
                                 seed=args.seed)
    fc_obs, fc_info = fc_env.reset()
    fc_action = fc_env.action_space.sample()
    fc_obs2, fc_r, fc_term, fc_trunc, fc_info2 = fc_env.step(fc_action)
    print(f"  forecast observation shape: {fc_obs.shape} (no_forecast + 1 feature), "
          f"step() OK, forecast feature value at reset: {fc_obs[-1]:.4f}")
    print(f"  forecast-safe start range (real BiLSTM predictions only): {fc_test_range}")

    # ---- Cap-only baseline: quick correctness check ----
    cap_action = cap_only_action(probe_env.inner)
    print(f"\ncap_only sanity check at reset: total requested "
          f"{cap_action.sum():.2f} kW (limit {probe_env.transformer_limit_kw} kW)")
    assert cap_action.sum() <= probe_env.transformer_limit_kw + 1e-6, \
        "cap_only_action exceeded the transformer limit - it should never be able to"

    # ---- Short validation training: no_forecast PPO only ----
    ppo_kwargs = dict(
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )
    print(f"\n===== SHORT VALIDATION TRAIN: no_forecast PPO, "
          f"{args.timesteps} timesteps, seed {args.seed} (NOT full training) =====")
    train_env = Monitor(ACNWeeklyEpisodeEnv(
        full_data, slot_roster, train_range, reward_weights=reward_weights, seed=args.seed))
    model = PPO("MlpPolicy", train_env, seed=args.seed, verbose=0, **ppo_kwargs)
    cb = EpisodeRewardCallback()
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    dt = time.time() - t0
    print(f"Training done in {dt / 60:.1f} min, {len(cb.episode_rewards)} episodes")

    n_check = max(1, len(cb.episode_rewards) // 5)
    early = float(np.mean(cb.episode_rewards[:n_check])) if cb.episode_rewards else float("nan")
    late = float(np.mean(cb.episode_rewards[-n_check:])) if cb.episode_rewards else float("nan")
    reward_increased = late > early
    print(f"Episode reward: first {n_check} eps mean={early:.2f}  "
          f"last {n_check} eps mean={late:.2f}  "
          f"({'INCREASED' if reward_increased else 'DID NOT INCREASE'})")

    # ---- Evaluate on held-out TEST weeks: random / cap-only / short-trained PPO ----
    lo, hi = test_range
    test_starts = sorted(set(np.linspace(lo, hi - 1, args.n_test_weeks).astype(int).tolist()))
    print(f"\nEvaluating on {len(test_starts)} held-out test weeks "
          f"(start steps {test_starts}, all >= {val_end} i.e. inside the test region)")

    eval_env = ACNWeeklyEpisodeEnv(full_data, slot_roster, test_range,
                                   reward_weights=reward_weights, seed=999)

    def random_action_fn(env, obs):
        return env.action_space.sample()

    def cap_action_fn(env, obs):
        return cap_only_action(env.inner)

    def ppo_action_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    results = {}
    for pname, fn in [("random", random_action_fn), ("cap_only", cap_action_fn),
                      ("ppo_short (50k)", ppo_action_fn)]:
        results[pname] = evaluate_policy_on_weeks(eval_env, test_starts, fn)

    print("\n===== TEST-WEEK COMPARISON (mean over held-out weeks) =====")
    print(f"{'Policy':16s} {'ViolRate':>10s} {'UnmetFrac':>10s} {'Delivered(kWh)':>15s}")
    for pname, r in results.items():
        print(f"{pname:16s} {100 * r['violation_rate']:9.2f}% "
              f"{100 * r['unmet_fraction']:9.2f}% {r['delivered_kwh']:15.1f}")

    print("\n===== VERDICT =====")
    viol_down = results["ppo_short (50k)"]["violation_rate"] < results["random"]["violation_rate"]
    print(f"Reward increased over training: {'YES' if reward_increased else 'NO'} "
          f"(first {n_check} eps {early:.2f} -> last {n_check} eps {late:.2f})")
    print(f"Violation rate went DOWN vs random policy: {'YES' if viol_down else 'NO'} "
          f"(random={100 * results['random']['violation_rate']:.2f}%, "
          f"ppo_short={100 * results['ppo_short (50k)']['violation_rate']:.2f}%)")
    print("\nThis was a SHORT validation run only - full multi-seed training is a separate task.")
