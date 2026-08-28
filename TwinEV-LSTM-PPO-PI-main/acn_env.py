# ==========================================
# ACN-Data demand-coupled EV charging environment
# ==========================================
# Replaces the flawed synthetic TwinEV environment for this use case.
# In twinev_env.py, delivered power was a function of the agent's
# action and the PI controller's OWN PREVIOUS OUTPUT - real demand
# never entered the loop, and the total was hard-clipped to a 150 kW
# cap, so "zero overloads" was trivially guaranteed regardless of
# policy and charging decisions had no real consequence.
#
# Here, every kW delivered must come from and reduce a real EV's
# remaining energy demand (drawn from ACN-Data session records), and
# the transformer limit is a measured signal, never an output clip -
# an agent that over-allocates WILL violate it. See the __main__
# self-test for a direct, assertion-based check of both properties.
# ==========================================

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from acn_data_loader import ACNSessionData, load_acn_sessions, make_synthetic_fixture

DEFAULT_REWARD_WEIGHTS = {
    "energy": 1.0,      # reward per kWh actually delivered
    "unmet": 1.0,        # penalty per kWh still unmet when an EV departs
    "violation": 1.0,    # penalty per kW drawn above the transformer limit
}


class ACNChargingEnv(gym.Env):
    """Gymnasium environment for EV charging control over cached
    ACN-Data sessions.

    Observation/action are indexed by a fixed set of physical charging
    slots (`data.slot_ids`, one entry per unique `spaceID` in the
    dataset by default) - each slot keeps the same index for the whole
    episode, whether or not an EV currently occupies it. This mirrors
    a real charging network, where a bay's identity and wiring are
    fixed even as different vehicles cycle through it.

    Observation (shape = 3 * n_slots + 4):
        [presence_0..presence_{n-1},                       in [0, 1]
         remaining_demand_norm_0..remaining_demand_norm_{n-1},  in [0, 1]
         time_remaining_norm_0..time_remaining_norm_{n-1},      in [0, 1]
         sin(hour_of_day), cos(hour_of_day),                in [-1, 1]
         sin(day_of_week), cos(day_of_week)]                in [-1, 1]
    A slot with no EV connected has presence=0 and the other two
    per-slot fields pinned to 0 (they are meaningless without
    presence, so the policy must gate on the presence bit rather than
    reading demand/time directly).

    The 4 trailing calendar features are cyclical encodings of the
    real wall-clock time at the current step (`data.start_time` plus
    `step_idx` timesteps - never just the raw step index, since
    `start_time` is not midnight/week-aligned in real ACN-Data). They
    give the policy access to daily/weekly arrival patterns even
    though the per-slot state alone carries no calendar signal, and
    are always present in the base observation (not behind a flag) so
    every training condition shares this same fair baseline; a
    forecast channel will be added separately later.

    Action (shape = n_slots, kW in [0, per_ev_max_kw]):
        requested charging power per slot. Ignored (treated as 0) for
        slots with no EV present.

    Default transformer_limit_kw=50.0 models a capacity-constrained
    site, not the real Caltech site's actual transformer rating. On
    the real 2019 Caltech cache, a realistic "charge every connected
    EV as fast as its remaining demand allows" (greedy) policy peaks
    at only 137.6 kW aggregate draw across the full year - so a 150 kW
    limit (a plausible real rating) is NEVER binding (0.00% of hours).
    50 kW is the smallest reference limit that keeps the safety
    problem live: greedy achievable draw exceeds it in ~10% of hours,
    making the constraint real but not hopeless, rather than either
    trivially satisfied or trivially impossible.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: ACNSessionData,
        transformer_limit_kw: float = 50.0,
        per_ev_max_kw: float = 6.6,
        reward_weights: Optional[dict] = None,
        max_demand_norm_kwh: float = 80.0,
        max_deadline_norm_steps: float = 48.0,
        episode_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        if transformer_limit_kw <= 0:
            raise ValueError("transformer_limit_kw must be positive")
        if per_ev_max_kw <= 0:
            raise ValueError("per_ev_max_kw must be positive")
        if max_demand_norm_kwh <= 0 or max_deadline_norm_steps <= 0:
            raise ValueError("normalization constants must be positive")
        if not data.slot_ids:
            raise ValueError("data has no charging slots (empty slot_ids)")
        if episode_len is not None and episode_len <= 0:
            raise ValueError("episode_len must be positive")

        self.data = data
        self.n_slots = len(data.slot_ids)
        self.slot_index = {sid: i for i, sid in enumerate(data.slot_ids)}
        self.dt_hours = data.timestep_minutes / 60.0
        self.transformer_limit_kw = float(transformer_limit_kw)
        self.per_ev_max_kw = float(per_ev_max_kw)
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        unknown_weights = set(self.reward_weights) - set(DEFAULT_REWARD_WEIGHTS)
        if unknown_weights:
            raise ValueError(f"unknown reward_weights key(s): {unknown_weights}")
        self.max_demand_norm_kwh = float(max_demand_norm_kwh)
        self.max_deadline_norm_steps = float(max_deadline_norm_steps)
        self.episode_len = int(episode_len or data.n_steps)

        self.action_space = spaces.Box(
            low=0.0, high=self.per_ev_max_kw,
            shape=(self.n_slots,), dtype=np.float32,
        )
        # 3 per-slot fields + 4 cyclical calendar features (sin/cos of
        # hour-of-day, sin/cos of day-of-week). The calendar features
        # are in [-1, 1]; the per-slot fields are in [0, 1].
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_slots * 3 + 4,), dtype=np.float32,
        )
        self._timestep_delta = pd.Timedelta(minutes=data.timestep_minutes)

        self._step_idx = 0
        self._remaining_kwh = np.zeros(self.n_slots, dtype=np.float64)
        self._departure_step = np.full(self.n_slots, -1, dtype=np.int64)
        self._present = np.zeros(self.n_slots, dtype=bool)

        # last-step diagnostics (also returned via info)
        self.last_delivered_kwh = np.zeros(self.n_slots, dtype=np.float64)
        self.last_violation_kw = 0.0

        # kWh from sessions dropped by _admit_arrivals() due to a same-step
        # slot collision (real, dense ACN-Data turnover: two sessions at
        # the same spaceID round to the same hourly grid step). This is
        # detected during reset()'s initial admission, before step() has
        # any reward/info channel to report it through - stashed here and
        # folded into the first step()'s unmet accounting so it's never
        # silently lost (see _admit_arrivals()).
        self._pending_collision_unmet_kwh = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        self._remaining_kwh[:] = 0.0
        self._departure_step[:] = -1
        self._present[:] = False
        self.last_delivered_kwh[:] = 0.0
        self.last_violation_kw = 0.0
        self._pending_collision_unmet_kwh = self._admit_arrivals(self._step_idx)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.n_slots,):
            raise ValueError(
                f"action shape {action.shape} does not match "
                f"({self.n_slots},)"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN/inf")

        # Per-EV physical rate limit: a real hard constraint, legitimate
        # to clip (a cable/EVSE cannot exceed its rated current).
        requested = np.clip(action, 0.0, self.per_ev_max_kw)
        requested = np.where(self._present, requested, 0.0)

        # Energy conservation: an EV cannot absorb more than its
        # remaining demand. This is what actually couples delivery to
        # real demand - the flaw being fixed.
        max_deliverable_kw = np.where(
            self._present, self._remaining_kwh / self.dt_hours, 0.0
        )
        delivered_kw = np.minimum(requested, max_deliverable_kw)

        # Transformer limit: measured, NEVER used to scale delivered_kw
        # down. A policy that over-allocates across slots really does
        # overload the transformer here.
        total_kw = float(delivered_kw.sum())
        violation_kw = max(0.0, total_kw - self.transformer_limit_kw)

        delivered_kwh = delivered_kw * self.dt_hours
        self._remaining_kwh = np.maximum(self._remaining_kwh - delivered_kwh, 0.0)
        self.last_delivered_kwh = delivered_kwh
        self.last_violation_kw = violation_kw

        reward = self.reward_weights["energy"] * float(delivered_kwh.sum())
        reward -= self.reward_weights["violation"] * violation_kw

        departing = self._present & (self._departure_step == self._step_idx)
        unmet_kwh = float(self._remaining_kwh[departing].sum())
        reward -= self.reward_weights["unmet"] * unmet_kwh

        # Fold in any collision-dropped kWh discovered when this step's
        # (or reset()'s, for step 0) arrivals were admitted - see
        # _admit_arrivals(). Without this, that demand would count toward
        # total requested kWh but never toward delivered or unmet,
        # silently breaking the requested == delivered + unmet invariant.
        unmet_kwh += self._pending_collision_unmet_kwh
        reward -= self.reward_weights["unmet"] * self._pending_collision_unmet_kwh
        self._pending_collision_unmet_kwh = 0.0

        self._present[departing] = False
        self._remaining_kwh[departing] = 0.0
        self._departure_step[departing] = -1

        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_len

        if truncated:
            # Episode horizon reached with EVs still connected: this is
            # a truncation, not a real departure, but leaving their
            # demand unsettled would silently drop it from the reward
            # accounting. Close it out as unmet so total energy
            # requested == total delivered + total unmet, exactly,
            # over any full episode.
            trailing_unmet = float(self._remaining_kwh[self._present].sum())
            reward -= self.reward_weights["unmet"] * trailing_unmet
            unmet_kwh += trailing_unmet
            self._present[:] = False
        else:
            self._pending_collision_unmet_kwh = self._admit_arrivals(self._step_idx)

        info = {
            "delivered_kwh": delivered_kwh.copy(),
            "total_delivered_kw": total_kw,
            "violation_kw": violation_kw,
            "unmet_kwh_this_step": unmet_kwh,
            "n_active": int(self._present.sum()),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _admit_arrivals(self, step_idx: int) -> float:
        """Admit sessions arriving at step_idx. Returns the kWh of any
        session that could NOT be admitted because its physical slot was
        still occupied by a still-connected EV (a same-step collision -
        real, dense ACN-Data turnover can round two adjacent sessions at
        the same spaceID onto the same hourly grid step). That EV keeps
        whichever session arrived first; the second genuinely has nowhere
        to charge in this simulation, so its full demand counts as
        unmet - the caller MUST fold the returned value into that step's
        unmet-energy accounting, or total requested kWh silently stops
        equalling delivered + unmet kWh over the episode.
        """
        collision_kwh = 0.0
        if step_idx >= len(self.data.arrivals_by_step):
            return collision_kwh
        for session_idx in self.data.arrivals_by_step[step_idx]:
            session = self.data.sessions[session_idx]
            slot = self.slot_index[session.slot_id]
            if self._present[slot]:
                collision_kwh += session.requested_kwh
                continue
            self._present[slot] = True
            self._remaining_kwh[slot] = session.requested_kwh
            self._departure_step[slot] = session.disconnect_step
        return collision_kwh

    def _obs(self) -> np.ndarray:
        presence = self._present.astype(np.float32)
        demand_norm = np.where(
            self._present,
            np.clip(self._remaining_kwh / self.max_demand_norm_kwh, 0.0, 1.0),
            0.0,
        ).astype(np.float32)
        steps_left = np.where(
            self._present,
            np.maximum(self._departure_step - self._step_idx, 0),
            0,
        )
        time_norm = np.where(
            self._present,
            np.clip(steps_left / self.max_deadline_norm_steps, 0.0, 1.0),
            0.0,
        ).astype(np.float32)

        calendar = self._calendar_features()
        return np.concatenate([presence, demand_norm, time_norm, calendar])

    def _calendar_features(self) -> np.ndarray:
        """Cyclical hour-of-day / day-of-week encoding of the real
        wall-clock time at the current step, anchored to the loader's
        tz-aware `data.start_time` - NOT derived from `_step_idx`
        alone, since `start_time` is not midnight/week-aligned in real
        ACN-Data (e.g. it starts mid-afternoon on a Tuesday)."""
        current_time = self.data.start_time + self._step_idx * self._timestep_delta
        hour_frac = current_time.hour + current_time.minute / 60.0
        dow = current_time.dayofweek  # Monday=0 .. Sunday=6

        hour_angle = 2.0 * math.pi * hour_frac / 24.0
        dow_angle = 2.0 * math.pi * dow / 7.0
        return np.array([
            math.sin(hour_angle), math.cos(hour_angle),
            math.sin(dow_angle), math.cos(dow_angle),
        ], dtype=np.float32)


# ==========================================
# Self-test
# ==========================================
# Uses a small SYNTHETIC fixture (make_synthetic_fixture) since no
# cached ACN-Data export exists in this repo yet - this only proves
# the environment's mechanics are correct, not anything about real
# ACN sessions. Point ACNChargingEnv at load_acn_sessions(<real cache
# path>) for actual experiments.

if __name__ == "__main__":
    PER_EV_MAX_KW = 6.6
    # Deliberately low relative to n_slots * PER_EV_MAX_KW so that an
    # all-max policy is guaranteed to overload it.
    TRANSFORMER_LIMIT_KW = 10.0

    data = make_synthetic_fixture(n_slots=4, timestep_minutes=60, seed=0)
    total_requested_kwh = sum(s.requested_kwh for s in data.sessions)
    print(f"Synthetic fixture: {len(data.sessions)} sessions across "
          f"{len(data.slot_ids)} slots, {data.n_steps} steps, "
          f"{total_requested_kwh:.1f} kWh total requested demand")

    # --------------------------------------------------
    # (a) all-max-power policy -> must violate the transformer limit
    # --------------------------------------------------
    env = ACNChargingEnv(
        data, transformer_limit_kw=TRANSFORMER_LIMIT_KW,
        per_ev_max_kw=PER_EV_MAX_KW,
    )
    obs, info = env.reset()
    total_violation_kw = 0.0
    n_violating_steps = 0
    for _ in range(env.episode_len):
        action = np.full(env.n_slots, PER_EV_MAX_KW, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_violation_kw += info["violation_kw"]
        if info["violation_kw"] > 0:
            n_violating_steps += 1
        if terminated or truncated:
            break

    assert total_violation_kw > 0, (
        "all-max-power policy produced ZERO transformer violations - "
        "the environment is not coupling delivered power to a real, "
        "unclipped transformer signal (this was exactly the original "
        "flaw: total output silently capped so violations can't happen)"
    )
    print(f"[PASS] (a) all-max-power policy: {n_violating_steps} violating "
          f"steps, {total_violation_kw:.1f} kW cumulative violation")

    # --------------------------------------------------
    # (b) zero-power policy -> must produce large unmet-energy penalty
    # --------------------------------------------------
    env = ACNChargingEnv(
        data, transformer_limit_kw=TRANSFORMER_LIMIT_KW,
        per_ev_max_kw=PER_EV_MAX_KW,
    )
    obs, info = env.reset()
    total_reward = 0.0
    total_unmet_kwh = 0.0
    total_violation_kw_b = 0.0
    for _ in range(env.episode_len):
        action = np.zeros(env.n_slots, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_unmet_kwh += info["unmet_kwh_this_step"]
        total_violation_kw_b += info["violation_kw"]
        if terminated or truncated:
            break

    assert total_unmet_kwh > 0.5 * total_requested_kwh, (
        "zero-power policy did not produce large unmet energy - demand "
        "is not actually persisting/being tracked against delivery"
    )
    assert np.isclose(total_unmet_kwh, total_requested_kwh, rtol=1e-6), (
        "zero-power policy's unmet energy does not equal total "
        "requested demand - some demand was lost or fabricated"
    )
    assert total_violation_kw_b == 0.0, (
        "zero-power policy produced a transformer violation - "
        "delivered power is not actually zero when action is zero"
    )
    print(f"[PASS] (b) zero-power policy: {total_unmet_kwh:.1f} kWh unmet "
          f"(== {total_requested_kwh:.1f} kWh total demand), "
          f"total reward {total_reward:.1f}")

    print("\nAll self-tests passed.")
