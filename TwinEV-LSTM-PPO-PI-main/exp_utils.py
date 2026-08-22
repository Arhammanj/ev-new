# ==========================================
# TwinEV v2 - Shared experiment utilities
# ==========================================
# Common metric functions, sequential test-set
# evaluation harness, and env variants used by
# experiments 1-7. Does not modify the original
# pipeline files.
# ==========================================

import os

import numpy as np
import pandas as pd
from gymnasium import spaces

from pi_controller import PIController
from twinev_env import (
    TwinEVEnv,
    MAX_GRID_KW,
    OVERLOAD_KW,
    MAX_PRICE,
    SOLAR_MAX_KW,
    MAX_QUEUE,
)

# ------------------------------------------------------------------
# TEST WINDOW - identical to the chronological test set used by
# TwinEVEnv (aligned to the 8568-sample forecast/data length):
#   Train 0-5997, Val 5997-7282, Test 7282-8568  (last 15%)
# ------------------------------------------------------------------

TEST_START = int(8568 * 0.85)      # 7282
TEST_END   = 8568                  # exclusive, aligned array length
TEST_LEN   = TEST_END - TEST_START

# ------------------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------------------


def load_aligned_data(data_path="ev_synthetic_hourly_load.csv",
                      forecast_path="lstm_full_predictions.csv"):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    fc = pd.read_csv(forecast_path)
    n = min(len(df), len(fc))
    return {
        "power": df["power_kw"].values[:n].astype(np.float32),
        "solar": df["solar_kw"].values[:n].astype(np.float32),
        "price": df["price_eur_kwh"].values[:n].astype(np.float32),
        "queue": df["ev_count"].values[:n].astype(np.float32),
        "forecast": fc["predicted"].values[:n].astype(np.float32),
        "n": n,
    }


# ------------------------------------------------------------------
# ENV VARIANTS (subclasses - original env untouched)
# ------------------------------------------------------------------


class NoForecastEnv(TwinEVEnv):
    """Observation = [load, price, solar, queue, sin, cos] - NO forecast."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self, idx):
        full = super()._get_obs(idx)
        return np.concatenate([full[:4], full[5:]]).astype(np.float32)


class NoPIEnv(TwinEVEnv):
    """Same as TwinEVEnv but PPO command is applied directly
    (no PI layer). Physical clamp identical to PI output clamp."""

    def step(self, action):
        idx = self.episode_start + self.current_step

        action_scalar = float(np.clip(action[0], 0.2, 1.0))
        setpoint_kw = action_scalar * MAX_GRID_KW
        actual_power = float(np.clip(setpoint_kw, 0.0, MAX_GRID_KW))
        self.actual_power = actual_power
        self.pi.tracking_error = 0.0

        grid_load = float(self.grid_load[idx])
        solar_gen = float(self.solar[idx])
        energy_price = float(self.price[idx])
        queue_len = float(self.ev_queue[idx])
        lstm_next = float(self.lstm_forecast[
            min(idx + 1, self.n_steps_total - 1)])

        reward, reward_info = self._compute_reward(
            actual_power=actual_power,
            setpoint_kw=setpoint_kw,
            grid_load=grid_load,
            solar_gen=solar_gen,
            energy_price=energy_price,
            queue_len=queue_len,
            lstm_forecast=lstm_next,
        )

        self.ep_rewards.append(reward)
        self.ep_actions.append(action_scalar)
        self.ep_actual.append(actual_power)
        self.ep_setpoints.append(setpoint_kw)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_idx = min(
            self.episode_start + self.current_step,
            self.n_steps_total - 1,
        )
        obs = self._get_obs(next_idx)
        info = {**reward_info, "step": self.current_step}
        return obs, reward, done, False, info


# ------------------------------------------------------------------
# SEQUENTIAL TEST-SET EVALUATION FOR RL POLICIES
# ------------------------------------------------------------------


def run_policy_sequential(policy, env_cls, start=None, end=None):
    """Roll the policy out over the exact chronological test window,
    starting from the beginning of the test set (no random episode
    starts; PI state carries continuously)."""
    start = TEST_START if start is None else start
    end = TEST_END if end is None else end

    env = env_cls(mode="test")
    env.episode_start = start
    env.max_steps = end - start
    env.current_step = 0
    env.pi.reset()
    env.actual_power = float(env.grid_load[start])

    obs = env._get_obs(start)
    power, setpoint, reward, rinfo = [], [], [], []
    for _ in range(end - start):
        action, _ = policy.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(action)
        power.append(info["actual_power"])
        setpoint.append(info["setpoint_kw"])
        reward.append(r)
        rinfo.append(info)
    return {
        "power": np.array(power),
        "setpoint": np.array(setpoint),
        "reward": np.array(reward),
        "reward_info": rinfo,
    }


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------


def control_metrics(power, setpoint=None, reward=None, solar=None,
                    price=None, raw=None, raw_peak=None):
    """Compute the full battery of physical/operational metrics.

    power : controller output profile over the test window (kW, hourly)
    raw   : uncontrolled demand over the same window (delivery %)
    """
    power = np.asarray(power, dtype=np.float64)

    hard = int(np.sum(power > MAX_GRID_KW))
    soft = int(np.sum(power > OVERLOAD_KW))
    peak = float(np.max(power))
    mean_power = float(np.mean(power))
    variance = float(np.var(power))
    energy = float(np.sum(power))

    if raw is not None:
        raw = np.asarray(raw, dtype=np.float64)
        delivery_pct = 100.0 * energy / (np.sum(raw) + 1e-9)
    else:
        delivery_pct = np.nan

    if raw_peak is None and raw is not None:
        raw_peak = float(np.max(raw))
    if raw_peak and raw_peak > 0:
        peak_reduction = 100.0 * (raw_peak - peak) / raw_peak
    else:
        peak_reduction = np.nan

    cost = np.nan
    if price is not None:
        price = np.asarray(price, dtype=np.float64)
        cost = float(np.sum(power[:len(price)] * price[:len(power)]))

    solar_util = np.nan
    if solar is not None:
        solar = np.asarray(solar, dtype=np.float64)
        solar_util = 100.0 * np.sum(
            np.minimum(power, solar)) / (np.sum(solar) + 1e-9)

    tracking_err = np.nan
    max_tracking_err = np.nan
    if setpoint is not None:
        setpoint = np.asarray(setpoint, dtype=np.float64)
        trk = np.abs(setpoint - power[:len(setpoint)])
        tracking_err = float(np.mean(trk))
        max_tracking_err = float(np.max(trk)) if len(trk) else np.nan

    ramp = np.abs(np.diff(power))
    mean_ramp = float(np.mean(ramp)) if len(ramp) else np.nan
    max_ramp = float(np.max(ramp)) if len(ramp) else np.nan

    total_reward = float(np.sum(reward)) if reward is not None else np.nan
    mean_reward = float(np.mean(reward)) if reward is not None else np.nan
    std_reward = float(np.std(reward)) if reward is not None else np.nan

    return {
        "hard_overloads": hard,
        "soft_overloads": soft,
        "peak_power_kw": round(peak, 2),
        "peak_reduction_pct": round(peak_reduction, 2),
        "mean_power_kw": round(mean_power, 2),
        "power_variance": round(variance, 2),
        "energy_delivered_kwh": round(energy, 2),
        "energy_delivery_pct": round(delivery_pct, 2),
        "charging_cost": round(cost, 2) if np.isfinite(cost) else np.nan,
        "solar_utilization_pct": (round(solar_util, 2)
                                  if np.isfinite(solar_util) else np.nan),
        "mean_tracking_error_kw": (round(tracking_err, 2)
                                   if np.isfinite(tracking_err) else np.nan),
        "max_tracking_error_kw": (round(max_tracking_err, 2)
                                  if np.isfinite(max_tracking_err) else np.nan),
        "mean_ramp_kw": round(mean_ramp, 2) if np.isfinite(mean_ramp) else np.nan,
        "max_ramp_kw": round(max_ramp, 2) if np.isfinite(max_ramp) else np.nan,
        "total_reward": round(total_reward, 2) if np.isfinite(total_reward) else np.nan,
        "mean_reward": round(mean_reward, 4) if np.isfinite(mean_reward) else np.nan,
        "std_reward": round(std_reward, 4) if np.isfinite(std_reward) else np.nan,
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def model_metrics_row(model_name, power, setpoint=None, reward=None,
                      solar=None, price=None, raw=None, extra=None):
    m = control_metrics(power, setpoint=setpoint, reward=reward,
                        solar=solar, price=price, raw=raw)
    row = {"model": model_name}
    row.update(m)
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------
# BASELINE PROFILES (evaluated over the same test window)
# ------------------------------------------------------------------


def baseline_profiles(state):
    test = {k: v[TEST_START:TEST_END] for k, v in state.items()}
    raw = test["power"]
    fcfs = np.array(raw)                                  # FCFS: passthrough
    rr = np.full_like(raw, np.clip(np.mean(raw), 0, MAX_GRID_KW))
    greedy = np.minimum(raw, OVERLOAD_KW)
    return {
        "Raw Demand": raw,
        "FCFS": fcfs,
        "Round Robin": rr,
        "Greedy": greedy,
    }
