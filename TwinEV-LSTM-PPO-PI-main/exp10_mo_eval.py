# ==========================================
# TwinEV v2 - EXPERIMENT 10
# MULTI-OBJECTIVE EVALUATION
# ==========================================
# Evaluates all MO models + fixed-cap baselines +
# Greedy + Original TwinEV on the exact same
# 1,286-hour test window with extended metrics
# and policy-response correlations.
# ==========================================

import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from exp_utils import (
    TwinEVEnv,
    ensure_dir,
    load_aligned_data,
    model_metrics_row,
    run_policy_sequential,
    TEST_START,
    TEST_END,
)
from mo_env import (
    MOFullEnv,
    MONoForecastEnv,
    MONoCostEnv,
    MONoSolarEnv,
    MONoSmoothnessEnv,
)

OUT_DIR = "results/mo_control"
MODEL_DIR = f"{OUT_DIR}/models"

MODELS = [
    ("MO_FULL", "MO_FULL", MOFullEnv),
    ("MO_FULL_s2", "MO_FULL", MOFullEnv),
    ("MO_FULL_s3", "MO_FULL", MOFullEnv),
    ("MO_NO_FORECAST", "MO_NO_FORECAST", MONoForecastEnv),
    ("MO_NO_COST", "MO_NO_COST", MONoCostEnv),
    ("MO_NO_SOLAR", "MO_NO_SOLAR", MONoSolarEnv),
    ("MO_NO_SMOOTHNESS", "MO_NO_SMOOTHNESS", MONoSmoothnessEnv),
]

FIXED_CAPS = [110, 120, 130, 140, 150]


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y))
    if n < 2 or np.std(x[:n]) == 0 or np.std(y[:n]) == 0:
        return np.nan
    return float(np.corrcoef(x[:n], y[:n])[0, 1])


def extended_row(model_name, power, setpoint=None, reward=None,
                 reward_info=None, solar=None, price=None, raw=None,
                 demand=None, forecast=None):
    row = model_metrics_row(model_name, power=power, setpoint=setpoint,
                            reward=reward, solar=solar, price=price,
                            raw=raw)
    p = np.asarray(power)
    row["hours_gt130"] = int(np.sum(p > 130.0))
    row["hours_lt130"] = int(np.sum(p < 130.0))
    row["distinct_power_levels"] = int(len(np.unique(np.round(p, 1))))
    row["corr_power_price"] = corr(p, price)
    row["corr_power_solar"] = corr(p, solar)
    row["corr_power_demand"] = corr(p, demand)
    row["corr_power_forecast"] = corr(p, forecast)
    if reward_info is not None:
        info_df = pd.DataFrame(reward_info)
        for comp in ["r_service", "r_cost", "r_solar", "r_grid",
                     "r_smooth", "r_fc"]:
            if comp in info_df.columns:
                v = info_df[comp].astype(float)
                row[f"{comp}_mean"] = round(float(v.mean()), 4)
                row[f"{comp}_std"] = round(float(v.std()), 4)
                row[f"{comp}_min"] = round(float(v.min()), 4)
                row[f"{comp}_max"] = round(float(v.max()), 4)
    return row


def main():
    ensure_dir(OUT_DIR)
    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]
    forecast_test = state["forecast"][TEST_START:TEST_END]

    rows = []

    # --- Fixed-cap baselines ---
    for cap in FIXED_CAPS:
        p = np.full(TEST_END - TEST_START, float(cap))
        rows.append(extended_row(
            f"Fixed {cap} kW", p, setpoint=None, reward=None,
            solar=solar_test, price=price_test, raw=raw_test,
            demand=raw_test, forecast=forecast_test,
            reward_info=None))

    # --- Greedy 130 ---
    greedy = np.minimum(raw_test, 130.0)
    rows.append(extended_row(
        "Greedy 130 kW", greedy, setpoint=None, reward=None,
        solar=solar_test, price=price_test, raw=raw_test,
        demand=raw_test, forecast=forecast_test, reward_info=None))

    # --- Original TwinEV ---
    full = PPO.load("results/models/exp1_modelB_full")
    out_full = run_policy_sequential(full, TwinEVEnv)
    rows.append(extended_row(
        "Original TwinEV", out_full["power"],
        setpoint=out_full["setpoint"], reward=out_full["reward"],
        solar=solar_test, price=price_test, raw=raw_test,
        demand=raw_test, forecast=forecast_test,
        reward_info=None))

    # --- MO models ---
    for name, group, env_cls in MODELS:
        path = f"{MODEL_DIR}/{name}.zip"
        if not os.path.exists(path):
            print(f"WARNING: {name} not trained yet; skipping.")
            continue
        model = PPO.load(path)
        out = run_policy_sequential(model, env_cls)
        row = extended_row(
            name, out["power"], setpoint=out["setpoint"],
            reward=out["reward"], reward_info=out["reward_info"],
            solar=solar_test, price=price_test, raw=raw_test,
            demand=raw_test, forecast=forecast_test)
        row["group"] = group
        rows.append(row)
        print(f"\n[{name}] service {row['energy_delivery_pct']:.2f}% | "
              f"cost {row['charging_cost']:,.0f} | "
              f"solar {row['solar_utilization_pct']:.1f}% | "
              f"hard {row['hard_overloads']} soft {row['soft_overloads']} | "
              f"distinct_levels {row['distinct_power_levels']} | "
              f"corr(price) {row['corr_power_price']:+.2f} "
              f"corr(solar) {row['corr_power_solar']:+.2f} "
              f"corr(demand) {row['corr_power_demand']:+.2f} "
              f"corr(fc) {row['corr_power_forecast']:+.2f}")
        if out["reward_info"] is not None:
            for comp in ["r_service", "r_cost", "r_solar", "r_grid",
                         "r_smooth", "r_fc"]:
                print(f"    {comp}: mean={row[comp+'_mean']:+.3f} "
                      f"[{row[comp+'_min']:+.3f},{row[comp+'_max']:+.3f}]")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/mo_results.csv", index=False)
    print(f"\nSaved results/mo_control/mo_results.csv "
          f"({len(df)} rows)")

    # Component statistics table (per MO model)
    comp_rows = []
    for name, group, env_cls in MODELS:
        path = f"{MODEL_DIR}/{name}.zip"
        if not os.path.exists(path):
            continue
        model = PPO.load(path)
        out = run_policy_sequential(model, env_cls)
        info_df = pd.DataFrame(out["reward_info"])
        for comp in ["r_service", "r_cost", "r_solar", "r_grid",
                     "r_smooth", "r_fc"]:
            v = info_df[comp].astype(float)
            comp_rows.append({
                "model": name, "component": comp,
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std()), 4),
                "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4),
            })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(f"{OUT_DIR}/mo_reward_components.csv", index=False)
    print("Saved results/mo_control/mo_reward_components.csv")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
