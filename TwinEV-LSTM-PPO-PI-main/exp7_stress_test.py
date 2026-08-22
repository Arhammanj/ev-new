# ==========================================
# TwinEV v2 - EXPERIMENT 7
# Stress Testing (no retraining)
# ==========================================
# Evaluate the already-trained FULL TwinEV
# policy under controlled scenario modifications
# of the test data:
#   normal | high_demand | low_solar |
#   high_price | demand_spike
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
    MAX_GRID_KW,
    MAX_PRICE,
    TEST_START,
    TEST_END,
)

MODEL_DIR = "results/models"

SCENARIOS = {
    "normal": {"description": "unmodified test conditions"},
    "high_demand": {"description": "EV demand +50%, queue +50%, forecast +50%"},
    "low_solar": {"description": "solar generation reduced to 30%"},
    "high_price": {"description": "electricity price x1.8 (capped at 0.55)"},
    "demand_spike": {"description": "brief +100 kW surge (6 h) on low-demand hours, crossing the overload threshold"},
}


def apply_scenario(env, scenario):
    """Modify the env arrays over the test window."""
    s = TEST_START
    e = TEST_END
    if scenario == "high_demand":
        env.grid_load[s:e] = np.clip(env.grid_load[s:e] * 1.5, 0, 400)
        env.ev_queue[s:e] = env.ev_queue[s:e] * 1.5
        env.lstm_forecast[s:e] = np.clip(env.lstm_forecast[s:e] * 1.5, 0, 400)
    elif scenario == "low_solar":
        env.solar[s:e] = env.solar[s:e] * 0.3
    elif scenario == "high_price":
        env.price[s:e] = np.clip(env.price[s:e] * 1.8, 0, MAX_PRICE)
    elif scenario == "demand_spike":
        # Surge (+100 kW for 6 h) on the lowest-demand hours so demand
        # crosses the overload threshold and the controller must react.
        order = np.argsort(env.grid_load[s:e])
        start = s + int(order[0])
        end = start + 6
        env.grid_load[start:end] = env.grid_load[start:end] + 100.0
        env.lstm_forecast[start:end] = env.lstm_forecast[start:end] + 100.0


def make_scenario_env(scenario):
    class _ScenarioEnv(TwinEVEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_scenario(self, scenario)

    _ScenarioEnv.__name__ = f"ScenarioEnv_{scenario}"
    return _ScenarioEnv


def main():
    ensure_dir("results/stress_tests")
    ensure_dir("results/profiles")

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]

    print("Loading FULL TwinEV policy (no retraining per scenario)...")
    model = PPO.load(f"{MODEL_DIR}/exp1_modelB_full")

    results = []
    for scenario, meta in SCENARIOS.items():
        print(f"\n===== Scenario: {scenario} =====")
        env_cls = make_scenario_env(scenario)
        out = run_policy_sequential(model, env_cls)
        pd.DataFrame({"power": out["power"]}).to_csv(
            f"results/profiles/exp7_{scenario}.csv", index=False)

        row = model_metrics_row(
            scenario,
            power=out["power"],
            setpoint=out["setpoint"],
            reward=out["reward"],
            solar=solar_test,   # original solar for utilization baseline
            price=price_test,   # original price for cost baseline
            raw=raw_test,       # original demand for delivery %
            extra={"description": meta["description"]},
        )
        results.append(row)
        print(f"peak {row['peak_power_kw']} kW | hard {row['hard_overloads']} | "
              f"soft {row['soft_overloads']} | energy {row['energy_delivered_kwh']} kWh | "
              f"total_reward {row['total_reward']}")

    df = pd.DataFrame(results)
    df.to_csv("results/stress_tests/stress_test_results.csv", index=False)
    print("\nSaved results/stress_tests/stress_test_results.csv")
    print(df.to_string())


if __name__ == "__main__":
    main()
