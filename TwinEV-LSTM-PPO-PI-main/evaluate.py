# ==========================================
# TwinEV v2 FINAL EVALUATION (aligned)
# Forecasting: ARIMA vs LSTM vs BiLSTM
# Control:     Raw vs Greedy vs TwinEV (RL+PI)
# All models evaluated on the IDENTICAL
# chronological test window:
#   trimmed-frame hours 7282-8568
#   (= original data hours 7474-8759,
#      the last 15% of the forecast targets)
# ==========================================

import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
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

SEQ_LEN = 24

MODEL_DIR = "results/models"

ensure_dir("results/comparison")
ensure_dir("results/rl")
ensure_dir("results/figures")


# ==========================================
# TARGET SERIES (identical to forecaster)
# ==========================================


def load_targets():
    """Reconstruct the exact series the forecasters were trained on."""
    df = pd.read_csv("ev_synthetic_hourly_load.csv", index_col=0,
                     parse_dates=True)
    df = df.copy()
    df["lag1"] = df["power_kw"].shift(1)
    df["lag24"] = df["power_kw"].shift(24)
    df["lag168"] = df["power_kw"].shift(168)
    df["rolling_mean_24"] = df["power_kw"].rolling(24).mean()
    df["rolling_std_24"] = df["power_kw"].rolling(24).std()
    df["rolling_mean_48"] = df["power_kw"].rolling(48).mean()
    df.dropna(inplace=True)
    power = df["power_kw"].values

    n_seq = len(power) - SEQ_LEN
    train_end = int(n_seq * 0.70)
    val_end = int(n_seq * 0.85)

    train_target = power[SEQ_LEN: SEQ_LEN + train_end]
    val_target = power[SEQ_LEN + train_end: SEQ_LEN + val_end]
    test_target = power[SEQ_LEN + val_end: SEQ_LEN + n_seq]
    return train_target, val_target, test_target


print("Loading aligned test window...")
train_target, val_target, test_target = load_targets()
print(f"Train {len(train_target)} | Val {len(val_target)} | "
      f"Test {len(test_target)}")

state = load_aligned_data()
raw_test = state["power"][TEST_START:TEST_END]
price_test = state["price"][TEST_START:TEST_END]
solar_test = state["solar"][TEST_START:TEST_END]

assert len(raw_test) == len(test_target) == TEST_END - TEST_START


# ==========================================
# ARIMA
# ==========================================

print("\nFitting ARIMA(5,1,0) on training segment...")
arima_fit = ARIMA(train_target, order=(5, 1, 0)).fit()
n_fc = len(val_target) + len(test_target)
arima_all = np.asarray(arima_fit.forecast(steps=n_fc))
arima_pred = arima_all[len(val_target):]


# ==========================================
# NEURAL FORECASTERS (pre-computed, aligned)
# ==========================================

bilstm_df = pd.read_csv("forecasts_bilstm_test.csv")
lstm_df = pd.read_csv("forecasts_lstm_test.csv")

assert np.allclose(bilstm_df["actual"].values, test_target, atol=1e-3), \
    "BiLSTM test file does not match reconstructed test window"
assert np.allclose(lstm_df["actual"].values, test_target, atol=1e-3), \
    "LSTM test file does not match reconstructed test window"


# ==========================================
# TWIN EV RL + PI
# ==========================================

print("\nEvaluating FULL TwinEV policy over the test window...")
rl_model = PPO.load(f"{MODEL_DIR}/exp1_modelB_full")
out_rl = run_policy_sequential(rl_model, TwinEVEnv)


# ==========================================
# METRICS
# ==========================================


def forecast_metrics(actual, pred):
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    rmse = math.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = float(np.mean(np.abs((actual - pred) / (actual + 1e-5))) * 100)
    return round(rmse, 3), round(mae, 3), round(mape, 3)


rows_fc = []
for name, pred in [("ARIMA", arima_pred),
                   ("LSTM", lstm_df["predicted"].values),
                   ("BiLSTM", bilstm_df["predicted"].values)]:
    rmse, mae, mape = forecast_metrics(test_target, pred)
    rows_fc.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape})
    print(f"{name:8s} RMSE {rmse:8.3f}  MAE {mae:8.3f}  MAPE {mape:7.3f}%")

pd.DataFrame(rows_fc).to_csv(
    "results/comparison/forecast_metrics.csv", index=False)

# Control metrics (Raw / Greedy / RL+PI)
greedy = np.minimum(raw_test, 130.0)
control_rows = [
    model_metrics_row("Raw Demand", power=raw_test, setpoint=raw_test,
                      solar=solar_test, price=price_test, raw=raw_test,
                      extra={"type": "baseline"}),
    model_metrics_row("Greedy 130 kW", power=greedy, setpoint=greedy,
                      solar=solar_test, price=price_test, raw=raw_test,
                      extra={"type": "baseline"}),
    model_metrics_row("TwinEV RL+PI", power=out_rl["power"],
                      setpoint=out_rl["setpoint"], reward=out_rl["reward"],
                      solar=solar_test, price=price_test, raw=raw_test,
                      extra={"type": "rl"}),
]
control_df = pd.DataFrame(control_rows)
control_df.to_csv("results/comparison/control_metrics_test.csv", index=False)
print()
print(control_df.to_string(index=False))


# ==========================================
# SAVE RL ROLLOUT
# ==========================================

pd.DataFrame({"power": out_rl["power"], "reward": out_rl["reward"]}).to_csv(
    "results/rl/test_power_rewards.csv", index=False)


# ==========================================
# FIGURES
# ==========================================

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test_target[:168], label="Actual", linewidth=2)
ax.plot(arima_pred[:168], label="ARIMA", linestyle="--")
ax.plot(lstm_df["predicted"].values[:168], label="LSTM", linestyle="--")
ax.plot(bilstm_df["predicted"].values[:168], label="BiLSTM", linestyle="--")
ax.set_title("Forecast Comparison - First Test Week")
ax.set_xlabel("Hour")
ax.set_ylabel("Power (kW)")
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/Fig01_forecast.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(raw_test[:168], label="Raw demand", alpha=0.8)
ax.plot(out_rl["power"][:168], label="TwinEV RL+PI", linewidth=2)
ax.axhline(130, color="orange", linestyle=":", label="Soft limit")
ax.axhline(150, color="red", linestyle=":", label="Hard limit")
ax.set_title("TwinEV RL+PI vs Raw Demand - First Test Week")
ax.set_xlabel("Hour")
ax.set_ylabel("Power (kW)")
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/Fig02_RL.png", dpi=300)
plt.close(fig)

print("\n===== EVALUATION COMPLETE =====")
print("Saved: results/comparison/forecast_metrics.csv")
print("Saved: results/comparison/control_metrics_test.csv")
print("Saved: results/rl/test_power_rewards.csv")
