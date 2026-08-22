# ==========================================
# TwinEV v2 - EXPERIMENT 6
# Forecasting Comparison: ARIMA vs LSTM vs BiLSTM
# ==========================================
# Same forecasting target (next-hour charging
# demand) and same chronological test window
# (last 15% of the aligned series).
# ==========================================

import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from exp_utils import ensure_dir

SEQ_LEN = 24


def load_target_series():
    """Reconstruct the trimmed series used by the forecasters
    and the target rows they predict."""
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
    power = df["power_kw"].values  # length 8592

    n_seq = len(power) - SEQ_LEN
    train_end = int(n_seq * 0.70)
    val_end = int(n_seq * 0.85)

    # Target rows: sequence i predicts power[row i+SEQ_LEN]
    train_target = power[SEQ_LEN: SEQ_LEN + train_end]
    val_target = power[SEQ_LEN + train_end: SEQ_LEN + val_end]
    test_target = power[SEQ_LEN + val_end: SEQ_LEN + n_seq]
    return train_target, val_target, test_target


def main():
    ensure_dir("results/forecasting")

    train_target, val_target, test_target = load_target_series()
    print(f"Train target: {len(train_target)} | Val: {len(val_target)} | "
          f"Test: {len(test_target)}")

    bilstm = pd.read_csv("forecasts_bilstm_test.csv")
    lstm = pd.read_csv("forecasts_lstm_test.csv")

    assert np.allclose(bilstm["actual"].values, test_target, atol=1e-3), \
        "BiLSTM test target does not match the reconstructed test window"
    assert np.allclose(lstm["actual"].values, test_target, atol=1e-3), \
        "LSTM test target does not match the reconstructed test window"

    print("Fitting ARIMA(5,1,0) on the training segment...")
    arima_fit = ARIMA(train_target, order=(5, 1, 0)).fit()
    n_fc = len(val_target) + len(test_target)
    arima_all = arima_fit.forecast(steps=n_fc)
    arima_test = np.asarray(arima_all[len(val_target):])

    rows = []
    for name, pred in [
        ("ARIMA", arima_test),
        ("LSTM", lstm["predicted"].values),
        ("BiLSTM", bilstm["predicted"].values),
    ]:
        pred = np.asarray(pred, dtype=np.float64)
        actual = np.asarray(test_target, dtype=np.float64)
        rmse = math.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mape = float(np.mean(np.abs((actual - pred) / (actual + 1e-5))) * 100)
        rows.append({"Model": name, "RMSE": round(rmse, 3),
                     "MAE": round(mae, 3), "MAPE": round(mape, 3)})
        print(f"{name:8s} RMSE {rmse:8.3f}  MAE {mae:8.3f}  MAPE {mape:7.3f}%")

    df = pd.DataFrame(rows)
    df.to_csv("results/forecasting/forecast_model_comparison.csv", index=False)
    print("\nSaved results/forecasting/forecast_model_comparison.csv")

    # Save aligned test forecasts for plotting
    pd.DataFrame({
        "actual": test_target,
        "arima": arima_test,
        "lstm": lstm["predicted"].values,
        "bilstm": bilstm["predicted"].values,
    }).to_csv("results/forecasting/forecast_test_window.csv", index=False)


if __name__ == "__main__":
    main()
