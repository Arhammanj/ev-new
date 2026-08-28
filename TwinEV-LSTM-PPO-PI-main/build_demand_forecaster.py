# ==========================================
# ACN-Data hourly demand forecaster + baseline ladder
# ==========================================
# Forecasting task, separate from ACNChargingEnv, but built on the
# same real Caltech 2019 cache and the same chronological-split
# discipline used throughout this project (train-only scaler fit,
# no shuffling, walk-forward-safe lag features).
#
# Target: hourly INCOMING demand = total requested kWh from sessions
# that ARRIVE in each hour (data.arrivals_by_step from the loader) -
# a forward-looking "how much demand is about to show up" signal,
# distinct from concurrency or delivered power.
#
# BiLSTM architecture is adapted from archive/lstm_forecast_torch.py
# (copied and retargeted, not imported - that module is archived
# reference code for a different, obsolete pipeline, not a live
# dependency).
# ==========================================

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

from acn_data_loader import load_acn_sessions
from metrics import mae, rmse, wape

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "acn_caltech_2019.csv"
DEMAND_CSV = "acn_hourly_demand.csv"
FORECASTS_CSV = "acn_demand_forecasts_test.csv"
SEQ_LEN = 24

FEATURE_COLS = [
    "demand", "lag1", "lag24", "lag168",
    "rolling_mean_24", "rolling_std_24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
LGBM_FEATURE_COLS = [c for c in FEATURE_COLS if c != "demand"]


# ==========================================
# 1. TARGET SERIES
# ==========================================

def build_target_series(data_path: str) -> pd.DataFrame:
    data = load_acn_sessions(data_path, timestep_minutes=60)
    n = data.n_steps
    demand = np.zeros(n, dtype=np.float64)
    for t in range(n):
        for idx in data.arrivals_by_step[t]:
            demand[t] += data.sessions[idx].requested_kwh
    timestamps = pd.date_range(data.start_time, periods=n, freq="h")
    return pd.DataFrame({"timestamp": timestamps, "demand": demand})


# ==========================================
# 2. FEATURE ENGINEERING (lag + calendar; all strictly past-referencing)
# ==========================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag1"] = df["demand"].shift(1)
    df["lag24"] = df["demand"].shift(24)
    df["lag168"] = df["demand"].shift(168)
    # shift(1) before rolling: window is [t-24 .. t-1], excludes row t
    # itself, so this can never leak the value being predicted.
    df["rolling_mean_24"] = df["demand"].shift(1).rolling(24).mean()
    df["rolling_std_24"] = df["demand"].shift(1).rolling(24).std()

    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    df = df.dropna().reset_index(drop=True)
    return df


# ==========================================
# 3. BiLSTM (architecture adapted from archive/lstm_forecast_torch.py)
# ==========================================

class Forecaster(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, bidirectional=True, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.drop3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.drop3(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


def build_sequences(arr: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len, 0])   # column 0 = "demand"
    return np.array(X), np.array(y)


def inverse_demand(scaled_vals: np.ndarray, scaler: MinMaxScaler, n_features: int) -> np.ndarray:
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, 0] = scaled_vals
    return scaler.inverse_transform(dummy)[:, 0]


def train_bilstm(X_train, y_train, X_val, y_val, device):
    model = Forecaster(X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss(delta=1.0)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    batch_size = 64
    n = len(Xtr)
    best_val, best_state, patience, best_epoch = float("inf"), None, 0, 0
    for epoch in range(100):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva), yva).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print(f"  early stop at epoch {epoch + 1} (best epoch {best_epoch + 1})")
                break

    model.load_state_dict(best_state)
    return model


def predict_bilstm(model, X, device, batch=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds).reshape(-1)


# ==========================================
# MAIN
# ==========================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Target series ----
    print("Building hourly incoming-demand target series...")
    series = build_target_series(DATA_PATH)
    series.to_csv(DEMAND_CSV, index=False)
    n = len(series)
    demand = series["demand"].values
    pct_zero = 100 * np.mean(demand == 0)
    print(f"Saved {DEMAND_CSV} ({n} hourly rows)")
    print(f"  mean={demand.mean():.3f} kWh  std={demand.std():.3f} kWh  "
          f"zero hours={pct_zero:.2f}%")

    # ---- 2. Chronological split (on the raw, un-featurized grid) ----
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    print(f"\nChronological split (N={n} hourly steps):")
    print(f"  train: [0, {train_end})          -> {series['timestamp'].iloc[0]} to "
          f"{series['timestamp'].iloc[train_end - 1]}")
    print(f"  val  : [{train_end}, {val_end})   -> {series['timestamp'].iloc[train_end]} to "
          f"{series['timestamp'].iloc[val_end - 1]}")
    print(f"  test : [{val_end}, {n})   -> {series['timestamp'].iloc[val_end]} to "
          f"{series['timestamp'].iloc[n - 1]}")
    test_len = n - val_end
    print(f"  test window length: {test_len} hours")

    actual_test = demand[val_end:n]

    # ---- 3. Simple baselines (no features/training needed) ----
    persistence = demand[val_end - 1:n - 1]
    seasonal_24 = demand[val_end - 24:n - 24]
    seasonal_168 = demand[val_end - 168:n - 168]
    assert len(persistence) == len(seasonal_24) == len(seasonal_168) == test_len

    # ---- 4. Features for LightGBM / BiLSTM ----
    feat_df = build_features(series)
    n_dropped = n - len(feat_df)
    print(f"\n{n_dropped} rows dropped to NaN lag/rolling warm-up "
          f"(first {n_dropped} hours) - test window starts well after this.")

    # re-express train/val boundaries in the post-dropna row index
    f_train_end = train_end - n_dropped
    f_val_end = val_end - n_dropped
    assert f_train_end > 0, "warm-up ate into the training window"

    # ---- 4a. LightGBM ----
    print("\nTraining LightGBM...")
    X_all = feat_df[LGBM_FEATURE_COLS].values
    y_all = feat_df["demand"].values
    lgbm = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=42, verbosity=-1,
    )
    lgbm.fit(X_all[:f_train_end], y_all[:f_train_end])
    lgbm_pred_all = lgbm.predict(X_all)
    lgbm_pred_test = lgbm_pred_all[f_val_end:]
    assert len(lgbm_pred_test) == test_len

    # ---- 4b. BiLSTM ----
    print("Training BiLSTM...")
    scaler = MinMaxScaler()
    scaler.fit(feat_df[FEATURE_COLS].iloc[:f_train_end])   # train-only fit
    scaled = scaler.transform(feat_df[FEATURE_COLS])

    X_seq, y_seq = build_sequences(scaled, SEQ_LEN)
    # sequence i predicts feat_df row (i + SEQ_LEN)
    seq_train_end = f_train_end - SEQ_LEN
    seq_val_end = f_val_end - SEQ_LEN
    assert seq_train_end > 0

    model = train_bilstm(
        X_seq[:seq_train_end], y_seq[:seq_train_end],
        X_seq[seq_train_end:seq_val_end], y_seq[seq_train_end:seq_val_end],
        device,
    )
    pred_scaled_test = predict_bilstm(model, X_seq[seq_val_end:], device)
    bilstm_pred_test = inverse_demand(pred_scaled_test, scaler, len(FEATURE_COLS))
    assert len(bilstm_pred_test) == test_len

    # sanity: the row this predicts should be exactly the same actual
    # values as the simple-baseline test window
    actual_from_seq = inverse_demand(y_seq[seq_val_end:], scaler, len(FEATURE_COLS))
    assert np.allclose(actual_from_seq, actual_test, atol=1e-6), \
        "BiLSTM test-window alignment does not match the canonical test window"

    # ---- 5. Metrics table ----
    models = {
        "Persistence (t-1)": persistence,
        "Seasonal-naive (t-24)": seasonal_24,
        "Seasonal-naive (t-168)": seasonal_168,
        "LightGBM": lgbm_pred_test,
        "BiLSTM": bilstm_pred_test,
    }

    rows = []
    for name, pred in models.items():
        rows.append({
            "Model": name,
            "RMSE": round(rmse(actual_test, pred), 4),
            "MAE": round(mae(actual_test, pred), 4),
            "WAPE": round(wape(actual_test, pred), 3),
        })
    table = pd.DataFrame(rows)
    print("\n===== TEST-WINDOW METRICS =====")
    print(table.to_string(index=False))

    # ---- 6. Verdict ----
    wape_bilstm = table.loc[table["Model"] == "BiLSTM", "WAPE"].iloc[0]
    wape_sn24 = table.loc[table["Model"] == "Seasonal-naive (t-24)", "WAPE"].iloc[0]
    wape_sn168 = table.loc[table["Model"] == "Seasonal-naive (t-168)", "WAPE"].iloc[0]
    best_sn_wape = min(wape_sn24, wape_sn168)
    best_sn_name = "t-24" if wape_sn24 <= wape_sn168 else "t-168"

    print(f"\n===== VERDICT =====")
    if wape_bilstm < best_sn_wape:
        print(f"BiLSTM WAPE ({wape_bilstm:.3f}%) BEATS the best seasonal-naive "
              f"({best_sn_name}, {best_sn_wape:.3f}%).")
    else:
        print(f"BiLSTM WAPE ({wape_bilstm:.3f}%) does NOT beat the best seasonal-naive "
              f"({best_sn_name}, {best_sn_wape:.3f}%). This is reported as-is, not tuned "
              "to force a different result.")

    # ---- 7. Save forecasts ----
    out = pd.DataFrame({
        "timestamp": series["timestamp"].iloc[val_end:n].values,
        "actual": actual_test,
        "persistence": persistence,
        "seasonal_naive_24": seasonal_24,
        "seasonal_naive_168": seasonal_168,
        "lightgbm": lgbm_pred_test,
        "bilstm": bilstm_pred_test,
    })
    out.to_csv(FORECASTS_CSV, index=False)
    print(f"\nSaved {FORECASTS_CSV} ({len(out)} rows)")
    table.to_csv("acn_demand_forecast_metrics.csv", index=False)
    print("Saved acn_demand_forecast_metrics.csv")


if __name__ == "__main__":
    main()
