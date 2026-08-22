# ==========================================
# TwinEV v2 — LSTM / BiLSTM Load Forecaster (PyTorch)
# ==========================================
# Drop-in replacement for lstm_forecast.py (which requires
# TensorFlow). Replicates the exact feature engineering,
# data split (70/15/15), SEQ_LEN=24, architecture
# (BiLSTM 128 -> LSTM 64 -> LSTM 32 -> Dense 32/16 -> 1),
# Huber loss, Adam, batch 64, early stopping on val loss.
#
# Usage:
#   python lstm_forecast_torch.py --variant bilstm
#   python lstm_forecast_torch.py --variant lstm
#
# Outputs:
#   forecasts_<variant>_test.csv   (test window: actual, predicted)
#   forecasts_<variant>_full.csv   (full-length:  actual, predicted)
#   twinev_lstm_model_<variant>.pth
# ==========================================

import argparse
import math
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)


def build_features(df):
    df = df.copy()
    df["lag1"] = df["power_kw"].shift(1)
    df["lag24"] = df["power_kw"].shift(24)
    df["lag168"] = df["power_kw"].shift(168)
    df["rolling_mean_24"] = df["power_kw"].rolling(24).mean()
    df["rolling_std_24"] = df["power_kw"].rolling(24).std()
    df["rolling_mean_48"] = df["power_kw"].rolling(48).mean()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df.dropna(inplace=True)
    return df


FEATURES = [
    "power_kw", "solar_kw", "price_eur_kwh", "ev_count", "soc_arrival",
    "lag1", "lag24", "lag168",
    "rolling_mean_24", "rolling_std_24", "rolling_mean_48",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

SEQ_LEN = 24


def build_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)


class Forecaster(nn.Module):
    def __init__(self, n_features, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        out1 = 128 * (2 if bidirectional else 1)
        self.lstm1 = nn.LSTM(
            input_size=n_features, hidden_size=128,
            bidirectional=bidirectional, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(
            input_size=out1, hidden_size=64,
            bidirectional=False, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(
            input_size=64, hidden_size=32,
            bidirectional=False, batch_first=True)
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
        x = x[:, -1, :]      # last timestep only (keras return_sequences=False)
        x = self.drop3(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


def inverse_target(scaled_vals, scaler, n_features):
    scaled_vals = np.asarray(scaled_vals).reshape(-1, 1)
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, 0] = scaled_vals[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def train_model(X_train, y_train, X_val, y_val, bidirectional, device):
    model = Forecaster(X_train.shape[2], bidirectional=bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss(delta=1.0)

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
    yva = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    batch_size = 64
    n = len(Xtr)
    best_val = float("inf")
    best_state = None
    patience, best_epoch = 0, 0
    epochs_done = 0
    for epoch in range(100):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva), yva).item())
        epochs_done = epoch + 1

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                for g in optimizer.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, 1e-6)
            if patience >= 15:
                print(f"  early stop at epoch {epoch + 1} (best epoch {best_epoch + 1})")
                break

    model.load_state_dict(best_state)
    return model, best_val, epochs_done


def predict_arrays(model, X, device, batch=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0).ravel()


def run(variant, device):
    tag = variant
    bidirectional = variant == "bilstm"

    print(f"===== FORECASTER: {tag.upper()} =====")
    df = pd.read_csv("ev_synthetic_hourly_load.csv", index_col=0, parse_dates=True)
    df = build_features(df)
    print(f"Data after feature engineering: {df.shape}")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, f"twinev_scaler_{tag}.pkl")

    X, y = build_sequences(scaled, SEQ_LEN)
    print(f"Sequences X: {X.shape}")

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    print("Training...")
    model, best_val, epochs = train_model(
        X_train, y_train, X_val, y_val,
        bidirectional=bidirectional, device=device)
    print(f"Best val loss {best_val:.4f} after {epochs} epochs")

    torch.save(model.state_dict(), f"twinev_lstm_model_{tag}.pth")
    print(f"Saved model: twinev_lstm_model_{tag}.pth")

    y_test_inv = inverse_target(y_test, scaler, len(FEATURES))
    y_pred_inv = inverse_target(
        predict_arrays(model, X_test, device), scaler, len(FEATURES))

    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-5))) * 100)
    print(f"TEST RMSE {rmse:.2f} kW | MAE {mae:.2f} kW | MAPE {mape:.2f}%")

    pd.DataFrame({"actual": y_test_inv, "predicted": y_pred_inv}).to_csv(
        f"forecasts_{tag}_test.csv", index=False)
    print(f"Saved forecasts_{tag}_test.csv")

    full_pred = predict_arrays(model, X, device)
    full_pred_inv = inverse_target(full_pred, scaler, len(FEATURES))
    y_full_inv = inverse_target(y, scaler, len(FEATURES))
    pd.DataFrame({"actual": y_full_inv, "predicted": full_pred_inv}).to_csv(
        f"forecasts_{tag}_full.csv", index=False)
    print(f"Saved forecasts_{tag}_full.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["bilstm", "lstm"], default="bilstm")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.makedirs("results/forecasts", exist_ok=True)
    run(args.variant, args.device)
