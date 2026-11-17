"""Build blocked OOF stepwise BiLSTM forecasts and the matching oracle table."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_demand_forecaster import build_target_series
from test_alternative_targets import (
    CALENDAR_COLS,
    LAG_FEATURE_COLS,
    SEQ_LEN,
    add_calendar,
    add_lag_features,
)


class StepwiseForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, bidirectional=True, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.drop3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, horizon)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def train_stepwise(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
) -> StepwiseForecaster:
    torch.manual_seed(seed)
    model = StepwiseForecaster(x_train.shape[2], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss(delta=1.0)
    x_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_va = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val, dtype=torch.float32, device=device)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    n = len(x_tr)
    batch_size = 64

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_va), y_va).item())
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("stepwise BiLSTM did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.best_epoch = best_epoch
    model.best_val_loss = best_loss
    return model


def predict_stepwise(
    model: StepwiseForecaster,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_frame(data_path: str, horizon: int, timestep_minutes: int) -> tuple[pd.DataFrame, np.ndarray]:
    series = build_target_series(data_path, timestep_minutes=timestep_minutes)
    demand = series["demand"].to_numpy(dtype=float)
    valid_end = len(demand) - horizon
    if valid_end <= SEQ_LEN + 1:
        raise ValueError("not enough data for a stepwise forecast target")
    targets = np.stack([demand[t + 1:t + 1 + horizon] for t in range(valid_end)])
    frame = pd.DataFrame({
        "abs_idx": np.arange(valid_end, dtype=int),
        "timestamp": series["timestamp"].iloc[:valid_end].to_numpy(),
        "feat_src": demand[:valid_end],
    })
    frame = add_calendar(frame)
    frame = add_lag_features(frame, "feat_src", steps_per_hour=60 // timestep_minutes)
    valid = frame["lag168"].notna().to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    targets = targets[frame["abs_idx"].to_numpy(dtype=int)]
    return frame, targets


def build_oracle_frame(data_path: str, horizon: int, timestep_minutes: int) -> pd.DataFrame:
    frame, targets = build_frame(data_path, horizon, timestep_minutes)
    oracle = pd.DataFrame({
        "hour_index": frame["abs_idx"].to_numpy(dtype=int),
        "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
    })
    for index in range(horizon):
        oracle[f"actual_h{index + 1}"] = targets[:, index]
        oracle[f"predicted_h{index + 1}"] = targets[:, index]
    oracle["actual_total"] = targets.sum(axis=1)
    oracle["predicted_total"] = targets.sum(axis=1)
    oracle["horizon_hours"] = horizon
    oracle["timestep_minutes"] = timestep_minutes
    oracle["input_last_index"] = oracle["hour_index"]
    return oracle


def metric_rows(actual: np.ndarray, predicted: np.ndarray, scope: str) -> list[dict]:
    rows = []
    for index in range(actual.shape[1]):
        a = actual[:, index]
        p = predicted[:, index]
        denom = float(np.abs(a).sum())
        rows.append({
            "scope": scope,
            "target": f"h{index + 1}",
            "n": int(len(a)),
            "mae_kwh": float(np.abs(a - p).mean()),
            "rmse_kwh": float(math.sqrt(np.mean((a - p) ** 2))),
            "wape_pct": float(np.abs(a - p).sum() / denom * 100.0) if denom > 0 else float("nan"),
        })
    a_total = actual.sum(axis=1)
    p_total = predicted.sum(axis=1)
    denom = float(np.abs(a_total).sum())
    rows.append({
        "scope": scope,
        "target": "total_h1_h6",
        "n": int(len(a_total)),
        "mae_kwh": float(np.abs(a_total - p_total).mean()),
        "rmse_kwh": float(math.sqrt(np.mean((a_total - p_total) ** 2))),
        "wape_pct": float(np.abs(a_total - p_total).sum() / denom * 100.0) if denom > 0 else float("nan"),
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="acn_caltech_2019.csv")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--timestep-minutes", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "forecast_stepwise_oof.csv")
    parser.add_argument("--oracle-output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "mpc" / "oracle_stepwise_h6.csv")
    parser.add_argument("--metrics-output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "mpc" / "forecast_stepwise_metrics.csv")
    parser.add_argument("--config-output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "mpc" / "stepwise_forecast_config.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace this new artifact")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if 60 % args.timestep_minutes != 0:
        raise ValueError("--timestep-minutes must divide one hour")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame, targets = build_frame(args.data_path, args.horizon, args.timestep_minutes)
    input_cols = LAG_FEATURE_COLS + CALENDAR_COLS
    n = len(frame)
    boundaries = np.linspace(0, n, args.folds + 1, dtype=int)
    predictions = np.full((n, args.horizon), np.nan, dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []

    for fold in range(1, args.folds):
        train_end = int(boundaries[fold])
        test_start = int(boundaries[fold])
        test_end = int(boundaries[fold + 1])
        if train_end <= SEQ_LEN + 1 or test_end <= test_start:
            continue
        input_scaler = MinMaxScaler().fit(frame[input_cols].iloc[:train_end])
        target_scaler = MinMaxScaler().fit(targets[:train_end])
        scaled_inputs = input_scaler.transform(frame[input_cols])
        scaled_targets = target_scaler.transform(targets)
        x_all = np.stack([
            scaled_inputs[i - SEQ_LEN:i]
            for i in range(SEQ_LEN, n)
        ])
        y_all = scaled_targets[SEQ_LEN:]
        seq_train_end = train_end - SEQ_LEN
        validation_start = max(1, seq_train_end - max(1, seq_train_end // 10))
        seq_test_start = max(test_start - SEQ_LEN, 0)
        seq_test_end = min(test_end - SEQ_LEN, len(y_all))
        if seq_test_end <= seq_test_start or validation_start <= 0:
            continue
        started = time.perf_counter()
        model = train_stepwise(
            x_all[:validation_start],
            y_all[:validation_start],
            x_all[validation_start:seq_train_end],
            y_all[validation_start:seq_train_end],
            device,
            args.seed + fold,
            args.max_epochs,
            args.patience,
        )
        pred_scaled = predict_stepwise(model, x_all[seq_test_start:seq_test_end], device)
        predictions[test_start:test_end] = target_scaler.inverse_transform(pred_scaled)[:test_end - test_start]
        elapsed = time.perf_counter() - started
        fold_rows.append({
            "fold": fold,
            "train_rows": train_end,
            "test_rows": test_end - test_start,
            "best_epoch": int(model.best_epoch),
            "best_val_loss": float(model.best_val_loss),
            "seconds": float(elapsed),
        })
        print(
            f"fold {fold}/{args.folds - 1}: train={train_end} test={test_end - test_start} "
            f"epoch={model.best_epoch} val={model.best_val_loss:.6f} in {elapsed:.1f}s"
        )

    finite = np.isfinite(predictions).all(axis=1)
    if finite.sum() == 0:
        raise RuntimeError("no finite stepwise OOF predictions were produced")

    abs_idx = frame["abs_idx"].to_numpy(dtype=int)
    actual = targets
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    output = pd.DataFrame({
        "hour_index": abs_idx[finite],
        "timestamp": timestamps.iloc[finite].reset_index(drop=True),
    })
    for index in range(args.horizon):
        output[f"actual_h{index + 1}"] = actual[finite, index]
        output[f"predicted_h{index + 1}"] = predictions[finite, index]
    output["actual_total"] = actual[finite].sum(axis=1)
    output["predicted_total"] = predictions[finite].sum(axis=1)
    output["horizon_hours"] = args.horizon
    output["timestep_minutes"] = args.timestep_minutes
    output["input_last_index"] = output["hour_index"] - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    oracle = build_oracle_frame(args.data_path, args.horizon, args.timestep_minutes)
    args.oracle_output.parent.mkdir(parents=True, exist_ok=True)
    oracle.to_csv(args.oracle_output, index=False)

    starts = [7429, 7713, 7997, 8281, 8565]
    test_rows = np.isin(abs_idx[finite], np.concatenate([np.arange(s, s + 168) for s in starts]))
    metrics = metric_rows(actual[finite][test_rows], predictions[finite][test_rows], "test_weeks")
    metrics += metric_rows(actual[finite], predictions[finite], "all_oof")
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.metrics_output, index=False)

    config = {
        "data_path": args.data_path,
        "horizon": args.horizon,
        "timestep_minutes": args.timestep_minutes,
        "folds": args.folds,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "feature_columns": input_cols,
        "sequence_length": SEQ_LEN,
        "output_rows": int(len(output)),
        "first_timestamp": str(output["timestamp"].min()),
        "last_timestamp": str(output["timestamp"].max()),
        "folds_detail": fold_rows,
    }
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2))
    print(f"saved {args.output} ({len(output)} rows)")
    print(f"saved {args.oracle_output} ({len(oracle)} rows)")
    print(f"saved {args.metrics_output}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
