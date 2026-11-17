"""Generate blocked expanding-fold BiLSTM forecasts for the training period."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_demand_forecaster import build_target_series, predict_bilstm, train_bilstm
from test_alternative_targets import (
    CALENDAR_COLS,
    LAG_FEATURE_COLS,
    SEQ_LEN,
    add_calendar,
    add_lag_features,
    build_rolling_6h_ahead,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="acn_caltech_2019.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "forecast_oof.npy")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--timestep-minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; use --force to replace this new OOF artifact")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if 60 % args.timestep_minutes != 0:
        raise ValueError("--timestep-minutes must divide one hour")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    series = build_target_series(args.data_path, timestep_minutes=args.timestep_minutes)
    demand = series["demand"].to_numpy()
    steps_per_hour = 60 // args.timestep_minutes
    horizon_steps = args.horizon * steps_per_hour
    target = build_rolling_6h_ahead(demand, horizon_steps)
    valid_end = len(demand) - horizon_steps
    frame = pd.DataFrame(
        {
            "timestamp": series["timestamp"].iloc[:valid_end],
            "target": target[:valid_end],
            "feat_src": demand[:valid_end],
        }
    )
    frame = add_calendar(frame)
    frame = add_lag_features(frame, "feat_src", steps_per_hour=steps_per_hour)
    frame = frame.dropna().reset_index(drop=True)
    input_cols = LAG_FEATURE_COLS + CALENDAR_COLS
    n = len(frame)
    boundaries = np.linspace(0, n, args.folds + 1, dtype=int)
    predictions = np.full(n, np.nan, dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(1, args.folds):
        train_end = boundaries[fold]
        test_start, test_end = boundaries[fold], boundaries[fold + 1]
        if train_end <= SEQ_LEN or test_end <= test_start:
            continue
        input_scaler = MinMaxScaler().fit(frame[input_cols].iloc[:train_end])
        target_scaler = MinMaxScaler().fit(frame[["target"]].iloc[:train_end])
        scaled_inputs = input_scaler.transform(frame[input_cols])
        scaled_target = target_scaler.transform(frame[["target"]]).ravel()
        X = np.array([scaled_inputs[i:i + SEQ_LEN] for i in range(n - SEQ_LEN)])
        y = scaled_target[SEQ_LEN:]
        seq_train_end = train_end - SEQ_LEN
        seq_test_start = max(test_start - SEQ_LEN, 0)
        seq_test_end = min(test_end - SEQ_LEN, len(y))
        if seq_test_end <= seq_test_start:
            continue
        validation_start = max(1, seq_train_end - max(1, seq_train_end // 10))
        model = train_bilstm(
            X[:validation_start],
            y[:validation_start],
            X[validation_start:seq_train_end],
            y[validation_start:seq_train_end],
            device,
        )
        pred_scaled = predict_bilstm(model, X[seq_test_start:seq_test_end], device)
        predictions[test_start:test_end] = target_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()[: test_end - test_start]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, predictions)
    print(f"Saved {args.output}; {np.isfinite(predictions).sum()} / {len(predictions)} rows predicted")


if __name__ == "__main__":
    main()
