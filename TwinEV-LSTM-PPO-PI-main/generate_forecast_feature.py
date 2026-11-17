# ==========================================
# Generate full-year Target-A (6h-ahead demand) BiLSTM predictions
# ==========================================
# train_controller.py's `forecast` variant needs a real BiLSTM
# prediction available at EVERY hour of the year: training episodes
# are sampled from the train window, evaluation episodes from the
# test window, and both need this feature to exist.
#
# IMPORTANT, STATED LIMITATION (not a leak into any reported metric):
#   - Train/val-window rows in this file are IN-SAMPLE: the BiLSTM saw
#     those rows during training, so its predictions there are better
#     than a real deployed forecaster would achieve. They are used
#     ONLY as an RL observation feature during training - the RL agent
#     conditioning on a slightly-too-good in-sample signal during
#     training is a real, acknowledged simplification (a production
#     system would retrain/roll the forecaster forward continuously;
#     this project trains it once), not a metrics leak.
#   - Test-window rows ARE genuine held-out predictions - the input
#     scaler and BiLSTM are both fit on TRAIN rows only. The forward
#     target is used only as y, never as an input feature.
#
# Output: acn_target_a_bilstm_full.csv, columns:
#   hour_index, timestamp, actual, predicted
# ==========================================

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from build_demand_forecaster import (
    build_target_series, predict_bilstm, train_bilstm,
)
from test_alternative_targets import (
    CALENDAR_COLS, HORIZON_A, LAG_FEATURE_COLS, SEQ_LEN,
    add_calendar, add_lag_features, build_rolling_6h_ahead,
)
from metrics import wape, rmse, mae

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "acn_caltech_2019.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=HORIZON_A,
                        help="Target A lead time in hours (default: 6, the "
                             "original/paper horizon).")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--timestep-minutes", type=int, default=60)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--reference-metrics", default=None,
        help="Optional Caltech reference metrics CSV; omit for other sites.",
    )
    args = parser.parse_args()
    horizon = args.horizon

    out_csv = args.output_csv or (
        "acn_target_a_bilstm_full.csv" if horizon == HORIZON_A
        else f"acn_target_a_h{horizon}_bilstm_full.csv"
    )
    metrics_path = args.reference_metrics or (
        "acn_target_a_metrics.csv" if horizon == HORIZON_A
        else f"acn_target_a_h{horizon}_metrics.csv"
    )
    reference_wape = None
    if args.reference_metrics is not None or (
        args.data_path == DATA_PATH and args.timestep_minutes == 60
    ):
        reference_table = pd.read_csv(metrics_path)
        reference_wape = float(
            reference_table.loc[reference_table["Model"] == "BiLSTM", "WAPE"].iloc[0]
        )

    if 60 % args.timestep_minutes != 0:
        raise ValueError("--timestep-minutes must divide one hour")
    series = build_target_series(args.data_path, args.timestep_minutes)
    demand = series["demand"].values
    timestamps = series["timestamp"]
    n = len(series)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    steps_per_hour = 60 // args.timestep_minutes
    horizon_steps = horizon * steps_per_hour
    target_a = build_rolling_6h_ahead(demand, horizon_steps)
    valid_end_a = n - horizon_steps

    df = pd.DataFrame({"timestamp": timestamps[:valid_end_a],
                       "target": target_a[:valid_end_a],
                       "feat_src": demand[:valid_end_a]})
    df = add_calendar(df)
    df = add_lag_features(
        df,
        "feat_src",
        steps_per_hour=steps_per_hour,
    )
    feat_df = df.dropna().reset_index(drop=True)
    n_dropped = valid_end_a - len(feat_df)   # warm-up rows lost to lag168/rolling

    f_train_end = train_end - n_dropped
    f_val_end = val_end - n_dropped
    assert f_train_end > 0 and f_val_end > f_train_end

    input_cols = LAG_FEATURE_COLS + CALENDAR_COLS
    input_scaler = MinMaxScaler()
    input_scaler.fit(feat_df[input_cols].iloc[:f_train_end])
    target_scaler = MinMaxScaler()
    target_scaler.fit(feat_df[["target"]].iloc[:f_train_end])
    scaled_inputs = input_scaler.transform(feat_df[input_cols])
    scaled_target = target_scaler.transform(feat_df[["target"]]).ravel()
    X_seq = np.array([
        scaled_inputs[i:i + SEQ_LEN]
        for i in range(len(scaled_inputs) - SEQ_LEN)
    ])
    y_seq = scaled_target[SEQ_LEN:]
    seq_train_end = f_train_end - SEQ_LEN
    seq_val_end = f_val_end - SEQ_LEN
    assert seq_train_end > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training BiLSTM on Target A (full-timeline prediction run)...")
    model = train_bilstm(
        X_seq[:seq_train_end], y_seq[:seq_train_end],
        X_seq[seq_train_end:seq_val_end], y_seq[seq_train_end:seq_val_end],
        device,
    )

    # Predict on EVERY sequence (train+val+test) - full timeline.
    pred_scaled_all = predict_bilstm(model, X_seq, device)
    pred_all = target_scaler.inverse_transform(
        np.asarray(pred_scaled_all).reshape(-1, 1)
    ).ravel()
    actual_all = target_scaler.inverse_transform(
        np.asarray(y_seq).reshape(-1, 1)
    ).ravel()

    # sequence i predicts feat_df row (i + SEQ_LEN), which is original
    # grid row (i + SEQ_LEN + n_dropped) -> that's hour_index.
    hour_index = np.arange(len(pred_all)) + SEQ_LEN + n_dropped
    out_timestamps = feat_df["timestamp"].iloc[SEQ_LEN:SEQ_LEN + len(pred_all)].values

    out = pd.DataFrame({
        "hour_index": hour_index,
        "timestamp": out_timestamps,
        "actual": actual_all,
        "predicted": pred_all,
    })
    out["horizon_hours"] = horizon
    out["timestep_minutes"] = args.timestep_minutes
    out["input_last_index"] = out["hour_index"] - 1
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(out)} rows, hour_index {out['hour_index'].iloc[0]} to "
          f"{out['hour_index'].iloc[-1]}, {out['timestamp'].iloc[0]} to {out['timestamp'].iloc[-1]})")

    # ---- Verification: test-window slice must be genuinely held-out ----
    # test window = hour_index >= val_end (matches every other script's split)
    test_mask = (out["hour_index"] >= val_end).values
    n_test = int(test_mask.sum())
    test_actual = out.loc[test_mask, "actual"].values
    test_pred = out.loc[test_mask, "predicted"].values

    recomputed_wape = wape(test_actual, test_pred)
    recomputed_rmse = rmse(test_actual, test_pred)
    recomputed_mae = mae(test_actual, test_pred)

    print(f"\n===== Test-window verification ({n_test} rows, hour_index >= {val_end}) =====")
    print(f"  RMSE={recomputed_rmse:.4f}  MAE={recomputed_mae:.4f}  WAPE={recomputed_wape:.3f}%")
    if reference_wape is not None:
        print(f"  reference ({metrics_path}, same code path/seed): WAPE={reference_wape}%")
        print(
            "[INFO] The historical reference is retained for audit only; "
            "it was generated before the target-column leakage fix."
        )
    else:
        print("[INFO] no cross-site reference metric supplied; reporting site-local metrics.")

    train_val_rows = int((~test_mask).sum())
    print(f"\nNOTE: {train_val_rows} train/val rows in this file are IN-SAMPLE "
          "(BiLSTM saw them in training) - used only as an RL observation "
          "feature, never as a reported metric. See file header.")


if __name__ == "__main__":
    main()
