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
#   - Test-window rows ARE genuine held-out predictions - the scaler
#     and BiLSTM are both fit on TRAIN rows only (see
#     test_alternative_targets.py / build_demand_forecaster.py), and
#     these test-window rows are never touched by training. This is
#     verified below: this file's test-window WAPE is asserted to
#     match the 23.799% already reported by test_alternative_targets.py
#     (saved in acn_target_a_metrics.csv) - the same code path, same
#     seed, same train-only scaler fit, just predicting a wider range
#     of rows this time. (acn_demand_forecasts_test.csv is a DIFFERENT
#     target - the original per-hour arrival demand, not Target A's
#     6h-ahead sum - so it is not a valid comparison file for this
#     target and is not used here.)
#
# Output: acn_target_a_bilstm_full.csv, columns:
#   hour_index, timestamp, actual, predicted
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from build_demand_forecaster import (
    build_sequences, build_target_series, inverse_demand, predict_bilstm, train_bilstm,
)
from test_alternative_targets import (
    CALENDAR_COLS, HORIZON_A, LAG_FEATURE_COLS, SEQ_LEN,
    add_calendar, add_lag_features, build_rolling_6h_ahead,
)
from metrics import wape, rmse, mae

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "acn_caltech_2019.csv"
OUT_CSV = "acn_target_a_bilstm_full.csv"
REFERENCE_WAPE = 23.799   # from test_alternative_targets.py / acn_target_a_metrics.csv


def main() -> None:
    series = build_target_series(DATA_PATH)
    demand = series["demand"].values
    timestamps = series["timestamp"]
    n = len(series)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    target_a = build_rolling_6h_ahead(demand, HORIZON_A)
    valid_end_a = n - HORIZON_A   # last HORIZON_A rows: window runs past the end of the data

    df = pd.DataFrame({"timestamp": timestamps[:valid_end_a],
                       "target": target_a[:valid_end_a],
                       "feat_src": demand[:valid_end_a]})
    df = add_calendar(df)
    df = add_lag_features(df, "feat_src")
    feat_df = df.dropna().reset_index(drop=True)
    n_dropped = valid_end_a - len(feat_df)   # warm-up rows lost to lag168/rolling

    f_train_end = train_end - n_dropped
    f_val_end = val_end - n_dropped
    assert f_train_end > 0 and f_val_end > f_train_end

    seq_cols = ["target"] + LAG_FEATURE_COLS + CALENDAR_COLS
    scaler = MinMaxScaler()
    scaler.fit(feat_df[seq_cols].iloc[:f_train_end])   # train-only fit - no leakage
    scaled = scaler.transform(feat_df[seq_cols])

    X_seq, y_seq = build_sequences(scaled, SEQ_LEN)
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
    pred_all = inverse_demand(pred_scaled_all, scaler, len(seq_cols))
    actual_all = inverse_demand(y_seq, scaler, len(seq_cols))

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
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} ({len(out)} rows, hour_index {out['hour_index'].iloc[0]} to "
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
    print(f"  reference (test_alternative_targets.py, same code path/seed): WAPE={REFERENCE_WAPE}%")

    assert abs(recomputed_wape - REFERENCE_WAPE) < 1.0, (
        f"test-window WAPE {recomputed_wape:.3f}% does not match the "
        f"already-established genuine held-out figure {REFERENCE_WAPE}% "
        "(tolerance 1.0 pp) - something about the split/features/training "
        "changed between runs; investigate before trusting this file"
    )
    print("[PASS] test-window predictions match the genuine held-out result "
          "within tolerance - confirmed NOT leaked/re-fit on test data.")

    train_val_rows = int((~test_mask).sum())
    print(f"\nNOTE: {train_val_rows} train/val rows in this file are IN-SAMPLE "
          "(BiLSTM saw them in training) - used only as an RL observation "
          "feature, never as a reported metric. See file header.")


if __name__ == "__main__":
    main()
