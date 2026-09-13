# ==========================================
# ACN-Data: are smoother, controller-relevant targets more forecastable?
# ==========================================
# The original target (build_demand_forecaster.py: hourly arrival
# demand) is 55.96% zero and no model beat WAPE ~100%. Tests two
# alternative targets, same real Caltech 2019 cache, same
# chronological split and model ladder (persistence, seasonal-naive
# t-24/t-168, LightGBM, BiLSTM).
#
#   Target A - rolling 6h-ahead demand: sum(arrival demand, t+1..t+6).
#     A forward-looking sum, so anything used to predict it must be
#     fully observed BY hour t: features are built from the raw
#     arrival-demand series (safe lags 1/24/168), NOT from target A's
#     own lag-1 (target_A[t-1] = sum(demand[t..t+5]), which itself
#     depends on demand[t+1..t+5] - future information relative to
#     hour t, so using it as a lag-1 "persistence" feature would leak).
#     "Persistence" for this target is instead the most recently
#     FULLY REALIZED 6h sum: sum(demand[t-5..t]).
#
#   Target B - concurrent connected demand: sum of requested_kwh over
#     all sessions connected at hour t (data.active_by_step), i.e. the
#     real load already on the controller's plate, independent of any
#     charging policy. This is a real-time-observable current-state
#     quantity (not forward-summed), so it's handled exactly like the
#     original target: persistence/seasonal-naive/lag features all
#     reference target_B's own past values directly - no special
#     leakage handling needed.
# ==========================================

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from acn_data_loader import load_acn_sessions
from build_demand_forecaster import (
    Forecaster, build_sequences, build_target_series, inverse_demand,
    predict_bilstm, train_bilstm,
)
from lightgbm import LGBMRegressor
from metrics import mae, rmse, wape
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "acn_caltech_2019.csv"
SEQ_LEN = 24
HORIZON_A = 6

CALENDAR_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
LAG_FEATURE_COLS = ["lag1", "lag24", "lag168", "rolling_mean_24", "rolling_std_24"]


# ==========================================
# Target construction
# ==========================================

def build_concurrent_demand_series(data) -> np.ndarray:
    """Target B: total requested_kwh of all sessions connected at each
    hour - the real load already on the controller's plate, computed
    directly from data (policy-independent: it's the ORIGINAL demand
    of connected sessions, not net of any hypothetical charging)."""
    n = data.n_steps
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        for idx in data.active_by_step[t]:
            out[t] += data.sessions[idx].requested_kwh
    return out


def build_rolling_6h_ahead(demand: np.ndarray, horizon: int) -> np.ndarray:
    """Target A: sum(demand[t+1 .. t+horizon]) for each t. The last
    `horizon` entries are NaN (their window runs past the end of the
    series) and must be dropped before evaluation."""
    s = pd.Series(demand)
    return s.rolling(horizon).sum().shift(-horizon).values


# ==========================================
# Calendar features (shared, series-independent)
# ==========================================

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def add_lag_features(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Lag/rolling features computed from `source_col`, all safely
    past-referencing (shift(1) before rolling excludes the current row)."""
    df["lag1"] = df[source_col].shift(1)
    df["lag24"] = df[source_col].shift(24)
    df["lag168"] = df[source_col].shift(168)
    df["rolling_mean_24"] = df[source_col].shift(1).rolling(24).mean()
    df["rolling_std_24"] = df[source_col].shift(1).rolling(24).std()
    return df


# ==========================================
# Model ladder (shared across all 3 targets)
# ==========================================

def run_ladder(
    name: str,
    timestamps: pd.Series,
    target: np.ndarray,
    feature_source: np.ndarray,
    persistence_override: np.ndarray | None,
    train_end: int,
    val_end: int,
    valid_end: int,
) -> tuple[pd.DataFrame, dict]:
    """Runs persistence/seasonal-naive/LightGBM/BiLSTM against `target`,
    evaluated on [val_end, valid_end) (valid_end <= len(target), lets
    a forward-looking target trim unevaluable trailing rows).

    `feature_source` is the series lag/rolling FEATURES are computed
    from - equal to `target` for targets that are real-time-observable
    (original, Target B), but a different, safely-past-only series for
    a forward-summed target (Target A), to avoid leaking the target's
    own future-dependent lag values back in as a feature.
    """
    n = valid_end
    df = pd.DataFrame({"timestamp": timestamps[:n], "target": target[:n],
                        "feat_src": feature_source[:n]})
    df = add_calendar(df)
    df = add_lag_features(df, "feat_src")
    feat_df = df.dropna().reset_index(drop=True)
    n_dropped = n - len(feat_df)

    f_train_end = train_end - n_dropped
    f_val_end = val_end - n_dropped
    assert f_train_end > 0, f"{name}: warm-up ate into the training window"

    test_len = n - val_end
    actual_test = target[val_end:n]

    if persistence_override is not None:
        persistence = persistence_override[val_end:n]
    else:
        persistence = target[val_end - 1:n - 1]
    seasonal_24 = target[val_end - 24:n - 24]
    seasonal_168 = target[val_end - 168:n - 168]
    assert len(persistence) == len(seasonal_24) == len(seasonal_168) == test_len

    # ---- LightGBM ----
    lgbm_cols = LAG_FEATURE_COLS + CALENDAR_COLS
    X_all = feat_df[lgbm_cols].values
    y_all = feat_df["target"].values
    lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                         random_state=42, verbosity=-1)
    lgbm.fit(X_all[:f_train_end], y_all[:f_train_end])
    lgbm_pred_test = lgbm.predict(X_all)[f_val_end:]
    assert len(lgbm_pred_test) == test_len

    # ---- BiLSTM ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_cols = ["target"] + lgbm_cols
    scaler = MinMaxScaler()
    scaler.fit(feat_df[seq_cols].iloc[:f_train_end])
    scaled = scaler.transform(feat_df[seq_cols])

    X_seq, y_seq = build_sequences(scaled, SEQ_LEN)
    seq_train_end = f_train_end - SEQ_LEN
    seq_val_end = f_val_end - SEQ_LEN
    assert seq_train_end > 0

    model = train_bilstm(
        X_seq[:seq_train_end], y_seq[:seq_train_end],
        X_seq[seq_train_end:seq_val_end], y_seq[seq_train_end:seq_val_end],
        device,
    )
    pred_scaled_test = predict_bilstm(model, X_seq[seq_val_end:], device)
    bilstm_pred_test = inverse_demand(pred_scaled_test, scaler, len(seq_cols))
    assert len(bilstm_pred_test) == test_len

    actual_from_seq = inverse_demand(y_seq[seq_val_end:], scaler, len(seq_cols))
    assert np.allclose(actual_from_seq, actual_test, atol=1e-6), \
        f"{name}: BiLSTM test-window alignment mismatch"

    models = {
        "Persistence": persistence,
        "Seasonal-naive (t-24)": seasonal_24,
        "Seasonal-naive (t-168)": seasonal_168,
        "LightGBM": lgbm_pred_test,
        "BiLSTM": bilstm_pred_test,
    }
    rows = []
    for mname, pred in models.items():
        rows.append({
            "Model": mname,
            "RMSE": round(rmse(actual_test, pred), 4),
            "MAE": round(mae(actual_test, pred), 4),
            "WAPE": round(wape(actual_test, pred), 3),
        })
    table = pd.DataFrame(rows)
    return table, {"actual": actual_test, **models}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=HORIZON_A,
                        help="Target A lead time in hours (default: 6, the "
                             "original/paper horizon). Only affects Target A "
                             "and its output filename; Target B is unchanged.")
    args = parser.parse_args()
    horizon = args.horizon
    metrics_a_path = ("acn_target_a_metrics.csv" if horizon == HORIZON_A
                       else f"acn_target_a_h{horizon}_metrics.csv")

    data = load_acn_sessions(DATA_PATH, timestep_minutes=60)
    series = build_target_series(DATA_PATH)   # original arrival-demand series + timestamps
    demand = series["demand"].values
    timestamps = series["timestamp"]
    n = len(series)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    print(f"Chronological split: train=[0,{train_end}) val=[{train_end},{val_end}) "
          f"test=[{val_end},{n})")

    # ==================== Target A ====================
    print("\n" + "=" * 60)
    print(f"TARGET A - rolling {horizon}h-ahead arrival demand")
    print("=" * 60)
    target_a = build_rolling_6h_ahead(demand, horizon)
    valid_end_a = n - horizon   # last `horizon` rows are NaN (future runs off the end)
    pct_zero_a = 100 * np.mean(target_a[:valid_end_a] == 0)
    print(f"valid range: [0, {valid_end_a}) ({n - valid_end_a} trailing rows dropped - "
          f"their window runs past the end of the data)")
    print(f"% zero hours: {pct_zero_a:.2f}%  mean={np.nanmean(target_a[:valid_end_a]):.3f}  "
          f"std={np.nanstd(target_a[:valid_end_a]):.3f}")

    # persistence override: most recently FULLY REALIZED `horizon`h sum,
    # sum(demand[t-horizon+1..t]) - safe (no future leakage), unlike target_a[t-1]
    trailing_horizon_sum = pd.Series(demand).rolling(horizon).sum().values
    table_a, _ = run_ladder(
        "Target A", timestamps, target_a, feature_source=demand,
        persistence_override=trailing_horizon_sum,
        train_end=train_end, val_end=val_end, valid_end=valid_end_a,
    )
    print(f"\ntest window for Target A: [{val_end}, {valid_end_a}) "
          f"({valid_end_a - val_end} hours, {n - valid_end_a} fewer than the "
          f"canonical 1,311h test window due to the {horizon}h-ahead trim)")
    print(table_a.to_string(index=False))

    # ==================== Target B ====================
    print("\n" + "=" * 60)
    print("TARGET B - concurrent connected demand")
    print("=" * 60)
    target_b = build_concurrent_demand_series(data)
    assert len(target_b) == n
    pct_zero_b = 100 * np.mean(target_b == 0)
    print(f"% zero hours: {pct_zero_b:.2f}%  mean={target_b.mean():.3f}  std={target_b.std():.3f}")

    table_b, _ = run_ladder(
        "Target B", timestamps, target_b, feature_source=target_b,
        persistence_override=None,
        train_end=train_end, val_end=val_end, valid_end=n,
    )
    print(f"\ntest window for Target B: [{val_end}, {n}) ({n - val_end} hours)")
    print(table_b.to_string(index=False))

    # ==================== Comparison vs original ====================
    orig = pd.read_csv("acn_demand_forecast_metrics.csv")
    orig_wape = dict(zip(orig["Model"], orig["WAPE"]))

    def verdict(name, table, pct_zero):
        wape_bilstm = table.loc[table["Model"] == "BiLSTM", "WAPE"].iloc[0]
        wape_lgbm = table.loc[table["Model"] == "LightGBM", "WAPE"].iloc[0]
        best_naive = table.loc[table["Model"].isin(
            ["Persistence", "Seasonal-naive (t-24)", "Seasonal-naive (t-168)"]), "WAPE"].min()
        best_overall = table["WAPE"].min()
        print(f"\n--- {name} verdict ---")
        print(f"  % zero hours: {pct_zero:.2f}% (original: 55.96%)")
        print(f"  best WAPE this target: {best_overall:.2f}% "
              f"(original best: {min(orig_wape.values()):.2f}%, LightGBM)")
        print(f"  (a) more forecastable than original? "
              f"{'YES' if best_overall < min(orig_wape.values()) else 'NO'}")
        print(f"  (b) LightGBM still best model? {lgbm_is_best(table)}")
        print(f"  (c) BiLSTM beats best naive? "
              f"{'YES' if wape_bilstm < best_naive else 'NO'} "
              f"(BiLSTM={wape_bilstm:.2f}%, best naive={best_naive:.2f}%)")

    def lgbm_is_best(table):
        return table.loc[table["WAPE"].idxmin(), "Model"] == "LightGBM"

    verdict(f"Target A ({horizon}h-ahead)", table_a, pct_zero_a)
    verdict("Target B (concurrent)", table_b, pct_zero_b)

    table_a.to_csv(metrics_a_path, index=False)
    table_b.to_csv("acn_target_b_metrics.csv", index=False)
    print(f"\nSaved {metrics_a_path}, acn_target_b_metrics.csv")


if __name__ == "__main__":
    main()
