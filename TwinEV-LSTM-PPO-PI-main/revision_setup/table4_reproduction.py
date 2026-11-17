"""Reproduce the forecast-alignment quantities used by diagnose_forecast.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_2x2_run import shared_setup
from lead_time_distribution import full_year_rollout


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def block_bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int = 24,
    resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    if len(x) < block_size:
        raise ValueError("Test-week rows are shorter than one 24-hour bootstrap block")
    rng = np.random.default_rng(seed)
    blocks = [
        np.arange(start, min(start + block_size, len(x)))
        for start in range(0, len(x), block_size)
    ]
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        values.append(corr(x[indices], y[indices]))
    lower, upper = np.nanquantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    dow = timestamp.dt.dayofweek
    result = frame.copy()
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    result["hour_of_week"] = dow * 24 + timestamp.dt.hour
    return result


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hour_sin"] = np.sin(2 * np.pi * result["hour_of_week"] / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour_of_week"] / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * (result["hour_of_week"] // 24) / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * (result["hour_of_week"] // 24) / 7.0)
    return result


def build_records(
    forecast: pd.DataFrame,
    draw: np.ndarray,
    timestamps: pd.Series,
    scope: str,
    horizons: tuple[int, ...],
    train_draw: np.ndarray,
) -> list[dict]:
    forecast = forecast.copy()
    forecast["timestamp_key"] = pd.to_datetime(forecast["timestamp"], utc=True)
    lookup = forecast.set_index("timestamp_key")["predicted"]
    rows = []
    for horizon in horizons:
        for index in range(len(draw) - horizon):
            key = pd.Timestamp(timestamps.iloc[index])
            if key not in lookup.index:
                continue
            future = draw[index + 1 : index + 1 + horizon]
            rows.append(
                {
                    "scope": scope,
                    "horizon_hours": horizon,
                    "timestamp": key,
                    "predicted": float(lookup.loc[key]),
                    "future_max_draw_kw": float(np.max(future)),
                    "future_violation": float(np.any(future > 50.0)),
                    "persistence": float(draw[index]),
                    "seasonal_naive_24": (
                        float(draw[index - 24]) if index >= 24 else np.nan
                    ),
                    "seasonal_naive_168": (
                        float(draw[index - 168]) if index >= 168 else np.nan
                    ),
                }
            )
    return rows


def residualize(frame: pd.DataFrame, train_mask: pd.Series, column: str) -> np.ndarray:
    means = frame.loc[train_mask].groupby("hour_of_week")[column].mean()
    return frame[column].to_numpy(float) - frame["hour_of_week"].map(means).to_numpy(float)


def evaluate_scope(
    records: list[dict],
    scope: str,
    output_rows: list[dict],
) -> None:
    frame = add_calendar(pd.DataFrame(records))
    if frame.empty:
        raise ValueError(f"No aligned records for {scope}")
    train_mask = frame["scope"].eq("full") & (
        pd.to_datetime(frame["timestamp"], utc=True)
        < pd.to_datetime(frame["timestamp"], utc=True).min()
        + pd.Timedelta(seconds=0)
    )
    # Full-series training boundary is set by the source timeline, not by
    # the filtered forecast rows.
    if scope == "full":
        boundary = frame["timestamp"].min() + (
            frame["timestamp"].max() - frame["timestamp"].min()
        ) * 0.70
        train_mask = pd.to_datetime(frame["timestamp"], utc=True) < boundary
    else:
        train_mask = pd.Series(False, index=frame.index)

    for horizon, group in frame.groupby("horizon_hours", sort=True):
        if scope == "test_weeks":
            # Training means are supplied by the full-series records below.
            train_group = None
        else:
            train_group = group.loc[train_mask.loc[group.index]]
        for target in ("future_max_draw_kw", "future_violation"):
            if scope == "test_weeks":
                # The test-week rows are residualized against their
                # chronological training-period hour-of-week means.
                train_group = frame.loc[
                    (frame["scope"] == "full")
                    & (pd.to_datetime(frame["timestamp"], utc=True)
                       < pd.to_datetime(frame["timestamp"], utc=True).min()
                       + (pd.to_datetime(frame["timestamp"], utc=True).max()
                          - pd.to_datetime(frame["timestamp"], utc=True).min()) * 0.70)
                    & (frame["horizon_hours"] == horizon)
                ]
            means = train_group.groupby("hour_of_week")[target].mean()
            y = group[target].to_numpy(float)
            y_resid = y - group["hour_of_week"].map(means).to_numpy(float)
            model = group["predicted"].to_numpy(float)
            model_means = train_group.groupby("hour_of_week")["predicted"].mean()
            model_resid = model - group["hour_of_week"].map(model_means).to_numpy(float)
            persistence = group["future_max_draw_kw"].shift(0).to_numpy(float)
            # Persistence is the draw observed at the forecast origin. It is
            # filled by the caller when present in the records.
            persistence = group["persistence"].to_numpy(float)
            persistence_means = train_group.groupby("hour_of_week")["persistence"].mean()
            persistence_resid = persistence - group["hour_of_week"].map(
                persistence_means
            ).to_numpy(float)
            output_rows.append(
                {
                    "scope": scope,
                    "horizon_hours": int(horizon),
                    "target": target,
                    "n": len(group),
                    "bilstm_raw_correlation": corr(model, y),
                    "bilstm_hour_of_week_residual_correlation": corr(model_resid, y_resid),
                    "persistence_raw_correlation": corr(persistence, y),
                    "persistence_hour_of_week_residual_correlation": corr(
                        persistence_resid, y_resid
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument("--forecast-csv", type=Path, default=ROOT / "acn_target_a_bilstm_full.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "table4_reproduction.csv",
    )
    args = parser.parse_args()

    forecast = pd.read_csv(args.forecast_csv)
    if not {"timestamp", "predicted"}.issubset(forecast.columns):
        raise ValueError("Forecast must contain timestamp and predicted")
    rollout = full_year_rollout(args.data_path, 50.0)
    full_setup = shared_setup()
    full_timestamps = pd.Series(
        pd.date_range(
            start=pd.to_datetime(forecast["timestamp"], utc=True).min()
            - pd.Timedelta(hours=int(forecast["hour_index"].min())),
            periods=len(rollout["achievable_kw"]),
            freq="h",
        )
    )
    # Use the data loader's actual timeline where available.
    start_time = pd.Timestamp(full_setup["full_data"].start_time)
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    full_timestamps = pd.Series(
        [
            start_time + step * pd.Timedelta(minutes=60)
            for step in range(len(rollout["achievable_kw"]))
        ]
    )
    full_draw = rollout["achievable_kw"]
    full_rows = build_records(
        forecast, full_draw, full_timestamps, "full", (1, 3, 6), full_draw
    )
    full_frame = pd.DataFrame(full_rows)
    if full_frame.empty:
        raise ValueError("No full-series forecast rows aligned with the rollout")
    full_frame["hour_of_week"] = (
        pd.to_datetime(full_frame["timestamp"], utc=True).dt.dayofweek * 24
        + pd.to_datetime(full_frame["timestamp"], utc=True).dt.hour
    )
    test_rows = []
    for week in [*__import__("lead_time_distribution").test_week_rollouts(50.0)]:
        timestamps = pd.Series(
            [pd.Timestamp(t).tz_localize("UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t)
             for t in week["timestamps"]]
        )
        rows = build_records(forecast, week["achievable_kw"], timestamps, "test_weeks", (1, 3, 6), full_draw)
        test_rows.extend(rows)
    test_frame = pd.DataFrame(test_rows)
    if test_frame.empty:
        raise ValueError("No test-week forecast rows aligned with the rollout")
    test_frame["hour_of_week"] = (
        pd.to_datetime(test_frame["timestamp"], utc=True).dt.dayofweek * 24
        + pd.to_datetime(test_frame["timestamp"], utc=True).dt.hour
    )
    full_draw_by_time = dict(zip(full_timestamps, full_draw))
    for lag, column in ((24, "seasonal_naive_24"), (168, "seasonal_naive_168")):
        test_frame[column] = test_frame["timestamp"].map(
            lambda value: full_draw_by_time.get(
                pd.Timestamp(value) - pd.Timedelta(hours=lag),
                np.nan,
            )
        )
    full_frame = add_model_features(full_frame)
    test_frame = add_model_features(test_frame)
    # LightGBM is a chronological calendar-only regression comparator.
    feature_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    output_rows: list[dict] = []
    for scope, frame in (("full", full_frame), ("test_weeks", test_frame)):
        for horizon, group in frame.groupby("horizon_hours"):
            train = full_frame[full_frame["horizon_hours"] == horizon].iloc[
                : int((full_frame["horizon_hours"] == horizon).sum() * 0.70)
            ]
            for target in ("future_max_draw_kw", "future_violation"):
                lgb = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    random_state=42,
                    verbosity=-1,
                )
                lgb.fit(train[feature_cols], train[target])
                lgb_prediction = lgb.predict(group[feature_cols])
                means = train.groupby("hour_of_week")[target].mean()
                y = group[target].to_numpy(float)
                yr = y - group["hour_of_week"].map(means).to_numpy(float)
                model_means = train.groupby("hour_of_week")["predicted"].mean()
                persistence_means = train.groupby("hour_of_week")["persistence"].mean()
                lgb_means = train.assign(lgb_prediction=lgb.predict(train[feature_cols])).groupby(
                    "hour_of_week"
                )["lgb_prediction"].mean()
                seasonal24_means = train.groupby("hour_of_week")["seasonal_naive_24"].mean()
                seasonal168_means = train.groupby("hour_of_week")["seasonal_naive_168"].mean()
                seasonal24 = group["seasonal_naive_24"].to_numpy(float)
                seasonal168 = group["seasonal_naive_168"].to_numpy(float)
                output_rows.append({
                    "scope": scope,
                    "horizon_hours": int(horizon),
                    "target": target,
                    "n": len(group),
                    "bilstm_raw_correlation": corr(group["predicted"].to_numpy(float), y),
                    "bilstm_hour_of_week_residual_correlation": corr(
                        group["predicted"].to_numpy(float)
                        - group["hour_of_week"].map(model_means).to_numpy(float),
                        yr,
                    ),
                    "lightgbm_raw_correlation": corr(lgb_prediction, y),
                    "lightgbm_hour_of_week_residual_correlation": corr(
                        lgb_prediction - group["hour_of_week"].map(lgb_means).to_numpy(float),
                        yr,
                    ),
                    "persistence_raw_correlation": corr(
                        group["persistence"].to_numpy(float), y
                    ),
                    "persistence_hour_of_week_residual_correlation": corr(
                        group["persistence"].to_numpy(float)
                        - group["hour_of_week"].map(persistence_means).to_numpy(float),
                        yr,
                    ),
                    "seasonal_naive_24_raw_correlation": corr(seasonal24, y),
                    "seasonal_naive_24_hour_of_week_residual_correlation": corr(
                        seasonal24
                        - group["hour_of_week"].map(seasonal24_means).to_numpy(float),
                        yr,
                    ),
                    "seasonal_naive_168_raw_correlation": corr(seasonal168, y),
                    "seasonal_naive_168_hour_of_week_residual_correlation": corr(
                        seasonal168
                        - group["hour_of_week"].map(seasonal168_means).to_numpy(float),
                        yr,
                    ),
                })
                if scope == "test_weeks":
                    residual = (
                        group["predicted"].to_numpy(float)
                        - group["hour_of_week"].map(model_means).to_numpy(float)
                    )
                    ci_lower, ci_upper = block_bootstrap_corr_ci(residual, yr)
                    output_rows[-1]["bilstm_residual_correlation_ci95_lower"] = ci_lower
                    output_rows[-1]["bilstm_residual_correlation_ci95_upper"] = ci_upper
                else:
                    output_rows[-1]["bilstm_residual_correlation_ci95_lower"] = np.nan
                    output_rows[-1]["bilstm_residual_correlation_ci95_upper"] = np.nan
                output_rows[-1]["residual_bootstrap_blocks_hours"] = 24
                output_rows[-1]["residual_bootstrap_resamples"] = 2000
    result = pd.DataFrame(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    result.to_csv(args.output.with_name("forecast_residual_correlation_extended.csv"), index=False)
    print(result.to_string(index=False))
    print("\nForecast-baseline definitions by horizon:")
    for horizon in (1, 3, 6):
        print(
            f"  {horizon}h: persistence = achievable_draw(t), lag 0, "
            f"compared with the target over t+1..t+{horizon}; "
            f"seasonal_naive_24 = achievable_draw(t-24), lag 24 hours, "
            f"compared with the same t+1..t+{horizon} target; "
            f"seasonal_naive_168 = achievable_draw(t-168), lag 168 hours, "
            f"compared with the same t+1..t+{horizon} target."
        )
    print("\nThe 0.90 target is future maximum achievable draw; the 0.81 target is future violation occurrence.")
    print("The prior n=8542 is the aligned full-series forecast sample and includes training-period rows.")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
