"""Residual forecast alignment audit against future six-hour peak draw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lead_time_distribution import full_year_rollout


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument("--forecast-csv", type=Path, default=ROOT / "acn_target_a_bilstm_full.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "forecast_residual_correlation.csv")
    parser.add_argument("--horizon-hours", type=int, default=6)
    args = parser.parse_args()

    forecast = pd.read_csv(args.forecast_csv)
    required = {"hour_index", "predicted"}
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError(f"Forecast is missing columns: {sorted(missing)}")
    rollout = full_year_rollout(args.data_path, 50.0)
    achievable = rollout["achievable_kw"]
    target = np.full(len(achievable), np.nan)
    for index in range(len(achievable) - args.horizon_hours):
        target[index] = np.max(achievable[index + 1 : index + 1 + args.horizon_hours])
    forecast["hour_index"] = forecast["hour_index"].astype(int)
    forecast["future_max_draw_kw"] = forecast["hour_index"].map(
        pd.Series(target, index=np.arange(len(target)))
    )
    forecast = forecast.dropna(subset=["future_max_draw_kw"]).copy()
    forecast = forecast[
        (forecast["hour_index"] >= 168)
        & (forecast["hour_index"] < len(target) - args.horizon_hours)
    ].copy()
    if forecast.empty:
        raise ValueError("No aligned forecast/future-draw rows remain")

    train_end = int(len(achievable) * 0.70)
    train_rows = forecast["hour_index"] < train_end
    if not train_rows.any():
        raise ValueError("No training rows available for hour-of-week means")
    forecast["hour_of_week"] = forecast["hour_index"] % 168
    means = forecast.loc[train_rows].groupby("hour_of_week")["future_max_draw_kw"].mean()
    forecast["future_seasonal_mean"] = forecast["hour_of_week"].map(means)
    forecast["forecast_seasonal_mean"] = forecast.loc[train_rows].groupby(
        "hour_of_week"
    )["predicted"].transform("mean")
    forecast["forecast_seasonal_mean"] = forecast["hour_of_week"].map(
        forecast.loc[train_rows].groupby("hour_of_week")["predicted"].mean()
    )
    forecast["seasonal_naive"] = forecast["hour_index"].map(
        pd.Series(target, index=np.arange(len(target))).shift(168)
    )
    forecast = forecast.dropna(
        subset=["future_seasonal_mean", "forecast_seasonal_mean", "seasonal_naive"]
    )
    y = forecast["future_max_draw_kw"].to_numpy(float)
    y_resid = y - forecast["future_seasonal_mean"].to_numpy(float)
    model = forecast["predicted"].to_numpy(float)
    model_resid = model - forecast["forecast_seasonal_mean"].to_numpy(float)
    seasonal = forecast["seasonal_naive"].to_numpy(float)
    seasonal_resid = seasonal - forecast["future_seasonal_mean"].to_numpy(float)
    result = pd.DataFrame(
        [
            {
                "target": "future_max_6h_achievable_draw_kw",
                "n": len(y),
                "bilstm_raw_correlation": corr(model, y),
                "bilstm_hour_of_week_residual_correlation": corr(model_resid, y_resid),
                "seasonal_naive_raw_correlation": corr(seasonal, y),
                "seasonal_naive_hour_of_week_residual_correlation": corr(
                    seasonal_resid, y_resid
                ),
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
