"""Convert the OOF NumPy predictions into the controller's forecast CSV schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_demand_forecaster import build_target_series
from test_alternative_targets import (
    CALENDAR_COLS,
    LAG_FEATURE_COLS,
    add_calendar,
    add_lag_features,
    build_rolling_6h_ahead,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument("--oof", type=Path, default=Path(__file__).parent / "forecast_oof.npy")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--timestep-minutes", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "forecast_oof.csv",
    )
    args = parser.parse_args()

    predictions = np.load(args.oof)
    series = build_target_series(args.data_path, timestep_minutes=args.timestep_minutes)
    steps_per_hour = 60 // args.timestep_minutes
    horizon_steps = args.horizon * steps_per_hour
    valid_end = len(series) - horizon_steps
    frame = pd.DataFrame(
        {
            "timestamp": series["timestamp"].iloc[:valid_end],
            "target": build_rolling_6h_ahead(
                series["demand"].to_numpy(), horizon_steps
            )[:valid_end],
            "feat_src": series["demand"].to_numpy()[:valid_end],
        }
    )
    frame = add_calendar(frame)
    frame = add_lag_features(frame, "feat_src", steps_per_hour=steps_per_hour)
    frame = frame.dropna().reset_index(drop=True)
    if len(predictions) != len(frame):
        raise ValueError(
            f"OOF length {len(predictions)} does not match aligned frame length {len(frame)}"
        )
    frame["predicted"] = predictions
    frame = frame[np.isfinite(frame["predicted"])].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    start = frame["timestamp"].min()
    frame["hour_index"] = (
        (frame["timestamp"] - start).dt.total_seconds() / 3600.0
    ).round().astype(int)
    frame["actual"] = frame["target"]
    frame["horizon_hours"] = args.horizon
    frame["timestep_minutes"] = args.timestep_minutes
    frame["input_last_index"] = frame["hour_index"] - 1
    result = frame[
        [
            "hour_index",
            "timestamp",
            "actual",
            "predicted",
            "horizon_hours",
            "timestep_minutes",
            "input_last_index",
        ]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output}; {len(result)} finite OOF rows")


if __name__ == "__main__":
    main()
