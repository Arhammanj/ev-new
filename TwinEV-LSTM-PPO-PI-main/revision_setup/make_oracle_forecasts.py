"""Create explicit oracle forecast CSVs from the existing real forecast targets.

The oracle is intentionally an upper-bound diagnostic: ``actual`` is copied
into ``predicted`` at the same timestamp.  It must never be used as a
deployment or leakage-free result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_demand_forecaster import build_target_series
from test_alternative_targets import build_rolling_6h_ahead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="acn_caltech_2019.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "oracle_forecasts")
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 3, 6])
    args = parser.parse_args()
    series = build_target_series(args.data_path, timestep_minutes=60)
    demand = series["demand"].to_numpy(dtype=float)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for horizon in args.horizons:
        target = build_rolling_6h_ahead(demand, horizon)
        valid = len(demand) - horizon
        out = pd.DataFrame({
            "hour_index": range(valid),
            "timestamp": series["timestamp"].iloc[:valid].values,
            "actual": target[:valid],
            "predicted": target[:valid],
            "horizon_hours": horizon,
            "timestep_minutes": 60,
        })
        out["input_last_index"] = out["hour_index"]
        path = args.output_dir / f"oracle_h{horizon}.csv"
        out.to_csv(path, index=False)
        print(f"saved {path} ({len(out)} rows; oracle predicted=actual)")


if __name__ == "__main__":
    main()
