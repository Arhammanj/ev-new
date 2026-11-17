"""Paired week-level comparisons for the MPC controllers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_rows(frame: pd.DataFrame, metric: str, pairs: list[tuple[str, str, str]]) -> list[dict]:
    rows = []
    for name, left, right in pairs:
        pivot = frame.pivot(index="test_start", columns="controller", values=metric)
        diff = (pivot[left] - pivot[right]).dropna().to_numpy(dtype=float)
        n = len(diff)
        mean = float(np.mean(diff)) if n else float("nan")
        std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        if n > 1:
            half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
            test = stats.ttest_rel(pivot.loc[pivot[left].notna() & pivot[right].notna(), left], pivot.loc[pivot[left].notna() & pivot[right].notna(), right])
            p_value = float(test.pvalue)
        else:
            half = 0.0
            p_value = float("nan")
        rows.append({
            "comparison": name,
            "metric": metric,
            "left": left,
            "right": right,
            "n_weeks": n,
            "mean_difference": mean,
            "std_difference": std,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "paired_t_p_value": p_value,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week-metrics", type=Path, default=Path(__file__).resolve().parent / "outputs" / "mpc" / "mpc_week_metrics.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "mpc" / "mpc_paired_comparisons.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.week_metrics)
    pairs = [
        ("bilstm_minus_no_forecast", "bilstm_mpc", "no_forecast_mpc"),
        ("oracle_minus_no_forecast", "oracle_mpc", "no_forecast_mpc"),
        ("oracle_minus_bilstm", "oracle_mpc", "bilstm_mpc"),
    ]
    rows = []
    for metric in ["violation_rate", "unmet_fraction", "delivered_kwh"]:
        rows.extend(paired_rows(frame, metric, pairs))
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
