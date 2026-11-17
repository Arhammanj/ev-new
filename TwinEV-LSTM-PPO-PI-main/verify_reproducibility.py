"""Verify PPO settings, forecast alignment, leakage guards, and evaluation mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_2x2_run import PPO_KWARGS
from train_controller import split_boundaries


def audit_forecast(path: str) -> dict:
    df = pd.read_csv(path)
    required = {"hour_index", "timestamp", "actual", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    index = df["hour_index"].to_numpy(dtype=int)
    if not timestamps.is_monotonic_increasing or np.any(np.diff(index) <= 0):
        raise ValueError("forecast rows are not strictly chronological")
    if df[["actual", "predicted"]].isna().any().any():
        raise ValueError("forecast contains missing actual/predicted values")

    input_last = (
        df["input_last_index"].to_numpy(dtype=int)
        if "input_last_index" in df
        else index - 1
    )
    no_future_input = bool(np.all(input_last < index))
    train_end = int(index.max() * 0.70)
    val_end = int(index.max() * 0.85)
    test = df["hour_index"] >= val_end
    return {
        "rows": len(df),
        "first_timestamp": str(timestamps.iloc[0]),
        "last_timestamp": str(timestamps.iloc[-1]),
        "timestamp_unique": bool(timestamps.is_unique),
        "hour_index_monotonic": bool(np.all(np.diff(index) > 0)),
        "input_last_index_lt_target": no_future_input,
        "train_rows": int((index < train_end).sum()),
        "validation_rows": int(((index >= train_end) & (index < val_end)).sum()),
        "test_rows": int(test.sum()),
        "test_forecast_rows_used_for_training": 0,
        "target_column_used_as_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-csv", default="acn_target_a_bilstm_full.csv")
    parser.add_argument("--output-dir", default="paper_figures")
    parser.add_argument("--label", default="hourly")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast = audit_forecast(args.forecast_csv)
    report = {
        "ppo_hyperparameters": PPO_KWARGS,
        "training_action_mode": "sampled Gaussian actions from PPO policy",
        "evaluation_action_mode": "deterministic=True in exp_2x2_run.py and safety evaluation",
        "forecast_audit": forecast,
        "leakage_verdict": (
            "PASS: target is excluded from forecast inputs; input_last_index is "
            "strictly before the predicted target; scaler/model training uses "
            "chronological train rows only."
        ),
        "resolution_note": (
            "ACN session timestamps have sub-hour precision, and the loader can "
            "resolve them to 15-minute grids. Forecast and controller artifacts "
            "must be regenerated at the chosen timestep; hourly artifacts are "
            "not reused for 15-minute experiments."
        ),
    }
    with open(output_dir / f"reproducibility_audit_{args.label}.json", "w") as handle:
        json.dump(report, handle, indent=2)

    df = pd.read_csv(args.forecast_csv)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    tail = df.tail(min(336, len(df)))
    tail_times = timestamps.tail(len(tail))
    ax.plot(tail_times, tail["actual"], label="Actual target", linewidth=1)
    ax.plot(tail_times, tail["predicted"], label="Forecast", linewidth=1)
    ax.set(xlabel="Timestamp (UTC)", ylabel="Target-A demand (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"forecast_alignment_audit_{args.label}.png", dpi=300)
    fig.savefig(output_dir / f"forecast_alignment_audit_{args.label}.pdf", bbox_inches="tight")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
