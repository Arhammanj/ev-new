"""Chronological six-hour violation prediction with and without the forecast."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lead_time_distribution import full_year_rollout


def block_bootstrap_auc_difference(
    labels: np.ndarray,
    calendar_probability: np.ndarray,
    forecast_probability: np.ndarray,
    block_size: int = 24,
    resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return observed and block-bootstrap 95% CI for forecast-minus-calendar AUC."""
    if len(labels) < block_size:
        raise ValueError("The test set is shorter than one 24-hour bootstrap block")
    observed = float(
        roc_auc_score(labels, forecast_probability)
        - roc_auc_score(labels, calendar_probability)
    )
    rng = np.random.default_rng(seed)
    blocks = [
        np.arange(start, min(start + block_size, len(labels)))
        for start in range(0, len(labels), block_size)
    ]
    differences = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        differences.append(
            roc_auc_score(sampled_labels, forecast_probability[indices])
            - roc_auc_score(sampled_labels, calendar_probability[indices])
        )
    if not differences:
        raise ValueError("No valid bootstrap samples contained both label classes")
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return observed, float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "acn_caltech_2019.csv"))
    parser.add_argument("--forecast-csv", type=Path, default=ROOT / "acn_target_a_bilstm_full.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "violation_prediction.csv")
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    args = parser.parse_args()

    forecast = pd.read_csv(args.forecast_csv)
    if not {"hour_index", "predicted"}.issubset(forecast.columns):
        raise ValueError("Forecast must contain hour_index and predicted")
    rollout = full_year_rollout(args.data_path, args.capacity_kw)
    achievable = rollout["achievable_kw"]
    forecast["hour_index"] = forecast["hour_index"].astype(int)
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True)
    hour = forecast["timestamp"].dt.hour + forecast["timestamp"].dt.minute / 60.0
    dow = forecast["timestamp"].dt.dayofweek
    forecast["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    forecast["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    forecast["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    forecast["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    calendar_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    rows = []
    for horizon in args.horizons:
        label = np.full(len(achievable), np.nan)
        for index in range(len(achievable) - horizon):
            label[index] = float(
                np.any(achievable[index + 1 : index + 1 + horizon] > args.capacity_kw)
            )
        current = forecast.copy()
        current["label"] = current["hour_index"].map(
            pd.Series(label, index=np.arange(len(label)))
        )
        current = current.dropna(subset=["label"]).copy()
        train_end = int(len(achievable) * 0.70)
        test_start = int(len(achievable) * 0.85)
        train = current[current["hour_index"] < train_end]
        test = current[current["hour_index"] >= test_start].sort_values("hour_index")
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            raise ValueError(
                f"Chronological train/test split lacks both classes at horizon {horizon}"
            )
        probabilities = {}
        for name, columns in {
            "calendar_only": calendar_cols,
            "calendar_plus_forecast": calendar_cols + ["predicted"],
        }.items():
            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                random_state=42,
                verbosity=-1,
            )
            model.fit(train[columns], train["label"].astype(int))
            probabilities[name] = model.predict_proba(test[columns])[:, 1]
        difference, ci_lower, ci_upper = block_bootstrap_auc_difference(
            test["label"].to_numpy(int),
            probabilities["calendar_only"],
            probabilities["calendar_plus_forecast"],
        )
        for name, columns in {
            "calendar_only": calendar_cols,
            "calendar_plus_forecast": calendar_cols + ["predicted"],
        }.items():
            probability = probabilities[name]
            predicted = probability >= 0.5
            rows.append(
                {
                    "model": name,
                    "capacity_kw": args.capacity_kw,
                    "horizon_hours": horizon,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_positive_rate_percent": 100.0 * float(test["label"].mean()),
                    "auc": float(roc_auc_score(test["label"], probability)),
                    "average_precision": float(average_precision_score(test["label"], probability)),
                    "precision_at_0_5": float(precision_score(test["label"], predicted, zero_division=0)),
                    "recall_at_0_5": float(recall_score(test["label"], predicted, zero_division=0)),
                    "auc_difference_forecast_minus_calendar": difference,
                    "auc_difference_ci95_lower": ci_lower,
                    "auc_difference_ci95_upper": ci_upper,
                    "bootstrap_blocks_hours": 24,
                    "bootstrap_resamples": 2000,
                }
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(
        "\nThe classifier uses chronological training/test splits. The forecast "
        "model's training-period forecast rows are the existing in-sample "
        "predictions documented by audit_report.md. AUC differences use "
        "24-hour block bootstrap intervals with 2000 resamples."
    )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
