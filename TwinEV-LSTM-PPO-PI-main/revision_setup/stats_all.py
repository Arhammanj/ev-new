"""Recompute cross-condition statistics from the saved PPO JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRS = {
    50.0: ROOT / "acn_exp_2x2_results",
    75.0: ROOT / "acn_exp_2x2_results_cap75",
    100.0: ROOT / "acn_exp_2x2_results_cap100",
}
OOF_DIR = ROOT / "revision_setup" / "outputs" / "step5"
OOF_CAPACITY = 50.0
OOF_TEST_STARTS = [7429, 7713, 7997, 8281, 8565]
FORECAST_FULL = ROOT / "acn_target_a_bilstm_full.csv"
FORECAST_OOF = ROOT / "revision_setup" / "forecast_oof.csv"
PAPER_PVALUES = {
    (50.0, "section_6_2"): (0.58, 0.63),
    (50.0, "section_8_2"): (0.50, 0.23),
    (75.0, "section_8_2"): (0.55, 0.18),
    (100.0, "section_8_2"): (0.37, 0.27),
}
COMPARISONS = [
    (50.0, "forecast_oof", "forecast", "OOF vs in-sample forecast"),
    (50.0, "forecast", "no_forecast", "in-sample forecast vs no_forecast"),
    (50.0, "forecast_oof", "no_forecast", "OOF vs no_forecast"),
]


def load_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for capacity, directory in RESULT_DIRS.items():
        files = sorted(directory.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No result JSON files found in {directory}")
        for path in files:
            with path.open(encoding="utf-8") as handle:
                row = json.load(handle)
            required = {"condition", "seed", "violation_rate", "unmet_fraction", "delivered_kwh"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            rows.append(
                {
                    "capacity": capacity,
                    "condition": row["condition"],
                    "seed": str(row["seed"]),
                    "violation": float(row["violation_rate"]),
                    "unmet": float(row["unmet_fraction"]),
                    "delivered": float(row["delivered_kwh"]),
                    "source_file": str(path),
                }
            )
    oof_files = sorted(OOF_DIR.glob("oof_h6_cap50_seed*.json"))
    if not oof_files:
        raise FileNotFoundError(f"No OOF result JSON files found in {OOF_DIR}")
    for path in oof_files:
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        required = {"condition", "seed", "violation_rate", "unmet_fraction", "delivered_kwh"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        if row["condition"] != "forecast":
            raise ValueError(f"{path} has unexpected condition {row['condition']!r}")
        if float(row.get("capacity_kw", -1.0)) != OOF_CAPACITY:
            raise ValueError(f"{path} has unexpected capacity {row.get('capacity_kw')!r}")
        if int(row.get("timesteps", -1)) != 300_000:
            raise ValueError(f"{path} has unexpected timesteps {row.get('timesteps')!r}")
        if list(row.get("test_starts", [])) != OOF_TEST_STARTS:
            raise ValueError(f"{path} has unexpected test_starts {row.get('test_starts')!r}")
        rows.append(
            {
                "capacity": OOF_CAPACITY,
                "condition": "forecast_oof",
                "seed": str(row["seed"]),
                "violation": float(row["violation_rate"]),
                "unmet": float(row["unmet_fraction"]),
                "delivered": float(row["delivered_kwh"]),
                "source_file": str(path),
            }
        )
    result = pd.DataFrame(rows)
    expected = {"cap_only", "no_forecast", "forecast"}
    for capacity, group in result.groupby("capacity"):
        conditions = set(group["condition"])
        missing = expected - conditions
        if missing:
            raise ValueError(f"{capacity:g} kW is missing conditions: {sorted(missing)}")
        extra = conditions - expected
        if extra - {"forecast_oof"}:
            raise ValueError(f"{capacity:g} kW has unexpected conditions: {sorted(extra - {'forecast_oof'})}")
        if "forecast_oof" in extra and capacity != OOF_CAPACITY:
            raise ValueError(f"{capacity:g} kW has forecast_oof but only {OOF_CAPACITY:g} kW is allowed")
        for condition in ("no_forecast", "forecast"):
            seeds = sorted(group.loc[group["condition"] == condition, "seed"])
            if seeds != ["1", "2", "3", "4", "5"]:
                raise ValueError(f"{capacity:g} kW {condition} seeds are {seeds}")
        if capacity == OOF_CAPACITY:
            seeds = sorted(group.loc[group["condition"] == "forecast_oof", "seed"])
            if seeds != ["1", "2", "3", "4", "5"]:
                raise ValueError(f"{capacity:g} kW forecast_oof seeds are {seeds}")
    return result


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    standard_error = sd / np.sqrt(len(values))
    if standard_error == 0.0:
        return mean, mean
    margin = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
    return mean - margin, mean + margin


def condition_summary(condition_group: pd.DataFrame, condition: str) -> dict:
    summary: dict = {}
    for metric in ("violation", "unmet", "delivered"):
        values = condition_group[metric].to_numpy(dtype=float)
        ci_low, ci_high = mean_ci(values)
        prefix = f"{condition}_{metric}"
        summary[f"{prefix}_n"] = len(values)
        summary[f"{prefix}_mean"] = float(np.mean(values))
        summary[f"{prefix}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        summary[f"{prefix}_ci95_low"] = ci_low
        summary[f"{prefix}_ci95_high"] = ci_high
    return summary


def paired_effects(group: pd.DataFrame) -> dict:
    paired = group[group["condition"].isin(["forecast", "no_forecast"])].pivot(
        index="seed", columns="condition", values=["violation", "unmet", "delivered"]
    )
    paired = paired.dropna()
    if len(paired) < 2:
        raise ValueError("At least two paired seeds are required for inference")
    output: dict = {"n_pairs": int(len(paired))}
    for metric in ("violation", "unmet", "delivered"):
        forecast = paired[(metric, "forecast")]
        no_forecast = paired[(metric, "no_forecast")]
        difference = (forecast - no_forecast).to_numpy()
        test = stats.ttest_rel(forecast, no_forecast)
        sd = float(np.std(difference, ddof=1))
        output[f"{metric}_forecast_mean"] = float(forecast.mean())
        output[f"{metric}_no_forecast_mean"] = float(no_forecast.mean())
        output[f"{metric}_difference_mean"] = float(difference.mean())
        output[f"{metric}_difference_sd"] = sd
        output[f"{metric}_paired_t_p"] = float(test.pvalue)
        output[f"{metric}_cohens_d"] = float(difference.mean() / sd) if sd else float("nan")
    return output


def welch_test(condition_a: np.ndarray, condition_b: np.ndarray) -> dict:
    condition_a = np.asarray(condition_a, dtype=float)
    condition_b = np.asarray(condition_b, dtype=float)
    n_a = len(condition_a)
    n_b = len(condition_b)
    if n_a < 2 or n_b < 2:
        raise ValueError("At least two observations per condition are required for Welch inference")
    mean_a = float(np.mean(condition_a))
    mean_b = float(np.mean(condition_b))
    sd_a = float(np.std(condition_a, ddof=1))
    sd_b = float(np.std(condition_b, ddof=1))
    difference = mean_a - mean_b
    standard_error = float(np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b))
    if standard_error == 0.0:
        ci_low = difference
        ci_high = difference
    else:
        denominator = (sd_a**2 / n_a) ** 2 / (n_a - 1) + (sd_b**2 / n_b) ** 2 / (n_b - 1)
        degrees_of_freedom = standard_error**4 / denominator
        margin = float(stats.t.ppf(0.975, degrees_of_freedom) * standard_error)
        ci_low = difference - margin
        ci_high = difference + margin
    p_value = float(stats.ttest_ind(condition_a, condition_b, equal_var=False).pvalue)
    return {
        "n_a": n_a,
        "mean_a": mean_a,
        "sd_a": sd_a,
        "n_b": n_b,
        "mean_b": mean_b,
        "sd_b": sd_b,
        "difference": difference,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_welch": p_value,
    }


def paired_p(group: pd.DataFrame, condition_a: str, condition_b: str, metric: str) -> float:
    paired = group[group["condition"].isin([condition_a, condition_b])].pivot(
        index="seed", columns="condition", values=metric
    )
    paired = paired.dropna()
    if len(paired) < 2:
        return float("nan")
    return float(stats.ttest_rel(paired[condition_a], paired[condition_b]).pvalue)


def build_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict] = []
    for capacity, condition_a, condition_b, label in COMPARISONS:
        group = rows[rows["capacity"] == capacity]
        for metric in ("violation", "unmet", "delivered"):
            values_a = group.loc[group["condition"] == condition_a, metric].to_numpy(dtype=float)
            values_b = group.loc[group["condition"] == condition_b, metric].to_numpy(dtype=float)
            result = welch_test(values_a, values_b)
            output.append(
                {
                    "capacity": capacity,
                    "comparison": label,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "metric": metric,
                    **result,
                    "paired_t_p_secondary": paired_p(group, condition_a, condition_b, metric),
                }
            )
    return pd.DataFrame(output)


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict] = []
    for capacity, group in rows.groupby("capacity", sort=True):
        summary = {"capacity": capacity}
        for condition, condition_group in group.groupby("condition", sort=True):
            summary.update(condition_summary(condition_group, condition))
        summary.update(paired_effects(group))
        summaries.append(summary)
    return pd.DataFrame(summaries)


def compare_pvalues(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in summary.iterrows():
        capacity = float(row["capacity"])
        for label, (paper_violation, paper_energy) in PAPER_PVALUES.items():
            label_capacity, section = label
            if label_capacity != capacity:
                continue
            for metric, paper_value, calculated in (
                ("violation", paper_violation, row["violation_paired_t_p"]),
                ("delivered", paper_energy, row["delivered_paired_t_p"]),
            ):
                rows.append(
                    {
                        "capacity": capacity,
                        "paper_section": section,
                        "metric": metric,
                        "paper_p": paper_value,
                        "calculated_p": calculated,
                        "absolute_difference": abs(paper_value - calculated),
                        "matches_within_1e-6": bool(np.isclose(paper_value, calculated, atol=1e-6)),
                    }
                )
    return pd.DataFrame(rows)


def compare_forecast_files() -> pd.DataFrame:
    full = pd.read_csv(FORECAST_FULL)
    oof = pd.read_csv(FORECAST_OOF)
    full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
    oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True)
    test_indices = {
        index
        for start in OOF_TEST_STARTS
        for index in range(start, start + 168)
    }
    test = full[full["hour_index"].isin(test_indices)].copy()
    if len(test) != len(test_indices):
        raise ValueError(f"Expected {len(test_indices)} test rows, found {len(test)}")
    merged = test.merge(oof, on="timestamp", how="inner", suffixes=("_full", "_oof"))
    if len(merged) != len(test):
        raise ValueError(f"Expected {len(test)} merged test rows, found {len(merged)}")
    predicted_difference = (merged["predicted_full"] - merged["predicted_oof"]).abs()
    actual_difference = (merged["actual_full"] - merged["actual_oof"]).abs()
    max_index = predicted_difference.idxmax()
    return pd.DataFrame(
        [
            {
                "test_week_rows": int(len(merged)),
                "max_abs_predicted_difference_kw": float(predicted_difference.max()),
                "max_abs_predicted_difference_at": merged.loc[max_index, "timestamp"].isoformat(),
                "full_predicted_at_max": float(merged.loc[max_index, "predicted_full"]),
                "oof_predicted_at_max": float(merged.loc[max_index, "predicted_oof"]),
                "mean_abs_predicted_difference_kw": float(predicted_difference.mean()),
                "max_abs_actual_difference": float(actual_difference.max()),
                "oof_model": (
                    "blocked expanding-fold BiLSTM (revision_setup/oof_forecast.py); "
                    "test weeks are in the final held-out fold"
                ),
            }
        ]
    )


def format_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    display = comparisons.copy()
    for metric in ("violation", "unmet"):
        mask = display["metric"] == metric
        for column in ("mean_a", "sd_a", "mean_b", "sd_b", "difference", "ci95_low", "ci95_high"):
            display.loc[mask, column] = display.loc[mask, column] * 100.0
    display["metric"] = display["metric"].map(
        {
            "violation": "violation (%)",
            "unmet": "unmet (%)",
            "delivered": "delivered (kWh)",
        }
    )
    display["ci95"] = display.apply(
        lambda row: f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]",
        axis=1,
    )
    display = display[
        [
            "capacity",
            "comparison",
            "metric",
            "n_a",
            "mean_a",
            "sd_a",
            "n_b",
            "mean_b",
            "sd_b",
            "difference",
            "ci95",
            "p_welch",
            "paired_t_p_secondary",
        ]
    ]
    for column in ("mean_a", "sd_a", "mean_b", "sd_b", "difference"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["p_welch"] = display["p_welch"].map(lambda value: f"{value:.6g}")
    display["paired_t_p_secondary"] = display["paired_t_p_secondary"].map(
        lambda value: f"{value:.6g}"
    )
    return display


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    rows.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    summary = build_summary(rows)
    summary.to_csv(args.output_dir / "stats_table.csv", index=False)
    comparison = compare_pvalues(summary)
    comparison.to_csv(args.output_dir / "paper_pvalue_comparison.csv", index=False)
    welch_comparisons = build_comparisons(rows)
    welch_comparisons.to_csv(args.output_dir / "stats_welch_comparisons.csv", index=False)
    oof_vs_no_forecast = welch_comparisons[
        (welch_comparisons["condition_a"] == "forecast_oof")
        & (welch_comparisons["condition_b"] == "no_forecast")
    ]
    oof_vs_no_forecast.to_csv(args.output_dir / "stats_oof_vs_no_forecast.csv", index=False)
    forecast_comparison = compare_forecast_files()
    forecast_comparison.to_csv(
        args.output_dir / "forecast_oof_vs_full_test_weeks.csv", index=False
    )

    print(rows.to_string(index=False))
    print("\nStatistics (t-based CIs; paired p secondary):")
    print(summary.to_string(index=False))
    print("\nPaper p-value comparison (paired secondary):")
    print(comparison.to_string(index=False))
    print("\nWelch comparisons (primary; paired p secondary):")
    print(format_comparisons(welch_comparisons).to_string(index=False))
    print("\nForecast CSV comparison (test weeks):")
    print(forecast_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
