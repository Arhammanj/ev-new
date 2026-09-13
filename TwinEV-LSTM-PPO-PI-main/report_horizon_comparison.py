# ==========================================
# Cross-horizon comparison: forecast accuracy (WAPE) vs. controller
# performance (violation rate, delivered kWh), for every forecast lead
# time that was actually trained (1h / 3h / 6h). Reads each horizon's
# exp_2x2_run.py/exp_2x2_cap_only.py results (see exp_2x2_run.py's
# horizon_dirs()) plus the matching Target A metrics CSV, and writes
# one row per horizon to acn_horizon_comparison.csv for
# make_paper_figures.py to plot.
# ==========================================

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from exp_2x2_run import BASE_HORIZON, horizon_dirs

HORIZONS = [1, 3, 6]


def metrics_path_for(horizon: int) -> str:
    return ("acn_target_a_metrics.csv" if horizon == BASE_HORIZON
            else f"acn_target_a_h{horizon}_metrics.csv")


def load_results(horizon: int) -> pd.DataFrame:
    _, results_dir = horizon_dirs(horizon)
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(
            f"no results found in {results_dir}/ - run exp_2x2_run.py "
            f"(--horizon {horizon}, both conditions, 5 seeds) and "
            f"exp_2x2_cap_only.py (--horizon {horizon}) first"
        )
    return pd.DataFrame(rows)


def summarize_condition(df: pd.DataFrame, condition: str) -> dict:
    sub = df[df["condition"] == condition]
    n = len(sub)
    return {
        "n_seeds": n,
        "viol_mean": sub["violation_rate"].mean() * 100,
        "viol_std": sub["violation_rate"].std(ddof=1) * 100 if n > 1 else float("nan"),
        "unmet_mean": sub["unmet_fraction"].mean() * 100,
        "unmet_std": sub["unmet_fraction"].std(ddof=1) * 100 if n > 1 else float("nan"),
        "kwh_mean": sub["delivered_kwh"].mean(),
        "kwh_std": sub["delivered_kwh"].std(ddof=1) if n > 1 else float("nan"),
    }


def main() -> None:
    out_rows = []
    for horizon in HORIZONS:
        metrics_path = metrics_path_for(horizon)
        metrics = pd.read_csv(metrics_path)
        wape_bilstm = float(metrics.loc[metrics["Model"] == "BiLSTM", "WAPE"].iloc[0])

        df = load_results(horizon)
        cap = df[df["condition"] == "cap_only"].iloc[0]
        nf = summarize_condition(df, "no_forecast")
        fc = summarize_condition(df, "forecast")

        viol_diff = nf["viol_mean"] - fc["viol_mean"]   # positive = forecast has fewer violations
        kwh_diff = fc["kwh_mean"] - nf["kwh_mean"]        # positive = forecast delivers more energy
        viol_pooled_std = float(np.sqrt(nf["viol_std"] ** 2 + fc["viol_std"] ** 2))
        kwh_pooled_std = float(np.sqrt(nf["kwh_std"] ** 2 + fc["kwh_std"] ** 2))
        forecast_beats_noise = (
            (abs(viol_diff) > viol_pooled_std and viol_diff > 0)
            or (abs(kwh_diff) > kwh_pooled_std and kwh_diff > 0)
        )

        out_rows.append({
            "horizon_h": horizon,
            "forecast_wape": wape_bilstm,
            "cap_only_viol_pct": 100 * cap["violation_rate"],
            "cap_only_kwh": cap["delivered_kwh"],
            "no_forecast_viol_mean_pct": nf["viol_mean"], "no_forecast_viol_std_pct": nf["viol_std"],
            "no_forecast_kwh_mean": nf["kwh_mean"], "no_forecast_kwh_std": nf["kwh_std"],
            "forecast_viol_mean_pct": fc["viol_mean"], "forecast_viol_std_pct": fc["viol_std"],
            "forecast_kwh_mean": fc["kwh_mean"], "forecast_kwh_std": fc["kwh_std"],
            "viol_diff_pct": viol_diff, "kwh_diff": kwh_diff,
            "forecast_beats_noise": forecast_beats_noise,
        })

        print(f"\n===== horizon={horizon}h =====")
        print(f"  forecast WAPE: {wape_bilstm:.2f}%")
        print(f"  no_forecast (n={nf['n_seeds']}): viol={nf['viol_mean']:.2f}+/-{nf['viol_std']:.2f}%  "
              f"delivered={nf['kwh_mean']:.1f}+/-{nf['kwh_std']:.1f} kWh")
        print(f"  forecast    (n={fc['n_seeds']}): viol={fc['viol_mean']:.2f}+/-{fc['viol_std']:.2f}%  "
              f"delivered={fc['kwh_mean']:.1f}+/-{fc['kwh_std']:.1f} kWh")
        print(f"  diff: viol={viol_diff:+.2f}pp (pooled std {viol_pooled_std:.2f}pp)  "
              f"kwh={kwh_diff:+.1f} (pooled std {kwh_pooled_std:.1f})")
        print(f"  forecast beats no_forecast beyond seed-to-seed noise? "
              f"{'YES' if forecast_beats_noise else 'NO'}")

    out = pd.DataFrame(out_rows)
    out.to_csv("acn_horizon_comparison.csv", index=False)
    print("\n===== FULL TABLE =====")
    print(out.to_string(index=False))
    print("\nSaved acn_horizon_comparison.csv")


if __name__ == "__main__":
    main()
