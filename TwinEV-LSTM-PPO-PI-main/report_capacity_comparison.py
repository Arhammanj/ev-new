# ==========================================
# Cross-capacity comparison: violation rate and delivered kWh for all
# three conditions (cap_only, no_forecast, forecast) across every
# transformer capacity that was trained/evaluated (40/50/60/75/100 kW,
# at the paper's primary 6h forecast horizon). Reads each capacity's
# exp_2x2_run.py/exp_2x2_cap_only.py results (see exp_2x2_run.py's
# run_dirs()) and writes one row per capacity to
# acn_capacity_comparison.csv for make_paper_figures.py to plot.
#
# Unlike the horizon sweep, no_forecast is NOT reused across capacities
# (capacity changes the environment's actual constraint/reward, so the
# policy is genuinely different) - both no_forecast and forecast are
# freshly trained at every non-base capacity.
# ==========================================

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from exp_2x2_run import BASE_HORIZON, run_dirs

CAPACITIES_KW = [50, 75, 100]


def load_results(capacity_kw: float) -> pd.DataFrame:
    _, results_dir = run_dirs(BASE_HORIZON, capacity_kw)
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(
            f"no results found in {results_dir}/ - run exp_2x2_run.py "
            f"(--capacity-kw {capacity_kw}, both conditions, 5 seeds) and "
            f"exp_2x2_cap_only.py (--capacity-kw {capacity_kw}) first"
        )
    return pd.DataFrame(rows)


def summarize_condition(df: pd.DataFrame, condition: str) -> dict:
    sub = df[df["condition"] == condition]
    n = len(sub)
    return {
        "n_seeds": n,
        "viol_mean": sub["violation_rate"].mean() * 100,
        "viol_std": sub["violation_rate"].std(ddof=1) * 100 if n > 1 else float("nan"),
        "kwh_mean": sub["delivered_kwh"].mean(),
        "kwh_std": sub["delivered_kwh"].std(ddof=1) if n > 1 else float("nan"),
    }


def main() -> None:
    out_rows = []
    for capacity_kw in CAPACITIES_KW:
        df = load_results(capacity_kw)
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
            "capacity_kw": capacity_kw,
            "cap_only_viol_pct": 100 * cap["violation_rate"],
            "cap_only_kwh": cap["delivered_kwh"],
            "no_forecast_viol_mean_pct": nf["viol_mean"], "no_forecast_viol_std_pct": nf["viol_std"],
            "no_forecast_kwh_mean": nf["kwh_mean"], "no_forecast_kwh_std": nf["kwh_std"],
            "forecast_viol_mean_pct": fc["viol_mean"], "forecast_viol_std_pct": fc["viol_std"],
            "forecast_kwh_mean": fc["kwh_mean"], "forecast_kwh_std": fc["kwh_std"],
            "viol_diff_pct": viol_diff, "kwh_diff": kwh_diff,
            "forecast_beats_noise": forecast_beats_noise,
        })

        print(f"\n===== capacity={capacity_kw}kW =====")
        print(f"  cap_only:    viol={100*cap['violation_rate']:.2f}%  delivered={cap['delivered_kwh']:.1f} kWh")
        print(f"  no_forecast (n={nf['n_seeds']}): viol={nf['viol_mean']:.2f}+/-{nf['viol_std']:.2f}%  "
              f"delivered={nf['kwh_mean']:.1f}+/-{nf['kwh_std']:.1f} kWh")
        print(f"  forecast    (n={fc['n_seeds']}): viol={fc['viol_mean']:.2f}+/-{fc['viol_std']:.2f}%  "
              f"delivered={fc['kwh_mean']:.1f}+/-{fc['kwh_std']:.1f} kWh")
        print(f"  diff: viol={viol_diff:+.2f}pp (pooled std {viol_pooled_std:.2f}pp)  "
              f"kwh={kwh_diff:+.1f} (pooled std {kwh_pooled_std:.1f})")
        print(f"  forecast beats no_forecast beyond seed-to-seed noise? "
              f"{'YES' if forecast_beats_noise else 'NO'}")

    out = pd.DataFrame(out_rows)
    out.to_csv("acn_capacity_comparison.csv", index=False)
    print("\n===== FULL TABLE =====")
    print(out.to_string(index=False))
    print("\nSaved acn_capacity_comparison.csv")


if __name__ == "__main__":
    main()
