# ==========================================
# Aggregate exp_2x2_run.py / exp_2x2_cap_only.py results into the
# paper's core-result table: mean +/- std across seeds per condition,
# plus the forecast-vs-no_forecast contrast against seed-to-seed noise.
# ==========================================

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from exp_2x2_run import RESULTS_DIR


def main() -> None:
    rows = []
    for path in sorted(glob.glob(f"{RESULTS_DIR}/*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows)

    print("===== RAW ROWS =====")
    print(df.to_string(index=False))

    cap = df[df["condition"] == "cap_only"].iloc[0]

    print("\n===== SUMMARY (mean +/- std across seeds) =====")
    summary = {}
    for cond in ["no_forecast", "forecast"]:
        sub = df[df["condition"] == cond]
        n = len(sub)
        summary[cond] = {
            "n_seeds": n,
            "viol_mean": sub["violation_rate"].mean() * 100,
            "viol_std": sub["violation_rate"].std(ddof=1) * 100 if n > 1 else float("nan"),
            "unmet_mean": sub["unmet_fraction"].mean() * 100,
            "unmet_std": sub["unmet_fraction"].std(ddof=1) * 100 if n > 1 else float("nan"),
            "kwh_mean": sub["delivered_kwh"].mean(),
            "kwh_std": sub["delivered_kwh"].std(ddof=1) if n > 1 else float("nan"),
        }
        s = summary[cond]
        print(f"{cond:12s} (n={s['n_seeds']} seeds): "
              f"viol={s['viol_mean']:.2f}+/-{s['viol_std']:.2f}%  "
              f"unmet={s['unmet_mean']:.2f}+/-{s['unmet_std']:.2f}%  "
              f"delivered={s['kwh_mean']:.1f}+/-{s['kwh_std']:.1f} kWh")
    print(f"{'cap_only':12s} (single run):     "
          f"viol={100*cap['violation_rate']:.2f}%  "
          f"unmet={100*cap['unmet_fraction']:.2f}%  "
          f"delivered={cap['delivered_kwh']:.1f} kWh")

    nf, fc = summary["no_forecast"], summary["forecast"]

    print("\n===== KEY CONTRAST: forecast vs no_forecast =====")
    viol_diff = nf["viol_mean"] - fc["viol_mean"]
    kwh_diff = fc["kwh_mean"] - nf["kwh_mean"]
    viol_pooled_std = float(np.sqrt(nf["viol_std"]**2 + fc["viol_std"]**2))
    kwh_pooled_std = float(np.sqrt(nf["kwh_std"]**2 + fc["kwh_std"]**2))
    print(f"Violation rate: no_forecast={nf['viol_mean']:.2f}%  forecast={fc['viol_mean']:.2f}%  "
          f"diff={viol_diff:+.2f}pp (forecast lower is better)")
    print(f"  seed-to-seed std: no_forecast={nf['viol_std']:.2f}pp  forecast={fc['viol_std']:.2f}pp  "
          f"pooled={viol_pooled_std:.2f}pp")
    print(f"  |diff| exceeds pooled std? {'YES' if abs(viol_diff) > viol_pooled_std else 'NO'}")

    print(f"\nDelivered kWh: no_forecast={nf['kwh_mean']:.1f}  forecast={fc['kwh_mean']:.1f}  "
          f"diff={kwh_diff:+.1f} kWh (forecast higher is better)")
    print(f"  seed-to-seed std: no_forecast={nf['kwh_std']:.1f}  forecast={fc['kwh_std']:.1f}  "
          f"pooled={kwh_pooled_std:.1f}")
    print(f"  |diff| exceeds pooled std? {'YES' if abs(kwh_diff) > kwh_pooled_std else 'NO'}")

    forecast_wins = (abs(viol_diff) > viol_pooled_std and viol_diff > 0) or \
                    (abs(kwh_diff) > kwh_pooled_std and kwh_diff > 0)
    print(f"\n-> Does forecast beat no_forecast beyond noise (fewer violations AND/OR more energy)? "
          f"{'YES' if forecast_wins else 'NO'}")

    print("\n===== Do either learned condition beat cap_only on any axis? =====")
    for cond, s in summary.items():
        beats_energy = s["kwh_mean"] > cap["delivered_kwh"]
        beats_viol = s["viol_mean"] < 100 * cap["violation_rate"]
        print(f"{cond:12s}: more energy than cap_only? {'YES' if beats_energy else 'NO'} "
              f"({s['kwh_mean']:.1f} vs {cap['delivered_kwh']:.1f})  |  "
              f"fewer violations than cap_only? {'YES' if beats_viol else 'NO'} "
              f"({s['viol_mean']:.2f}% vs {100*cap['violation_rate']:.2f}%)")


if __name__ == "__main__":
    main()
