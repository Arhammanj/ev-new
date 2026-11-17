"""Run requested-energy lead-time X on supported real ACN sites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acn_env import ACNChargingEnv
from acn_data_loader import load_acn_sessions
from capacity_x_sweep import score
from exp_2x2_run import shared_setup
from train_controller import ACNWeeklyEpisodeEnv, PER_EV_MAX_KW


SITE_FILES = {
    "Caltech": ROOT / "acn_caltech_2019.csv",
    "JPL": ROOT / "acn_jpl_2019.csv",
    "Office001": ROOT / "acn_office001_2019.csv",
}


def requested_coverage(path: Path) -> tuple[int, int, float]:
    df = pd.read_csv(path, usecols=["requested_kWh"])
    values = pd.to_numeric(df["requested_kWh"], errors="coerce")
    positive = int((values > 0).sum())
    return len(df), positive, 100.0 * positive / len(df) if len(df) else 0.0


def full_rollout(path: Path) -> dict:
    data = load_acn_sessions(str(path), timestep_minutes=60)
    env = ACNChargingEnv(data, transformer_limit_kw=50.0, episode_len=data.n_steps)
    env.reset()
    achievable, presence, remaining, departure = [], [], [], []
    for _ in range(data.n_steps):
        present = env.true_presence().copy()
        remaining_kwh = env.true_remaining_kwh().copy()
        draw = np.where(
            present,
            np.minimum(PER_EV_MAX_KW, remaining_kwh / env.dt_hours),
            0.0,
        )
        achievable.append(float(draw.sum()))
        presence.append(present)
        remaining.append(remaining_kwh)
        departure.append(env._departure_step.copy())
        _, _, terminated, truncated, _ = env.step(draw.astype(np.float32))
        if terminated or truncated:
            break
    return {
        "achievable_kw": np.asarray(achievable),
        "presence": np.asarray(presence),
        "remaining": np.asarray(remaining),
        "departure_step": np.asarray(departure),
        "dt_hours": float(env.dt_hours),
    }


def select_capacity(rollout: dict) -> float:
    return float(np.quantile(rollout["achievable_kw"], 0.90))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    status_rows, score_rows = [], []
    for site, path in SITE_FILES.items():
        total, positive, coverage = requested_coverage(path)
        status = "supported" if coverage > 0 else "excluded_no_requested_energy"
        full = full_rollout(path)
        capacity = select_capacity(full)
        status_rows.append(
            {
                "site": site,
                "file": str(path),
                "raw_sessions": total,
                "positive_requested_sessions": positive,
                "requested_energy_coverage_percent": coverage,
                "capacity_rule": "90th percentile full-grid greedy draw",
                "selected_capacity_kw": capacity,
                "status": status,
            }
        )
        if status != "supported":
            continue
        setup = shared_setup(
            data_path=str(path),
            use_forecast=False,
            timestep_minutes=60,
        )
        weeks = []
        for start in setup["test_starts"]:
            env = ACNWeeklyEpisodeEnv(
                setup["full_data"],
                setup["slot_roster"],
                setup["test_range"],
                transformer_limit_kw=capacity,
                seed=999,
            )
            env.reset(start_step=start)
            achievable, presence, remaining, departure = [], [], [], []
            for _ in range(env.episode_len):
                present = env.inner.true_presence().copy()
                remaining_kwh = env.inner.true_remaining_kwh().copy()
                draw = np.where(
                    present,
                    np.minimum(PER_EV_MAX_KW, remaining_kwh / env.inner.dt_hours),
                    0.0,
                )
                achievable.append(float(draw.sum()))
                presence.append(present)
                remaining.append(remaining_kwh)
                departure.append(env.inner._departure_step.copy())
                _, _, terminated, truncated, _ = env.step(draw.astype(np.float32))
                if terminated or truncated:
                    break
            weeks.append(
                {
                    "achievable_kw": np.asarray(achievable),
                    "presence": np.asarray(presence),
                    "remaining": np.asarray(remaining),
                    "departure_step": np.asarray(departure),
                    "dt_hours": float(env.inner.dt_hours),
                }
            )
        score_rows.extend(score([full], capacity, f"{site}_full_year"))
        score_rows.extend(score(weeks, capacity, f"{site}_test_weeks"))

    status_df = pd.DataFrame(status_rows)
    score_df = pd.DataFrame(score_rows)
    status_df.to_csv(args.output_dir / "site_capacity_status.csv", index=False)
    score_df.to_csv(args.output_dir / "site_x_scores.csv", index=False)

    supported = score_df[score_df["scope"].str.endswith("_full_year")]
    fig, axis = plt.subplots(figsize=(8, 5))
    for site, group in supported.groupby(supported["scope"].str.replace("_full_year", "", regex=False)):
        axis.plot(
            group["horizon_hours"],
            group["energy_X_mean"],
            marker="o",
            label=site,
        )
    axis.set_xlabel("Lead time [hours]")
    axis.set_ylabel("Mean energy-based X")
    axis.set_title("Cross-site requested-energy lead-time score")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "site_x_comparison.png", dpi=200)
    plt.close(fig)
    print(status_df.to_string(index=False))
    print("\n" + score_df.to_string(index=False))


if __name__ == "__main__":
    main()
