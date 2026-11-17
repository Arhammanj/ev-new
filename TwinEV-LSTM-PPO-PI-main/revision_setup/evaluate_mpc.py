"""Evaluate Oracle-MPC and stepwise-BiLSTM-MPC on the shared held-out weeks."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_2x2_run import shared_setup
from train_controller import ACNWeeklyEpisodeEnv
from mpc_optimizer import StepwiseForecastLookup, mpc_action_for_env

REQUIRED_TEST_STARTS = [7429, 7713, 7997, 8281, 8565]


class ZeroForecast:
    def __init__(self, horizon: int = 6) -> None:
        self.horizon = horizon

    def __call__(self, timestamp) -> np.ndarray:
        return np.zeros(self.horizon, dtype=float)


def evaluate_controller(
    env: ACNWeeklyEpisodeEnv,
    controller: str,
    forecast,
    test_starts: list[int],
    capacity_kw: float,
    horizon_hours: int,
) -> tuple[list[dict], list[dict]]:
    week_rows = []
    step_rows = []
    for start_step in test_starts:
        obs, _ = env.reset(start_step=start_step)
        start_time = str(env.inner.current_time())
        total_requested = float(sum(s.requested_kwh for s in env.inner.data.sessions))
        total_delivered = 0.0
        total_unmet = 0.0
        n_violating = 0
        sum_violation_kw = 0.0
        max_violation_kw = 0.0
        solver_failures = 0
        solve_times = []
        reserve_sums = []
        horizon_values = []
        for _ in range(env.episode_len):
            step_index = int(env.inner._step_idx)
            decision_time = str(env.inner.current_time())
            started = time.perf_counter()
            result = mpc_action_for_env(
                env,
                forecast,
                horizon_hours=horizon_hours,
                capacity_kw=capacity_kw,
            )
            solve_times.append(time.perf_counter() - started)
            horizon_values.append(result.horizon_steps)
            reserve_sums.append(float(result.reserve_kw[1:].sum()) if len(result.reserve_kw) > 1 else 0.0)
            if not result.success:
                solver_failures += 1
            obs, reward, terminated, truncated, info = env.step(result.action_kw)
            total_delivered += float(info["delivered_kwh"].sum())
            total_unmet += float(info["unmet_kwh_this_step"])
            violation = float(info["violation_kw"])
            if violation > 0:
                n_violating += 1
            sum_violation_kw += violation
            max_violation_kw = max(max_violation_kw, violation)
            step_rows.append({
                "controller": controller,
                "test_start": start_step,
                "step": step_index,
                "timestamp": decision_time,
                "action_kw": float(result.action_kw.sum()),
                "horizon_steps": int(result.horizon_steps),
                "reserve_kw": float(result.reserve_kw[1:].sum()) if len(result.reserve_kw) > 1 else 0.0,
                "solve_success": bool(result.success),
                "solve_status": int(result.status),
                "solve_time_s": float(solve_times[-1]),
                "violation_kw": violation,
                "delivered_kwh": float(info["delivered_kwh"].sum()),
                "unmet_kwh": float(info["unmet_kwh_this_step"]),
            })
            if terminated or truncated:
                break
        week_rows.append({
            "controller": controller,
            "test_start": start_step,
            "start_time": start_time,
            "violation_rate": n_violating / env.episode_len,
            "unmet_fraction": total_unmet / total_requested if total_requested > 0 else 0.0,
            "delivered_kwh": total_delivered,
            "total_requested_kwh": total_requested,
            "unmet_kwh": total_unmet,
            "n_violating_steps": n_violating,
            "sum_violation_kw": sum_violation_kw,
            "max_violation_kw": max_violation_kw,
            "solver_failures": solver_failures,
            "mean_solve_time_s": float(np.mean(solve_times)) if solve_times else float("nan"),
            "total_solve_time_s": float(np.sum(solve_times)),
            "mean_reserve_kw": float(np.mean(reserve_sums)) if reserve_sums else 0.0,
            "mean_horizon_steps": float(np.mean(horizon_values)) if horizon_values else 0.0,
        })
    return week_rows, step_rows


def aggregate_rows(week_rows: list[dict]) -> list[dict]:
    metrics = [
        "violation_rate",
        "unmet_fraction",
        "delivered_kwh",
        "total_requested_kwh",
        "unmet_kwh",
        "sum_violation_kw",
        "max_violation_kw",
        "solver_failures",
        "mean_solve_time_s",
        "total_solve_time_s",
        "mean_reserve_kw",
        "mean_horizon_steps",
    ]
    rows = []
    frame = pd.DataFrame(week_rows)
    for controller, group in frame.groupby("controller", sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            n = int(len(values))
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            if n > 1:
                half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
                ci_low, ci_high = mean - half, mean + half
            else:
                ci_low = ci_high = mean
            rows.append({
                "controller": controller,
                "metric": metric,
                "n_weeks": n,
                "mean": mean,
                "std": std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    return rows


def load_ppo_rows(ppo_dir: Path, test_starts: list[int]) -> list[dict]:
    rows = []
    patterns = {
        "ppo_oof": "oof_h6_cap50_seed*.json",
        "ppo_no_forecast": "no_forecast_restricted_h6_cap50_seed*.json",
    }
    for controller, pattern in patterns.items():
        for path in sorted(ppo_dir.glob(pattern)):
            result = json.loads(path.read_text())
            if result.get("test_starts") != test_starts:
                continue
            rows.append({
                "controller": controller,
                "source": str(path),
                "violation_rate": float(result["violation_rate"]),
                "unmet_fraction": float(result["unmet_fraction"]),
                "delivered_kwh": float(result["delivered_kwh"]),
            })
    return rows


def aggregate_external(ppo_rows: list[dict]) -> list[dict]:
    rows = []
    frame = pd.DataFrame(ppo_rows)
    if frame.empty:
        return rows
    for controller, group in frame.groupby("controller", sort=False):
        for metric in ["violation_rate", "unmet_fraction", "delivered_kwh"]:
            values = group[metric].to_numpy(dtype=float)
            n = int(len(values))
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            if n > 1:
                half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
                ci_low, ci_high = mean - half, mean + half
            else:
                ci_low = ci_high = mean
            rows.append({
                "controller": controller,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="acn_caltech_2019.csv")
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--oracle-csv", type=Path, default=Path(__file__).resolve().parent / "outputs" / "mpc" / "oracle_stepwise_h6.csv")
    parser.add_argument("--bilstm-csv", type=Path, default=Path(__file__).resolve().parent / "forecast_stepwise_oof.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "mpc")
    parser.add_argument("--ppo-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "step5")
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    setup = shared_setup(use_forecast=False, data_path=args.data_path)
    test_starts = list(REQUIRED_TEST_STARTS)
    for start in test_starts:
        if not (setup["test_range"][0] <= start < setup["test_range"][1]):
            raise ValueError(f"test start {start} is outside {setup['test_range']}")
    env = ACNWeeklyEpisodeEnv(
        setup["full_data"],
        setup["slot_roster"],
        setup["test_range"],
        transformer_limit_kw=args.capacity_kw,
        episode_len=setup["episode_len"],
        symmetric_action=False,
        hard_safety_filter=False,
        forecast=None,
        seed=args.seed,
    )
    forecast_factories = {
        "oracle_mpc": lambda: StepwiseForecastLookup(args.oracle_csv, horizon=args.horizon_hours),
        "bilstm_mpc": lambda: StepwiseForecastLookup(args.bilstm_csv, horizon=args.horizon_hours),
        "no_forecast_mpc": lambda: ZeroForecast(horizon=args.horizon_hours),
    }
    all_week_rows = []
    all_step_rows = []
    for controller, factory in forecast_factories.items():
        forecast = factory()
        week_rows, step_rows = evaluate_controller(
            env,
            controller,
            forecast,
            test_starts,
            args.capacity_kw,
            args.horizon_hours,
        )
        all_week_rows.extend(week_rows)
        all_step_rows.extend(step_rows)
        print(f"evaluated {controller} on {len(test_starts)} weeks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    week_frame = pd.DataFrame(all_week_rows)
    week_frame.to_csv(args.output_dir / "mpc_week_metrics.csv", index=False)
    step_frame = pd.DataFrame(all_step_rows)
    step_frame.to_csv(args.output_dir / "mpc_step_diagnostics.csv", index=False)
    aggregate = aggregate_rows(all_week_rows)
    aggregate_frame = pd.DataFrame(aggregate)
    aggregate_frame.to_csv(args.output_dir / "mpc_aggregate.csv", index=False)

    ppo_rows = load_ppo_rows(args.ppo_dir, test_starts)
    ppo_aggregate = aggregate_external(ppo_rows)
    if ppo_aggregate:
        pd.DataFrame(ppo_aggregate).to_csv(args.output_dir / "mpc_vs_ppo.csv", index=False)

    summary = {
        "config": {
            "data_path": args.data_path,
            "capacity_kw": args.capacity_kw,
            "horizon_hours": args.horizon_hours,
            "test_starts": test_starts,
            "oracle_csv": str(args.oracle_csv),
            "bilstm_csv": str(args.bilstm_csv),
            "episode_len": setup["episode_len"],
        },
        "mpc_aggregate": aggregate,
        "ppo_aggregate": ppo_aggregate,
        "week_metrics": all_week_rows,
    }
    (args.output_dir / "mpc_summary.json").write_text(json.dumps(summary, indent=2))
    print(aggregate_frame.to_string(index=False))
    print(f"saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
