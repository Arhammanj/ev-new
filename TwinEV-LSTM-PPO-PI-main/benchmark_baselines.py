"""Constraint-respecting scheduling benchmarks for the ACN experiments.

This module evaluates two non-learning references on the same held-out weeks:
an earliest-deadline-first scheduler and a rolling-horizon MPC-style scheduler.
Both use current measured EV state and enforce the transformer limit directly.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from exp_2x2_run import shared_setup
from train_controller import (
    ACNWeeklyEpisodeEnv,
    PER_EV_MAX_KW,
    evaluate_policy_on_weeks,
)


def _available(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner = env.inner
    present = inner.true_presence()
    remaining = inner.true_remaining_kwh()
    deadline_steps = inner._departure_step.copy()
    max_kw = np.where(
        present,
        np.minimum(inner.per_ev_max_kw, remaining / inner.dt_hours),
        0.0,
    )
    return present, deadline_steps, max_kw


def earliest_deadline_action(env) -> np.ndarray:
    """Allocate capacity to connected EVs with the nearest departure."""
    present, deadlines, max_kw = _available(env)
    order = np.argsort(np.where(present, deadlines, np.iinfo(np.int64).max))
    action = np.zeros(env.inner.n_slots, dtype=np.float64)
    budget = env.transformer_limit_kw - 1e-4
    for slot in order:
        amount = min(float(max_kw[slot]), budget)
        action[slot] = max(0.0, amount)
        budget -= amount
        if budget <= 0:
            break
    return action.astype(np.float32)


def rolling_horizon_mpc_action(env, horizon_hours: int = 6) -> np.ndarray:
    """Solve the first action of a receding-horizon scheduling LP.

    The horizon is represented by deadline urgency: energy that must be
    served per remaining hour is maximized at the current step, with a small
    deadline tie-break. Re-solving this LP after every environment step makes
    it a receding-horizon MPC benchmark.
    """
    present, deadlines, max_kw = _available(env)
    remaining = env.inner.true_remaining_kwh()
    steps_left = np.maximum(1, deadlines - env.inner._step_idx)
    urgency = np.where(present, remaining / steps_left, 0.0)
    urgency += np.where(present, 1.0 / np.maximum(1, deadlines), 0.0) * 1e-6
    lookahead_factor = np.minimum(
        1.0, max(1, horizon_hours) / np.maximum(1, steps_left)
    )
    weights = urgency * lookahead_factor
    bounds = [(0.0, float(value)) for value in max_kw]
    result = linprog(
        c=-weights,
        A_ub=np.ones((1, len(max_kw))),
        b_ub=[env.transformer_limit_kw - 1e-4],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"MPC allocation failed: {result.message}")
    return np.asarray(result.x, dtype=np.float32)


def multistep_mpc_action(env, horizon_hours: int = 6) -> np.ndarray:
    """Solve a finite-horizon LP using only the current EV state.

    Future arrivals are intentionally excluded: the controller predicts the
    evolution of vehicles already connected, then re-solves after each step.
    The returned first action is therefore a genuine receding-horizon action,
    rather than a one-step urgency score.
    """
    present, deadlines, max_kw = _available(env)
    n_slots = len(max_kw)
    horizon = max(1, int(horizon_hours))
    remaining = env.inner.true_remaining_kwh()
    step_idx = env.inner._step_idx
    dt = env.inner.dt_hours
    n_vars = n_slots * horizon
    objective = np.zeros(n_vars, dtype=np.float64)
    bounds = []
    for t in range(horizon):
        for slot in range(n_slots):
            active = bool(present[slot] and step_idx + t < deadlines[slot])
            upper = float(max_kw[slot]) if active else 0.0
            bounds.append((0.0, upper))
            # Prefer energy for EVs with less time remaining. The small
            # tie-break rewards useful current delivery without changing
            # feasibility or making the objective depend on future data.
            steps_left = max(1, int(deadlines[slot] - (step_idx + t)))
            objective[t * n_slots + slot] = -(
                1.0 / steps_left + 1e-3 / (t + 1)
            )

    constraints = []
    limits = []
    for t in range(horizon):
        row = np.zeros(n_vars, dtype=np.float64)
        row[t * n_slots:(t + 1) * n_slots] = 1.0
        constraints.append(row)
        limits.append(env.transformer_limit_kw - 1e-4)
    for slot in range(n_slots):
        row = np.zeros(n_vars, dtype=np.float64)
        for t in range(horizon):
            row[t * n_slots + slot] = dt
        constraints.append(row)
        limits.append(float(remaining[slot]))

    result = linprog(
        c=objective,
        A_ub=np.asarray(constraints),
        b_ub=np.asarray(limits),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"multi-step MPC allocation failed: {result.message}")
    return np.asarray(result.x[:n_slots], dtype=np.float32)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument(
        "--mpc-mode", choices=["one_step", "multi_step"], default="multi_step",
        help="Use the historical urgency LP or the finite-horizon LP.",
    )
    parser.add_argument("--output", default="acn_benchmark_results.json")
    parser.add_argument("--data-path", default="acn_caltech_2019.csv")
    args = parser.parse_args()

    setup = shared_setup(use_forecast=False, data_path=args.data_path)
    env = ACNWeeklyEpisodeEnv(
        setup["full_data"],
        setup["slot_roster"],
        setup["test_range"],
        transformer_limit_kw=args.capacity_kw,
        seed=999,
    )
    results = {}
    mpc_fn = (
        multistep_mpc_action
        if args.mpc_mode == "multi_step"
        else rolling_horizon_mpc_action
    )
    for name, fn in (
        ("earliest_deadline", lambda e, _: earliest_deadline_action(e)),
        (f"{args.mpc_mode}_mpc", lambda e, _: mpc_fn(e, args.horizon_hours)),
    ):
        results[name] = evaluate_policy_on_weeks(env, setup["test_starts"], fn)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(pd.DataFrame(results).T.to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    run()
