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


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-kw", type=float, default=50.0)
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--output", default="acn_benchmark_results.json")
    args = parser.parse_args()

    setup = shared_setup()
    env = ACNWeeklyEpisodeEnv(
        setup["full_data"],
        setup["slot_roster"],
        setup["test_range"],
        transformer_limit_kw=args.capacity_kw,
        seed=999,
    )
    results = {}
    for name, fn in (
        ("earliest_deadline", lambda e, _: earliest_deadline_action(e)),
        (
            "rolling_horizon_mpc",
            lambda e, _: rolling_horizon_mpc_action(e, args.horizon_hours),
        ),
    ):
        results[name] = evaluate_policy_on_weeks(env, setup["test_starts"], fn)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(pd.DataFrame(results).T.to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    run()
