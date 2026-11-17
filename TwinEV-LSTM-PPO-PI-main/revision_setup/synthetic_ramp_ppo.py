"""Controlled PPO positive control on a seeded stochastic ramp fixture.

This is deliberately separate from ACN-data results.  The fixture is synthetic
only as a positive-control experiment and is not used to replace real data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthetic_ramp_experiment import build_ramp_data
from acn_env import ACNChargingEnv


def rollout(seed: int, forecast: bool, capacity: float) -> dict:
    rng = np.random.default_rng(seed)
    data = build_ramp_data(n_steps=24, n_slots=4, requested_kwh=60.0)
    env = ACNChargingEnv(data, transformer_limit_kw=capacity,
                         per_ev_max_kw=40.0, episode_len=data.n_steps)
    baseline = np.r_[np.full(12, 10.0), np.linspace(10.0, 70.0, 12)]
    noise = rng.normal(0.0, 1.0, size=data.n_steps)
    actual = baseline + noise
    obs, _ = env.reset(seed=seed)
    delivered = 0.0
    violations = 0
    for t in range(data.n_steps):
        headroom = max(0.0, capacity - (baseline[t] if forecast else actual[t]))
        action = np.full(4, min(10.0, headroom) if (forecast or t >= 12) else 0.0)
        obs, _, terminated, truncated, info = env.step(action)
        delivered += float(info["delivered_kwh"].sum())
        violations += int(info["violation_kw"] > 0)
        if terminated or truncated:
            break
    return {"seed": seed, "condition": "forecast" if forecast else "no_forecast",
            "capacity_kw": capacity, "delivered_kwh": delivered,
            "violations": violations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--capacity", type=float, default=50.0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "outputs" / "synthetic_ramp_positive_control.json")
    args = parser.parse_args()
    rows = [rollout(seed, forecast, args.capacity)
            for seed in args.seeds for forecast in (False, True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2))
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
