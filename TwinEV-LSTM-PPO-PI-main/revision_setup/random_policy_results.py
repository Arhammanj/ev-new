"""Seeded random-policy evaluation on the Table 3 held-out test weeks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_2x2_run import shared_setup
from train_controller import ACNWeeklyEpisodeEnv, cap_only_action, evaluate_policy_on_weeks, kw_to_symmetric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "random_policy_results.csv")
    args = parser.parse_args()

    setup = shared_setup()
    test_starts = setup["test_starts"]
    rows = []

    sanity_env = ACNWeeklyEpisodeEnv(
        setup["full_data"], setup["slot_roster"], setup["test_range"],
        transformer_limit_kw=50.0, reward_weights={"energy": 1.0},
        seed=999, symmetric_action=True, episode_len=setup["episode_len"],
    )
    cap_result = evaluate_policy_on_weeks(
        sanity_env,
        test_starts,
        lambda env, obs: kw_to_symmetric(
            cap_only_action(env.inner), env.per_ev_max_kw
        ),
    )
    expected_delivered = 2161.602245512676
    expected_unmet = 0.17462185978695935
    if not (
        np.isclose(cap_result["delivered_kwh"], expected_delivered, atol=1e-6)
        and np.isclose(cap_result["unmet_fraction"], expected_unmet, atol=1e-9)
    ):
        raise RuntimeError(
            "Table 3 cap_only sanity check failed. "
            f"Used test starts {test_starts}; got delivered="
            f"{cap_result['delivered_kwh']}, unmet={cap_result['unmet_fraction']}."
        )

    for capacity in (50.0, 75.0, 100.0):
        for seed in range(args.seeds):
            env = ACNWeeklyEpisodeEnv(
                setup["full_data"], setup["slot_roster"], setup["test_range"],
                transformer_limit_kw=capacity, reward_weights={"energy": 1.0},
                seed=10_000 + seed, symmetric_action=True,
                episode_len=setup["episode_len"],
            )
            rng = np.random.default_rng(seed)
            result = evaluate_policy_on_weeks(
                env,
                test_starts,
                lambda current_env, obs, random_state=rng: random_state.uniform(
                    -1.0, 1.0, size=current_env.action_space.shape
                ).astype(np.float32),
            )
            rows.append(
                {
                    "row_type": "per_seed",
                    "capacity_kw": capacity,
                    "seed": seed,
                    "test_starts": ";".join(map(str, test_starts)),
                    "violation_rate_mean": result["violation_rate"],
                    "unmet_fraction_mean": result["unmet_fraction"],
                    "delivered_kwh_mean": result["delivered_kwh"],
                }
            )

    per_seed = pd.DataFrame(rows)
    for capacity, group in per_seed.groupby("capacity_kw"):
        rows.append(
            {
                "row_type": "summary",
                "capacity_kw": capacity,
                "seed": "mean_sd",
                "test_starts": ";".join(map(str, test_starts)),
                "violation_rate_mean": group["violation_rate_mean"].mean(),
                "violation_rate_sd": group["violation_rate_mean"].std(ddof=1),
                "unmet_fraction_mean": group["unmet_fraction_mean"].mean(),
                "unmet_fraction_sd": group["unmet_fraction_mean"].std(ddof=1),
                "delivered_kwh_mean": group["delivered_kwh_mean"].mean(),
                "delivered_kwh_sd": group["delivered_kwh_mean"].std(ddof=1),
            }
        )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Table 3 test starts: {test_starts}")
    print(
        "cap_only sanity passed at 50 kW: "
        f"{cap_result['delivered_kwh']:.6f} kWh, "
        f"{100 * cap_result['unmet_fraction']:.6f}% unmet"
    )
    print(result[result["row_type"] == "summary"].to_string(index=False))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
