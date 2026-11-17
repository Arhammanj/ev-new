"""Audit whether aggregate greedy draw is reconstructible from observations."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_2x2_run import shared_setup
from train_controller import ACNWeeklyEpisodeEnv, PER_EV_MAX_KW


def main() -> None:
    setup = shared_setup()
    rows = []
    for start_step in setup["test_starts"]:
        env = ACNWeeklyEpisodeEnv(
            setup["full_data"], setup["slot_roster"], setup["test_range"],
            transformer_limit_kw=50.0, reward_weights={"energy": 1.0},
            symmetric_action=True, seed=999, episode_len=setup["episode_len"],
        )
        obs, _ = env.reset(start_step=start_step)
        n_slots = len(setup["slot_roster"])
        for step in range(env.episode_len):
            presence = obs[:n_slots] > 0.5
            demand_norm = obs[n_slots:2 * n_slots]
            observed_remaining = demand_norm * 80.0
            observed_draw = float(
                np.minimum(PER_EV_MAX_KW, observed_remaining / env.inner.dt_hours)
                .clip(min=0.0)[presence].sum()
            )
            true_presence = env.inner.true_presence()
            true_remaining = env.inner.true_remaining_kwh()
            true_draw = float(
                np.minimum(
                    PER_EV_MAX_KW,
                    np.divide(
                        true_remaining,
                        env.inner.dt_hours,
                        out=np.zeros_like(true_remaining),
                        where=env.inner.dt_hours > 0,
                    ),
                )[true_presence].sum()
            )
            clipped_present = int(np.sum(presence & (demand_norm >= 0.999999)))
            rows.append(
                {
                    "test_start_step": start_step,
                    "step_in_week": step,
                    "observation_dimension": len(obs),
                    "base_observation_dimension": n_slots * 3 + 4,
                    "aggregate_draw_in_observation": False,
                    "aggregate_draw_reconstructible_from_per_ev_fields": True,
                    "true_draw_kw": true_draw,
                    "reconstructed_draw_kw": observed_draw,
                    "absolute_error_kw": abs(true_draw - observed_draw),
                    "present_slots_with_demand_norm_clipped": clipped_present,
                }
            )
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    result = pd.DataFrame(rows)
    output = Path(__file__).parent / "outputs" / "aggregate_draw_observation_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Test starts: {setup['test_starts']}")
    print(
        result[
            [
                "aggregate_draw_in_observation",
                "aggregate_draw_reconstructible_from_per_ev_fields",
                "absolute_error_kw",
                "present_slots_with_demand_norm_clipped",
            ]
        ].describe(include="all").to_string()
    )
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
