# ==========================================
# Instrument per-step energy-reward vs violation-penalty magnitudes
# ==========================================
# No training - just rolls an ALL-MAX (uncapped, "try to charge
# everything as hard as possible") policy over the same 5 held-out
# test weeks used throughout this diagnosis, at the ORIGINAL reward
# weights (energy=1, unmet=1, violation=1), and reports the mean
# energy-reward term vs mean violation-penalty term during BUSY hours
# (defined as steps where this uncapped policy's total draw actually
# exceeds the transformer limit - i.e. violation_kw > 0). This is the
# direct, policy-independent test of "does charging during a surge
# look net-negative in reward terms" - using all-max rather than the
# already-collapsed PPO policy, since a policy that already barely
# charges would trivially show near-zero of both terms and answer
# nothing.
# ==========================================

from __future__ import annotations

import numpy as np

from acn_data_loader import load_acn_sessions
from train_controller import (
    DATA_PATH, EPISODE_LEN, ACNWeeklyEpisodeEnv, split_boundaries,
)

ORIGINAL_WEIGHTS = {"energy": 1.0, "unmet": 1.0, "violation": 1.0}
N_TEST_WEEKS = 5


def main() -> None:
    full_data = load_acn_sessions(DATA_PATH, timestep_minutes=60)
    slot_roster = full_data.slot_ids
    n_steps = full_data.n_steps
    train_end, val_end = split_boundaries(n_steps)
    test_range = (val_end, n_steps - EPISODE_LEN)
    lo, hi = test_range
    test_starts = sorted(set(np.linspace(lo, hi - 1, N_TEST_WEEKS).astype(int).tolist()))
    print(f"Held-out test weeks: {test_starts}")
    print(f"Reward weights under test: {ORIGINAL_WEIGHTS}")

    env = ACNWeeklyEpisodeEnv(full_data, slot_roster, test_range,
                             reward_weights=ORIGINAL_WEIGHTS, seed=999)

    all_energy_terms, all_violation_terms = [], []
    busy_energy_terms, busy_violation_terms = [], []
    n_steps_total = 0
    n_busy = 0

    for start_step in test_starts:
        obs, _ = env.reset(start_step=start_step)
        for _ in range(env.episode_len):
            action = np.full(env.inner.n_slots, env.per_ev_max_kw, dtype=np.float32)   # all-max
            obs, reward, terminated, truncated, info = env.step(action)

            energy_term = ORIGINAL_WEIGHTS["energy"] * float(info["delivered_kwh"].sum())
            violation_term = ORIGINAL_WEIGHTS["violation"] * float(info["violation_kw"])

            n_steps_total += 1
            all_energy_terms.append(energy_term)
            all_violation_terms.append(violation_term)

            if info["violation_kw"] > 0:
                n_busy += 1
                busy_energy_terms.append(energy_term)
                busy_violation_terms.append(violation_term)

            if terminated or truncated:
                break

    print(f"\n{n_steps_total} total steps simulated, {n_busy} were BUSY "
          f"(all-max policy's draw exceeded the {env.transformer_limit_kw} kW limit, "
          f"{100*n_busy/n_steps_total:.1f}% of steps)")

    print("\n===== ALL STEPS =====")
    print(f"  mean energy-reward term:     {np.mean(all_energy_terms):+.3f}")
    print(f"  mean violation-penalty term: {-np.mean(all_violation_terms):+.3f}")

    print("\n===== BUSY (violating) STEPS ONLY =====")
    mean_e = np.mean(busy_energy_terms)
    mean_v = np.mean(busy_violation_terms)
    print(f"  mean energy-reward term:     {mean_e:+.3f}  (per step, this step's kWh delivered)")
    print(f"  mean violation-penalty term: {-mean_v:+.3f}  (per step, -violation_weight * excess kW)")
    print(f"  net per busy step (energy - violation): {mean_e - mean_v:+.3f}")
    print(f"  ratio |violation term| / energy term: {mean_v / mean_e:.2f}x")

    if mean_v > mean_e:
        print(f"\nCONFIRMED: during busy hours, the violation penalty "
              f"({mean_v:.2f}) outweighs the energy reward ({mean_e:.2f}) for "
              f"the SAME step's aggressive charging - net reward for pushing "
              f"through a surge is NEGATIVE. This supports the hypothesis: a "
              f"rational reward-maximizing policy should indeed back off "
              f"during busy hours at these weights.")
    else:
        print(f"\nNOT CONFIRMED: energy reward ({mean_e:.2f}) already exceeds "
              f"the violation penalty ({mean_v:.2f}) during busy hours at the "
              f"original weights - the per-step imbalance hypothesis does not "
              f"explain the collapse; look elsewhere (e.g. value-function/"
              f"reward-scale issues already flagged from the unmet sweep).")


if __name__ == "__main__":
    main()
