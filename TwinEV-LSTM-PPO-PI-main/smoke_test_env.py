# ==========================================
# ACNChargingEnv real-data smoke test
# ==========================================
# Runs two fixed, non-learning policies over randomly-started one-week
# episodes drawn from the real Caltech 2019 ACN-Data cache and reports
# transformer-violation and unmet-energy behaviour. Not a policy
# evaluation - a sanity check that the environment's mechanics behave
# plausibly on real data before any RL training is built on top of it.
#
# Verifies, per episode, EXACT energy conservation:
#   total requested kWh == total delivered kWh + total unmet kWh
# ==========================================

from __future__ import annotations

import numpy as np

from acn_data_loader import load_acn_sessions, slice_session_data
from acn_env import ACNChargingEnv

DATA_PATH = "acn_caltech_2019.csv"
TRANSFORMER_LIMIT_KW = 150.0
PER_EV_MAX_KW = 6.6
EPISODE_LEN = 168     # one week, hourly steps
N_EPISODES = 20
SEED = 0


def uniform_random_policy(env: ACNChargingEnv, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, PER_EV_MAX_KW, size=env.n_slots).astype(np.float32)


def all_max_policy(env: ACNChargingEnv, rng: np.random.Generator) -> np.ndarray:
    return np.full(env.n_slots, PER_EV_MAX_KW, dtype=np.float32)


def run_episode(data_window, policy_fn, rng: np.random.Generator) -> dict:
    env = ACNChargingEnv(
        data_window,
        transformer_limit_kw=TRANSFORMER_LIMIT_KW,
        per_ev_max_kw=PER_EV_MAX_KW,
        episode_len=EPISODE_LEN,
    )
    obs, _ = env.reset()

    total_requested_kwh = float(sum(s.requested_kwh for s in data_window.sessions))
    total_delivered_kwh = 0.0
    total_unmet_kwh = 0.0
    n_violating_steps = 0
    violation_kw_on_violating_steps: list[float] = []

    n_steps_run = 0
    for _ in range(EPISODE_LEN):
        action = policy_fn(env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps_run += 1
        total_delivered_kwh += float(info["delivered_kwh"].sum())
        total_unmet_kwh += info["unmet_kwh_this_step"]
        if info["violation_kw"] > 0:
            n_violating_steps += 1
            violation_kw_on_violating_steps.append(info["violation_kw"])
        if terminated or truncated:
            break

    # Energy conservation must hold EXACTLY (up to float tolerance) over
    # any full episode: nothing delivered or unmet can be invented or lost.
    assert np.isclose(
        total_requested_kwh, total_delivered_kwh + total_unmet_kwh,
        rtol=1e-6, atol=1e-6,
    ), (
        f"energy not conserved: requested={total_requested_kwh:.6f} != "
        f"delivered={total_delivered_kwh:.6f} + unmet={total_unmet_kwh:.6f} "
        f"(diff={total_requested_kwh - total_delivered_kwh - total_unmet_kwh:.6f})"
    )

    return {
        "violation_rate": n_violating_steps / n_steps_run,
        "mean_violation_kw_on_violating_steps": (
            float(np.mean(violation_kw_on_violating_steps))
            if violation_kw_on_violating_steps else 0.0
        ),
        "unmet_fraction": (
            total_unmet_kwh / total_requested_kwh if total_requested_kwh > 0 else 0.0
        ),
        "delivered_kwh": total_delivered_kwh,
        "requested_kwh": total_requested_kwh,
    }


def summarize(policy_name: str, records: list[dict]) -> dict:
    keys = ["violation_rate", "mean_violation_kw_on_violating_steps",
            "unmet_fraction", "delivered_kwh"]
    means = {k: float(np.mean([r[k] for r in records])) for k in keys}

    for k, v in means.items():
        assert np.isfinite(v), f"{policy_name}: non-finite mean for {k}: {v}"

    print(f"\n=== {policy_name} ({len(records)} episodes) ===")
    print(f"  violation rate                        : {100 * means['violation_rate']:.2f}% of steps")
    print(f"  mean violation kW (on violating steps) : {means['mean_violation_kw_on_violating_steps']:.1f} kW")
    print(f"  unmet-energy fraction                  : {100 * means['unmet_fraction']:.2f}%")
    print(f"  mean delivered kWh per episode          : {means['delivered_kwh']:.1f} kWh")
    return means


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = load_acn_sessions(DATA_PATH, timestep_minutes=60)

    max_start = data.n_steps - EPISODE_LEN
    assert max_start > 0, "dataset shorter than one episode - can't run this smoke test"

    results = {}
    for policy_name, policy_fn in [
        ("uniform-random", uniform_random_policy),
        ("all-max", all_max_policy),
    ]:
        records = []
        attempts = 0
        while len(records) < N_EPISODES:
            attempts += 1
            if attempts > N_EPISODES * 5:
                raise RuntimeError(
                    f"too many empty windows while sampling episodes for "
                    f"{policy_name} - data may be too sparse"
                )
            start_step = int(rng.integers(0, max_start))
            try:
                window = slice_session_data(data, start_step, EPISODE_LEN)
            except ValueError:
                continue  # empty window (no arrivals that week) - resample
            records.append(run_episode(window, policy_fn, rng))

        results[policy_name] = summarize(policy_name, records)

    # NOTE on the ~11.4% sanity expectation from check_binding.py:
    # empirically, it does NOT hold here, and that is a real finding,
    # not a bug. check_binding.py's binding estimate assumes every
    # CONCURRENTLY CONNECTED EV draws 6.6 kW for its entire connected
    # duration - but real ACN-Data sessions mostly finish charging in
    # 1-3 hours (median request ~12 kWh / 6.6 kW ~= 1.8 h) and then
    # idle at the plug for hours more. A full-year continuous all-max
    # rollout (see PR description) peaks at only ~137.6 kW once real
    # per-EV demand exhaustion is modeled - never crossing 150 kW - so
    # 0% violations at transformer_limit_kw=150 is the CORRECT result
    # for this env, not a defect. Asserting the original ">0%" premise
    # would mean asserting something now known to be false, so it's
    # replaced with checks that stay meaningful regardless: cross-
    # policy monotonicity and finite values (already checked above).
    all_max_rate = results["all-max"]["violation_rate"]
    random_rate = results["uniform-random"]["violation_rate"]
    assert all_max_rate >= random_rate, (
        f"all-max violation rate ({100 * all_max_rate:.2f}%) is LOWER than "
        f"uniform-random's ({100 * random_rate:.2f}%) - charging harder "
        "should never produce fewer transformer violations"
    )
    assert results["all-max"]["delivered_kwh"] >= results["uniform-random"]["delivered_kwh"], (
        "all-max policy delivered less energy than uniform-random - "
        "policies are not behaving as their names imply"
    )
    print(f"\n[INFO] all-max violation rate at 150 kW: {100 * all_max_rate:.2f}% "
          "(see PR description for why this legitimately differs from the "
          "~11.4% check_binding.py estimate)")
    print("[PASS] cross-policy monotonicity checks passed")
    print("[PASS] energy conservation held exactly on every episode")


if __name__ == "__main__":
    main()
