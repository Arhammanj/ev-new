# ==========================================
# cap_only reference cell for the 2x2 experiment - non-learning,
# deterministic, single run. Evaluated on the exact same
# forecast-safe shared test weeks as the safe-RL conditions (see
# exp_2x2_run.py's shared_setup()).
# ==========================================

from __future__ import annotations

import argparse

from train_controller import cap_only_action, evaluate_policy_on_weeks, kw_to_symmetric
from exp_2x2_run import (
    BASE_HORIZON, BASE_CAPACITY_KW, shared_setup, make_env, append_result_row, run_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[1, 3, 6], default=BASE_HORIZON,
                        help="Forecast lead time in hours - cap_only is a non-learning, "
                             "forecast-independent baseline, but the shared test weeks "
                             "shift slightly per horizon, so it's cheap to re-evaluate.")
    parser.add_argument("--capacity-kw", type=float, default=BASE_CAPACITY_KW,
                        help="Transformer capacity limit in kW - cap_only adapts to "
                             "whatever limit its env is built with (cap_only_action() "
                             "reads env.transformer_limit_kw dynamically), so this is "
                             "always a cheap non-learning re-evaluation, never training.")
    args = parser.parse_args()
    _, results_dir = run_dirs(args.horizon, args.capacity_kw)

    setup = shared_setup(args.horizon)
    print(f"Shared test weeks: {setup['test_starts']}")
    eval_env = make_env(setup, "no_forecast", seed=999, allowed_start_range=setup["test_range"],
                        capacity_kw=args.capacity_kw)
    # eval_env is built with symmetric_action=True (make_env always sets this) so its
    # step() expects external actions in [-1,1] and will re-map them - cap_only_action()
    # returns physical kW, so invert the mapping before submitting it (same pattern as
    # sweep_reward_weights_symmetric.py's cap_only_symmetric_fn) or the wrapper's own
    # clip-to-[-1,1] silently mangles the physical action and produces bogus violations.
    result = evaluate_policy_on_weeks(
        eval_env, setup["test_starts"],
        lambda env, obs: kw_to_symmetric(cap_only_action(env.inner), env.per_ev_max_kw))

    row = {
        "condition": "cap_only", "seed": "na", "horizon": args.horizon,
        "capacity_kw": args.capacity_kw,
        "violation_rate": result["violation_rate"],
        "unmet_fraction": result["unmet_fraction"],
        "delivered_kwh": result["delivered_kwh"],
    }
    append_result_row(row, results_dir)
    print(f"\n===== RESULT: cap_only horizon={args.horizon}h capacity={args.capacity_kw}kW =====")
    print(f"  violation_rate={100*result['violation_rate']:.2f}%  "
          f"unmet_fraction={100*result['unmet_fraction']:.2f}%  "
          f"delivered_kwh={result['delivered_kwh']:.1f}")


if __name__ == "__main__":
    main()
