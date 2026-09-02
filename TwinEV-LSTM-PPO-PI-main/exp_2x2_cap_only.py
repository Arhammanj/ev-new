# ==========================================
# cap_only reference cell for the 2x2 experiment - non-learning,
# deterministic, single run. Evaluated on the exact same
# forecast-safe shared test weeks as the safe-RL conditions (see
# exp_2x2_run.py's shared_setup()).
# ==========================================

from __future__ import annotations

from train_controller import cap_only_action, evaluate_policy_on_weeks, kw_to_symmetric
from exp_2x2_run import shared_setup, make_env, append_result_row


def main() -> None:
    setup = shared_setup()
    print(f"Shared test weeks: {setup['test_starts']}")
    eval_env = make_env(setup, "no_forecast", seed=999, allowed_start_range=setup["test_range"])
    # eval_env is built with symmetric_action=True (make_env always sets this) so its
    # step() expects external actions in [-1,1] and will re-map them - cap_only_action()
    # returns physical kW, so invert the mapping before submitting it (same pattern as
    # sweep_reward_weights_symmetric.py's cap_only_symmetric_fn) or the wrapper's own
    # clip-to-[-1,1] silently mangles the physical action and produces bogus violations.
    result = evaluate_policy_on_weeks(
        eval_env, setup["test_starts"],
        lambda env, obs: kw_to_symmetric(cap_only_action(env.inner), env.per_ev_max_kw))

    row = {
        "condition": "cap_only", "seed": "na",
        "violation_rate": result["violation_rate"],
        "unmet_fraction": result["unmet_fraction"],
        "delivered_kwh": result["delivered_kwh"],
    }
    append_result_row(row)
    print(f"\n===== RESULT: cap_only =====")
    print(f"  violation_rate={100*result['violation_rate']:.2f}%  "
          f"unmet_fraction={100*result['unmet_fraction']:.2f}%  "
          f"delivered_kwh={result['delivered_kwh']:.1f}")


if __name__ == "__main__":
    main()
