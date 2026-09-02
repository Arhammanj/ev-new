# ==========================================
# Sanity-check the forecast integration BEFORE trusting the 2x2's null
# result (forecast doesn't beat no_forecast beyond seed noise).
# ==========================================
# Diagnosis only - no retuning, no rerunning the 2x2. Distinguishes
# three independent failure modes, any one of which would explain the
# null result WITHOUT it meaning "anticipation doesn't help":
#
#   1. Is the forecast feature informative in the observation? (not
#      accidentally zeroed / clipped flat / mis-normalized)
#   2. Is the forecast predictive of what the controller actually
#      needs - future transformer-limit PRESSURE (total achievable
#      draw vs the 50 kW cap) - or does it predict something else?
#      NOTE: acn_target_a_bilstm_full.csv's "predicted" column is a
#      6h-ahead forecast of sum(NEW ARRIVAL kWh demand over the next
#      6h) (see generate_forecast_feature.py / build_demand_forecaster
#      .build_target_series()) - a cumulative NEW-DEMAND proxy, not a
#      direct forecast of instantaneous transformer draw. Part 2 tests
#      whether that proxy actually leads the moments the controller
#      binds.
#   3. Does this data even offer exploitable lead time - are the EVs
#      present at a violation-risk hour already connected (with spare
#      system capacity) in the preceding hours, or do they arrive too
#      late / find no earlier slack for pre-charging to be physically
#      possible regardless of forecast quality?
#
# "Transformer-limit pressure" / "achievable draw" here = the GREEDY,
# UNCAPPED total kW every present EV would draw if charged as fast as
# its own physical/energy limits allow (min(per_ev_max_kw,
# remaining_kwh/dt) per slot, summed) - the same "greedy achievable"
# quantity acn_env.py's docstring uses to justify the 50 kW limit
# choice, and the same pre-clip target cap_only_action() computes
# before its proportional scale-down.
# ==========================================

from __future__ import annotations

import numpy as np

from exp_2x2_run import shared_setup
from train_controller import ACNWeeklyEpisodeEnv, TRANSFORMER_LIMIT_KW, PER_EV_MAX_KW

HORIZONS = [1, 3, 6]


def greedy_rollout(setup, start_step: int):
    """Run the GREEDY (uncapped) reference policy over one test week,
    recording per-hour: wall-clock timestamp, achievable_kw (pre-limit
    total draw), violation_kw, presence, and remaining_kwh - all
    BEFORE that hour's delivery (i.e. the state the controller would
    have had to act on)."""
    env = ACNWeeklyEpisodeEnv(
        setup["full_data"], setup["slot_roster"], setup["test_range"],
        seed=0, symmetric_action=False,  # raw-kW action space, no rescale mapping involved
    )
    env.reset(start_step=start_step)

    timestamps, achievable_kw, violation_kw = [], [], []
    presence, remaining = [], []
    for _ in range(env.episode_len):
        present = env.inner.true_presence().copy()
        remaining_kwh = env.inner.true_remaining_kwh().copy()
        max_deliverable = np.where(
            present, np.minimum(PER_EV_MAX_KW, remaining_kwh / env.inner.dt_hours), 0.0)
        timestamps.append(env.inner.current_time())
        achievable_kw.append(float(max_deliverable.sum()))
        violation_kw.append(max(0.0, float(max_deliverable.sum()) - TRANSFORMER_LIMIT_KW))
        presence.append(present)
        remaining.append(remaining_kwh)

        _, _, terminated, truncated, _ = env.step(max_deliverable.astype(np.float32))
        if terminated or truncated:
            break

    return {
        "timestamps": timestamps,
        "achievable_kw": np.array(achievable_kw),
        "violation_kw": np.array(violation_kw),
        "presence": np.array(presence),   # (T, n_slots) bool
        "remaining": np.array(remaining),  # (T, n_slots) float
    }


def part1_forecast_variance(setup, weeks):
    print("=" * 70)
    print("PART 1 - Is the forecast feature informative in the observation?")
    print("=" * 70)
    fc = setup["fc"]
    vals = []
    for wk in weeks:
        for ts in wk["timestamps"]:
            vals.append(fc(ts))
    vals = np.array(vals)
    print(f"n={len(vals)} hours across {len(weeks)} test weeks")
    print(f"mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")
    pct = np.percentile(vals, [1, 10, 25, 50, 75, 90, 99])
    print(f"percentiles [1,10,25,50,75,90,99] = {np.round(pct, 4).tolist()}")
    frac_at_lower = float(np.mean(vals <= -0.999))
    frac_at_upper = float(np.mean(vals >= 2.999))
    frac_exact_zero = float(np.mean(vals == 0.0))
    print(f"fraction saturated at clip floor (<=-0.999): {100*frac_at_lower:.1f}%")
    print(f"fraction saturated at clip ceiling (>=2.999): {100*frac_at_upper:.1f}%")
    print(f"fraction exactly 0.0: {100*frac_exact_zero:.1f}%")
    near_constant = vals.std() < 0.01
    print(f"\n-> Near-constant / saturated (std < 0.01)? {'YES - BROKEN' if near_constant else 'NO - varies normally'}")
    return vals


def pearson(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def part2_predictive_alignment(setup, weeks):
    print("\n" + "=" * 70)
    print("PART 2 - Is the forecast predictive of future transformer pressure?")
    print("=" * 70)
    fc = setup["fc"]
    print("Correlating forecast(t) [normalized 6h-ahead NEW-ARRIVAL-demand "
          "prediction] against ACTUAL future achievable draw / violations "
          "over the next 1h/3h/6h (greedy-uncapped reference).")

    for h in HORIZONS:
        fvals, y_max, y_mean, y_violsum, y_violany = [], [], [], [], []
        for wk in weeks:
            T = len(wk["achievable_kw"])
            for i in range(T - h):
                fvals.append(fc(wk["timestamps"][i]))
                window = wk["achievable_kw"][i + 1: i + 1 + h]
                vwindow = wk["violation_kw"][i + 1: i + 1 + h]
                y_max.append(window.max())
                y_mean.append(window.mean())
                y_violsum.append(vwindow.sum())
                y_violany.append(1.0 if vwindow.max() > 0 else 0.0)
        fvals = np.array(fvals)
        r_max = pearson(fvals, np.array(y_max))
        r_mean = pearson(fvals, np.array(y_mean))
        r_violsum = pearson(fvals, np.array(y_violsum))
        r_violany = pearson(fvals, np.array(y_violany))
        base_rate = float(np.mean(y_violany))
        print(f"\n  horizon={h}h (n={len(fvals)}):")
        print(f"    corr(forecast, MAX achievable_kw in window)   = {r_max:+.3f}")
        print(f"    corr(forecast, MEAN achievable_kw in window)  = {r_mean:+.3f}")
        print(f"    corr(forecast, SUM violation_kw in window)    = {r_violsum:+.3f}")
        print(f"    corr(forecast, violation-occurs-in-window)    = {r_violany:+.3f} "
              f"(base rate {100*base_rate:.1f}% of windows contain a violation)")
    print("\n-> If these correlations are weak/near-zero despite Part 1 showing "
          "real variance, the forecast is informative about SOMETHING but not "
          "aligned with when the controller actually binds.")


def part3_lead_time_feasibility(setup, weeks):
    print("\n" + "=" * 70)
    print("PART 3 - Does the data offer exploitable lead time for pre-charging?")
    print("=" * 70)
    total_hours = sum(len(wk["achievable_kw"]) for wk in weeks)
    total_violation_hours = sum(int((wk["violation_kw"] > 0).sum()) for wk in weeks)
    print(f"Violation-risk hours (greedy achievable > {TRANSFORMER_LIMIT_KW} kW): "
          f"{total_violation_hours} / {total_hours} hours "
          f"({100*total_violation_hours/total_hours:.1f}%)")

    for h in HORIZONS:
        n_pairs = 0
        n_present_earlier = 0
        slacks = []
        for wk in weeks:
            T = len(wk["achievable_kw"])
            for i in range(T):
                if wk["violation_kw"][i] <= 0:
                    continue
                present_at_i = wk["presence"][i]
                contributing = np.where(present_at_i & (wk["remaining"][i] > 0))[0]
                j = i - h
                for s in contributing:
                    n_pairs += 1
                    if j >= 0 and wk["presence"][j][s]:
                        n_present_earlier += 1
                        slack = max(0.0, TRANSFORMER_LIMIT_KW - wk["achievable_kw"][j])
                        slacks.append(slack)
        frac_present = n_present_earlier / n_pairs if n_pairs else float("nan")
        mean_slack = float(np.mean(slacks)) if slacks else float("nan")
        median_slack = float(np.median(slacks)) if slacks else float("nan")
        print(f"\n  {h}h before each violation-risk hour (n={n_pairs} "
              f"violation-hour x contributing-EV pairs):")
        print(f"    already connected {h}h earlier: {100*frac_present:.1f}% "
              f"({'NO LEAD TIME - cars mostly arrive too late' if frac_present < 0.3 else 'lead time generally exists'})")
        if slacks:
            print(f"    among those: system slack {h}h earlier "
                  f"(={TRANSFORMER_LIMIT_KW}kW - achievable_kw) "
                  f"mean={mean_slack:.1f} kW  median={median_slack:.1f} kW "
                  f"({'ROOM to pre-charge' if median_slack > 2.0 else 'system ALREADY near/at capacity - no room to shift load earlier'})")
        else:
            print("    (no cases with lead time to evaluate slack)")

    print("\n-> Anticipation is physically possible only where BOTH lead time "
          "exists AND meaningful preceding-hour slack exists. If either is "
          "absent, no controller - however well-forecasted - could have "
          "avoided the violation by pre-charging.")


def main() -> None:
    print("Loading full-year ACN data + forecast lookup + building greedy "
          "reference rollouts over the held-out test weeks...")
    setup = shared_setup()
    print(f"Test weeks (forecast-safe, same as the 2x2): {setup['test_starts']}\n")

    weeks = [greedy_rollout(setup, start) for start in setup["test_starts"]]

    part1_forecast_variance(setup, weeks)
    part2_predictive_alignment(setup, weeks)
    part3_lead_time_feasibility(setup, weeks)

    print("\n" + "=" * 70)
    print("This is a DIAGNOSIS ONLY - no retuning, no 2x2 rerun. See each "
          "part's verdict line above to determine which premise (if any) "
          "fails before treating the 2x2's null result as final.")
    print("=" * 70)


if __name__ == "__main__":
    main()
