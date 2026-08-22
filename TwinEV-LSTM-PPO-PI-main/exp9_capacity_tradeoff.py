# ==========================================
# TwinEV v2 - EXPERIMENT 9
# GRID-CAPACITY / SERVICE TRADE-OFF ANALYSIS
# ==========================================
# A. Theoretical service ceiling vs power cap
# B. Greedy controller sweep over caps
# C. Service70 policy diagnostics under caps
#
# Uses the exact same 1,286-hour test window
# (hours 7282-8568) and the existing metric
# definitions.  No existing result or model is
# modified.  Service70 is the already-trained
# model from Experiment 8 (no retraining).
# ==========================================

import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from exp_utils import (
    ensure_dir,
    load_aligned_data,
    model_metrics_row,
    TEST_START,
    TEST_END,
)
from service_env import ServiceConstrainedEnv

OUT = "results/capacity_tradeoff"
PLOT_OUT = "results/plots"

CAPS = [110, 120, 125, 130, 135, 140, 145, 150]
SERVICE70_MODEL = "results/service_constrained/models/twinev_service70.zip"
HARD_LIMIT = 150.0
SOFT_LIMIT = 130.0


def sequential_policy_run(policy, cap, state, start, end):
    """Roll the trained Service70 policy over the test window with the
    PI output clamp set to `cap`.  The trained policy is unchanged;
    only the physical delivery clamp varies (diagnostic only)."""
    env = ServiceConstrainedEnv(mode="test", service_target=0.70)
    env.episode_start = start
    env.max_steps = end - start
    env.current_step = 0
    env.pi.reset()
    env.pi.output_max = float(cap)      # diagnostic: alter physical clamp
    env.actual_power = float(env.grid_load[start])

    obs = env._get_obs(start)
    power, setpoint, reward = [], [], []
    for _ in range(end - start):
        action, _ = policy.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(action)
        power.append(info["actual_power"])
        setpoint.append(info["setpoint_kw"])
        reward.append(r)
    return np.array(power), np.array(setpoint), np.array(reward)


def main():
    ensure_dir(OUT)
    ensure_dir(PLOT_OUT)

    state = load_aligned_data()
    raw_test = state["power"][TEST_START:TEST_END]
    price_test = state["price"][TEST_START:TEST_END]
    solar_test = state["solar"][TEST_START:TEST_END]
    n_hours = TEST_END - TEST_START
    available = float(np.sum(raw_test))
    raw_peak = float(np.max(raw_test))

    # ------------------------------------------
    # EXPERIMENT A - THEORETICAL SERVICE CEILING
    # ------------------------------------------
    theo_rows = []
    for cap in CAPS:
        max_energy = cap * n_hours
        theo_rows.append({
            "power_cap_kw": cap,
            "test_hours": n_hours,
            "total_available_energy_kwh": round(available, 2),
            "maximum_possible_energy_kwh": round(max_energy, 2),
            "maximum_possible_service_pct": round(
                100.0 * max_energy / available, 2),
        })
    theo_df = pd.DataFrame(theo_rows)
    theo_df.to_csv(f"{OUT}/theoretical_service_ceiling.csv", index=False)
    print("Saved results/capacity_tradeoff/theoretical_service_ceiling.csv")
    print(theo_df.to_string(index=False))

    import matplotlib.pyplot as plt
    from plot_style import set_style
    set_style()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(theo_df["power_cap_kw"],
            theo_df["maximum_possible_service_pct"], "o-",
            color="#1f77b4", linewidth=2)
    ax.axvline(SOFT_LIMIT, color="orange", linestyle=":",
               label=f"Soft limit ({SOFT_LIMIT:.0f} kW)")
    ax.axvline(HARD_LIMIT, color="red", linestyle="--",
               label=f"Hard limit ({HARD_LIMIT:.0f} kW)")
    for _, r in theo_df.iterrows():
        ax.annotate(f"{r['maximum_possible_service_pct']:.1f}%",
                    (r["power_cap_kw"], r["maximum_possible_service_pct"]),
                    textcoords="offset points", xytext=(4, -14), fontsize=8)
    ax.set_xlabel("Power cap (kW)")
    ax.set_ylabel("Maximum possible service (%)")
    ax.set_title("Theoretical Service Ceiling vs Grid Power Cap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(PLOT_OUT, "theoretical_service_ceiling.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/plots/theoretical_service_ceiling.png")

    # ------------------------------------------
    # EXPERIMENT B - GREEDY SWEEP
    # ------------------------------------------
    greedy_rows = []
    for cap in CAPS:
        greedy = np.minimum(raw_test, float(cap))
        row = model_metrics_row(
            f"Greedy_{cap}kW",
            power=greedy,
            setpoint=greedy,
            reward=None,
            solar=solar_test,
            price=price_test,
            raw=raw_test,
            extra={"power_cap_kw": cap},
        )
        row["energy_service_pct"] = row["energy_delivery_pct"]
        greedy_rows.append(row)
    greedy_df = pd.DataFrame(greedy_rows)
    greedy_df.to_csv(f"{OUT}/greedy_capacity_sweep.csv", index=False)
    print("\nSaved results/capacity_tradeoff/greedy_capacity_sweep.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(greedy_df["power_cap_kw"], greedy_df["energy_service_pct"],
            "s-", color="#2e8b57", linewidth=2, label="Greedy service (%)")
    ax.plot(theo_df["power_cap_kw"], theo_df["maximum_possible_service_pct"],
            "o--", color="#1f77b4", linewidth=2,
            label="Theoretical ceiling (%)")
    ax.plot(greedy_df["power_cap_kw"], greedy_df["soft_overloads"],
            "^-", color="#d62728", linewidth=2, label="Soft overloads")
    ax.set_xlabel("Power cap (kW)")
    ax.set_ylabel("Service (%)  /  soft overloads (events)")
    ax.set_title("Greedy Controller: Service and Soft Overloads vs Cap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(PLOT_OUT, "greedy_service_vs_cap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/plots/greedy_service_vs_cap.png")

    # ------------------------------------------
    # EXPERIMENT C - Service70 DIAGNOSTICS
    # ------------------------------------------
    print("\nLoading Service70 model for diagnostics...")
    policy = PPO.load(SERVICE70_MODEL)

    service_rows = []
    for cap in CAPS:
        p, sp, rw = sequential_policy_run(policy, cap, state,
                                          TEST_START, TEST_END)
        row = model_metrics_row(
            f"Service70_cap{cap}kW",
            power=p,
            setpoint=sp,
            reward=rw,
            solar=solar_test,
            price=price_test,
            raw=raw_test,
            extra={"power_cap_kw": cap},
        )
        row["energy_service_pct"] = row["energy_delivery_pct"]
        row["req_gt130_pct"] = round(100.0 * np.mean(sp > SOFT_LIMIT), 2)
        row["req_at150_pct"] = round(100.0 * np.mean(sp >= HARD_LIMIT - 0.5), 2)
        row["delivered_at_cap_pct"] = round(
            100.0 * np.mean(p >= cap - 0.5), 2)
        service_rows.append(row)
        print(f"  cap {cap} kW -> service {row['energy_service_pct']:.2f}% | "
              f"hard {row['hard_overloads']} soft {row['soft_overloads']} | "
              f"req>130 {row['req_gt130_pct']:.1f}% | "
              f"req=150 {row['req_at150_pct']:.1f}% | "
              f"delivered@cap {row['delivered_at_cap_pct']:.1f}%")
    service_df = pd.DataFrame(service_rows)
    service_df.to_csv(f"{OUT}/service70_capacity_analysis.csv", index=False)
    print("\nSaved results/capacity_tradeoff/service70_capacity_analysis.csv")

    # ------------------------------------------
    # FINAL SUMMARY TABLE
    # ------------------------------------------
    summary_rows = []
    for i, cap in enumerate(CAPS):
        summary_rows.append({
            "power_cap_kw": cap,
            "theoretical_service_pct": theo_df.iloc[i][
                "maximum_possible_service_pct"],
            "greedy_service_pct": greedy_df.iloc[i]["energy_service_pct"],
            "twinev_service70_pct": service_df.iloc[i]["energy_service_pct"],
            "hard_overloads": int(service_df.iloc[i]["hard_overloads"]),
            "soft_overloads": int(service_df.iloc[i]["soft_overloads"]),
            "energy_delivered_mwh": round(
                service_df.iloc[i]["energy_delivered_kwh"] / 1000.0, 3),
            "charging_cost_eur": service_df.iloc[i]["charging_cost"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUT}/capacity_tradeoff_summary.csv", index=False)
    print("\nSaved results/capacity_tradeoff/capacity_tradeoff_summary.csv")
    print(summary_df.to_string(index=False))

    # Bonus combined plot for the report
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(theo_df["power_cap_kw"],
            theo_df["maximum_possible_service_pct"], "o--",
            label="Theoretical ceiling", color="#1f77b4")
    ax.plot(greedy_df["power_cap_kw"], greedy_df["energy_service_pct"],
            "s-", label="Greedy (min(raw, cap))", color="#2e8b57")
    ax.plot(service_df["power_cap_kw"], service_df["energy_service_pct"],
            "^-", label="TwinEV Service70 (trained at 150 kW)",
            color="#d62728")
    ax.set_xlabel("Grid power cap (kW)")
    ax.set_ylabel("Energy service ratio (%)")
    ax.set_title("Grid-Capacity vs Service Trade-Off (TwinEV v2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(PLOT_OUT, "capacity_tradeoff_summary.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/plots/capacity_tradeoff_summary.png")

    print("\nExperiment 9 complete.")


if __name__ == "__main__":
    main()
