import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from exp_utils import TwinEVEnv, run_policy_sequential


def ramp_metrics(p):
    d = np.abs(np.diff(p))
    return float(d.mean()), float(d.max())


def main():
    model = PPO.load("results/models/exp1_modelB_full")
    out = run_policy_sequential(model, TwinEVEnv)
    sp = out["setpoint"]
    actual_pi = out["power"]
    direct = np.clip(sp, 0, 150)

    rows = []
    for name, p in [("direct_setpoint_no_pi", direct),
                    ("pi_output", actual_pi)]:
        mr, xr = ramp_metrics(p)
        rows.append({
            "command": name,
            "mean_ramp_kw": round(mr, 2),
            "max_ramp_kw": round(xr, 2),
            "power_variance": round(float(np.var(p)), 2),
            "peak_power_kw": round(float(np.max(p)), 2),
            "mean_power_kw": round(float(np.mean(p)), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/ablation/pi_isolated_same_command.csv", index=False)
    print(df.to_string(index=False))
    print("Saved results/ablation/pi_isolated_same_command.csv")


if __name__ == "__main__":
    main()
