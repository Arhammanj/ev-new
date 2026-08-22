# ==========================================
# TwinEV v2 - FULL RETRAIN DRIVER
# Retrains all PPO models under the CORRECTED
# forecast-aligned environment, then regenerates
# every result table / profile / figure.
#
# Resume-safe: any model whose .zip already
# exists is skipped, so the driver can be
# relaunched after an interruption.
#
# Usage:  python retrain_driver.py
# ==========================================

import os
import subprocess
import sys
import time

LOG_DIR = "train_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "retrain.log")

MODEL_DIR = "results/models"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, tag):
    log(f">>> START {tag}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable] + cmd,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    dt = time.time() - t0
    out_tail = "\n".join((proc.stdout or "").splitlines()[-15:])
    err_tail = "\n".join((proc.stderr or "").splitlines()[-15:])
    with open(os.path.join(LOG_DIR, f"retrain_{tag}.out"),
              "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
    with open(os.path.join(LOG_DIR, f"retrain_{tag}.err"),
              "w", encoding="utf-8") as f:
        f.write(proc.stderr or "")
    if proc.returncode != 0:
        log(f"!!! FAILED {tag} (rc={proc.returncode}, "
            f"{dt/60:.1f} min)\n{err_tail}")
        raise SystemExit(1)
    log(f"<<< DONE {tag} in {dt/60:.1f} min\n{out_tail}")


def have(path):
    return os.path.exists(path)


# ==========================================
# PHASE 1 - core ablation models
# ==========================================

CORE = [
    ("exp1_modelA_noforecast", ["noforecast"]),
    ("exp1_modelB_full",       ["twin"]),
    ("exp2_modelC_no_pi",      ["nopi"]),
]
for name, extra in CORE:
    if have(f"{MODEL_DIR}/{name}.zip"):
        log(f"-- skip {name} (exists)")
        continue
    env_cls = extra[0]
    run(["train_single.py", name, env_cls,
         f"{MODEL_DIR}/{name}", "--seed", "1"], f"train_{name}")

# ==========================================
# PHASE 2-5 - experiment trainings
# (each script trains only what is missing)
# ==========================================

run(["exp4_reward_sensitivity.py", "--skip-train"], "exp4")
run(["exp5_seed_stability.py",     "--skip-train"], "exp5")
run(["exp8_service_constrained.py", "--skip-train"], "exp8")
run(["exp10_mo_train.py",          "--skip-train"], "exp10_train")

# ==========================================
# PHASE 6 - regenerate ALL results
# ==========================================

run(["exp1_forecast_ablation.py", "--skip-train"], "exp1_eval")
run(["exp2_pi_ablation.py",       "--skip-train"], "exp2_eval")
run(["exp2b_pi_isolation.py"],                     "exp2b")
run(["exp3_comparison.py"],                        "exp3")
run(["exp6_forecast_comparison.py"],               "exp6")
run(["exp7_stress_test.py"],                       "exp7")
run(["exp9_capacity_tradeoff.py"],                 "exp9")
run(["exp10_mo_eval.py"],                          "exp10_eval")
run(["evaluate.py"],                               "evaluate")
run(["compare_ablation.py"],                       "cmp_ablation")
run(["compare_ablation_final.py"],                 "cmp_ablation_final")
run(["compare_all_models.py"],                     "cmp_all_models")

log("ALL DONE - results regenerated under the corrected environment.")
