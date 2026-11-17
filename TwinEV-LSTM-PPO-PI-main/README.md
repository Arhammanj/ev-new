# Safe-RL EV Charging Control on Real ACN-Data

A reinforcement-learning charging controller for electric-vehicle charging stations,
trained and evaluated on **real session records** from Caltech's ACN-Data (2019),
under a hard transformer-capacity constraint it must learn to respect. Benchmarked
against a non-learning hard-cap controller, with and without a 6-hours-ahead BiLSTM
demand forecast.

**Headline finding:** the safe-RL controller does not beat a simple hard-cap baseline
on this problem, and a 6h-ahead demand forecast provides no measurable benefit to it
— but a targeted diagnostic (§ Forecast diagnostic below) shows this is because the
*data* offers no exploitable lead time for anticipation, not because the forecast or
its integration is broken. The full reasoning trail — including two rejected
hypotheses and one invalidated environment along the way — is in
[`PROJECT_LOG.md`](PROJECT_LOG.md).

## Why this project reads the way it does

This started as a synthetic-data digital twin (see [Origin](#origin-a-rejected-synthetic-environment)
below) and was rebuilt from scratch once the synthetic environment turned out to make
the safety problem impossible to fail. What's kept here isn't just the final numbers —
it's the process: a policy collapse traced to a root cause in the action-space
parameterization rather than the reward, a reward sweep run to its ceiling and
reported as such, and a null result on the forecast that was stress-tested with an
independent diagnostic before being accepted. `PROJECT_LOG.md` documents all of it in
order, including the dead ends.

## Results (locked config: `energy=1, unmet=2, violation=6`, 300k PPO steps, 5 seeds)

| Condition | Violation rate | Unmet fraction | Delivered kWh |
|---|---|---|---|
| `cap_only` (non-learning hard cap) | 0.00% | 17.46% | 2161.6 |
| Safe-RL PPO, no forecast (n=5 seeds) | 3.29 ± 0.94% | 27.61 ± 1.30% | 1884.3 ± 34.9 |
| Safe-RL PPO, + forecast (n=5 seeds) | 3.00 ± 0.53% | 27.35 ± 1.21% | 1899.0 ± 24.3 |

- The forecast vs. no-forecast difference (−0.29pp violations, +14.7 kWh) is within
  seed-to-seed noise on both axes — no measurable benefit at this budget.
- Neither learned condition beats `cap_only` on either axis.
- See `PROJECT_LOG.md` §6–7 for the full multi-seed table, the reward-weight sweep
  that led to this config, and the forecast-integration diagnostic that explains
  *why* the forecast doesn't help (spoiler: the arrival surges that cause violations
  happen too fast for any forecast horizon to give a controller room to pre-charge).

### Seed stability in the legacy FULL TwinEV pipeline (not the newer ACN reviewer experiments)

The original FULL TwinEV pipeline — the archived synthetic LSTM+PPO+PI setup, not
the real-ACN `ACNChargingEnv` experiments above — was repeated with five
independent PPO seeds on the same test set. The seed spread separates cleanly
into stable and unstable metrics. Peak power was tight at **115.54 ± 2.64 kW**
(CV 2.3%), while delivered energy was **109,712.15 ± 5,201.74 kWh** (CV 4.7%)
and evaluation reward was **4,108.67 ± 458.74** (CV 11.2%). Seeds 4–5 delivered
about 8% more energy than seeds 1–3 with roughly 12x lower power variance
(33 vs 395 kW²), so they converged to a visibly smoother charging profile
rather than adding noise around one common solution. Evaluation reward tracked
delivered energy (r = 0.97) and anti-tracked power variance (r = -0.98);
final training reward was tight (349.13 ± 5.92) but nearly uncorrelated with
evaluation reward (r = 0.14), so it is not a reliable model-selection proxy.

**Interpretation:** on this legacy pipeline, single-seed energy/reward comparisons
can move by roughly 5–11% and are not trustworthy; multi-seed reporting is
required. Peak power and the overload count are seed-stable. The zero-overload
result (0 hard, 0 soft for all five seeds) is **not** evidence of learned
safety, however: the archived FULL TwinEV environment hard-clips delivered power
at 150 kW, making zero overloads true by construction. This experiment is
therefore reported only as a training-variability diagnostic for the older FULL
TwinEV pipeline, separate from the ACN reviewer experiments above.

Figure:
[`paper_figures/legacy_exp5_twinev_seed_stability.pdf`](paper_figures/legacy_exp5_twinev_seed_stability.pdf)
([PNG](results/plots/twinev_seed_stability.png)). Per-seed data and aggregate
summary:
[`results/statistics/twinev_seed_stability.csv`](results/statistics/twinev_seed_stability.csv)
and
[`results/statistics/twinev_seed_stability_summary.csv`](results/statistics/twinev_seed_stability_summary.csv).
Regenerate the figure from the saved CSV without retraining:
`python plot_exp5_seed_stability.py`.

## Architecture

```
ACN-Data session records (real 2019 Caltech charging sessions)
        |
        v
ACNChargingEnv  (acn_env.py)
  - per-slot presence / remaining demand / time-to-departure
  - physical clipping only: per-EV rate limit, energy conservation
  - transformer limit is a MEASURED signal, never used to clip output
        |
        v
ACNWeeklyEpisodeEnv  (train_controller.py)
  - fixed obs/action space across randomly sampled one-week episodes
  - symmetric [-1, 1] external action space -> kw = (a+1)/2 * per_ev_max_kw
  - optional BiLSTM 6h-ahead forecast feature appended to the observation
        |
        +--------------------------+
        v                          v
  cap_only baseline          PPO (Stable-Baselines3)
  (greedy + proportional     safe-RL controller
   scale-down to the limit)  (energy/unmet/violation reward)
        |                          |
        +------------+-------------+
                     v
        evaluated on identical held-out test weeks
```

## Repository structure

**Core pipeline (current, active):**
| File | Purpose |
|---|---|
| `acn_data_loader.py` | Loads and caches ACN-Data session records into fixed per-slot arrays |
| `acn_env.py` | `ACNChargingEnv` — the demand-coupled Gymnasium environment; has its own runnable self-test |
| `train_controller.py` | `ACNWeeklyEpisodeEnv` wrapper, the symmetric action-space mapping, `cap_only_action()`, evaluation harness |
| `build_demand_forecaster.py`, `generate_forecast_feature.py`, `test_alternative_targets.py` | BiLSTM 6h-ahead demand forecaster and its held-out feature file |
| `validate_symmetric_action.py` | Validates the symmetric action-space fix against the action-floor patch it replaced |
| `sweep_reward_weights_symmetric.py` | The reward-weight sweep that locked `energy=1, unmet=2, violation=6` |
| `exp_2x2_run.py`, `exp_2x2_cap_only.py`, `exp_2x2_report.py` | The full multi-seed {forecast, no_forecast} × {cap_only, safe-RL} experiment and its aggregation report |
| `diagnose_forecast.py` | Forecast diagnostic plus continuous flexibility analysis, saturation-onset lead time, 50/75/100 kW criterion sweep, distributions, and joint demand/headroom plots |
| `exp_safety_filtered_ppo.py` | Compares ordinary penalty PPO against PPO with a hard capacity projection safety layer |
| `benchmark_baselines.py` | Evaluates earliest-deadline-first and rolling-horizon MPC-style scheduling baselines |
| `synthetic_ramp_experiment.py` | Reproducible gradual-demand ramp fixture for testing whether anticipation can help when lead time exists |
| `instrument_action_collapse.py`, `instrument_reward_terms.py` | Diagnostics that found the action-space-asymmetry root cause and ruled out reward-content explanations |

**Legacy (superseded, kept for history — not part of the current pipeline):**
`baseline.py`, `pi_controller.py`, `evaluate.py`, `lstm_forecast.py`/`lstm_forecast_torch.py`,
`exp5_seed_stability.py`, and `plot_exp5_seed_stability.py` at the repo root, plus
everything under `archive/`, belong to the original synthetic digital-twin design
(see below) and are not imported by anything in the current pipeline. They're left
in place rather than deleted so the project's history stays traceable; a future
cleanup pass would move the root-level ones into `archive/` too.

### Origin: a rejected synthetic environment

The project originally simulated EV charging on fully synthetic hourly data
(`archive/generate_data.py`, `archive/twinev_env.py`) with an LSTM forecaster feeding
a PPO agent whose output passed through a PI controller. That environment had a
structural flaw: delivered power depended on the agent's action **and the PI
controller's own previous output**, never on real physical demand, and the total was
hard-clipped to a 150 kW cap — which made "zero grid overloads" trivially true
regardless of what the policy did. The safety problem being studied literally could
not fail, so no result from it meant anything. Rather than patch around this, the
environment was rebuilt on real ACN-Data with delivery coupled to actual session
demand and an unclipped transformer-limit signal (`acn_env.py`) — see `PROJECT_LOG.md`
§0–1 for the full reasoning.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Validate the symmetric action-space fix against the prior action-floor patch
python validate_symmetric_action.py

# 2. Reward-weight sweep (locks energy=1, unmet=2, violation=6)
python sweep_reward_weights_symmetric.py

# 3. Full 2x2, multi-seed (5 seeds x {no_forecast, forecast}, 300k steps each)
python exp_2x2_cap_only.py
for seed in 1 2 3 4 5; do
  python exp_2x2_run.py --condition no_forecast --seed $seed
  python exp_2x2_run.py --condition forecast    --seed $seed
done
python exp_2x2_report.py

# 4. Forecast-integration diagnostic (no training required)
python diagnose_forecast.py
```

Additional reviewer experiments:

```bash
# Hard safety layer versus ordinary penalty PPO (also writes loadable .zip checkpoints)
python exp_safety_filtered_ppo.py --seed 1 --timesteps 300000

# Publication-scale five-seed aggregate (run sequentially)
python exp_safety_filtered_ppo.py --seeds 1 2 3 4 5 --timesteps 300000

# Constraint-respecting conventional scheduling benchmarks
python benchmark_baselines.py --capacity-kw 50 --horizon-hours 6

# Historical one-step LP (the default is now finite-horizon MPC)
python benchmark_baselines.py --capacity-kw 50 --horizon-hours 6 --mpc-mode one_step

# Synthetic gradual-ramp sanity experiment
python synthetic_ramp_experiment.py
```

The PPO runner already supports `--horizon 1`, `--horizon 3`, and
`--horizon 6` for forecast experiments. To pull another ACN site, set
`ACN_SITE=jpl` or `ACN_SITE=office001` before running `pull_acn_data.py`;
the loader and evaluation pipeline use the downloaded CSV's site-neutral
schema. JPL retained requested-energy values for 96.8% of its 2019 rows in
the verification run; Office001 retained none, so Office001 falls back to
`kWhDelivered` and is not directly comparable to Caltech's requested-demand
experiments without an explicit demand-assumption change.

Cross-site forecast artifacts can be generated without comparing against
Caltech's reference metric:

```bash
python generate_forecast_feature.py --data-path acn_jpl_2019.csv \
  --horizon 6 --output-csv acn_jpl_target_a_h6_bilstm_full.csv
python exp_2x2_run.py --condition forecast --seed 1 --data-path acn_jpl_2019.csv \
  --forecast-csv acn_jpl_target_a_h6_bilstm_full.csv
python exp_2x2_cap_only.py --data-path acn_jpl_2019.csv
```

Use the analogous Office001 paths for a technically runnable comparison, but
report its fallback-to-delivered-demand assumption separately.

### Reproducibility, leakage, and temporal resolution

The locked PPO configuration is:

```text
learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10
gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01
vf_coef=0.5, max_grad_norm=0.5, policy net=[256, 256, 128]
```

PPO samples Gaussian actions during training. Reported policy evaluation uses
`deterministic=True`; it does not sample from the learned action distribution.
The safety-filter comparison uses the same deterministic evaluation convention.

Run the audit and save its alignment figure under `paper_figures/`:

```bash
python verify_reproducibility.py
```

Forecast timestamps are normalized to UTC, strictly increasing, unique, and
aligned to the same grid as the environment. The forecast target is excluded
from the BiLSTM input sequence; only past arrival-demand lags, rolling
statistics, and calendar features are inputs. The scaler and model are fit on
chronological training rows only. Test rows are evaluated after training and
are never used as training targets or inputs. The audit writes
`reproducibility_audit.json` and `forecast_alignment_audit.{png,pdf}`.

The cached ACN session timestamps have sub-hour precision, so 15-minute
experiments are feasible through the loader and environment. Generate a
matching 15-minute forecast and run the same controller at that resolution:

```bash
python generate_forecast_feature.py --timestep-minutes 15 \
  --output-csv acn_target_a_15m_bilstm_full.csv
python exp_2x2_cap_only.py --timestep-minutes 15
python exp_2x2_run.py --condition forecast --seed 1 --timestep-minutes 15 \
  --forecast-csv acn_target_a_15m_bilstm_full.csv
```

The 15-minute runner uses a 168-hour episode (`672` steps) and separate
checkpoint/result directories. Compare real-data grid pressure before training:

```bash
python resolution_comparison.py
```

This writes hourly-versus-15-minute metrics and
`resolution_comparison.{png,pdf}` under `paper_figures/`. Hourly forecast
artifacts are never reused for 15-minute runs.

The benchmark's default MPC is a receding finite-horizon linear allocation
using only currently connected EVs, their remaining demand/deadlines,
per-EV limits, and the capacity constraint. It does not use future actual
sessions or a forecast trajectory. The historical urgency-weighted one-step
LP remains available with `--mpc-mode one_step`. The synthetic ramp script
also runs both policies through `ACNChargingEnv`; its external background
load remains a controlled fixture rather than real ACN data.

The diagnostic also writes `acn_flexibility_records.csv`,
`acn_flexibility_summary.csv`, `acn_saturation_periods.csv`, and
`acn_flexibility_distributions.png`. The flexibility criterion reports
controllable EV demand, cumulative headroom energy over each lead-time
window, usable pre-charge energy, and a continuous flexibility fraction at
50, 75, and 100 kW. Saturation periods are detected as contiguous
above-capacity intervals; useful lead time counts only pre-onset hours where
the EV was already present and positive headroom was available.

`exp_2x2_run.py` trains in resumable timestep chunks (`--chunk-timesteps`, default
150k) and checkpoints to `acn_exp_2x2_checkpoints/` — re-running the same command
continues from the last checkpoint rather than restarting, and results are written
one JSON file per (condition, seed) to `acn_exp_2x2_results/` so parallel runs never
race on a shared file.

## Data

[ACN-Data](https://ev.caltech.edu/dataset) session records from Caltech's 2019
charging network, cached locally as `acn_caltech_2019.csv`. Split chronologically
(70% train / 15% validation-buffer, unused / 15% test) — the same boundaries are used
by every script in the pipeline, and the test weeks are held out from all training.
The transformer limit (50 kW) is a deliberate scenario choice, not the site's real
rating: a greedy "charge everyone as fast as possible" policy peaks at 137.6 kW
across the full year, so a more realistic 150 kW limit is never binding. 50 kW is the
smallest limit that keeps the constraint genuinely live (violated in ~8-10% of hours
under a greedy policy) without being hopeless.

## Tech stack

Python, [Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO),
PyTorch (BiLSTM forecaster), pandas / NumPy / scikit-learn.

## Status and honest limitations

- Reward-weight tuning is at its ceiling in this setup: a 10x change in the violation
  penalty moved the violation rate by under 1 percentage point.
- Neither safe-RL condition beats the `cap_only` hard-cap reference on delivered
  energy or violation rate, at 300k steps / 5 seeds.
- The forecast doesn't help *on this data* — this is not a general claim about
  forecast-augmented control; § `diagnose_forecast.py` traces it to the arrival
  dynamics offering no exploitable lead time, not to a broken or misaligned forecast.
- Safety-filtered PPO has been implemented and verified in focused tests and a
  2,048-step smoke comparison; a publication-scale multi-seed run is still pending.
- EDF and finite-horizon MPC benchmarks are implemented at 50 kW. Existing
  benchmark JSON files were produced by the historical one-step mode; rerun the
  benchmark with its default mode for finite-horizon results.
- JPL and Office001 data downloads and loader/environment checks passed. Full
  flexibility diagnostics and controller experiments on those sites remain pending;
  Office001 has no requested-energy field and is therefore not directly comparable.
- The synthetic ramp is a reproducible controlled fixture and now includes an
  ACN/Gym environment rollout; it is not a substitute for a real-data study.
- Shorter-horizon controller experiments (1h and 3h) and additional independent
  controller runs remain pending. New safety-comparison runs save loadable PPO
  binaries under `acn_safety_filtered_checkpoints/`; older sidecar-only runs
  cannot be reconstructed without retraining.
