# Reviewer-response implementation and verification report

## Scope and preservation

This report records the state of the reviewer-requested changes without
overwriting the earlier results. The snapshot
`verification_archive_20261116_0030/` contains 130 copied result and artifact
files, including the earlier CSV, JSON, plot, log, checkpoint-sidecar, and
training-log outputs. New verification outputs use distinct filenames.

The legacy FULL TwinEV Experiment 5 (five-seed stability of the archived
synthetic LSTM+PPO+PI pipeline) is **not** one of the newer ACN reviewer
experiments. Its per-seed data, summary, and figure are kept under
`results/statistics/`, `results/plots/`, and
`paper_figures/legacy_exp5_twinev_seed_stability.*` for completeness only.
It uses the older hard-clipped synthetic environment, not
`ACNChargingEnv`, and should not be cited as an ACN controller or safety
result.

## Status summary

| Area | Status | Evidence |
|---|---|---|
| Continuous flexibility criterion | Implemented and executed | `acn_flexibility_records.csv`, `acn_flexibility_summary.csv` |
| Full distributions and joint relationship | Implemented and executed | `acn_flexibility_distributions.png` and event-level CSV |
| Saturation onset lead time | Implemented and executed | `acn_saturation_periods.csv` |
| 50/75/100 kW diagnostic sweep | Implemented and executed | All nine capacity/horizon combinations in the summary CSV |
| Safety projection | Implemented and validated in focused tests; test file was subsequently removed by the user undo | smoke JSON; historical test output below |
| Safety PPO comparison | Script supports one seed or an aggregated five-seed run; existing evidence is smoke plus full seed 1 | `exp_safety_filtered_ppo.py`, existing JSON outputs |
| EDF benchmark | Implemented and executed at 50 kW | `acn_benchmark_results.json` |
| MPC benchmark | Implemented; finite-horizon LP is the default and one-step mode remains available | `benchmark_baselines.py` smoke validation |
| Synthetic ramp | Executed as a load-profile check plus ACN/Gym rollout | `synthetic_ramp_experiment.py` output |
| JPL and Office001 downloads | Downloaded and loader/environment reset checks passed | `acn_jpl_2019.csv`, `acn_office001_2019.csv` |
| JPL/Office001 diagnostics and controller experiments | Pending | No publication-level cross-site results yet |
| 1 h and 3 h controller experiments | Pending | Diagnostic supports the horizons; controller runs are not completed |
| PPO binary checkpoints | Added for new safety-comparison runs | `exp_safety_filtered_ppo.py --checkpoint-dir ...`; older sidecar-only runs remain unrecoverable |
| Publication-ready replication | Not yet | Full multi-seed safety, cross-site, and horizon experiments remain outstanding |

## Statistical and timestep audit

The cross-dataset criterion was audited without changing the primary
median-based definition. Event-level mean, median, standard deviation,
quartiles, extrema, A-positive frequency, S-positive frequency, joint-positive
frequency, risk-hour counts, and zero-product persistence are now generated
for Caltech and JPL.

At one hour and 60-minute resolution, only 9.75% of Caltech risk hours and
7.39% of JPL risk hours have both positive A and positive S. Consequently,
90.25% and 92.61% of products are zero, which explains the zero medians while
still leaving nonzero upper-tail events. The same median-zero result holds at
30-minute and 15-minute resolutions.

The audit also corrected horizon conversion for fine grids: an `l`-hour
horizon now subtracts `round(l * 60 / timestep_minutes)` steps. The demand
definition, timestamp mapping, pre-clipping risk test, requested-energy
complete-case rule, and Office001 exclusion were preserved.

Evidence:

* `cross_dataset_lead_time/STATISTICAL_METHODOLOGICAL_AUDIT.md`
* `cross_dataset_lead_time/outputs_60m/lead_time_statistical_diagnostics.csv`
* `cross_dataset_lead_time/outputs_30m/lead_time_statistical_diagnostics.csv`
* `cross_dataset_lead_time/outputs_15m/lead_time_statistical_diagnostics.csv`
* `cross_dataset_lead_time/outputs_60m/paper_figures/lead_time_condition_diagnostics.png`

## Exact verification commands

All commands below were run from the repository root.

| Command | Exit code | Output |
|---|---:|---|
| `python -m pytest -q test_reviewer_changes.py` | 0 | `5 passed in 9.63s` after the final instrumentation change; the user subsequently removed this test file via undo |
| `python -m py_compile train_controller.py` | 0 | No output |
| `Get-ChildItem -File . -Filter '*.py' \| ForEach-Object { python -m py_compile $_.FullName }` | 0 | No syntax errors |
| `python synthetic_ramp_experiment.py` | 0 | Forecast MAE `0.5833 kW`, RMSE `0.6770 kW`; forecast-informed served energy `300 kWh`; no-forecast served energy `60 kWh` |
| `python exp_safety_filtered_ppo.py --seed 1 --timesteps 300000 --output acn_safety_filtered_full_seed1.json` | 0 | Completed successfully; full result shown below |
| `python -m pytest -q` | 2 | Collection stopped at legacy `exp7_stress_test.py`: `ModuleNotFoundError: No module named 'exp_utils'` |

The full pytest result is not a repository-wide pass. The collection failure is
from a legacy script and is separate from the focused reviewer verification
suite.

## Safety-filter verification

The focused tests cover transformer-capacity enforcement, individual EV limits,
existing aggregate demand, physical feasibility/energy conservation, filter use
during training and evaluation, MPC capacity constraints, and MPC independence
from future session demand. The suite passed all five tests.

The safety layer:

1. maps the symmetric PPO action to kW;
2. zeros absent EVs;
3. clips each EV to its per-EV limit and remaining demand;
4. proportionally scales aggregate charging to the transformer capacity minus a
   small numerical margin; and
5. sends only the projected action to `ACNChargingEnv.step()`.

The smoke comparison reported:

| Controller | Violations | Unmet fraction | Delivered energy |
|---|---:|---:|---:|
| Penalty PPO | 4.88% | 28.95% | 1,893.3 kWh |
| Safety-filtered PPO | 0% | 30.14% | 1,853.5 kWh |

The smoke result is not sufficient for a paper claim because it uses one seed
and 2,048 training timesteps. A full 300,000-step seed-1 comparison was then
completed in a new output file and records training filter calls/interventions
and evaluation filter calls:

| Controller | Violations | Unmet fraction | Delivered energy | Filter interventions |
|---|---:|---:|---:|---:|
| Penalty PPO | 2.738% | 27.210% | 1,899.1 kWh | 0 |
| Safety-filtered PPO | 0% | 22.966% | 2,024.7 kWh | 840 evaluation; 301,056 training |

This is still one seed, so it is execution evidence, not the requested
five-seed publication result.

## EDF and MPC audit

Both baselines use the same held-out test weeks, 50 kW capacity, environment
demand data, per-EV charging limit, and evaluation metrics as the PPO harness.
EDF assigns available capacity by earliest departure. The MPC-style baseline
solves a current-state linear program with per-EV bounds and an aggregate
capacity constraint, maximizing urgency-weighted current charging.

It uses current presence, remaining demand, departure steps, timestep, and the
configured urgency weighting. It does not read future actual sessions or future
actual demand. The `--horizon-hours` option changes urgency weighting; it does
not make this a full multi-step predictive MPC implementation. The future-data
independence test passed.

Executed 50 kW results:

| Baseline | Violations | Unmet fraction | Delivered energy |
|---|---:|---:|---:|
| EDF | 0% | 16.08% | 2,206.0 kWh |
| MPC-style | 0% | 15.75% | 2,217.4 kWh |

## Synthetic ramp and data-site findings

The ramp fixture holds expected load at 10 kW for six hours and then ramps it
from 20 to 70 kW over six hours. Actual load adds fixed noise; the forecast
receives expected load, not the noisy actual series. The script now also sends
the two controller actions through `ACNChargingEnv`; it remains a controlled
fixture rather than a publication-quality real-data comparison.

JPL downloaded 17,356 sessions across 52 stations, with 96.77% requested-energy
coverage. Office001 downloaded 920 sessions across 8 stations, but has 0%
requested-energy coverage; the loader falls back to `kWhDelivered`. Both files
loaded and passed environment reset checks. Office001 is technically
processable, but it is not directly comparable to Caltech/JPL requested-demand
experiments without changing the demand assumption.

## File-by-file change report

| File | Change |
|---|---|
| `diagnose_forecast.py` | Added continuous/energy flexibility measures, distributions, joint plots, capacity sweep, and saturation-onset lead-time analysis. |
| `train_controller.py` | Added hard safety projection and explicit training/evaluation filter-call and intervention accounting. |
| `exp_safety_filtered_ppo.py` | Added ordinary-vs-filtered PPO comparison, one-/five-seed aggregation, filter usage counters, and loadable PPO checkpoint persistence. |
| `benchmark_baselines.py` | Added EDF, historical one-step LP, and current-state finite-horizon MPC baselines with shared evaluation metrics. |
| `synthetic_ramp_experiment.py` | Added forecast-error, served-energy, violation reporting, and ACN/Gym rollout for the ramp fixture. |
| `pull_acn_data.py` | Added `ACN_SITE` selection for additional ACN sites. |
| `README.md` | Replaced outdated “not yet tried” statements with implemented, executed, pending, and comparability limitations. |
| `test_reviewer_changes.py` | Focused reviewer verification tests were run successfully, but the file was subsequently removed by the user undo and is not currently present. |
| `REVIEWER_PROGRESS_REPORT.md` | This status, evidence, command, and change report. |

## Remaining work before publication claims

Do not describe the following as completed experiments yet:

* five-seed, 300,000-step safety-filtered PPO versus penalty PPO;
* full controller experiments at 1 h, 3 h, and 6 h forecast horizons;
* full flexibility diagnostics and controller comparisons for JPL and Office001;
* publication-scale benchmark replication with the new finite-horizon MPC;
* restoration or reproduction from the historical sidecar-only PPO runs.

## Mathematical and physical lead-time audit

The criterion algebra matches the recorded Equation (2) form:

`X(l) = median_h[A(l,h) * I(S(l,h) > 0)]`.

The audit confirms consistent 6.6 kW clipping, pre-clipping greedy risk
identification, future-risk contributor selection, anchor connectivity checks,
and strict positive-slack indication. The corrected horizon-to-grid-step
conversion was tested at hourly, 30-minute, and 15-minute resolutions.

Two physical-model inconsistencies remain documented rather than marked fixed:

1. The cross-dataset implementation repeatedly uses total `requested_kWh`,
   whereas the environment/original diagnostic uses evolving remaining energy.
2. The cross-dataset grid admits overlapping sessions in the same `spaceID`;
   runtime checks found duplicate-space steps in both Caltech and JPL.

Consequently, the current scores are defensible as request-based retrospective
screening diagnostics, but not yet as exact environment-equivalent
instantaneous-demand results. No corrected physical result has been claimed.

Detailed evidence and source locations:

* `cross_dataset_lead_time/MATHEMATICAL_PHYSICAL_AUDIT.md`
* `cross_dataset_lead_time/cross_dataset_lead_time.py`
* `diagnose_forecast.py`
* `acn_env.py`

## Transformer-capacity sensitivity

The exact median lead-time criterion was evaluated at 25, 50, 75, 100, 125,
and 150 kW on the common 60-minute grid for Caltech and JPL. The 50 kW result
remains the baseline. The sweep reports the score, risk-hour count, A/S
frequencies, joint positive frequency, and event-level distributions.

Caltech develops nonzero median scores at higher capacities for selected
horizons: for example, `X(3h)=0.120` at 150 kW and `X(6h)=0.077`. JPL remains
median-zero across the tested capacities and horizons, although its joint
positive frequency increases from 5.18% to 17.55% at one hour between 25 and
150 kW. This demonstrates that capacity changes available flexibility, while
the median score must be interpreted together with the changing number of
risk hours.

Evidence:

* `cross_dataset_lead_time/CAPACITY_SWEEP_REPORT.md`
* `cross_dataset_lead_time/capacity_sweep.py`
* `cross_dataset_lead_time/capacity_sweep_outputs/capacity_sweep_scores.csv`
* `cross_dataset_lead_time/capacity_sweep_outputs/capacity_sweep_statistics.csv`
* `cross_dataset_lead_time/capacity_sweep_outputs/paper_figures/capacity_lead_time_scores.png`
* `cross_dataset_lead_time/capacity_sweep_outputs/paper_figures/capacity_lead_time_heatmap.png`

The approved reproducibility run was executed separately at
`cross_dataset_lead_time/capacity_sweep_outputs_approved_20261117/` with the
same six capacities, 60-minute grid, and 1--10 hour horizons. It produced ten
rows per capacity/dataset combination, passed the focused four-test suite, and
did not overwrite the earlier sweep outputs.
