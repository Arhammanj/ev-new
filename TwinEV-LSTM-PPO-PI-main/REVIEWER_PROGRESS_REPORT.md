# Reviewer-response implementation and verification report

## Scope and preservation

This report records the state of the reviewer-requested changes without
overwriting the earlier results. The snapshot
`verification_archive_20261116_0030/` contains 130 copied result and artifact
files, including the earlier CSV, JSON, plot, log, checkpoint-sidecar, and
training-log outputs. New verification outputs use distinct filenames.

## Status summary

| Area | Status | Evidence |
|---|---|---|
| Continuous flexibility criterion | Implemented and executed | `acn_flexibility_records.csv`, `acn_flexibility_summary.csv` |
| Full distributions and joint relationship | Implemented and executed | `acn_flexibility_distributions.png` and event-level CSV |
| Saturation onset lead time | Implemented and executed | `acn_saturation_periods.csv` |
| 50/75/100 kW diagnostic sweep | Implemented and executed | All nine capacity/horizon combinations in the summary CSV |
| Safety projection | Implemented and validated in focused tests; test file was subsequently removed by the user undo | smoke JSON; historical test output below |
| Safety PPO comparison | Executed as a 2,048-step smoke run and a full 300,000-step seed-1 run | `acn_safety_filtered_smoke.json`, `acn_safety_filtered_full_seed1.json` |
| EDF benchmark | Implemented and executed at 50 kW | `acn_benchmark_results.json` |
| MPC benchmark | Implemented and executed at 50 kW; one-step MPC-style LP, not full multi-step MPC | `acn_benchmark_results.json` |
| Synthetic ramp | Executed as a load-profile sanity check | `synthetic_ramp_experiment.py` output |
| JPL and Office001 downloads | Downloaded and loader/environment reset checks passed | `acn_jpl_2019.csv`, `acn_office001_2019.csv` |
| JPL/Office001 diagnostics and controller experiments | Pending | No publication-level cross-site results yet |
| 1 h and 3 h controller experiments | Pending | Diagnostic supports the horizons; controller runs are not completed |
| PPO binary checkpoints | Not available | Existing checkpoint directories contain timestep sidecars only |
| Publication-ready replication | Not yet | Full multi-seed safety, cross-site, and horizon experiments remain outstanding |

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
receives expected load, not the noisy actual series. This avoids privileged
access to future actual noise, but the script is a controlled load-profile
check rather than an ACN/Gym controller rollout. Its current allocation logic
therefore does not establish a publication-quality controller comparison.

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
| `exp_safety_filtered_ppo.py` | Added ordinary-vs-filtered PPO comparison and persisted filter usage counters. |
| `benchmark_baselines.py` | Added EDF and current-state MPC-style LP baselines with shared evaluation metrics. |
| `synthetic_ramp_experiment.py` | Added forecast-error, served-energy, and violation reporting for the ramp fixture. |
| `pull_acn_data.py` | Added `ACN_SITE` selection for additional ACN sites. |
| `README.md` | Replaced outdated “not yet tried” statements with implemented, executed, pending, and comparability limitations. |
| `test_reviewer_changes.py` | Focused reviewer verification tests were run successfully, but the file was subsequently removed by the user undo and is not currently present. |
| `REVIEWER_PROGRESS_REPORT.md` | This status, evidence, command, and change report. |

## Remaining work before publication claims

Do not describe the following as completed experiments yet:

* five-seed, 300,000-step safety-filtered PPO versus penalty PPO;
* full controller experiments at 1 h, 3 h, and 6 h forecast horizons;
* full flexibility diagnostics and controller comparisons for JPL and Office001;
* an actual ACN/Gym synthetic-ramp controller rollout;
* a full multi-step MPC implementation and publication-scale benchmark;
* restoration or reproduction from binary PPO checkpoints.
