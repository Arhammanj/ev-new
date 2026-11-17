# Reviewer Response Tracker

Statuses: `open`, `partial`, `addressed`.
Update a point with `python reviewer_response_tracker.py --id R1-2 --status addressed --response "..." --artifacts path/to/file`.

| ID | Status | Reviewer point | Response / evidence | Artifacts |
|---|---|---|---|---|
| R1-1 | open | Distinguish the criterion from flexibility, demand response, and prior forecasting/MPC work. | Pending. | - |
| R1-2 | addressed | Validate the criterion on additional real-world datasets or independent scenarios. | Caltech and JPL were evaluated with requested-energy complete cases; Office001 was retained as unsupported because requested energy is absent. | cross_dataset_lead_time\outputs\lead_time_scores.csv, cross_dataset_lead_time\CROSS_DATASET_REPORT.md |
| R1-3 | partial | Quantify how the score changes with transformer capacity. | Approved exact capacity sweep completed at 25, 50, 75, 100, 125, and 150 kW. Capacity changes risk frequency and joint flexibility; physical-model caveats remain. | cross_dataset_lead_time\capacity_sweep_outputs_approved_20261117\capacity_sweep_scores.csv, cross_dataset_lead_time\capacity_sweep_outputs_approved_20261117\capacity_sweep_statistics.csv, cross_dataset_lead_time\capacity_sweep_outputs_approved_20261117\paper_figures\capacity_lead_time_heatmap.png |
| R1-4 | open | Justify the median/binary-slack formulation and compare continuous or energy-based flexibility. | Pending. | - |
| R1-5 | partial | Compare against safety-filtered RL, constrained optimization, or MPC. | MPC and safety-filter infrastructure exists; the complete comparative experiment is not yet reported. | - |
| R1-6 | partial | Increase seeds and distinguish no significance from no practical benefit. | Reproducibility and multi-seed aggregation support were added; expanded independent runs remain pending. | - |
| R1-7 | open | Evaluate one-, three-, and six-hour forecasts in the controller. | Pending. | - |
| R1-8 | partial | Discuss charging-rate, SOC, deadlines, and finer control intervals. | Real Caltech 15-minute loading and forecast-resolution support were verified; SOC/deadline and rate sensitivity remain pending. | - |
| R1-9 | addressed | Provide joint distributions of availability and slack. | Statistical audit completed across 60, 30, and 15 minute grids; joint A/S frequencies and zero-product persistence are reported. | cross_dataset_lead_time\STATISTICAL_METHODOLOGICAL_AUDIT.md, cross_dataset_lead_time\outputs_60m\lead_time_statistical_diagnostics.csv, cross_dataset_lead_time\outputs_60m\paper_figures\lead_time_condition_diagnostics.png |
| R1-10 | partial | Clarify whether the criterion is a screening metric, necessary condition, or performance predictor. | The cross-dataset report frames it as a screening/necessary-condition diagnostic; prospective controller selection is not yet demonstrated. | - |
| R2-1 | open | State what information is needed to compute the criterion. | Pending. | - |
| R2-2 | open | Quantify rapid deployment or grid stress in the introduction. | Pending. | - |
| R2-3 | open | Explain why forecasting is widely adopted before criticizing its premise. | Pending. | - |
| R2-4 | open | Link each contribution to the supporting questions. | Pending. | - |
| R2-5 | open | Signpost the distinction from prediction-assisted and safe-RL studies. | Pending. | - |
| R2-6 | open | Expand the motivation for the actionability gap. | Pending. | - |
| R2-7 | partial | Describe ACN-Data usage patterns and site context. | Dataset characteristics and arrival/departure/energy figures were generated; manuscript prose remains to be updated. | - |
| R2-8 | addressed | Justify hourly resolution. | The README and resolution comparison document hourly versus real 15-minute support. | resolution_comparison.py, paper_figures\resolution_comparison.png |
| R2-9 | open | Contrast measured capacity with synthetic environments. | Pending. | - |
| R2-10 | open | Explain why the selected aggregate target is more forecastable. | Pending. | - |
| R2-11 | partial | Provide the full reward function and learning trade-offs. | PPO configuration and evaluation behavior are documented; full manuscript reward exposition remains pending. | - |
| R2-12 | partial | Explain residual violations under reward penalties. | The implementation documents penalty-based evaluation and safety-filter infrastructure; manuscript discussion remains pending. | - |
| R2-13 | open | Improve Figure 3 axis label and horizon markers. | Pending. | - |
| R2-14 | addressed | Justify median slack instead of mean slack. | The report preserves the median criterion and explains its distributional interpretation. | cross_dataset_lead_time\CROSS_DATASET_REPORT.md |
| R2-15 | partial | Discuss why capacity findings persist across capacities. | Approved capacity sweep quantifies risk-hour frequency, slack, joint A/S frequency, and the original median score across six capacities; manuscript interpretation remains pending. | cross_dataset_lead_time\capacity_sweep_outputs_approved_20261117\capacity_sweep_statistics.csv, cross_dataset_lead_time\capacity_sweep_outputs_approved_20261117\paper_figures\capacity_flexibility_relationship.png |
| R2-16 | partial | Give concrete next steps for wider datasets and adaptive horizons. | Cross-dataset evaluation and multi-resolution support provide concrete next steps; adaptive horizons remain future work. | - |
| R2-17 | open | Change 'controllable demand' to 'controllable EV demand'. | Pending. | - |
| R3-1 | partial | Add learning curves and MPC with BiLSTM and oracle forecasts. | Finite-horizon MPC and rollout infrastructure were added; the requested synthetic-site comparison is not yet reported. | - |
| R3-2 | open | Test whether the forecast adds information beyond calendar features. | Pending. | - |
| R3-3 | open | Count lead time from saturated-period onset. | Pending. | - |
| R3-4 | addressed | Explain the median and positive-slack threshold; report full distributions. | Full event-level distributions show that the median is zero because 90.25% of Caltech and 92.61% of JPL one-hour products are zero; nonzero means and upper tails are retained. | cross_dataset_lead_time\STATISTICAL_METHODOLOGICAL_AUDIT.md, cross_dataset_lead_time\outputs_60m\lead_time_statistical_diagnostics.csv, cross_dataset_lead_time\outputs_60m\risk_hour_zero_persistence.csv |
| R3-5 | partial | Run controllers with and without forecast and score JPL/Office001. | JPL was scored using requested energy; Office001 is scientifically unsupported for requested-demand scoring; synthetic controller comparison remains pending. | - |
| R3-6 | open | Add aggregate draw to the PPO observation. | Pending. | - |
| R3-7 | partial | Increase seeds because the confidence interval spans the capacity-sweep room. | Multi-seed aggregation support exists; additional independent runs remain pending. | - |
| R3-8 | partial | Use a safety layer to isolate forecast effects. | Safety-filtered controller support was added; the complete comparison is not yet reported. | - |
| R3-9 | open | Reconcile inconsistent p-values and capacity-sweep statements. | Pending. | - |
| R3-10 | addressed | Document PPO hyperparameters, deterministic actions, and train/test forecast separation. | PPO settings, sampled versus deterministic evaluation, timestamp alignment, and leakage safeguards are documented. | verify_reproducibility.py, README.md |
| R3-11 | partial | Correct or run the promised one- and three-hour controller forecasts. | Forecast generation supports multiple horizons; the controller comparison for all promised horizons remains pending. | - |
| R3-12 | open | Restore random-policy and reward-weight-sweep reporting. | Pending. | - |
| R3-13 | partial | Explain the synthetic ramp arrival model and imposed curve. | Synthetic-ramp infrastructure was audited; a fuller manuscript explanation remains pending. | - |
| R4-1 | partial | Justify 50 kW, per-EV power, aggregation, and capacity sensitivity. | Approved exact criterion sweep completed at 25-150 kW with 50 kW retained as baseline. Site-specific transformer justification and physical-model correction remain pending. | cross_dataset_lead_time\capacity_sweep_outputs_approved_20261117\run_configuration.json, cross_dataset_lead_time\capacity_sweep_outputs_approved_20261117\paper_figures\capacity_lead_time_scores.png |
| R4-2 | addressed | Justify hourly resolution and repeat at 15-minute intervals. | Real Caltech 15-minute feasibility and hourly-versus-15-minute comparison artifacts were generated. | resolution_comparison.py, paper_figures\resolution_comparison.png |
| R4-3 | addressed | Explain variables mathematically and report joint/event-level distributions. | The criterion implementation, event records, statistical diagnostics, and condition figures are available. | cross_dataset_lead_time\outputs\lead_time_event_records.csv, cross_dataset_lead_time\outputs\paper_figures\lead_time_condition_diagnostics.png |
| R4-4 | partial | Justify six-hour aggregate demand and verify no forecast input/target leakage. | Input/target leakage was removed and alignment audits were added; peak/violation forecasting targets remain future work. | - |
| R4-5 | partial | Add rule-based, optimization, or MPC benchmarks. | MPC and cap-only infrastructure exists; a complete benchmark table is not yet reported. | - |
| R4-6 | partial | Increase runs, statistical comparisons, temporal periods, and PPO reproducibility details. | Reproducibility audits, configuration capture, and multi-seed support were added; expanded periods/runs remain pending. | - |
| R4-7 | addressed | Validate on an independent real-world dataset. | JPL was evaluated independently; Office001 was transparently excluded from requested-demand scoring because coverage is zero. | cross_dataset_lead_time\outputs\dataset_verification.csv, cross_dataset_lead_time\CROSS_DATASET_REPORT.md |
| R4-8 | partial | Reconcile forecasting horizons and capacity-sweep claims. | Resolution and forecast documentation were updated; manuscript tables and all controller horizon runs remain pending. | - |

## Reproduction commands

```powershell
python cross_dataset_lead_time\cross_dataset_lead_time.py
python -m pytest -q cross_dataset_lead_time\test_cross_dataset_lead_time.py
python resolution_comparison.py
python verify_reproducibility.py --forecast-csv acn_target_a_bilstm_full.csv --label hourly
```
