# Transformer-Capacity Sensitivity Report

## Experiment

The original median lead-time criterion was evaluated without changing its
formula:

```text
X(l,C) = median_h [ A(l,h,C) * I(S(l,h,C) > 0) ]
```

The sweep uses the requested-energy complete-case sessions, 60-minute grid,
6.6 kW per-EV limit, and capacities of 25, 50, 75, 100, 125, and 150 kW.
Office001 remains unsupported because it has no requested-energy coverage.

## Results

Risk-hour counts decrease as capacity increases, while positive-slack and
joint availability/slack frequencies generally increase. The median remains
zero for most capacity/horizon combinations because the event product is still
concentrated at zero.

Selected results:

| Dataset | Capacity | Risk hours | X(1h) | X(3h) | X(6h) | X(10h) |
|---|---:|---:|---:|---:|---:|---:|
| Caltech | 25 kW | 4,176 | 0.000 | 0.000 | 0.000 | 0.000 |
| Caltech | 50 kW | 2,780 | 0.000 | 0.000 | 0.000 | 0.000 |
| Caltech | 75 kW | 2,372 | 0.000 | 0.000 | 0.000 | 0.000 |
| Caltech | 100 kW | 2,012 | 0.000 | 0.000 | 0.000 | 0.000 |
| Caltech | 125 kW | 1,436 | 0.000 | 0.000 | 0.043 | 0.000 |
| Caltech | 150 kW | 758 | 0.000 | 0.120 | 0.077 | 0.000 |
| JPL | 25 kW | 3,800 | 0.000 | 0.000 | 0.000 | 0.000 |
| JPL | 50 kW | 3,219 | 0.000 | 0.000 | 0.000 | 0.000 |
| JPL | 75 kW | 3,000 | 0.000 | 0.000 | 0.000 | 0.000 |
| JPL | 100 kW | 2,840 | 0.000 | 0.000 | 0.000 | 0.000 |
| JPL | 125 kW | 2,726 | 0.000 | 0.000 | 0.000 | 0.000 |
| JPL | 150 kW | 2,615 | 0.000 | 0.000 | 0.000 | 0.000 |

At 1-hour lead time, the joint `A > 0` and `S > 0` frequency changes from
8.76% to 22.96% for Caltech between 25 and 150 kW, and from 5.18% to 17.55%
for JPL. These joint percentages are more informative than the median alone
when the criterion distribution is concentrated at zero.

## Interpretation

The capacity relationship is not monotonic in the median score at every
horizon. Increasing capacity reduces the number of risk events, increases
available slack, and can expose a nonzero upper tail. However, the median
remains zero until enough risk-hour products become positive. At high
capacities, the remaining risk hours are a selected subset of more severe
events, so score comparisons must always include risk-hour counts.

The results establish that capacity changes the physical opportunity for
anticipatory charging, but they do not by themselves prove that forecasting
improves controller performance. The criterion remains a screening diagnostic.

## Physical caveat

This sweep reuses the current request-based cross-dataset implementation.
The separate mathematical/physical audit identified two limitations that are
not silently changed here:

1. Total requested energy is used at each active timestep rather than evolving
   remaining energy.
2. Same-space overlapping sessions are not resolved as environment collisions.

Therefore, this is a capacity sensitivity analysis of the existing
request-based diagnostic, not a claim of exact physical equivalence to the
controller environment.

## Outputs

- `capacity_sweep_scores.csv`
- `capacity_sweep_statistics.csv`
- `capacity_sweep_event_records.csv`
- `capacity_sweep_comparison.csv`
- `dataset_verification.csv`
- `run_configuration.json`
- `paper_figures/capacity_lead_time_scores.{png,pdf}`
- `paper_figures/capacity_lead_time_heatmap.{png,pdf}`
- `paper_figures/capacity_flexibility_relationship.{png,pdf}`

## Reproduction

```powershell
python cross_dataset_lead_time\capacity_sweep.py
python -m pytest -q cross_dataset_lead_time\test_cross_dataset_lead_time.py
```
