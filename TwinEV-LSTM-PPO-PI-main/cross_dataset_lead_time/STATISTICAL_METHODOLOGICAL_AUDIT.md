# Statistical and Methodological Audit

## Result

The original median criterion is not zero because A is absent. It is zero
because the joint condition is uncommon and the criterion uses a median:

```text
X(l) = median_h [ A(l,h) * I(S(l,h) > 0) ]
```

At a one-hour lead time:

| Resolution | Dataset | Risk hours | Mean | Median | P25 | P75 | A>0 | S>0 | Both>0 | Zero product |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 min | Caltech | 2,780 | 0.0382 | 0 | 0 | 0 | 99.96% | 9.78% | 9.75% | 90.25% |
| 60 min | JPL | 3,219 | 0.0176 | 0 | 0 | 0 | 99.63% | 7.77% | 7.39% | 92.61% |
| 30 min | Caltech | 5,225 | 0.0363 | 0 | 0 | 0 | 99.83% | 10.16% | 9.99% | 90.01% |
| 30 min | JPL | 6,216 | 0.0153 | 0 | 0 | 0 | 99.23% | 8.04% | 7.27% | 92.73% |
| 15 min | Caltech | 10,084 | 0.0358 | 0 | 0 | 0 | 99.78% | 10.34% | 10.12% | 89.88% |
| 15 min | JPL | 12,205 | 0.0156 | 0 | 0 | 0 | 99.15% | 8.18% | 7.32% | 92.68% |

The complete horizon-by-horizon statistics are in
`outputs_60m/lead_time_statistical_diagnostics.csv`,
`outputs_30m/lead_time_statistical_diagnostics.csv`, and
`outputs_15m/lead_time_statistical_diagnostics.csv`.

The product distribution is concentrated at zero, with nonzero products
reaching at most 0.875 in the one-hour rows shown above. Therefore the
median-zero conclusion is supported by the observed distribution, while the
mean and upper-tail values show that some individual periods do have
exploitable flexibility.

## Persistence across horizons

The same risk step is not universally zero at every horizon. The fraction of
risk steps whose product is zero at all ten tested horizons is:

| Resolution | Caltech | JPL |
|---|---:|---:|
| 60 min | 15.47% | 28.11% |
| 30 min | 15.14% | 31.10% |
| 15 min | 13.87% | 30.92% |

Thus, the zero median is primarily caused by the concentration of each
horizon's product distribution at zero, not by one small set of universally
unexploitable risk hours.

## Demand-estimation audit

The implementation uses:

```text
P_i = min(6.6 kW, requested_kWh / timestep_hours)
greedy_draw = sum(P_i for connected EVs)
risk = greedy_draw > transformer_capacity
```

This is the same demand definition used by the existing diagnostic semantics.
The audit confirms:

- Caltech and JPL use only positive `requested_kWh` complete cases.
- `kWhDelivered` is never substituted for requested energy.
- Connection steps use floor mapping and disconnect steps use ceil mapping.
- An EV is active on the half-open interval `[connect_step, disconnect_step)`.
- Risk is identified from greedy achievable draw before transformer clipping.
- The same `active_by_step` EV indices are used for greedy demand and A.
- A is computed retrospectively from the future risk hour's contributing EVs;
  this is the intended diagnostic definition, not a causal forecast input.
- The lead-time anchor is now converted correctly from hours to grid steps:
  `round(horizon_hours * 60 / timestep_minutes)`.

The last item fixes a timestep-specific implementation bug: before this audit,
30-minute and 15-minute runs subtracted one raw step for a one-hour horizon.
Hourly behavior was unchanged.

## Resolution sensitivity

All three runs used the same 50 kW capacity, 6.6 kW per-EV limit, requested
energy complete-case rule, and 1–10 hour horizons. The median remained zero
for every tested horizon for both Caltech and JPL at 60, 30, and 15 minutes.

Risk-hour counts increase at finer grids because the same interval-level demand
definition is evaluated more frequently and uses `requested_kWh /
timestep_hours`. These counts should not be interpreted as independent physical
surges without an additional continuous-power or energy-delivery model.

## Office001

Office001 remains excluded from primary requested-demand scoring because all
920 rows have zero/absent requested-energy values. Valid diagnostics include
arrival and departure distributions, session duration, occupancy, and delivered
energy. Requested-demand lead-time scores, requested-demand risk hours, and
requested-energy headroom conclusions cannot be drawn without a defensible
requested-energy field.

## Artifacts and reproduction

Each resolution directory contains:

- `lead_time_statistical_diagnostics.csv`
- `risk_hour_zero_persistence.csv`
- `lead_time_event_records.csv`
- `run_configuration.json`
- `paper_figures/lead_time_condition_diagnostics.png` and `.pdf`
- Existing comparison and dataset-characteristic CSV/figure outputs

Run from the repository root:

```powershell
python cross_dataset_lead_time\cross_dataset_lead_time.py --timestep-minutes 60 --output-dir cross_dataset_lead_time\outputs_60m
python cross_dataset_lead_time\cross_dataset_lead_time.py --timestep-minutes 30 --output-dir cross_dataset_lead_time\outputs_30m
python cross_dataset_lead_time\cross_dataset_lead_time.py --timestep-minutes 15 --output-dir cross_dataset_lead_time\outputs_15m
python -m pytest -q cross_dataset_lead_time\test_cross_dataset_lead_time.py
```

## Scientific defensibility

The existing median-based results are scientifically defensible as a strict
screening criterion, provided the paper reports the concentrated distribution
and does not claim that no individual charging periods have flexibility. Mean,
percentile, and joint A/S statistics are descriptive supplements; they do not
replace the original median criterion.
