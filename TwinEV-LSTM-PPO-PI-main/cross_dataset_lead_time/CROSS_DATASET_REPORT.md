# Cross-Dataset Lead-Time Criterion Report

## Scope

This experiment uses the existing ACN session caches and does not download or
modify the original project results. It evaluates the lead-time criterion on a
common 60-minute grid, 50 kW transformer capacity, 6.6 kW per-EV limit, and
lead times from 1 through 10 hours.

The implementation is:

```text
A(l,h) = fraction of EVs contributing to risk hour h
          that are already connected at h-l
S(l,h) = max(0, capacity - greedy achievable draw at h-l)
X(l)   = median_h [ A(l,h) * I(S(l,h) > 0) ]
```

Risk hours are hours where greedy achievable draw exceeds 50 kW. Greedy draw
is computed before transformer clipping as the sum of
`min(6.6 kW, requested_kWh / timestep_hours)` over connected EVs.

## Dataset verification

| Dataset | Sessions | File size | Stations/spaces | Requested coverage | Missing requested rows |
|---|---:|---:|---:|---:|---:|
| Caltech | 10,609 | 1,605,353 bytes | 54 / 54 | 85.47% | 1,542 |
| JPL | 17,356 | 2,674,279 bytes | 52 / 52 | 96.77% | 560 |
| Office001 | 920 | 140,438 bytes | 8 / 8 | 0.00% | 920 |

All files contain `sessionID`, `stationID`, `spaceID`, `siteID`,
`connectionTime`, `disconnectTime`, `doneChargingTime`, `kWhDelivered`, and
`requested_kWh`. `doneChargingTime` is missing for 466 Caltech, 276 JPL,
and 38 Office001 rows.

Caltech and JPL scores are complete-case estimates using only rows with
positive `requested_kWh`; rows without requested energy are excluded. The
loader's delivered-energy fallback is deliberately not used. Office001 is
not assigned a primary score because substituting `kWhDelivered` for requested
demand would change the scientific quantity.

## Results

The primary scores are in `outputs/lead_time_scores.csv` and
`outputs/comparison_table.csv`. In the executed run:

| Dataset | Risk hours | X(1h) through X(10h) |
|---|---:|---|
| Caltech | 2,780 | 0.0 at every tested horizon |
| JPL | 3,219 | 0.0 at every tested horizon |
| Office001 | unsupported | not computed |

The zero median means that, for at least half of risk-hour observations, the
product of prior connectivity and positive prior headroom is zero. This is
consistent with a criterion that requires both vehicles to be present early
and usable transformer slack to coexist; it is not a missing-value default.

## Generated artifacts

- `outputs/dataset_verification.csv`
- `outputs/lead_time_scores.csv`
- `outputs/lead_time_event_records.csv`
- `outputs/dataset_characteristics.csv`
- `outputs/comparison_table.csv`
- `outputs/run_configuration.json`
- `outputs/paper_figures/lead_time_score_vs_horizon.{png,pdf}`
- `outputs/paper_figures/dataset_characteristics.{png,pdf}`
- `outputs/paper_figures/arrival_departure_energy_distributions.{png,pdf}`
- `outputs/paper_figures/headroom_risk_diagnostics.{png,pdf}`

## Reproduction

From the repository root:

```powershell
python cross_dataset_lead_time\cross_dataset_lead_time.py
python -m pytest -q cross_dataset_lead_time\test_cross_dataset_lead_time.py
```

To change the common grid:

```powershell
python cross_dataset_lead_time\cross_dataset_lead_time.py --timestep-minutes 15
```

## Interpretation for Reviewer 2

The implementation demonstrates that the same formal criterion can be applied
to Caltech and JPL and yields the same qualitative result: the median score is
zero across 1–10 hours under the common 50 kW scenario. This supports
cross-dataset consistency for the two datasets with requested-energy data.
Office001 cannot support an equivalent demand-based criterion because its
requested-energy coverage is zero. It contributes valid arrival, departure,
duration, occupancy, and delivered-energy diagnostics, but not a scientifically
equivalent requested-demand lead-time score. A claim covering all three sites
would therefore be too strong; the defensible claim is consistency across
Caltech/JPL and explicit non-identifiability for Office001.
