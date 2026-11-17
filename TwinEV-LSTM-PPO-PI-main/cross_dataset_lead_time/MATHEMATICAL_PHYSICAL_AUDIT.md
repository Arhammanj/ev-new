# Mathematical and Physical Audit of the Lead-Time Criterion

## Audit scope

This audit compares the criterion as documented in the repository with the
cross-dataset implementation and the original diagnostic implementation. No
criterion formula or prior result was changed.

The manuscript source itself is not present in the workspace: no `.tex`,
`.docx`, or manuscript `.pdf` was found. Therefore, the equation comparison
below is against the Equation (2) form recorded in the existing code and
reports:

```text
X(l) = median_h [ A(l,h) * I(S(l,h) > 0) ]
```

An exact manuscript equation-number comparison remains conditional on
receiving the manuscript source or PDF.

## Equation-by-equation comparison

### Demand proxy

The implementation computes, for each active EV:

```text
P_i(t) = min(6.6 kW, requested_kWh_i / timestep_hours)
D(t) = sum_i P_i(t)
```

This is implemented in
`cross_dataset_lead_time/cross_dataset_lead_time.py:132-142`.
The original environment uses the same power cap and timestep conversion at
`acn_env.py:190-205`, but applies it to **remaining** energy.

`requested_kWh / timestep_hours` is not a measurement of instantaneous
charging demand. It is the power required to deliver the entire outstanding
energy within one timestep, capped at 6.6 kW. It is a valid conservative
achievable-draw proxy only if `requested_kWh` means remaining energy at that
time. In the cross-dataset code it means total session request and is reused
at every active step. This can overstate demand for multi-step sessions.

### Per-EV limit

The 6.6 kW limit is applied consistently in the cross-dataset greedy series,
the original diagnostic rollout, the environment, and controller constants:

- `cross_dataset_lead_time/cross_dataset_lead_time.py:41,138`
- `diagnose_forecast.py:69-74`
- `acn_env.py:121-123,193-196`
- `train_controller.py:33-34`

The limit is a power limit. It does not by itself establish that the EV can
accept that power for the full timestep, because that also depends on
remaining energy and departure time.

### Risk event

The cross-dataset implementation defines:

```text
R(h) = 1[D(h) > capacity]
```

at `cross_dataset_lead_time/cross_dataset_lead_time.py:163-165`.
The original diagnostic uses the same pre-clipping condition at
`diagnose_forecast.py:117` and the rollout constructs achievable draw before
the environment step at `diagnose_forecast.py:53-74`.

This matches a greedy, uncapped achievable-draw risk definition. It is not
actual delivered power under a controller and should be described as
capacity pressure or risk, not an observed violation.

### Future risk-hour EV population

For each risk step, the cross-dataset implementation sets:

```text
contributing = active_by_step[risk_step]
```

filtered to positive requests at
`cross_dataset_lead_time/cross_dataset_lead_time.py:169-173`.
Because the grid is built only from positive requested-energy rows, the
filter is redundant for supported datasets. This identifies the EVs active at
the future risk hour under the implementation's occupancy model.

The original diagnostic instead identifies:

```text
presence[risk_step] & (remaining[risk_step] > 0)
```

at `diagnose_forecast.py:120-122`. Thus the two implementations agree on
presence only when every active session has positive remaining energy and
there are no physical-space collisions.

### Availability term A(l,h)

The cross-dataset implementation computes:

```text
A(l,h) = count(contributing EVs present at anchor h-l)
         / count(contributing EVs)
```

at `cross_dataset_lead_time/cross_dataset_lead_time.py:176-183`.
The numerator uses membership in `active_by_step[anchor]`, so the same session
indices selected as future contributors are tested at the anchor. This is
mathematically aligned with the stated fraction under the implementation's
occupancy representation.

### Slack term S(l,h)

The implementation computes:

```text
S(l,h) = max(0, capacity - D(h-l))
```

at `cross_dataset_lead_time/cross_dataset_lead_time.py:165` and
`174-186`. The indicator is strictly positive slack:

```text
I(S(l,h) > 0)
```

at `cross_dataset_lead_time/cross_dataset_lead_time.py:186`.
This matches the recorded criterion. It does not measure the amount of energy
that can actually be shifted, and it does not reserve headroom for other EVs
over the complete lead-time window. The original diagnostic's continuous
extension explicitly sums headroom over the window at
`diagnose_forecast.py:128-137`; that is a different supplementary metric, not
the original median criterion.

### Final score

The implementation takes the median of event products at
`cross_dataset_lead_time/cross_dataset_lead_time.py:208-209`. This matches the
recorded Equation (2) form exactly, subject to the demand and occupancy
assumptions above.

## Physical correctness findings

### 1. Total request versus remaining energy: inconsistency

The cross-dataset audit uses each session's total `requested_kwh` at every
active grid step. The environment and original diagnostic use mutable
remaining energy:

- `acn_env.py:193-205`
- `acn_env.py:296-305`
- `diagnose_forecast.py:69-74`

Therefore, the cross-dataset `D(t)` is not the same physical quantity as the
environment's instantaneous achievable draw after energy has already been
delivered. It is an upper-envelope proxy based on the original request.

Recommended correction for a future physical reimplementation: compute
remaining energy under an explicitly declared reference delivery policy, or
define the diagnostic as a request-based upper bound and rename it
accordingly. Do not silently substitute one for the other.

### 2. Same-space overlap: inconsistency

`load_grid` stores every requested-energy session in `active_by_step` and does
not resolve overlapping sessions sharing a `spaceID`
(`cross_dataset_lead_time/cross_dataset_lead_time.py:113-124`). The environment
uses collision handling when a new arrival finds an occupied slot
(`acn_env.py:289-298`).

A runtime audit found duplicate physical-space occupancy in the complete-case
grids:

| Dataset | 60 min duplicate steps | 30 min | 15 min | Maximum simultaneous sessions in one space |
|---|---:|---:|---:|---:|
| Caltech | 409 | 235 | 110 | 3 at 60 min; 2 at finer grids |
| JPL | 1,414 | 1,470 | 1,059 | 3 at 60 min; 2 at finer grids |

This means the cross-dataset risk and A calculations can count physically
incompatible simultaneous sessions. The results are therefore not fully
equivalent to the environment's physical occupancy model.

Recommended correction: explicitly resolve or exclude overlapping sessions by
the same physical-space rule used by the environment before claiming
environment-equivalent physical demand.

### 3. Resolution equivalence

The horizon conversion is now tested and uses:

```text
horizon_steps = round(horizon_hours * 60 / timestep_minutes)
```

at `cross_dataset_lead_time/cross_dataset_lead_time.py:168`.
This makes the requested lead-time duration consistent in wall-clock hours.

However, the physical experiments are not otherwise equivalent across
resolutions because:

1. `requested_kWh / timestep_hours` changes with grid size.
2. Sessions are admitted using floor connection and ceil disconnect mapping.
3. The request-based demand proxy is evaluated more frequently at finer grids.
4. Same-space overlap counts differ by grid.

Thus, 60/30/15-minute results are algorithmically comparable sensitivity runs,
not identical physical experiments.

## Future-information audit

The calculation intentionally uses future information to define the diagnostic:

- It first identifies future risk steps from the complete time series.
- It selects the EVs contributing at that future step.
- It then checks their earlier anchor presence and headroom.

This is not leakage into a forecast model; it is retrospective labeling of
whether a risk event had exploitable lead time. It would be future leakage if
the resulting future risk population, A, S, or event label were supplied to a
controller or forecast at the anchor time. The paper should explicitly call
this a retrospective feasibility diagnostic.

The timestamp construction itself is chronological and uses the grid start
plus the step index at `cross_dataset_lead_time/cross_dataset_lead_time.py:191-195`.

## Overall conclusion

The algebra of the recorded median criterion is implemented correctly:
future risk contributors are identified, anchor connectivity is measured for
those same session indices, nonnegative headroom is computed before clipping,
and the median product is taken.

The physical interpretation is weaker than the algebra because the
cross-dataset implementation:

1. uses total requested energy rather than evolving remaining energy; and
2. permits overlapping sessions in one physical space.

The existing results are defensible as a **request-based retrospective
screening diagnostic**, not yet as an exact reproduction of the environment's
physical instantaneous demand process. No corrected physical results should be
claimed until a revised implementation is tested against explicit occupancy
and remaining-energy rules.

## Validation evidence

The previously tested horizon conversion remains covered by
`cross_dataset_lead_time/test_cross_dataset_lead_time.py`, and the focused
suite passes. This audit introduces no criterion or result changes.
