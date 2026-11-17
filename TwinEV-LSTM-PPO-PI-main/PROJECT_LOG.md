# Project Log: Safe-RL EV Charging Control on Real ACN-Data

A chronological record of the investigation, kept honest on purpose: hypotheses that
were wrong are left in, dead ends are labeled as dead ends, and every numeric claim
below is traceable to a script in this repo. This is the backbone for the method
section — if a paper/report is written from this project, the story should follow
this log's actual order of discovery, not a cleaned-up version of it.

## 0. Starting point: the synthetic environment was invalid

The original `twinev_env.py` pipeline (archived, not deleted — see `archive/`) had a
structural flaw: delivered power was a function of the agent's action and the PI
controller's own *previous output*, never real demand, and the total was hard-clipped
to a 150 kW cap. That clip made "zero overloads" trivially true regardless of policy —
the safety problem being studied couldn't actually fail, so no result from it was
meaningful. This was caught and the whole environment was rebuilt rather than patched.

## 1. `acn_env.py` — a demand-coupled environment on real data (2026-08-27/28)

Replaced the synthetic pipeline with `ACNChargingEnv`: every kWh delivered must come
from and reduce a real EV session's remaining demand (drawn from cached 2019 Caltech
ACN-Data records), and the transformer limit is a measured signal that is **never**
used to scale delivered power down — an agent that over-allocates really does violate
it. The environment's own `__main__` self-test asserts both properties directly (an
all-max policy must produce violations; a zero-power policy's unmet energy must equal
total demand exactly).

The 50 kW transformer limit is a deliberate choice, not the site's real rating: a
greedy "charge everyone as fast as possible" policy peaks at 137.6 kW across the full
year, so a realistic 150 kW limit is never binding. 50 kW is the smallest limit that
keeps the constraint live without being hopeless — greedy achievable draw exceeds it
in ~8–10% of hours (see Part 3 below for the exact figure on the test weeks).

Also built in this phase: the BiLSTM demand forecaster (Target A: a 6h-ahead sum of
new-arrival kWh demand, 23.8% held-out WAPE — `build_demand_forecaster.py`,
`generate_forecast_feature.py`) and the Track B safe-RL controller scaffold
(`train_controller.py`, PPO via Stable-Baselines3).

## 2. First collapse: the controller learned to do nothing (2026-08-29)

The first short validation run (`energy=1, unmet=1, violation=1`, 50k steps) collapsed
to a near-zero-charging policy: **86.5 kWh delivered vs. `cap_only`'s 2173.8 kWh,
97% unmet demand, 0% violations** — it satisfied the reward by simply not acting.

Two independent reward-content hypotheses were tested and **both refuted**:
- `sweep_reward_weights.py`: raising the unmet-energy penalty (configs A–D, unmet
  weight 3→10) did not fix the collapse.
- Per-step reward-term instrumentation (`instrument_reward_terms.py`) showed the
  energy reward actually *outweighs* the violation penalty 3:1 during demand surges —
  the reward was not structurally biased toward inaction.

This ruled out "the reward discourages charging" and pointed at training dynamics
instead of reward design.

## 3. Root cause found: action-space asymmetry (`instrument_action_collapse.py`, 2026-08-29)

SB3 PPO's continuous-action policy is an unbounded Gaussian, initialized with mean
≈ 0. The environment's action box was `[0, per_ev_max_kw]` = `[0, 6.6]` kW — an
**asymmetric** range whose bottom, not its middle, sits at 0. An untrained (and,
it turned out, a *trained*) policy's mean stayed at or below 0, so essentially every
sampled action got clipped to zero charging before it ever reached the environment.
This was not a reward problem at all.

A quick mitigation — `action_floor_kw`, forcing a minimum non-zero request on every
present slot — confirmed the diagnosis: it moved delivered energy from 86.5 → 919
kWh. But it's blunt: it forces every connected EV to draw at least that floor
regardless of whether it needs to, which is physically arbitrary and caps how good
the policy can get.

## 4. The principled fix: symmetric action-space rescale (this session, 2026-08-30)

Instead of forcing a floor, give the policy a **symmetric** external action space,
`a ∈ [-1, 1]` per slot — matching the Gaussian's natural init — and map it internally
to the physical range:

```
kw = (a + 1) / 2 * per_ev_max_kw
```

`a = -1 → 0 kW`, `a = 0 → per_ev_max_kw / 2` (the policy's natural init point now
lands at the **middle** of the usable range, not the bottom), `a = +1 → per_ev_max_kw`.
This is a pure reparameterization of the RL-agent/environment interface — all physical
clipping (per-EV max, demand-bounded delivery) is applied identically afterward, on
the mapped kW value. Implemented as `ACNWeeklyEpisodeEnv(symmetric_action=True)` plus
`symmetric_to_kw()` / `kw_to_symmetric()` in `train_controller.py`.

**Validation** (`validate_symmetric_action.py`; 50k steps, seed 1, `no_forecast`,
original 1/1/1 weights, 5 held-out test weeks):

| Policy | Violation rate | Unmet fraction | Delivered kWh |
|---|---|---|---|
| random | 4.76% | 30.60% | 1865.2 |
| cap_only | 0.00% | 18.11% | 2173.8 |
| action-floor patch (prior fix) | — | — | 919.0 |
| **symmetric-action PPO (this fix)** | **5.48%** | **27.53%** | **1955.6** |

More than double the floor patch's delivered energy, at a violation rate close to
random's own baseline. The trained action distribution confirmed the mechanism: mean
3.49 kW / std 0.44 kW against a [0, 6.6] kW range (midpoint 3.30 kW), with **0%** of
actions landing within 10% of either bound — the policy uses the interior of the
range instead of collapsing to a corner. **Adopted as the fix; the action-floor patch
was retired** (left in the code, unused, for anyone re-deriving this history).

## 5. Reward-weight sweep on the *working* controller (this session)

All of Phase 2's reward sweeps were run on a collapsed policy and are void — they
were diagnosing the wrong layer. Re-ran holding `energy=1`, sweeping `unmet`/
`violation` (`sweep_reward_weights_symmetric.py`; 50k steps, seed 1):

| Config | Violation % | Unmet % | Delivered kWh |
|---|---|---|---|
| random | 5.24 | 30.91 | 1883.2 |
| cap_only | 0.00 | 18.11 | 2173.8 |
| S0 (unmet=1, viol=1) | 5.48 | 27.53 | 1955.6 |
| S1 (unmet=1, viol=3) | 5.24 | 27.95 | 1940.0 |
| S2 (unmet=1, viol=6) | 4.88 | 28.33 | 1916.9 |
| **S3 (unmet=2, viol=6)** | **4.76** | 28.29 | 1926.7 |
| S4 (unmet=2, viol=10) | 4.64 | 28.64 | 1913.5 |

A 10× change in the violation weight moved violations by less than 1 percentage
point. Reward-weight tuning had hit a ceiling in this range, and no config beat
`cap_only` on either axis. **S3 was locked** as the best balance (first config
below the random baseline on violations, without giving up meaningful delivery) —
with the explicit caveat that this is a weak effect, likely near the limit of what
reward-shaping alone can do here; a different lever (more training budget, more
seeds, or an architectural safety layer) looked more promising than further reward
retuning.

## 6. The full 2×2, multi-seed (this session)

Locked config: S3 weights, `symmetric_action=True`, 300k timesteps (up from 50k),
5 seeds (1–5) per learned condition, same held-out test weeks throughout
(`exp_2x2_run.py`, `exp_2x2_cap_only.py`, `exp_2x2_report.py`). `cap_only + forecast`
is not a meaningful cell — a hard cap has no mechanism to use a forecast — so three
real conditions were run.

| Condition | Violation rate | Unmet fraction | Delivered kWh |
|---|---|---|---|
| cap_only (single, deterministic) | 0.00% | 17.46% | 2161.6 |
| safe-RL, no_forecast (n=5) | 3.29 ± 0.94% | 27.61 ± 1.30% | 1884.3 ± 34.9 |
| safe-RL, forecast (n=5) | 3.00 ± 0.53% | 27.35 ± 1.21% | 1899.0 ± 24.3 |

- Forecast vs. no_forecast: −0.29pp violations, +14.7 kWh delivered — **both within
  pooled seed-to-seed noise** (1.08pp and 42.5 kWh respectively). No measurable
  forecast benefit at this budget/seed count.
- Neither learned condition beat `cap_only` on **either** axis.
- One real positive: extending the training budget alone (50k → 300k) measurably
  helped — violation rate dropped from S3's 4.76% (50k) to ~3.0–3.3% (300k) at
  comparable delivered energy. The extra budget bought something; the forecast
  feature, on its own, did not.

This is a **null result on the forecast's value** and a **negative result on safe-RL
vs. a simple hard cap**, at this point in the investigation.

## 7. Before trusting the null: sanity-checking the forecast integration (this session)

A null result is only informative if the thing being tested for was actually usable.
`diagnose_forecast.py` checked three independent premises, any one of which failing
would mean "the forecast didn't help" for a boring/uninteresting reason rather than a
real finding:

**Part 1 — is the forecast feature informative? PASS.** Over the 5 test weeks (840
hours), the normalized feature has mean 0.18, std 0.23, range [-0.01, 1.13], 0%
saturated at either clip bound, 0% exactly zero. Not broken, not flat.

**Part 2 — is it predictive of what the controller needs? PASS, strongly.**
Correlated forecast(t) against actual future greedy-achievable draw / violations:

| Horizon | corr(forecast, max achievable kW) | corr(forecast, violation occurs) |
|---|---|---|
| 1h | +0.40 | +0.33 |
| 3h | +0.72 | +0.63 |
| 6h | +0.90 | +0.81 |

The forecast is strongly aligned with transformer pressure at its native 6h horizon.
Not a misaligned-target problem.

**Part 3 — does the data offer exploitable lead time? FAILS.** At violation-risk
hours (8.5% of test-week hours: 71/840), checking the EVs actually drawing power
there:

| Lead time before violation | % of contributing EVs already connected | System slack then (median) |
|---|---|---|
| 1h | 64.3% | 0.0 kW — already saturated |
| 3h | 12.2% | 0.0 kW — mostly not there yet |
| 6h | 0.7% | 43.4 kW — plenty of slack, but ~no EVs present |

At the only horizon where the relevant EVs are actually connected (1h), the system
has already run out of spare capacity. At the horizon where spare capacity exists
(6h), the EVs that will cause the violation haven't arrived. Violations here are
driven by short, sharp arrival surges, not slow-building load a controller could see
coming and smooth out in advance.

### Reframed conclusion

The forecast integration is **not** broken and **not** misaligned. The null result
should not be reported as "forecasting doesn't help" — it should be reported as:
**this data's arrival dynamics leave no physical window for anticipatory
pre-charging, so a controller gains nothing from a forecast no matter how accurate,
not because it can't act on one.** That is a materially more precise and more
interesting claim than a generic negative result, and it is the honest one this
investigation supports.

## Status as of 2026-08-30

- **Fixed and adopted:** symmetric `[-1, 1]` action-space rescale (§4). Action-floor
  patch retired.
- **Locked:** reward weights S3 (`energy=1, unmet=2, violation=6`) (§5), with the
  caveat that reward-shaping alone looks close to its ceiling in this setup.
- **Established (multi-seed, §6):** neither safe-RL condition beats the `cap_only`
  hard-cap reference on delivered energy or violation rate; forecast provides no
  measurable benefit at 300k steps / 5 seeds.
- **Established (diagnostic, §7):** the forecast is sound; the *data* doesn't offer
  exploitable anticipation lead time. This reframes §6's null result.
- **Open / not yet done:** An explicit safety layer / action-masking approach has
  not been tried as an alternative to reward-shaping, which is the more promising
  lever now that reward tuning is exhausted. This has not been run — it is a
  candidate next step, not a result.

## Multi-horizon forecast sweep (2026-09-13)

Tested the §7 hypothesis directly: trained real BiLSTM forecasters and PPO
controllers (5 seeds each, same 300k-timestep/S3-reward-weight protocol as the
6h result) at **1h and 3h** horizons too, not just 6h. `test_alternative_targets.py`
/ `generate_forecast_feature.py` / `exp_2x2_run.py` / `exp_2x2_cap_only.py` all
take a `--horizon {1,3,6}` argument now; `report_horizon_comparison.py` aggregates
the three horizons into `acn_horizon_comparison.csv` and
`make_paper_figures.py`'s new Figure 5 plots it. (`no_forecast`/`cap_only` don't
depend on horizon, so those were re-evaluated on each horizon's test weeks rather
than retrained — only the `forecast` condition needed fresh PPO training per
horizon.)

| Horizon | Forecast WAPE | no_forecast viol% | forecast viol% | no_forecast kWh | forecast kWh | Δ kWh | beats seed noise? |
|---|---|---|---|---|---|---|---|
| 1h | 105.69% | 3.36±0.89% | 3.21±0.50% | 1884.3±34.4 | 1895.6±24.7 | +11.3 | NO |
| 3h | 35.61%  | 3.36±0.89% | 3.02±0.43% | 1885.3±35.1 | 1897.2±18.7 | +12.0 | NO |
| 6h | 23.80%  | 3.29±0.94% | 3.00±0.53% | 1884.3±34.9 | 1899.0±24.3 | +14.7 | NO |

**Result: the null result is robust across the whole tested accuracy range.**
Forecast never beats no_forecast beyond seed-to-seed noise, at any horizon —
despite forecast WAPE spanning 23.8% (6h) to 105.7% (1h), a >4x range in
accuracy. This directly answers §7's open question: a shorter-horizon forecast
does not "realistically help more" — it helps just as little as the 6h one did.

**Caveat on the WAPE trend itself:** WAPE improves monotonically with horizon
(105.7% → 35.6% → 23.8%), but this is not pure evidence that "more lead time is
inherently more predictable." Target A is a *rolling sum* of arrival demand
over the horizon window; at h=1 it collapses to the raw per-hour arrival-demand
series (56%-zero, same as the original hourly target, hence WAPE in the same
~100% range), while at h=6 summing six hours smooths out that zero-inflation
and noise. Part of the accuracy gain with horizon is a target-definition
smoothing artifact of the rolling-sum construction, not solely a lead-time
effect — worth stating explicitly when writing this up, so the finding isn't
over-read as "longer lead time is intrinsically easier to forecast" in general.

## Capacity sweep (2026-09-13)

Repeated the 2x2 controller comparison (`cap_only` / `no_forecast` / `forecast`,
fixed at the paper's primary 6h horizon) across three transformer capacity
levels: **50 kW** (the paper's original limit, reused as-is) plus **75 kW** and
**100 kW**, newly trained. Per explicit requirement, `no_forecast` and
`forecast` were **retrained from scratch at every capacity** (5 seeds each,
same 300k-timestep/S3-reward-weight protocol) rather than evaluating a
50kW-trained policy under a different limit — capacity changes the
environment's actual constraint and reward (unlike forecast horizon, which
only changes `no_forecast`'s *evaluation* weeks, not its training), so a
mismatched-capacity evaluation would not be a fair comparison. `exp_2x2_run.py`
/ `exp_2x2_cap_only.py` / `exp_2x2_report.py` now take a `--capacity-kw`
argument (`run_dirs()` generalizes the earlier `horizon_dirs()` to key on both
axes); `report_capacity_comparison.py` aggregates the three capacities into
`acn_capacity_comparison.csv`, plotted by `make_paper_figures.py`'s new Figure 6.
(A `40 kW` capacity was also started but stopped and dropped mid-training at
the user's request before it produced any checkpoint — no partial result
exists for it.)

| Capacity | cap_only viol% / kWh | no_forecast viol% / kWh | forecast viol% / kWh | Δ kWh | beats seed noise? |
|---|---|---|---|---|---|
| 50 kW  | 0.00% / 2161.6 | 3.29±0.94% / 1884.3±34.9 | 3.00±0.53% / 1899.0±24.3 | +14.7 | NO |
| 75 kW  | 0.00% / 2234.6 | 1.17±0.26% / 2066.0±21.5 | 1.26±0.26% / 2085.1±10.5 | +19.1 | NO |
| 100 kW | 0.00% / 2245.0 | 0.05±0.07% / 2093.5±20.6 | 0.07±0.07% / 2110.0±14.6 | +16.5 | NO |

**Result: as capacity is relaxed, violations for both learned conditions fall
sharply and monotonically (3.0-3.3% at 50kW → ~1.2% at 75kW → ~0.05-0.07% at
100kW) and delivered energy rises accordingly** — the environment/reward
respond to capacity exactly as expected, a useful sanity check on the whole
setup. **But the forecast-vs-no_forecast comparison itself is unchanged by
capacity: forecast never beats no_forecast beyond seed-to-seed noise at any of
the three levels**, including at 75kW/100kW where `no_forecast` actually edges
out `forecast` on violation rate (by less than one pooled std either way, so
this is noise, not a reversal). This is the same qualitative result as the
horizon sweep: the null finding on forecast usefulness is robust to varying
capacity, not an artifact of the specific 50kW limit chosen for the paper.

**On "beats seed noise" — same caveat as the horizon sweep, restated for
clarity:** this is not a t-test, confidence interval, or p-value. It is
`abs(mean_forecast - mean_no_forecast) > sqrt(std_forecast**2 + std_no_forecast**2)`
across n=5 seeds per condition (sample std, ddof=1) — a hand-rolled
effect-size-vs-spread heuristic (`report_capacity_comparison.py`, same logic as
`report_horizon_comparison.py`). It does not divide by `sqrt(n)`, so it is
actually a *stricter* bar than a proper standard-error-based test would be —
worth citing precisely rather than as "not statistically significant" if this
goes in the paper.

## Appendix: Legacy Experiment 5 (older FULL TwinEV pipeline)

Experiment 5 is the five-seed repetition of the original FULL TwinEV
LSTM+PPO+PI pipeline, **not** the newer ACN reviewer experiments. Results:
delivered energy 109,712.15 ± 5,201.74 kWh (CV 4.7%), peak power
115.54 ± 2.64 kW (CV 2.3%), and evaluation reward 4,108.67 ± 458.74
(CV 11.2%). Seeds 4-5 deliver about 8% more energy with roughly 12x lower
power variance than seeds 1-3, so the seed spread reflects different
smoothing strategies rather than noise around one solution. Final training
reward is tight (349.13 ± 5.92) and nearly uncorrelated with evaluation
reward (r = 0.14), so it is not a model-selection signal. Multi-seed
reporting is required for energy/reward claims on this pipeline. The
zero-overload result is by construction (the archived environment
hard-clips at 150 kW) and is not a learned-safety result. Figure:
paper_figures/legacy_exp5_twinev_seed_stability.pdf; data:
results/statistics/twinev_seed_stability.csv.
