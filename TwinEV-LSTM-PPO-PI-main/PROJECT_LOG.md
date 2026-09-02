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
- **Open / not yet done:** a shorter-horizon forecast (e.g. 1h-ahead, matching the
  only horizon where any lead time exists at all) has not been tried and might
  realistically help more than the 6h target used here. An explicit safety
  layer / action-masking approach has not been tried as an alternative to
  reward-shaping, which is the more promising lever now that reward tuning is
  exhausted. Neither of these has been run — they are candidate next steps, not
  results.
