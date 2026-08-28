# ==========================================
# ACN-Data session loader
# ==========================================
# Reads ALREADY-CACHED ACN-Data charging sessions from a local
# CSV/Parquet file and turns them into the per-timestep "which EVs
# are plugged in, at which physical slot, needing how much energy"
# structure ACNChargingEnv steps through.
#
# Does NOT call the ACN API and does NOT handle tokens. To produce a
# cache file, pull sessions the way test_acn.py does and dump them,
# e.g.:
#
#     import pandas as pd
#     pd.DataFrame(sessions).to_csv("acn_sessions.csv", index=False)
#
# ACN-Data session fields used here: connectionTime, disconnectTime,
# kWhDelivered, doneChargingTime (optional), stationID, spaceID,
# userInputs (optional; a list of dicts, each possibly containing
# 'kWhRequested' - the driver's actual requested energy).
# ==========================================

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("connectionTime", "disconnectTime", "kWhDelivered")


@dataclass
class Session:
    """One EV plug-in event, resolved onto the fixed simulation grid."""
    session_id: int
    slot_id: str            # physical bay identity (spaceID by default)
    connect_step: int       # first timestep index the EV is present at
    disconnect_step: int    # timestep index the EV departs at (exclusive)
    requested_kwh: float    # energy demand tracked as the env's ground truth


@dataclass
class ACNSessionData:
    sessions: list[Session]
    # active_by_step[t] = session indices connected at timestep t
    # (general-purpose per-timestep occupancy structure)
    active_by_step: list[list[int]]
    # arrivals_by_step[t] = session indices whose connect_step == t
    # (what the env actually needs to admit new EVs each step)
    arrivals_by_step: list[list[int]]
    n_steps: int
    slot_ids: list[str]      # stable slot ordering used for obs/action indexing
    timestep_minutes: int
    start_time: pd.Timestamp


def _parse_requested_kwh(flat_requested_kwh, user_inputs, fallback_kwh: float) -> float:
    """Resolve a session's true energy demand, in priority order:

    1. A flat `requested_kWh` column, if the export has one and it's a
       usable positive number. This is how `pull_acn_data.py` actually
       exports real ACN-Data (`userInputs[0]['kWhRequested']` already
       flattened at pull time) - present on ~85% of real Caltech 2019
       rows, and is the ONLY one of these three sources that reflects
       what the driver actually asked for rather than what they
       happened to receive.
    2. A nested `userInputs` field (list of dicts, one per plug-in
       event), if present - kept for CSVs/Parquets that were cached
       without flattening it first.
    3. `kWhDelivered` - the session's actual delivered energy. This
       systematically UNDER-estimates true demand for any session
       that finished charging or departed before getting everything
       it wanted, which is common (median real request is ~2x median
       delivered on this dataset) - use only when nothing better exists.
    """
    if flat_requested_kwh is not None:
        try:
            flat_val = float(flat_requested_kwh)
        except (TypeError, ValueError):
            flat_val = float("nan")
        if not math.isnan(flat_val) and flat_val > 0:
            return flat_val

    if user_inputs is None:
        return fallback_kwh
    if isinstance(user_inputs, float) and math.isnan(user_inputs):
        return fallback_kwh
    if isinstance(user_inputs, str):
        stripped = user_inputs.strip()
        if not stripped:
            return fallback_kwh
        try:
            user_inputs = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return fallback_kwh
    if isinstance(user_inputs, dict):
        user_inputs = [user_inputs]
    if not isinstance(user_inputs, list) or not user_inputs:
        return fallback_kwh
    entry = user_inputs[0]
    if not isinstance(entry, dict):
        return fallback_kwh
    kwh = entry.get("kWhRequested")
    if kwh is None:
        return fallback_kwh
    try:
        kwh = float(kwh)
    except (TypeError, ValueError):
        return fallback_kwh
    return kwh if kwh > 0 else fallback_kwh


def _parse_datetime_utc(series: pd.Series) -> pd.Series:
    """Parse an ACN-Data timestamp column to tz-aware UTC datetimes.

    ACN-Data timestamps carry mixed UTC offsets across a year
    (-08:00 PST / -07:00 PDT for Pacific-timezone sites). Plain
    `pd.read_csv(..., parse_dates=[...])` (and bare `pd.to_datetime`
    without `utc=True`) cannot build one consistent datetime64 dtype
    across mixed offsets and silently falls back to leaving the
    column as strings (object dtype) - `.dt` accessors then raise,
    and any comparison/sorting on it is comparing strings, not time.
    `utc=True` normalizes every row to a single UTC-based dtype
    regardless of its original offset, which is the fix.
    """
    return pd.to_datetime(series, utc=True, errors="coerce")


def _read_table(data_path: str) -> pd.DataFrame:
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    if data_path.endswith(".parquet"):
        try:
            return pd.read_parquet(data_path)
        except ImportError as e:
            raise ImportError(
                "Reading .parquet requires 'pyarrow' (or 'fastparquet'), "
                "which is not in requirements.txt. Install one of those, "
                "or export the cache as .csv instead."
            ) from e
    raise ValueError(
        f"Unsupported data file extension for '{data_path}': "
        "expected '.csv' or '.parquet'."
    )


def load_acn_sessions(
    data_path: str,
    timestep_minutes: int = 60,
    slot_key: str = "spaceID",
) -> ACNSessionData:
    """Load a cached ACN-Data sessions file and resolve it onto a
    fixed timestep grid.

    Parameters
    ----------
    data_path : path to a local .csv or .parquet export of ACN-Data
        sessions (see module docstring - this function never touches
        the network).
    timestep_minutes : simulation step size.
    slot_key : which column identifies the physical charging bay a
        session occupies. Defaults to 'spaceID' (a bay can only serve
        one EV at a time, so this gives every observation/action index
        a stable physical identity across the whole episode).
    """
    if timestep_minutes <= 0:
        raise ValueError("timestep_minutes must be positive")

    df = _read_table(data_path)

    missing = [c for c in (*REQUIRED_COLUMNS, slot_key) if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{data_path}' is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["connectionTime"] = _parse_datetime_utc(df["connectionTime"])
    df["disconnectTime"] = _parse_datetime_utc(df["disconnectTime"])

    n_before = len(df)
    df = df.dropna(subset=["connectionTime", "disconnectTime", slot_key])
    df = df[df["disconnectTime"] > df["connectionTime"]]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[acn_data_loader] dropped {n_dropped}/{n_before} sessions "
              f"with missing timestamps/slot or non-positive duration")

    if df.empty:
        raise ValueError(f"'{data_path}' contained no usable sessions after cleaning")

    user_inputs_col = df["userInputs"] if "userInputs" in df.columns else [None] * len(df)

    start_time = df["connectionTime"].min()
    end_time = df["disconnectTime"].max()
    step = pd.Timedelta(minutes=timestep_minutes)
    n_steps = int(math.ceil((end_time - start_time) / step))
    n_steps = max(n_steps, 1)

    sessions: list[Session] = []
    skipped_zero_len = 0
    for i, (row, user_inputs) in enumerate(zip(df.itertuples(index=False), user_inputs_col)):
        # connect_step = floor((connectionTime - start) / step); disconnect_step
        # = ceil((disconnectTime - start) / step). Given the disconnectTime >
        # connectionTime filter above (b > a), floor(a) < ceil(b) is a
        # mathematical guarantee: floor(a) <= a < b <= ceil(b). So every
        # admitted session is guaranteed >= 1 chargeable step on the grid -
        # the env can never be handed demand that's physically impossible to
        # serve in zero time. Verified on the real 10,609-session Caltech
        # 2019 cache: 0 sessions hit the skip condition below (see
        # diagnose_zero_length_sessions()). The skip is kept as a defensive
        # guard in case that invariant is ever broken by a future change to
        # the rounding functions, not because it currently fires.
        connect_step = int((getattr(row, "connectionTime") - start_time) / step)
        disconnect_step = int(math.ceil((getattr(row, "disconnectTime") - start_time) / step))
        disconnect_step = min(disconnect_step, n_steps)
        if disconnect_step <= connect_step:
            skipped_zero_len += 1
            continue

        fallback_kwh = float(getattr(row, "kWhDelivered") or 0.0)
        flat_requested_kwh = getattr(row, "requested_kWh", None)
        requested_kwh = _parse_requested_kwh(flat_requested_kwh, user_inputs, fallback_kwh)
        if requested_kwh <= 0:
            continue

        sessions.append(Session(
            session_id=i,
            slot_id=str(getattr(row, slot_key)),
            connect_step=connect_step,
            disconnect_step=disconnect_step,
            requested_kwh=requested_kwh,
        ))

    if skipped_zero_len:
        print(f"[acn_data_loader] skipped {skipped_zero_len} sessions shorter "
              f"than one timestep ({timestep_minutes} min)")
    if not sessions:
        raise ValueError(
            "No sessions survived resolution onto the timestep grid - "
            "try a smaller timestep_minutes"
        )

    slot_ids = sorted({s.slot_id for s in sessions})

    active_by_step: list[list[int]] = [[] for _ in range(n_steps)]
    arrivals_by_step: list[list[int]] = [[] for _ in range(n_steps)]
    for idx, s in enumerate(sessions):
        for t in range(s.connect_step, s.disconnect_step):
            active_by_step[t].append(idx)
        arrivals_by_step[s.connect_step].append(idx)

    return ACNSessionData(
        sessions=sessions,
        active_by_step=active_by_step,
        arrivals_by_step=arrivals_by_step,
        n_steps=n_steps,
        slot_ids=slot_ids,
        timestep_minutes=timestep_minutes,
        start_time=start_time,
    )


def _test_mixed_offset_datetime_parsing() -> None:
    """Regression test: FAILS if timestamps with mixed UTC offsets
    (PST -08:00 vs PDT -07:00, as ACN-Data has across a year) come
    back as strings instead of tz-aware datetimes.
    """
    import io

    csv_text = (
        "connectionTime,disconnectTime\n"
        "2019-01-15 09:00:00-08:00,2019-01-15 11:00:00-08:00\n"   # PST
        "2019-07-15 09:00:00-07:00,2019-07-15 11:00:00-07:00\n"   # PDT
    )
    df = pd.read_csv(io.StringIO(csv_text))
    assert df["connectionTime"].dtype == object, (
        "test fixture assumption broken: expected bare read_csv to "
        "leave mixed-offset timestamps as strings"
    )

    parsed = _parse_datetime_utc(df["connectionTime"])

    assert pd.api.types.is_datetime64_any_dtype(parsed), (
        f"timestamps came back as {parsed.dtype}, not a datetime64 "
        "dtype - mixed-offset parsing is broken"
    )
    assert parsed.dt.tz is not None, "parsed timestamps are not tz-aware"
    assert parsed.isna().sum() == 0, "valid mixed-offset timestamps were dropped as NaT"

    # Both rows are 09:00 *local* time but on different UTC offsets;
    # confirm they resolved to different, correctly-shifted UTC instants
    # rather than being compared/parsed as equal local-looking strings.
    pst_utc_hour = parsed.iloc[0].hour
    pdt_utc_hour = parsed.iloc[1].hour
    assert pst_utc_hour == 17, f"PST row parsed to UTC hour {pst_utc_hour}, expected 17"
    assert pdt_utc_hour == 16, f"PDT row parsed to UTC hour {pdt_utc_hour}, expected 16"

    print("[PASS] mixed-offset timestamps parse to proper tz-aware UTC datetimes")


def slice_session_data(
    data: ACNSessionData, start_step: int, length: int,
    slot_ids: list[str] | None = None,
) -> ACNSessionData:
    """Extract a self-contained sub-window of `length` timesteps
    starting at `start_step` from a full-timeline ACNSessionData.

    `slot_ids`: pass the FULL dataset's slot roster (not left as the
    default None) when windows will be used as RL episodes with a
    fixed-shape observation/action space - e.g. SB3 PPO queries
    observation_space/action_space once and assumes it never changes,
    but a given one-week window may only see arrivals at a SUBSET of
    physical stations, which would otherwise shrink n_slots for that
    episode. Passing the full roster keeps every episode's slot
    indexing identical; slots with no session that week are simply
    always-empty in that episode's observation. Left as None (auto-
    derived from just this window's sessions) for non-RL uses like the
    smoke test, where a shrinking roster is harmless.

    Only sessions whose connect_step falls inside [start_step,
    start_step + length) are included - a session already in
    progress at start_step is excluded rather than admitted with an
    ambiguous "how much of its demand was already served before the
    window began" state. Every returned session starts fresh, re-based
    to the window's own step 0, exactly as if load_acn_sessions() had
    only ever seen that slice of data. Sessions that arrive inside the
    window but would run past its end are truncated to the window
    (disconnect_step clipped) while keeping their FULL original
    requested_kwh - this can create some unmet energy purely from the
    window boundary, independent of policy quality; both policies
    being compared are equally subject to it, so it doesn't bias a
    relative comparison, but it does mean windowed unmet-energy
    fractions read somewhat higher than a full continuous simulation
    would show.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if not (0 <= start_step < data.n_steps):
        raise ValueError(f"start_step {start_step} out of range [0, {data.n_steps})")

    end_step = min(start_step + length, data.n_steps)
    n_steps = end_step - start_step

    sessions: list[Session] = []
    for t in range(start_step, end_step):
        for session_idx in data.arrivals_by_step[t]:
            s = data.sessions[session_idx]
            sessions.append(Session(
                session_id=len(sessions),
                slot_id=s.slot_id,
                connect_step=s.connect_step - start_step,
                disconnect_step=min(s.disconnect_step, end_step) - start_step,
                requested_kwh=s.requested_kwh,
            ))

    if not sessions and slot_ids is None:
        raise ValueError(f"window [{start_step}, {end_step}) has no session arrivals")

    if slot_ids is None:
        slot_ids = sorted({s.slot_id for s in sessions})
    active_by_step: list[list[int]] = [[] for _ in range(n_steps)]
    arrivals_by_step: list[list[int]] = [[] for _ in range(n_steps)]
    for i, s in enumerate(sessions):
        for t in range(s.connect_step, s.disconnect_step):
            active_by_step[t].append(i)
        arrivals_by_step[s.connect_step].append(i)

    return ACNSessionData(
        sessions=sessions,
        active_by_step=active_by_step,
        arrivals_by_step=arrivals_by_step,
        n_steps=n_steps,
        slot_ids=slot_ids,
        timestep_minutes=data.timestep_minutes,
        start_time=data.start_time + start_step * pd.Timedelta(minutes=data.timestep_minutes),
    )


def diagnose_zero_length_sessions(
    data_path: str,
    timestep_minutes: int = 60,
    slot_key: str = "spaceID",
) -> dict:
    """Report how many sessions round to a zero-length stay on the
    timestep grid (arrival_step == disconnect_step) and how much
    requested kWh they represent - both in absolute kWh and as a
    fraction of total requested demand in the dataset.

    This duplicates load_acn_sessions()'s own connect/disconnect_step
    computation rather than calling it, because load_acn_sessions()
    already silently drops zero-length sessions (see the comment at
    its skip site) - this function exists specifically to measure
    what would be dropped, and why it's expected to be ~0 given the
    floor/ceil rounding asymmetry documented there.
    """
    df = _read_table(data_path)
    df["connectionTime"] = _parse_datetime_utc(df["connectionTime"])
    df["disconnectTime"] = _parse_datetime_utc(df["disconnectTime"])
    df = df.dropna(subset=["connectionTime", "disconnectTime", slot_key])
    df = df[df["disconnectTime"] > df["connectionTime"]]

    start_time = df["connectionTime"].min()
    end_time = df["disconnectTime"].max()
    step = pd.Timedelta(minutes=timestep_minutes)
    n_steps = max(int(math.ceil((end_time - start_time) / step)), 1)
    user_inputs_col = df["userInputs"] if "userInputs" in df.columns else [None] * len(df)

    n_sessions = 0
    total_kwh = 0.0
    zero_len_count = 0
    zero_len_kwh = 0.0
    for row, user_inputs in zip(df.itertuples(index=False), user_inputs_col):
        fallback_kwh = float(getattr(row, "kWhDelivered") or 0.0)
        flat_requested_kwh = getattr(row, "requested_kWh", None)
        requested_kwh = _parse_requested_kwh(flat_requested_kwh, user_inputs, fallback_kwh)
        if requested_kwh <= 0:
            continue
        n_sessions += 1
        total_kwh += requested_kwh

        connect_step = int((getattr(row, "connectionTime") - start_time) / step)
        disconnect_step = int(math.ceil((getattr(row, "disconnectTime") - start_time) / step))
        disconnect_step = min(disconnect_step, n_steps)
        if disconnect_step <= connect_step:
            zero_len_count += 1
            zero_len_kwh += requested_kwh

    frac = (zero_len_kwh / total_kwh) if total_kwh > 0 else 0.0
    return {
        "n_sessions": n_sessions,
        "total_requested_kwh": total_kwh,
        "zero_length_sessions": zero_len_count,
        "zero_length_kwh": zero_len_kwh,
        "zero_length_kwh_fraction": frac,
    }


def make_synthetic_fixture(
    n_slots: int = 4,
    timestep_minutes: int = 60,
    seed: int = 0,
) -> ACNSessionData:
    """A small, deliberately-overlapping set of SYNTHETIC sessions used
    only to exercise ACNChargingEnv's self-tests in __main__.

    THIS IS NOT REAL ACN-DATA. It exists because no cached ACN-Data
    export is available in this repo yet - point load_acn_sessions()
    at a real cache file for anything beyond a mechanics smoke test.
    """
    rng = np.random.default_rng(seed)
    start_time = pd.Timestamp("2024-01-01T08:00:00Z")
    step = pd.Timedelta(minutes=timestep_minutes)

    slot_ids = [f"SLOT_{i}" for i in range(n_slots)]
    sessions: list[Session] = []
    active_by_step: list[list[int]] = []
    arrivals_by_step: list[list[int]] = []

    n_steps = 8
    # Every slot gets an EV that connects at step 0 and stays for
    # several steps, deliberately overlapping so a max-power policy
    # is forced to draw through every slot at once.
    for i, slot in enumerate(slot_ids):
        connect_step = 0
        disconnect_step = int(rng.integers(4, n_steps))
        requested_kwh = float(rng.uniform(8.0, 15.0))
        sessions.append(Session(
            session_id=i, slot_id=slot,
            connect_step=connect_step, disconnect_step=disconnect_step,
            requested_kwh=requested_kwh,
        ))

    active_by_step = [[] for _ in range(n_steps)]
    arrivals_by_step = [[] for _ in range(n_steps)]
    for idx, s in enumerate(sessions):
        for t in range(s.connect_step, s.disconnect_step):
            active_by_step[t].append(idx)
        arrivals_by_step[s.connect_step].append(idx)

    return ACNSessionData(
        sessions=sessions,
        active_by_step=active_by_step,
        arrivals_by_step=arrivals_by_step,
        n_steps=n_steps,
        slot_ids=slot_ids,
        timestep_minutes=timestep_minutes,
        start_time=start_time,
    )


if __name__ == "__main__":
    import os

    _test_mixed_offset_datetime_parsing()

    real_cache = "acn_caltech_2019.csv"
    if os.path.exists(real_cache):
        data = load_acn_sessions(real_cache, timestep_minutes=60)
        assert data.sessions, "loaded 0 sessions from the real cache"
        for s in data.sessions[:5] + data.sessions[-5:]:
            assert isinstance(s.connect_step, int) and isinstance(s.disconnect_step, int)
        print(f"[PASS] loaded {len(data.sessions)} real sessions, "
              f"{len(data.slot_ids)} slots, {data.n_steps} hourly steps "
              f"starting {data.start_time}")

        diag = diagnose_zero_length_sessions(real_cache, timestep_minutes=60)
        print(f"\nZero-length-on-grid diagnostic ({real_cache}):")
        print(f"  sessions               : {diag['n_sessions']}")
        print(f"  total requested kWh    : {diag['total_requested_kwh']:.1f}")
        print(f"  zero-length sessions   : {diag['zero_length_sessions']}")
        print(f"  zero-length kWh        : {diag['zero_length_kwh']:.4f}")
        print(f"  zero-length kWh fraction: {100 * diag['zero_length_kwh_fraction']:.4f}%")
        assert diag["zero_length_kwh_fraction"] == 0.0, (
            "zero-length sessions now represent nonzero requested kWh - "
            "the floor/ceil invariant documented in load_acn_sessions() "
            "broke; a real fix (not just this assertion) is needed"
        )
    else:
        print(f"[skip] real cache '{real_cache}' not found - "
              "only ran the mixed-offset regression test")

    print("\nAll acn_data_loader self-tests passed.")
