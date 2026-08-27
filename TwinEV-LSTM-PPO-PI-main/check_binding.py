# ==========================================
# ACN-Data transformer-binding diagnostic
# ==========================================
# One question: does a station-wide transformer capacity constraint
# actually bind on the real ACN-Data (Caltech, 2019) session record,
# or is concurrency so low that any realistic limit is never
# threatened? Answering this picks the transformer_limit_kw for
# ACNChargingEnv - too high and the safety problem is trivial (never
# binds, exactly the flaw the new environment was built to avoid);
# too low and it's unachievable regardless of policy.
#
# Full year (not one month): a single busy month can look artificially
# binding or artificially slack depending on season: 2019 in ACN-Data
# has Caltech's known long summer/fall ramp-up in EV adoption, so
# June alone under- or over-states the annual picture.
# ==========================================

import numpy as np
import pandas as pd

DATA_PATH = "acn_caltech_2019.csv"
PER_EV_KW = 6.6   # typical ACN EVSE level-2 rate (matches acn_env.py's default)

# Reference points already used elsewhere in this project (twinev_env.py):
# MAX_GRID_KW = 150 (hard limit), OVERLOAD_KW = 130 (soft limit).
REFERENCE_LIMITS_KW = [50, 75, 100, 130, 150, 200]

VIOLATION_TARGETS = [0.05, 0.10, 0.20]   # 5% / 10% / 20% of hours


def load_sessions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Mixed UTC offsets across the year (PST -08:00 in winter, PDT
    # -07:00 in summer) - bare parse_dates=[...] / to_datetime without
    # utc=True cannot build one consistent dtype across that and
    # silently leaves the column as strings. utc=True fixes it.
    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True, errors="coerce")
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True, errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["connectionTime", "disconnectTime"])
    df = df[df["disconnectTime"] > df["connectionTime"]]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped}/{n_before} sessions with bad timestamps")

    assert pd.api.types.is_datetime64_any_dtype(df["connectionTime"]), (
        "connectionTime did not parse to a datetime dtype - "
        "mixed-offset parsing regression"
    )
    assert df["connectionTime"].dt.tz is not None, "connectionTime is not tz-aware"

    return df.reset_index(drop=True)


def hourly_concurrency(df: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Full-year hourly concurrency timeline via an O(n log n) sweep
    instead of an O(n_hours * n_sessions) nested scan: for each grid
    hour t, a session counts as connected iff connectionTime <= t <
    disconnectTime, so a session occupies the half-open index range
    [searchsorted(t >= connectionTime), searchsorted(t >= disconnectTime))
    on the hourly grid - a difference array over those ranges, cumsum'd,
    gives the same result as the naive per-hour boolean scan.
    """
    start = df["connectionTime"].min().floor("h")
    end = df["disconnectTime"].max().ceil("h")
    idx = pd.date_range(start, end, freq="h")
    grid = idx.values

    start_bin = np.searchsorted(grid, df["connectionTime"].values, side="left")
    end_bin = np.searchsorted(grid, df["disconnectTime"].values, side="left")
    start_bin = np.clip(start_bin, 0, len(grid))
    end_bin = np.clip(end_bin, 0, len(grid))

    diff = np.zeros(len(grid) + 1, dtype=np.int64)
    np.add.at(diff, start_bin, 1)
    np.add.at(diff, end_bin, -1)
    concurrent = np.cumsum(diff)[:-1]

    return idx, concurrent


def main() -> None:
    df = load_sessions(DATA_PATH)
    print(f"Loaded {len(df)} sessions, {df['stationID'].nunique()} stations, "
          f"{df['spaceID'].nunique()} spaces")
    print(f"Date range: {df['connectionTime'].min()} to {df['disconnectTime'].max()}")

    idx, concurrent = hourly_concurrency(df)
    demand_kw = concurrent * PER_EV_KW

    n_hours = len(idx)
    peak_evs = int(concurrent.max())
    mean_evs = float(concurrent.mean())
    peak_kw = float(demand_kw.max())
    mean_kw = float(demand_kw.mean())

    print(f"\nFull year: {n_hours} hourly steps ({n_hours / 24:.0f} days)")
    print(f"Peak concurrent EVs : {peak_evs}")
    print(f"Mean concurrent EVs : {mean_evs:.2f}")
    print(f"Peak aggregate demand (@ {PER_EV_KW} kW/EV): {peak_kw:.1f} kW")
    print(f"Mean aggregate demand (@ {PER_EV_KW} kW/EV): {mean_kw:.1f} kW")

    print("\nAggregate demand percentiles (kW):")
    percentiles = [50, 75, 90, 95, 99]
    pct_values = np.percentile(demand_kw, percentiles)
    for p, v in zip(percentiles, pct_values):
        print(f"  p{p:<3d}: {v:7.1f} kW")

    print("\nHours exceeding reference limits (twinev_env.py uses 130/150 kW):")
    for limit in REFERENCE_LIMITS_KW:
        n_exceed = int((demand_kw > limit).sum())
        print(f"  > {limit:4d} kW: {n_exceed:5d} / {n_hours} hours "
              f"({100 * n_exceed / n_hours:5.2f}%)")

    print("\n===== VERDICT =====")
    print("Transformer limit (kW) that would be violated in ~X% of hours:")
    # P(demand > T) = target  <=>  T = (1 - target) quantile of demand
    verdict_limits = {}
    for target in VIOLATION_TARGETS:
        q = 100 * (1 - target)
        limit_kw = float(np.percentile(demand_kw, q))
        verdict_limits[target] = limit_kw
        print(f"  ~{int(target * 100):>2d}% of hours violate at limit = {limit_kw:7.1f} kW "
              f"(p{q:.0f} of aggregate demand)")

    if peak_kw > max(REFERENCE_LIMITS_KW):
        print(f"\nPeak aggregate demand ({peak_kw:.0f} kW) comfortably exceeds all "
              f"reference limits up to {max(REFERENCE_LIMITS_KW)} kW: the constraint "
              "binds naturally on this data - a realistic site transformer rating "
              "can be used directly.")
    elif peak_kw > min(REFERENCE_LIMITS_KW):
        print(f"\nPeak aggregate demand ({peak_kw:.0f} kW) exceeds some but not all "
              "reference limits: pick a limit near the p90-p95 aggregate-demand "
              f"range above ({verdict_limits[0.10]:.0f}-{verdict_limits[0.05]:.0f} kW) "
              "for a constraint that binds occasionally but is not structurally "
              "impossible.")
    else:
        print(f"\nPeak aggregate demand ({peak_kw:.0f} kW) is low relative to typical "
              "site transformer ratings: concurrency on this data rarely threatens "
              "capacity at a realistic rating. Per the design note, use a "
              f"deliberately tighter limit (e.g. {verdict_limits[0.10]:.0f} kW, the "
              "level that binds in ~10% of hours) and justify it in the paper as a "
              "capacity-constrained/retrofit scenario, not the site's real rating.")


if __name__ == "__main__":
    main()
