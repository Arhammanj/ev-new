"""Cross-dataset lead-time criterion audit on the cached ACN datasets.

The primary criterion follows the existing diagnostic semantics without using
the loader's Office001 fallback:

  A(l, h) = fraction of EVs contributing to risk hour h that were present
            at anchor h-l
  S(l, h) = max(0, capacity_kw - greedy achievable draw at h-l)
  X(l)    = median_h[A(l,h) * I(S(l,h) > 0)]

Risk is a greedy achievable draw above the configured transformer capacity.
All scores use the same hourly grid, 6.6 kW per-EV limit, and 50 kW capacity.
Office001 has no requested-energy field, so its primary score is reported as
unsupported rather than using kWhDelivered as demand.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from acn_data_loader import ACNSessionData, Session  # noqa: E402


DATASETS = {
    "Caltech": REPO_ROOT / "acn_caltech_2019.csv",
    "JPL": REPO_ROOT / "acn_jpl_2019.csv",
    "Office001": REPO_ROOT / "acn_office001_2019.csv",
}
CAPACITY_KW = 50.0
PER_EV_MAX_KW = 6.6
HORIZONS_HOURS = tuple(range(1, 11))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = OUTPUT_DIR / "paper_figures"


def raw_verification(name: str, path: Path) -> dict:
    df = pd.read_csv(path)
    parsed = {
        col: pd.to_datetime(df[col], utc=True, errors="coerce")
        for col in ("connectionTime", "disconnectTime", "doneChargingTime")
        if col in df
    }
    requested = pd.to_numeric(df.get("requested_kWh"), errors="coerce")
    delivered = pd.to_numeric(df["kWhDelivered"], errors="coerce")
    return {
        "dataset": name,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sessions_raw": len(df),
        "columns": "|".join(df.columns),
        "date_start_connection_utc": str(parsed["connectionTime"].min()),
        "date_end_connection_utc": str(parsed["connectionTime"].max()),
        "date_start_disconnect_utc": str(parsed["disconnectTime"].min()),
        "date_end_disconnect_utc": str(parsed["disconnectTime"].max()),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_done_charging": int(df["doneChargingTime"].isna().sum()),
        "stations": int(df["stationID"].nunique()),
        "spaces": int(df["spaceID"].nunique()),
        "requested_energy_positive": int((requested > 0).sum()),
        "requested_energy_coverage": float((requested > 0).mean()),
        "delivered_energy_positive": int((delivered > 0).sum()),
        "delivered_kwh_total": float(delivered.sum()),
        "energy_basis": (
            "requested_kWh"
            if (requested > 0).any()
            else "unsupported: requested_kWh absent for every session"
        ),
        "criterion_supported": bool((requested > 0).any()),
        "complete_case_sessions": int((requested > 0).sum()),
        "excluded_missing_requested_sessions": int((~(requested > 0)).sum()),
    }


def load_grid(path: Path, timestep_minutes: int = 60):
    raw = pd.read_csv(path)
    requested = pd.to_numeric(raw["requested_kWh"], errors="coerce")
    raw = raw.loc[requested > 0].copy()
    raw["requested_kWh"] = requested.loc[raw.index]
    raw["connectionTime"] = pd.to_datetime(raw["connectionTime"], utc=True, errors="coerce")
    raw["disconnectTime"] = pd.to_datetime(raw["disconnectTime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["connectionTime", "disconnectTime", "spaceID"])
    raw = raw[raw["disconnectTime"] > raw["connectionTime"]]
    if raw.empty:
        raise ValueError(f"{path} has no requested-energy complete cases")

    start = pd.to_datetime(pd.read_csv(path)["connectionTime"], utc=True).min()
    end = pd.to_datetime(pd.read_csv(path)["disconnectTime"], utc=True).max()
    step = pd.Timedelta(minutes=timestep_minutes)
    n_steps = max(1, int(np.ceil((end - start) / step)))
    sessions = []
    for row_index, row in raw.iterrows():
        connect = int((row["connectionTime"] - start) / step)
        disconnect = min(n_steps, int(np.ceil((row["disconnectTime"] - start) / step)))
        if disconnect > connect:
            sessions.append(Session(
                session_id=int(row_index),
                slot_id=str(row["spaceID"]),
                connect_step=connect,
                disconnect_step=disconnect,
                requested_kwh=float(row["requested_kWh"]),
            ))
    slot_ids = sorted({s.slot_id for s in sessions})
    slot_index = {slot: i for i, slot in enumerate(slot_ids)}
    active = [[] for _ in range(n_steps)]
    arrivals = [[] for _ in range(n_steps)]
    for index, session in enumerate(sessions):
        for step_index in range(session.connect_step, session.disconnect_step):
            active[step_index].append(index)
        arrivals[session.connect_step].append(index)
    return ACNSessionData(
        sessions=sessions,
        active_by_step=active,
        arrivals_by_step=arrivals,
        n_steps=n_steps,
        slot_ids=slot_ids,
        timestep_minutes=timestep_minutes,
        start_time=start,
    )


def greedy_series(data) -> tuple[np.ndarray, np.ndarray]:
    dt_hours = data.timestep_minutes / 60.0
    achievable = np.zeros(data.n_steps, dtype=float)
    connected = np.zeros(data.n_steps, dtype=float)
    for step, indices in enumerate(data.active_by_step):
        connected[step] = len(indices)
        achievable[step] = sum(
            min(PER_EV_MAX_KW, data.sessions[idx].requested_kwh / dt_hours)
            for idx in indices
        )
    return achievable, connected


def criterion_for_dataset(
    name: str,
    data,
    capacity_kw: float = CAPACITY_KW,
    horizons: tuple[int, ...] = HORIZONS_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    score_rows = []
    if name == "Office001":
        for horizon in horizons:
            score_rows.append({
                "dataset": name,
                "horizon_hours": horizon,
                "risk_hours": np.nan,
                "criterion_x": np.nan,
                "criterion_status": "unsupported_no_requested_energy",
            })
        return pd.DataFrame(), pd.DataFrame(score_rows)
    achievable, connected = greedy_series(data)
    risk = achievable > capacity_kw
    headroom = np.maximum(0.0, capacity_kw - achievable)

    for horizon in horizons:
        values = []
        horizon_steps = int(round(horizon * 60 / data.timestep_minutes))
        for risk_step in np.flatnonzero(risk):
            contributing = [
                idx for idx in data.active_by_step[risk_step]
                if data.sessions[idx].requested_kwh > 0
            ]
            anchor = int(risk_step - horizon_steps)
            if not contributing or anchor < 0:
                a = 0.0
                slack = 0.0
            else:
                present_indices = set(data.active_by_step[anchor])
                a = sum(
                    idx in present_indices
                    for idx in contributing
                ) / len(contributing)
                slack = float(headroom[anchor])
            indicator = float(slack > 0.0)
            x_value = a * indicator
            values.append(x_value)
            event_rows.append({
                "dataset": name,
                "risk_step": int(risk_step),
                "risk_timestamp_utc": str(
                    data.start_time + pd.Timedelta(
                        minutes=risk_step * data.timestep_minutes
                    )
                ),
                "horizon_hours": horizon,
                "horizon_steps": horizon_steps,
                "anchor_step": anchor,
                "contributing_evs": len(contributing),
                "connected_fraction_A": a,
                "headroom_S_kw": slack,
                "headroom_positive": bool(indicator),
                "criterion_product": x_value,
            })
        score_rows.append({
            "dataset": name,
            "horizon_hours": horizon,
            "risk_hours": int(risk.sum()),
            "criterion_x": float(np.median(values)) if values else np.nan,
            "criterion_status": "computed_complete_case",
        })
    return pd.DataFrame(event_rows), pd.DataFrame(score_rows)


def statistical_diagnostics(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize event-level products and identify persistent zero-risk hours."""
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = []
    group_columns = ["dataset", "horizon_hours"]
    if "capacity_kw" in events.columns:
        group_columns.insert(1, "capacity_kw")
    for group_key, group in events.groupby(group_columns, sort=True):
        if len(group_columns) == 2:
            dataset, horizon = group_key
            capacity = None
        else:
            dataset, capacity, horizon = group_key
        product = group["criterion_product"].to_numpy(dtype=float)
        summary = {
            "dataset": dataset,
            "horizon_hours": int(horizon),
            "mean_score": float(np.mean(product)),
            "median_score": float(np.median(product)),
            "std_score": float(np.std(product)),
            "p25_score": float(np.quantile(product, 0.25)),
            "p75_score": float(np.quantile(product, 0.75)),
            "min_score": float(np.min(product)),
            "max_score": float(np.max(product)),
            "pct_A_positive": float(np.mean(group["connected_fraction_A"] > 0) * 100),
            "pct_S_positive": float(np.mean(group["headroom_positive"]) * 100),
            "pct_both_positive": float(np.mean(product > 0) * 100),
            "pct_zero_product": float(np.mean(product == 0) * 100),
            "risk_hours_evaluated": int(len(group)),
        }
        if capacity is not None:
            summary["capacity_kw"] = float(capacity)
        grouped.append(summary)
    diagnostics = pd.DataFrame(grouped)
    persistence_columns = ["dataset", "risk_step"]
    if "capacity_kw" in events.columns:
        persistence_columns.insert(1, "capacity_kw")
    persistence = (
        events.groupby(persistence_columns, as_index=False)
        .agg(
            risk_timestamp_utc=("risk_timestamp_utc", "first"),
            horizons_evaluated=("horizon_hours", "nunique"),
            horizons_with_positive_product=(
                "criterion_product",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            horizons_with_zero_product=(
                "criterion_product",
                lambda values: int(np.sum(np.asarray(values) == 0)),
            ),
        )
    )
    persistence["zero_for_all_evaluated_horizons"] = (
        persistence["horizons_with_positive_product"] == 0
    )
    return diagnostics, persistence


def occupancy_counts(path: Path, timestep_minutes: int = 60) -> np.ndarray:
    raw = pd.read_csv(path)
    start = pd.to_datetime(raw["connectionTime"], utc=True, errors="coerce").min()
    end = pd.to_datetime(raw["disconnectTime"], utc=True, errors="coerce").max()
    n_steps = max(1, int(np.ceil((end - start) / pd.Timedelta(minutes=timestep_minutes))))
    counts = np.zeros(n_steps, dtype=float)
    for _, row in raw.iterrows():
        connect = pd.to_datetime(row["connectionTime"], utc=True, errors="coerce")
        disconnect = pd.to_datetime(row["disconnectTime"], utc=True, errors="coerce")
        if pd.isna(connect) or pd.isna(disconnect) or disconnect <= connect:
            continue
        first = int((connect - start) / pd.Timedelta(minutes=timestep_minutes))
        last = min(n_steps, int(np.ceil((disconnect - start) / pd.Timedelta(minutes=timestep_minutes))))
        counts[first:last] += 1
    return counts


def dataset_characteristics(name: str, path: Path, data, timestep_minutes: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    connection = pd.to_datetime(raw["connectionTime"], utc=True, errors="coerce")
    disconnect = pd.to_datetime(raw["disconnectTime"], utc=True, errors="coerce")
    delivered = pd.to_numeric(raw["kWhDelivered"], errors="coerce")
    requested = pd.to_numeric(raw["requested_kWh"], errors="coerce")
    if data is None:
        connected = occupancy_counts(path, timestep_minutes)
        achievable = np.full_like(connected, np.nan)
    else:
        achievable, connected = greedy_series(data)
    demand_stats = {
        "mean_greedy_kw": float(achievable.mean()) if data is not None else np.nan,
        "p95_greedy_kw": float(np.quantile(achievable, .95)) if data is not None else np.nan,
        "risk_hour_fraction": float(np.mean(achievable > CAPACITY_KW)) if data is not None else np.nan,
        "headroom_median_kw": float(np.median(np.maximum(0, CAPACITY_KW - achievable)))
        if data is not None else np.nan,
    }
    return pd.DataFrame([{
        "dataset": name,
        "sessions": len(raw),
        "median_session_duration_hours": float(
            (disconnect - connection).dt.total_seconds().median() / 3600
        ),
        "p95_session_duration_hours": float(
            (disconnect - connection).dt.total_seconds().quantile(.95) / 3600
        ),
        "median_requested_kwh": float(requested.median()) if (requested > 0).any() else np.nan,
        "median_delivered_kwh": float(delivered.median()),
        "median_connected_evs": float(np.median(connected)),
        "p95_connected_evs": float(np.quantile(connected, .95)),
        **demand_stats,
        "requested_energy_supported": bool((requested > 0).all()),
    }])


def save_figures(
    scores: pd.DataFrame,
    characteristics: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for dataset, group in scores.groupby("dataset"):
        valid = group[group["criterion_status"].str.startswith("computed")]
        if not valid.empty:
            ax.plot(valid["horizon_hours"], valid["criterion_x"], marker="o", label=dataset)
        else:
            ax.plot([], [], label=f"{dataset} (unsupported)")
    ax.set(xlabel="Lead time (hours)", ylabel="X(l)", title="Lead-time criterion by dataset")
    ax.set_xticks(list(HORIZONS_HOURS))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "lead_time_score_vs_horizon.png", dpi=300)
    fig.savefig(figure_dir / "lead_time_score_vs_horizon.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for dataset, group in characteristics.groupby("dataset"):
        axes[0, 0].bar(dataset, group["median_session_duration_hours"].iloc[0])
        axes[0, 1].bar(dataset, group["median_connected_evs"].iloc[0])
        axes[1, 0].bar(dataset, group["p95_greedy_kw"].iloc[0])
        axes[1, 1].bar(dataset, group["risk_hour_fraction"].iloc[0])
    axes[0, 0].set_title("Median session duration (h)")
    axes[0, 1].set_title("Median connected EVs")
    axes[1, 0].set_title("P95 greedy achievable draw (kW)")
    axes[1, 1].set_title("Risk-hour fraction")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "dataset_characteristics.png", dpi=300)
    fig.savefig(figure_dir / "dataset_characteristics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for dataset, frame in distributions.items():
        axes[0, 0].hist(
            frame["arrival_hour"], bins=np.arange(25) - .5, alpha=.35, label=dataset
        )
        axes[0, 1].hist(
            frame["departure_hour"], bins=np.arange(25) - .5, alpha=.35, label=dataset
        )
        axes[1, 0].hist(
            frame["duration_hours"], bins=30, alpha=.35, label=dataset
        )
        energy_col = "requested_kwh" if frame["requested_kwh"].notna().any() else "delivered_kwh"
        axes[1, 1].hist(
            frame[energy_col].dropna(), bins=30, alpha=.35,
            label=f"{dataset} ({energy_col.replace('_kwh', '')})"
        )
    axes[0, 0].set_title("Arrival hour")
    axes[0, 1].set_title("Departure hour")
    axes[1, 0].set_title("Session duration (hours)")
    axes[1, 1].set_title("Energy distribution")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "arrival_departure_energy_distributions.png", dpi=300)
    fig.savefig(figure_dir / "arrival_departure_energy_distributions.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    labels = list(characteristics["dataset"])
    axes[0].bar(labels, characteristics["p95_greedy_kw"])
    axes[1].bar(labels, characteristics["headroom_median_kw"])
    axes[2].bar(labels, characteristics["risk_hour_fraction"])
    axes[0].set_title("P95 greedy draw (kW)")
    axes[1].set_title("Median headroom (kW)")
    axes[2].set_title("Risk-hour fraction")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "headroom_risk_diagnostics.png", dpi=300)
    fig.savefig(figure_dir / "headroom_risk_diagnostics.pdf")
    plt.close(fig)

    if not diagnostics.empty:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
        for dataset, group in diagnostics.groupby("dataset"):
            axes[0].plot(group["horizon_hours"], group["pct_A_positive"], marker="o", label=dataset)
            axes[1].plot(group["horizon_hours"], group["pct_S_positive"], marker="o", label=dataset)
            axes[2].plot(group["horizon_hours"], group["pct_both_positive"], marker="o", label=dataset)
        for ax, title in zip(
            axes,
            ("% A > 0", "% S > 0", "% A > 0 and S > 0"),
        ):
            ax.set_title(title)
            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel("Risk hours (%)")
            ax.set_ylim(-2, 102)
            ax.grid(alpha=.25)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "lead_time_condition_diagnostics.png", dpi=300)
        fig.savefig(figure_dir / "lead_time_condition_diagnostics.pdf")
        plt.close(fig)


def raw_distributions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    connect = pd.to_datetime(raw["connectionTime"], utc=True, errors="coerce")
    disconnect = pd.to_datetime(raw["disconnectTime"], utc=True, errors="coerce")
    requested = pd.to_numeric(raw["requested_kWh"], errors="coerce")
    delivered = pd.to_numeric(raw["kWhDelivered"], errors="coerce")
    return pd.DataFrame({
        "arrival_hour": connect.dt.hour,
        "departure_hour": disconnect.dt.hour,
        "duration_hours": (disconnect - connect).dt.total_seconds() / 3600,
        "requested_kwh": requested.where(requested > 0),
        "delivered_kwh": delivered,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-kw", type=float, default=CAPACITY_KW)
    parser.add_argument("--timestep-minutes", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir
    figure_dir = output_dir / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    verification = []
    events = []
    scores = []
    characteristics = []
    distributions = {}
    for name, path in DATASETS.items():
        verification.append(raw_verification(name, path))
        data = None if name == "Office001" else load_grid(path, args.timestep_minutes)
        if data is None:
            event_df, score_df = criterion_for_dataset(
                name,
                type("UnsupportedGrid", (), {"n_steps": 0})(),
                args.capacity_kw,
            )
        else:
            event_df, score_df = criterion_for_dataset(name, data, args.capacity_kw)
        if not event_df.empty:
            events.append(event_df)
        scores.append(score_df)
        characteristics.append(
            dataset_characteristics(name, path, data, args.timestep_minutes)
        )
        distributions[name] = raw_distributions(path)

    verification_df = pd.DataFrame(verification)
    scores_df = pd.concat(scores, ignore_index=True)
    characteristics_df = pd.concat(characteristics, ignore_index=True)
    events_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    diagnostics_df, persistence_df = statistical_diagnostics(events_df)
    verification_df.to_csv(output_dir / "dataset_verification.csv", index=False)
    scores_df.to_csv(output_dir / "lead_time_scores.csv", index=False)
    characteristics_df.to_csv(output_dir / "dataset_characteristics.csv", index=False)
    events_df.to_csv(output_dir / "lead_time_event_records.csv", index=False)
    diagnostics_df.to_csv(output_dir / "lead_time_statistical_diagnostics.csv", index=False)
    persistence_df.to_csv(output_dir / "risk_hour_zero_persistence.csv", index=False)
    scores_df.pivot(index="horizon_hours", columns="dataset", values="criterion_x").to_csv(
        output_dir / "comparison_table.csv"
    )
    summary = {
        "configuration": {
            "capacity_kw": args.capacity_kw,
            "per_ev_max_kw": PER_EV_MAX_KW,
            "timestep_minutes": args.timestep_minutes,
            "horizons_hours": list(HORIZONS_HOURS),
            "risk_definition": "greedy achievable draw > capacity",
            "criterion": "median_h[A(l,h) * I(S(l,h) > 0)]",
            "demand_definition": "sum(min(6.6, requested_kWh / timestep_hours))",
            "timestamp_mapping": "floor connection step; ceil disconnect step; active on [connect, disconnect)",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "office001_limitation": (
            "Office001 has zero requested_kWh coverage. Its primary lead-time "
            "criterion is intentionally not computed; kWhDelivered is not treated "
            "as requested demand."
        ),
    }
    with open(output_dir / "run_configuration.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    save_figures(scores_df, characteristics_df, distributions, diagnostics_df, figure_dir)
    print(verification_df.to_string(index=False))
    print(scores_df.to_string(index=False))
    print(f"Saved outputs and figures under {output_dir}")


if __name__ == "__main__":
    main()
