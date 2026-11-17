import numpy as np
import pandas as pd

from cross_dataset_lead_time import (
    criterion_for_dataset,
    greedy_series,
    statistical_diagnostics,
)


class Session:
    def __init__(self, session_id, requested_kwh):
        self.session_id = session_id
        self.requested_kwh = requested_kwh


class Data:
    timestep_minutes = 60
    n_steps = 3
    start_time = pd.Timestamp("2019-01-01T00:00:00Z")

    def __init__(self):
        self.sessions = [Session(0, 10.0), Session(1, 10.0)]
        self.active_by_step = [[0], [0, 1], [1]]


def test_greedy_series_uses_per_ev_rate_and_presence():
    achievable, connected = greedy_series(Data())
    np.testing.assert_allclose(achievable, [6.6, 13.2, 6.6])
    np.testing.assert_allclose(connected, [1, 2, 1])


def test_office001_is_not_scored_without_requested_energy():
    _, scores = criterion_for_dataset("Office001", Data(), horizons=(1, 2))
    assert scores["criterion_x"].isna().all()
    assert set(scores["criterion_status"]) == {"unsupported_no_requested_energy"}


def test_horizon_is_converted_from_hours_to_grid_steps():
    data = Data()
    data.timestep_minutes = 30
    data.n_steps = 5
    data.active_by_step = [[0], [0], [0, 1], [0, 1], [1]]
    events, scores = criterion_for_dataset("Caltech", data, capacity_kw=10, horizons=(1,))
    assert scores.iloc[0]["criterion_status"] == "computed_complete_case"
    assert set(events["horizon_steps"]) == {2}


def test_statistical_diagnostics_preserves_capacity_groups():
    events = pd.DataFrame([
        {
            "dataset": "Caltech",
            "capacity_kw": 50.0,
            "risk_step": 1,
            "risk_timestamp_utc": "2019-01-01T01:00:00Z",
            "horizon_hours": 1,
            "connected_fraction_A": 1.0,
            "headroom_positive": True,
            "criterion_product": 1.0,
        },
        {
            "dataset": "Caltech",
            "capacity_kw": 100.0,
            "risk_step": 1,
            "risk_timestamp_utc": "2019-01-01T01:00:00Z",
            "horizon_hours": 1,
            "connected_fraction_A": 0.0,
            "headroom_positive": True,
            "criterion_product": 0.0,
        },
    ])
    diagnostics, _ = statistical_diagnostics(events)
    assert set(diagnostics["capacity_kw"]) == {50.0, 100.0}
