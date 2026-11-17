from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpc_optimizer import ForecastLookupError, StepwiseForecastLookup, mpc_action_for_env, solve_mpc_action


def test_single_ev_respects_power_and_remaining_energy() -> None:
    result = solve_mpc_action(
        present=np.array([True]),
        remaining_kwh=np.array([10.0]),
        departure_step=np.array([5]),
        step_idx=0,
        dt_hours=1.0,
        per_ev_max_kw=6.6,
        capacity_kw=50.0,
        future_demand_kwh=np.zeros(5),
        horizon_hours=6,
    )
    assert result.success
    assert 0.0 <= result.action_kw[0] <= 6.6 + 1e-9
    assert result.action_kw.sum() <= 50.0


def test_capacity_is_enforced() -> None:
    result = solve_mpc_action(
        present=np.ones(6, dtype=bool),
        remaining_kwh=np.full(6, 100.0),
        departure_step=np.full(6, 5),
        step_idx=0,
        dt_hours=1.0,
        per_ev_max_kw=6.6,
        capacity_kw=20.0,
        future_demand_kwh=np.zeros(5),
        horizon_hours=6,
    )
    assert result.success
    assert result.action_kw.sum() <= 20.0 - 1e-4 + 1e-6


def test_nearest_deadline_is_prioritized() -> None:
    result = solve_mpc_action(
        present=np.array([True, True]),
        remaining_kwh=np.array([10.0, 10.0]),
        departure_step=np.array([0, 5]),
        step_idx=0,
        dt_hours=1.0,
        per_ev_max_kw=6.6,
        capacity_kw=6.6,
        future_demand_kwh=np.zeros(5),
        horizon_hours=6,
    )
    assert result.success
    assert result.action_kw[0] > result.action_kw[1]
    assert result.action_kw.sum() <= 6.6 + 1e-6


def test_future_reserve_is_capped_at_capacity() -> None:
    result = solve_mpc_action(
        present=np.array([True]),
        remaining_kwh=np.array([30.0]),
        departure_step=np.array([5]),
        step_idx=0,
        dt_hours=1.0,
        per_ev_max_kw=6.6,
        capacity_kw=50.0,
        future_demand_kwh=np.array([70.0, 50.0, 0.0, 0.0, 0.0]),
        horizon_hours=6,
    )
    assert result.success
    assert result.reserve_kw[0] == 0.0
    assert np.all(result.reserve_kw <= 50.0 - 1e-4 + 1e-9)
    assert result.action_kw.sum() <= 50.0


def test_horizon_is_clipped_to_episode_end() -> None:
    result = solve_mpc_action(
        present=np.array([True]),
        remaining_kwh=np.array([10.0]),
        departure_step=np.array([10]),
        step_idx=0,
        dt_hours=1.0,
        per_ev_max_kw=6.6,
        capacity_kw=50.0,
        future_demand_kwh=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        horizon_hours=6,
        episode_len=1,
    )
    assert result.success
    assert result.horizon_steps == 1


def test_environment_adapter_returns_first_action() -> None:
    class Inner:
        _step_idx = 0
        episode_len = 168
        dt_hours = 1.0
        per_ev_max_kw = 6.6
        _departure_step = np.array([5])

        def current_time(self):
            return pd.Timestamp("2024-01-01T00:00:00Z")

        def true_presence(self):
            return np.array([True])

        def true_remaining_kwh(self):
            return np.array([10.0])

    class Env:
        episode_len = 168
        inner = Inner()

    class Forecast:
        def __call__(self, timestamp):
            assert timestamp == pd.Timestamp("2024-01-01T00:00:00Z")
            return np.zeros(6)

    result = mpc_action_for_env(Env(), Forecast(), horizon_hours=6, capacity_kw=50.0)
    assert result.success
    assert result.horizon_steps == 6
    assert 0.0 <= result.action_kw[0] <= 6.6 + 1e-6


def test_inactive_ev_gets_zero_action() -> None:
    result = solve_mpc_action(
        present=np.array([False, True]),
        remaining_kwh=np.array([10.0, 10.0]),
        departure_step=np.array([5, 5]),
        step_idx=0,
        dt_hours=1.0,
        per_ev_max_kw=6.6,
        capacity_kw=50.0,
        future_demand_kwh=np.zeros(5),
        horizon_hours=6,
    )
    assert result.success
    assert result.action_kw[0] == 0.0
    assert result.action_kw[1] > 0.0


def test_energy_beyond_request_is_not_served() -> None:
    result = solve_mpc_action(
        present=np.array([True]),
        remaining_kwh=np.array([1.0]),
        departure_step=np.array([5]),
        step_idx=0,
        dt_hours=1.0,
        per_ev_max_kw=6.6,
        capacity_kw=50.0,
        future_demand_kwh=np.zeros(5),
        horizon_hours=6,
    )
    assert result.success
    assert result.action_kw[0] <= 1.0 + 1e-6


def test_forecast_lookup_clips_negative_values_and_requires_exact_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    frame = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
        "predicted_h1": [-1.0, 1.0],
        "predicted_h2": [2.0, 2.0],
        "predicted_h3": [3.0, 3.0],
        "predicted_h4": [4.0, 4.0],
        "predicted_h5": [5.0, 5.0],
        "predicted_h6": [6.0, 6.0],
    })
    frame.to_csv(path, index=False)
    lookup = StepwiseForecastLookup(path, horizon=6)
    values = lookup(pd.Timestamp("2024-01-01T00:00:00Z"))
    assert values[0] == 0.0
    assert values[1] == 2.0
    with pytest.raises(ForecastLookupError):
        lookup(pd.Timestamp("2024-01-01T02:00:00Z"))


def test_forecast_lookup_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    columns = [f"predicted_h{i}" for i in range(1, 7)]
    frame = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        **{column: [1.0, 2.0] for column in columns},
    })
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        StepwiseForecastLookup(path, horizon=6)


def test_generated_artifacts_cover_required_test_weeks() -> None:
    oracle = ROOT / "revision_setup" / "outputs" / "mpc" / "oracle_stepwise_h6.csv"
    bilstm = ROOT / "revision_setup" / "forecast_stepwise_oof.csv"
    if not oracle.exists() or not bilstm.exists():
        pytest.skip("stepwise forecast artifacts have not been generated yet")
    starts = [7429, 7713, 7997, 8281, 8565]
    required = set()
    from_acn_start = pd.Timestamp("2019-01-01 17:41:45+00:00")
    for start in starts:
        for offset in range(168):
            required.add(from_acn_start + pd.Timedelta(hours=start + offset))
    for path in (oracle, bilstm):
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        assert frame["timestamp"].is_unique
        assert frame["timestamp"].is_monotonic_increasing
        return_cols = [f"predicted_h{i}" for i in range(1, 7)]
        assert set(return_cols).issubset(frame.columns)
        values = frame.loc[frame["timestamp"].isin(required), return_cols].to_numpy(dtype=float)
        assert len(values) == 840
        assert np.isfinite(values).all()


def test_bilstm_artifact_uses_only_past_inputs() -> None:
    path = ROOT / "revision_setup" / "forecast_stepwise_oof.csv"
    if not path.exists():
        pytest.skip("stepwise forecast artifact has not been generated yet")
    frame = pd.read_csv(path)
    assert {"hour_index", "input_last_index", "horizon_hours", "timestep_minutes"}.issubset(frame.columns)
    assert (frame["input_last_index"] < frame["hour_index"]).all()
    assert set(frame["horizon_hours"].unique()) == {6}
    assert set(frame["timestep_minutes"].unique()) == {60}


def test_oracle_artifact_is_aligned_and_uses_actual_future() -> None:
    path = ROOT / "revision_setup" / "outputs" / "mpc" / "oracle_stepwise_h6.csv"
    if not path.exists():
        pytest.skip("oracle stepwise artifact has not been generated yet")
    frame = pd.read_csv(path)
    for index in range(1, 7):
        assert np.allclose(frame[f"actual_h{index}"], frame[f"predicted_h{index}"])
    assert (frame["input_last_index"] == frame["hour_index"]).all()
