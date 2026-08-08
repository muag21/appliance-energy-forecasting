"""Tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from appliance_energy import evaluation


@pytest.fixture
def train() -> pd.Series:
    """Seasonal series with noise, long enough for a daily scale factor."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2016-01-01", periods=720, freq="h")
    signal = 100 + 40 * np.sin(2 * np.pi * np.arange(720) / 24)
    return pd.Series(signal + rng.normal(0, 5, 720), index=index)


@pytest.fixture
def test_series(train) -> pd.Series:
    index = pd.date_range(train.index[-1] + pd.Timedelta("1h"), periods=336, freq="h")
    signal = 100 + 40 * np.sin(2 * np.pi * np.arange(336) / 24)
    return pd.Series(signal, index=index)


# ---------------------------------------------------------------------------
# MASE
# ---------------------------------------------------------------------------

def test_mase_is_zero_for_a_perfect_forecast(train, test_series):
    assert evaluation.mase(test_series, test_series, train) == pytest.approx(0.0)


def test_mase_scale_is_the_in_sample_seasonal_naive_mae(train, test_series):
    """A forecast whose MAE equals the scale must score exactly one."""
    scale = np.abs(train.to_numpy()[24:] - train.to_numpy()[:-24]).mean()
    shifted = test_series + scale

    assert evaluation.mase(test_series, shifted, train) == pytest.approx(1.0)


def test_mase_is_invariant_to_rescaling_the_series(train, test_series):
    """MASE is scale free: multiplying every quantity leaves it unchanged."""
    pred = test_series + 7.0

    base = evaluation.mase(test_series, pred, train)
    scaled = evaluation.mase(test_series * 6, pred * 6, train * 6)

    assert base == pytest.approx(scaled)


def test_mase_rejects_a_series_shorter_than_the_season(test_series):
    short = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ValueError):
        evaluation.mase(test_series, test_series, short)


def test_mase_returns_nan_for_a_constant_training_series(test_series):
    constant = pd.Series(np.full(200, 50.0))
    assert np.isnan(evaluation.mase(test_series, test_series, constant))


# ---------------------------------------------------------------------------
# RMSE and MAE
# ---------------------------------------------------------------------------

def test_rmse_is_never_below_mae(train, test_series):
    rng = np.random.default_rng(1)
    pred = test_series + rng.normal(0, 20, len(test_series))

    row = evaluation.evaluate_forecast("noisy", test_series, pred, train)
    assert row["RMSE"] >= row["MAE"] - 1e-9


def test_rmse_penalises_a_single_large_error_more_than_mae(train, test_series):
    spread = test_series.copy()
    spread.iloc[:10] += 10.0

    spike = test_series.copy()
    spike.iloc[0] += 100.0

    a = evaluation.evaluate_forecast("spread", test_series, spread, train)
    b = evaluation.evaluate_forecast("spike", test_series, spike, train)

    assert a["MAE"] == pytest.approx(b["MAE"])
    assert b["RMSE"] > a["RMSE"]


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

def test_bias_is_positive_when_over_forecasting(train, test_series):
    assert evaluation.bias(test_series, test_series + 10.0) == pytest.approx(10.0)
    assert evaluation.bias(test_series, test_series - 10.0) == pytest.approx(-10.0)


def test_bias_cancels_for_symmetric_errors(train, test_series):
    pred = test_series.copy()
    pred.iloc[::2] += 10.0
    pred.iloc[1::2] -= 10.0

    assert evaluation.bias(test_series, pred) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_evaluate_forecast_rejects_a_short_forecast(train, test_series):
    """Forecast length must match the test period."""
    truncated = test_series.iloc[:-24]

    with pytest.raises(ValueError):
        evaluation.evaluate_forecast("short", test_series, truncated, train)


def test_evaluate_all_is_sorted_by_mase(train, test_series):
    rng = np.random.default_rng(2)
    forecasts = {
        "good": test_series + rng.normal(0, 2, len(test_series)),
        "poor": test_series + rng.normal(0, 60, len(test_series)),
        "perfect": test_series.copy(),
    }

    table = evaluation.evaluate_all(forecasts, test_series, train)

    assert list(table.columns) == ["model", "MAE", "RMSE", "MASE", "Bias"]
    assert table["MASE"].is_monotonic_increasing
    assert table["model"].iloc[0] == "perfect"


def test_errors_by_horizon_has_one_row_per_lead_time(train, test_series):
    forecasts = {"naive": test_series + 5.0}
    table = evaluation.errors_by_horizon(forecasts, test_series, horizon=24)

    assert len(table) == 24
    assert table.index.name == "lead_time"
    assert table["naive"].round(6).eq(5.0).all()
