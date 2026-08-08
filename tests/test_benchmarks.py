"""Tests for the benchmark forecasters and the rolling-origin driver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from appliance_energy.models import benchmarks


@pytest.fixture
def series() -> pd.Series:
    index = pd.date_range("2016-01-01", periods=1000, freq="h")
    values = 100 + 40 * np.sin(2 * np.pi * np.arange(1000) / 24)
    return pd.Series(values, index=index)


@pytest.fixture
def split(series):
    return series.iloc[:-336], series.iloc[-336:]


def test_mean_forecast_is_flat_at_the_history_mean(split):
    train, _ = split
    out = benchmarks.mean_forecast(train, 24)

    assert len(out) == 24
    assert np.allclose(out, train.mean())


def test_naive_forecast_repeats_the_last_observation(split):
    train, _ = split
    out = benchmarks.naive_forecast(train, 24)
    assert np.allclose(out, train.iloc[-1])


def test_seasonal_naive_repeats_the_last_cycle(split):
    train, _ = split
    out = benchmarks.seasonal_naive_forecast(train, 24, seasonality=24)
    assert np.allclose(out, train.iloc[-24:].to_numpy())


def test_seasonal_naive_recycles_beyond_one_period(split):
    """Past one season the forecast repeats the same cycle again."""
    train, _ = split
    out = benchmarks.seasonal_naive_forecast(train, 72, seasonality=24)

    assert np.allclose(out[:24], out[24:48])
    assert np.allclose(out[:24], out[48:72])


def test_seasonal_naive_rejects_a_short_history():
    short = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ValueError):
        benchmarks.seasonal_naive_forecast(short, 5, seasonality=24)


def test_drift_extrapolates_a_linear_series():
    index = pd.date_range("2016-01-01", periods=100, freq="h")
    linear = pd.Series(np.arange(100, dtype=float) * 3.0, index=index)

    out = benchmarks.drift_forecast(linear, 5)
    expected = np.array([297.0 + 3.0 * s for s in range(1, 6)])

    assert np.allclose(out, expected)


# ---------------------------------------------------------------------------
# Rolling-origin driver
# ---------------------------------------------------------------------------

def test_rolling_origin_covers_the_test_index_exactly(series, split):
    train, test = split
    out = benchmarks.rolling_origin_forecast(
        series, test.index, 24, benchmarks.naive_forecast
    )

    assert len(out) == len(test)
    assert out.index.equals(test.index)
    assert not out.isna().any()


def test_rolling_origin_never_sees_the_current_block(series, split):
    """Perturbing a block's actuals must not change that block's forecast."""
    train, test = split

    base = benchmarks.rolling_origin_forecast(
        series, test.index, 24, benchmarks.naive_forecast
    )

    tampered = series.copy()
    tampered.loc[test.index[:24]] += 5_000.0      # first block only

    alt = benchmarks.rolling_origin_forecast(
        tampered, test.index, 24, benchmarks.naive_forecast
    )

    assert np.allclose(base.iloc[:24], alt.iloc[:24]), "Block saw its own actuals."
    assert not np.allclose(base.iloc[24:48], alt.iloc[24:48]), (
        "Later blocks should react to released observations."
    )


def test_rolling_origin_naive_is_constant_within_each_block(series, split):
    train, test = split
    out = benchmarks.rolling_origin_forecast(
        series, test.index, 24, benchmarks.naive_forecast
    )

    for start in range(0, len(out), 24):
        block = out.iloc[start : start + 24]
        assert block.nunique() == 1


def test_rolling_origin_handles_a_ragged_final_block(series):
    """A test window that is not a whole multiple of the horizon still works."""
    test_index = series.index[-50:]
    out = benchmarks.rolling_origin_forecast(
        series, test_index, 24, benchmarks.naive_forecast
    )

    assert len(out) == 50
    assert out.index.equals(test_index)


def test_benchmark_suite_exposes_the_five_required_models():
    suite = benchmarks.benchmark_suite()
    assert set(suite) == {
        "mean",
        "naive",
        "seasonal_naive_daily",
        "seasonal_naive_weekly",
        "drift",
    }


def test_seasonal_naive_daily_is_exact_on_a_pure_daily_cycle(series, split):
    """On a noiseless 24-hour cycle the daily seasonal naive is error free."""
    train, test = split
    out = benchmarks.rolling_origin_forecast(
        series, test.index, 24, benchmarks.benchmark_suite()["seasonal_naive_daily"]
    )

    assert np.allclose(out.to_numpy(), test.to_numpy(), atol=1e-9)
