"""Tests for feature construction, focused on leakage guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from appliance_energy import features


@pytest.fixture
def series() -> pd.Series:
    """Strictly increasing series: each value identifies its own position."""
    index = pd.date_range("2016-01-01", periods=1000, freq="h")
    return pd.Series(np.arange(1000, dtype=float), index=index, name="Appliances")


@pytest.fixture
def exog(series) -> pd.DataFrame:
    return pd.DataFrame(
        {"T_out": np.arange(len(series), dtype=float) * 10.0}, index=series.index
    )


# ---------------------------------------------------------------------------
# Lag anchoring
# ---------------------------------------------------------------------------

def test_lag_features_are_anchored_to_the_origin(series):
    """lag_k at (t, h) must equal y[t - h + 1 - k], not y[t - k]."""
    frame = features.build_supervised_frame(series, max_horizon=24)

    positions = {ts: i for i, ts in enumerate(series.index)}

    for h in (1, 6, 24):
        rows = frame[frame["horizon"] == h]
        sample = rows.iloc[[0, len(rows) // 2, -1]]

        for ts, row in sample.iterrows():
            t = positions[ts]
            for k in (1, 24, 168):
                assert row[f"lag_{k}"] == pytest.approx(series.iloc[t - h + 1 - k])


def test_no_feature_references_the_forecast_window(series):
    """Every target-derived feature must come from strictly before the origin.

    Perturbing the target from a cutoff onwards must leave untouched every
    feature belonging to a row whose origin lies at or before that cutoff.
    """
    cutoff = 800
    perturbed = series.copy()
    perturbed.iloc[cutoff:] += 10_000.0

    base = features.build_supervised_frame(series, max_horizon=24)
    alt = features.build_supervised_frame(perturbed, max_horizon=24)

    positions = {ts: i for i, ts in enumerate(series.index)}
    target_pos = np.array([positions[ts] for ts in base.index])
    origin_pos = target_pos - base["horizon"].to_numpy() + 1

    # Rows whose entire history predates the perturbation.
    safe = origin_pos <= cutoff
    feature_cols = [c for c in features.feature_columns(base) if c != "horizon"]

    left = base.loc[safe, feature_cols].to_numpy()
    right = alt.loc[safe, feature_cols].to_numpy()

    assert np.allclose(left, right), "A feature responded to a future target value."


def test_rolling_window_ends_before_the_origin(series):
    """roll_mean_w at (t, h) averages y over [t-h-w+1, t-h]."""
    frame = features.build_supervised_frame(
        series, max_horizon=24, roll_windows=(3, 24)
    )
    positions = {ts: i for i, ts in enumerate(series.index)}

    for h in (1, 12, 24):
        row = frame[frame["horizon"] == h].iloc[-1]
        t = positions[row.name]

        for w in (3, 24):
            expected = series.iloc[t - h - w + 1 : t - h + 1].mean()
            assert row[f"roll_mean_{w}"] == pytest.approx(expected)


def test_rolling_std_is_finite_and_nonnegative(series):
    rng = np.random.default_rng(0)
    noisy = series + rng.normal(0, 5, len(series))

    frame = features.build_supervised_frame(noisy, max_horizon=6)
    std_cols = [c for c in frame.columns if c.startswith("roll_std_")]

    assert (frame[std_cols] >= 0).all().all()
    assert np.isfinite(frame[std_cols].to_numpy()).all()


# ---------------------------------------------------------------------------
# Exogenous handling
# ---------------------------------------------------------------------------

def test_origin_exog_is_lagged_but_future_exog_is_not(series, exog):
    """Origin covariates come from before the origin; future ones are realised."""
    frame = features.build_supervised_frame(
        series, max_horizon=24, exog_origin=exog, exog_future=exog
    )
    positions = {ts: i for i, ts in enumerate(series.index)}

    for h in (1, 24):
        row = frame[frame["horizon"] == h].iloc[-1]
        t = positions[row.name]

        assert row["T_out_origin"] == pytest.approx(exog["T_out"].iloc[t - h])
        assert row["T_out_future"] == pytest.approx(exog["T_out"].iloc[t])


def test_origin_exog_ignores_the_forecast_window(series, exog):
    cutoff = 900
    perturbed = exog.copy()
    perturbed.iloc[cutoff:] += 1e6

    base = features.build_supervised_frame(series, max_horizon=24, exog_origin=exog)
    alt = features.build_supervised_frame(
        series, max_horizon=24, exog_origin=perturbed
    )

    positions = {ts: i for i, ts in enumerate(series.index)}
    origin_pos = np.array([positions[ts] for ts in base.index]) - base["horizon"].to_numpy() + 1
    safe = origin_pos <= cutoff

    assert np.allclose(
        base.loc[safe, "T_out_origin"], alt.loc[safe, "T_out_origin"]
    )


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def test_calendar_features_match_the_timestamp():
    index = pd.date_range("2016-01-02", periods=48, freq="h")  # Saturday
    cal = features.add_calendar_features(index)

    assert (cal["hour"].to_numpy() == index.hour).all()
    assert cal["is_weekend"].iloc[0] == 1                 # Saturday
    assert cal["is_weekend"].iloc[24] == 1                # Sunday
    assert cal["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_cyclic_encoding_lies_on_the_unit_circle():
    index = pd.date_range("2016-01-01", periods=200, freq="h")
    cal = features.add_calendar_features(index)

    radius = cal["hour_sin"] ** 2 + cal["hour_cos"] ** 2
    assert np.allclose(radius, 1.0)


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------

def test_supervised_frame_has_no_missing_values(series, exog):
    frame = features.build_supervised_frame(
        series, max_horizon=24, exog_origin=exog
    )
    assert not frame.isna().any().any()
    assert len(frame) > 0


def test_every_horizon_is_represented(series):
    frame = features.build_supervised_frame(series, max_horizon=24)
    assert sorted(frame["horizon"].unique()) == list(range(1, 25))


def test_target_column_matches_the_series(series):
    frame = features.build_supervised_frame(series, max_horizon=6)
    assert np.allclose(frame["target"].to_numpy(), series.loc[frame.index].to_numpy())


# ---------------------------------------------------------------------------
# Rolling-origin row selection
# ---------------------------------------------------------------------------

def test_rolling_origin_pairs_cycle_through_the_horizon():
    index = pd.date_range("2016-01-01", periods=72, freq="h")
    pairs = features.rolling_origin_pairs(index, horizon=24)

    assert pairs["horizon"].tolist()[:3] == [1, 2, 3]
    assert pairs["horizon"].iloc[23] == 24
    assert pairs["horizon"].iloc[24] == 1          # second origin restarts
    assert len(pairs) == len(index)


def test_select_rows_returns_one_row_per_test_timestamp(series):
    frame = features.build_supervised_frame(series, max_horizon=24)
    test_index = series.index[-48:]

    pairs = features.rolling_origin_pairs(test_index, horizon=24)
    rows = features.select_rows(frame, pairs)

    assert len(rows) == len(test_index)
    assert rows.index.equals(test_index)
    assert rows["horizon"].tolist() == pairs["horizon"].tolist()
