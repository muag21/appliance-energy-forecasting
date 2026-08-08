"""Feature construction for direct multi-horizon forecasting.

Design note
-----------
Every row of the supervised table corresponds to a ``(target timestamp, horizon)``
pair.  For a row with target timestamp ``t`` and horizon ``h`` the forecast
origin is ``o = t - h + 1``, meaning the most recent observable target value is
``y[o - 1] = y[t - h]``.

All target-derived features are therefore anchored to the *origin*, not to the
target timestamp:

    lag_k          = y[o - k]     = y.shift(h - 1 + k)
    roll_mean_w    = mean of y over [o - w, o - 1]  = y.shift(h).rolling(w).mean()

This makes leakage structurally impossible: no feature can reference the target
inside the forecast window, whatever horizon is requested.  Contrast this with
target-anchored lags (``y.shift(1)``), which silently convert a 24-hour-ahead
task into a one-step-ahead task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Calendar features
# --------------------------------------------------------------------------

def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features, known arbitrarily far in advance."""
    out = pd.DataFrame(index=index)

    hour = index.hour
    dow = index.dayofweek

    out["hour"] = hour
    out["dayofweek"] = dow
    out["is_weekend"] = (dow >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return out


# --------------------------------------------------------------------------
# Supervised table
# --------------------------------------------------------------------------

def build_supervised_frame(
    y: pd.Series,
    max_horizon: int = config.HORIZON,
    lag_offsets=config.LAG_OFFSETS,
    roll_windows=config.ROLL_WINDOWS,
    exog_origin: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
    horizons=None,
) -> pd.DataFrame:
    """Build the long ``(timestamp, horizon)`` supervised table.

    Parameters
    ----------
    y
        Target series on a regular hourly index.
    exog_origin
        Covariates observed at the origin.  Lagged by ``h`` so that only values
        available before the forecast is issued enter the row.  Use this for
        indoor sensors and for any weather channel treated as unknown.
    exog_future
        Covariates whose realised future values are supplied unshifted.  Any
        column passed here makes the resulting forecast CONDITIONAL, and the
        report must say so.
    horizons
        Explicit iterable of horizons.  Defaults to ``1 .. max_horizon``.

    Returns
    -------
    DataFrame indexed by target timestamp, with a ``horizon`` column, a
    ``target`` column and one column per feature.  Rows with incomplete
    history are dropped.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DatetimeIndex")

    horizons = range(1, max_horizon + 1) if horizons is None else horizons
    calendar = add_calendar_features(y.index)

    blocks = []

    for h in horizons:
        cols: dict[str, pd.Series] = {}

        # Target lags, measured backwards from the origin.
        for k in lag_offsets:
            cols[f"lag_{k}"] = y.shift(h - 1 + k)

        # Rolling statistics over the window ending at the last observable point.
        base = y.shift(h)
        for w in roll_windows:
            cols[f"roll_mean_{w}"] = base.rolling(w).mean()
            cols[f"roll_std_{w}"] = base.rolling(w).std()

        block = pd.DataFrame(cols, index=y.index)

        if exog_origin is not None and len(exog_origin.columns):
            lagged = exog_origin.shift(h).add_suffix("_origin")
            block = block.join(lagged)

        if exog_future is not None and len(exog_future.columns):
            block = block.join(exog_future.add_suffix("_future"))

        block = block.join(calendar)
        block["horizon"] = h
        block["target"] = y

        blocks.append(block)

    frame = pd.concat(blocks).sort_index(kind="stable")

    return frame.dropna()


def select_rows(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Select the ``(timestamp, horizon)`` rows listed in ``pairs``.

    ``pairs`` must have a DatetimeIndex and a ``horizon`` column.  Rows are
    returned in the order given by ``pairs``.
    """
    keyed = frame.set_index("horizon", append=True)
    wanted = pd.MultiIndex.from_arrays(
        [pairs.index, pairs["horizon"].to_numpy()], names=keyed.index.names
    )
    return keyed.loc[wanted].reset_index(level="horizon")


def rolling_origin_pairs(
    test_index: pd.DatetimeIndex, horizon: int = config.HORIZON
) -> pd.DataFrame:
    """Map each test timestamp to its horizon under the rolling-origin protocol.

    With ``horizon=24`` the first test day is forecast at horizons 1..24 from
    the first origin, the second day at horizons 1..24 from the second origin,
    and so on.
    """
    horizons = [(i % horizon) + 1 for i in range(len(test_index))]
    return pd.DataFrame({"horizon": horizons}, index=test_index)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature column names, excluding the target."""
    return [c for c in frame.columns if c != "target"]
