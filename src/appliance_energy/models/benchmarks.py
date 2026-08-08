"""Simple benchmark forecasters and the rolling-origin evaluation driver.

Every forecaster shares the signature ``f(history, horizon) -> np.ndarray``,
where ``history`` contains all observations strictly before the forecast
origin.  This uniform interface is what lets the rolling-origin driver give
each model exactly the same information set.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


Forecaster = Callable[[pd.Series, int], np.ndarray]


def mean_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(history.mean()), horizon)


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(history.iloc[-1]), horizon)


def seasonal_naive_forecast(
    history: pd.Series, horizon: int, seasonality: int
) -> np.ndarray:
    """Recursive seasonal naive: repeat the last complete seasonal cycle."""
    if len(history) < seasonality:
        raise ValueError("History shorter than the seasonal period.")

    values = list(history.to_numpy(dtype=float))
    out = []

    for _ in range(horizon):
        out.append(values[-seasonality])
        values.append(out[-1])

    return np.asarray(out)


def drift_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Extrapolate the straight line through the first and last observation."""
    if len(history) < 2:
        raise ValueError("Drift requires at least two observations.")

    slope = (history.iloc[-1] - history.iloc[0]) / (len(history) - 1)
    steps = np.arange(1, horizon + 1)

    return float(history.iloc[-1]) + slope * steps


def benchmark_suite(daily: int = 24, weekly: int = 168) -> dict[str, Forecaster]:
    """The five required benchmarks, ready for the rolling-origin driver."""
    return {
        "mean": mean_forecast,
        "naive": naive_forecast,
        "seasonal_naive_daily": lambda h, n: seasonal_naive_forecast(h, n, daily),
        "seasonal_naive_weekly": lambda h, n: seasonal_naive_forecast(h, n, weekly),
        "drift": drift_forecast,
    }


# --------------------------------------------------------------------------
# Rolling-origin driver
# --------------------------------------------------------------------------

def origin_blocks(test_index: pd.DatetimeIndex, horizon: int):
    """Yield the consecutive forecast blocks of the rolling-origin protocol."""
    for start in range(0, len(test_index), horizon):
        yield test_index[start : start + horizon]


def rolling_origin_forecast(
    y: pd.Series,
    test_index: pd.DatetimeIndex,
    horizon: int,
    forecaster: Forecaster,
) -> pd.Series:
    """Run ``forecaster`` from each origin and concatenate the blocks.

    At each origin the forecaster receives every observation strictly before
    the first timestamp of the block, including test observations released by
    earlier blocks.  This mirrors operational use, where yesterday's readings
    are available when today's forecast is issued.
    """
    pieces = []

    for block in origin_blocks(test_index, horizon):
        history = y.loc[y.index < block[0]]
        values = np.asarray(forecaster(history, len(block)), dtype=float)

        if len(values) != len(block):
            raise ValueError("Forecaster returned the wrong number of values.")

        pieces.append(pd.Series(values, index=block))

    return pd.concat(pieces)
