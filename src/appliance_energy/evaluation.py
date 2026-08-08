"""Forecast evaluation metrics.

Sign convention: ``Bias`` is the mean of ``prediction - actual``, so a positive
value indicates systematic over-forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, compatible across scikit-learn versions.

    ``mean_squared_error(..., squared=False)`` was removed in scikit-learn 1.6,
    so the square root is taken explicitly.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mase(y_true, y_pred, y_train, seasonality: int = config.DAILY_PERIOD) -> float:
    """Mean absolute scaled error.

    The scale is the in-sample mean absolute error of the seasonal naive
    forecast on the training series, following Hyndman and Koehler (2006).
    A value below one indicates the forecast beats a seasonal naive rule
    applied in sample.
    """
    y_train = np.asarray(pd.Series(y_train).astype(float))

    if len(y_train) <= seasonality:
        raise ValueError("Training series is shorter than the seasonal period.")

    scale = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()

    if not np.isfinite(scale) or scale == 0:
        return float("nan")

    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def bias(y_true, y_pred) -> float:
    """Mean forecast error; positive indicates over-forecasting."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def evaluate_forecast(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = config.DAILY_PERIOD,
) -> dict:
    """Compute the four required metrics for a single forecast."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float).reindex(y_true.index)

    if y_pred.isna().any():
        raise ValueError(f"Forecast '{name}' has missing values on the test index.")

    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": bias(y_true, y_pred),
    }


def evaluate_all(
    forecasts: dict[str, pd.Series],
    y_true: pd.Series,
    y_train: pd.Series,
    seasonality: int = config.DAILY_PERIOD,
) -> pd.DataFrame:
    """Evaluate every forecast on a common test index, sorted by MASE."""
    rows = [
        evaluate_forecast(name, y_true, pred, y_train, seasonality)
        for name, pred in forecasts.items()
    ]
    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)


def errors_by_horizon(
    forecasts: dict[str, pd.Series],
    y_true: pd.Series,
    horizon: int = config.HORIZON,
) -> pd.DataFrame:
    """Mean absolute error as a function of lead time.

    Under the rolling-origin protocol this exposes how quickly each model
    degrades as the forecast moves away from the origin, which the aggregate
    metrics conceal.
    """
    lead = pd.Series(
        [(i % horizon) + 1 for i in range(len(y_true))], index=y_true.index
    )

    out = {}
    for name, pred in forecasts.items():
        abs_err = (pd.Series(pred).reindex(y_true.index) - y_true).abs()
        out[name] = abs_err.groupby(lead).mean()

    result = pd.DataFrame(out)
    result.index.name = "lead_time"
    return result
