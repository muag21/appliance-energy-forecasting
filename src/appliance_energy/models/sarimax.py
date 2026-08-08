"""SARIMAX estimation and rolling-origin forecasting.

Parameters are estimated once on the training sample.  At each subsequent
origin the state filter is extended with the newly released observations via
``MLEResults.append(..., refit=False)``.  This is a fixed-parameter
rolling-origin evaluation: the Kalman filter conditions on all data up to the
origin, but the estimated coefficients are held at their training values.
Re-estimating fourteen times with a seasonal period of 24 costs a great deal
of compute for a negligible change in coefficients over a two-week window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .. import config


def fit_sarimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None = None,
    order=config.SARIMAX_ORDER,
    seasonal_order=config.SARIMAX_SEASONAL_ORDER,
    trend: str = config.SARIMAX_TREND,
    maxiter: int = config.SARIMAX_MAXITER,
):
    """Estimate SARIMAX on the training sample.

    ``maxiter`` defaults to 500 rather than statsmodels' 50. The near-unit-root
    seasonal AR term produces a flat likelihood surface on which the default
    iteration budget terminates prematurely; 500 iterations converge reliably.

    Convergence warnings are captured rather than silenced so that they can be
    reported honestly in the write-up.
    """
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=False, maxiter=maxiter)

    for w in caught:
        print(f"  [SARIMAX] {w.category.__name__}: {str(w.message)[:120]}")

    return result


def rolling_origin_sarimax(
    result,
    y: pd.Series,
    test_index: pd.DatetimeIndex,
    horizon: int = config.HORIZON,
    exog: pd.DataFrame | None = None,
) -> pd.Series:
    """Forecast the test period from successive origins, extending the state.

    ``exog`` supplies the covariate path over the forecast window.  Passing the
    realised test-set weather produces a conditional forecast; passing a proxy
    such as persisted or climatological weather produces an operational one.
    """
    pieces = []
    state = result

    for start in range(0, len(test_index), horizon):
        block = test_index[start : start + horizon]

        exog_block = None if exog is None else exog.loc[block]

        forecast = state.get_forecast(steps=len(block), exog=exog_block)
        mean = pd.Series(np.asarray(forecast.predicted_mean), index=block)
        pieces.append(mean)

        # Release the block's true observations and extend the filter.
        new_endog = y.loc[block]
        new_exog = None if exog is None else exog.loc[block]
        state = state.append(new_endog, exog=new_exog, refit=False)

    return pd.concat(pieces)


def persisted_exog(
    exog: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    horizon: int = config.HORIZON,
) -> pd.DataFrame:
    """Operational covariate proxy: the last observed value before each origin.

    Weather forecasts are not part of this dataset, so persistence stands in
    for whatever the operator would actually know at the origin.  Comparing a
    model driven by this proxy against the same model driven by realised
    weather isolates the cost of covariate uncertainty.
    """
    out = pd.DataFrame(index=test_index, columns=exog.columns, dtype=float)

    for start in range(0, len(test_index), horizon):
        block = test_index[start : start + horizon]
        last = exog.loc[exog.index < block[0]].iloc[-1]
        out.loc[block] = last.to_numpy()

    return out
