"""Stationarity diagnostics.

ADF and KPSS are reported together because they test complementary nulls:
ADF's null is a unit root, KPSS's null is stationarity.  Agreement across the
two gives a defensible basis for the differencing orders in SARIMAX; the
awkward cases are precisely those where they disagree, and those should be
reported rather than resolved by habit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series: pd.Series, regression: str = "c") -> dict:
    stat, pvalue, lags, nobs, crit, _ = adfuller(
        series.dropna(), regression=regression, autolag="AIC"
    )
    return {
        "test": "ADF",
        "null": "unit root present",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "lags_used": int(lags),
        "crit_5pct": float(crit["5%"]),
        "reject_null_5pct": bool(pvalue < 0.05),
    }


def kpss_test(series: pd.Series, regression: str = "c", nlags="auto") -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # p-value is interpolated at the bounds
        stat, pvalue, lags, crit = kpss(
            series.dropna(), regression=regression, nlags=nlags
        )
    return {
        "test": "KPSS",
        "null": "series is stationary",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "lags_used": int(lags),
        "crit_5pct": float(crit["5%"]),
        "reject_null_5pct": bool(pvalue < 0.05),
    }


def seasonal_strength(series: pd.Series, period: int = 24) -> float:
    """Strength of seasonality on the Wang, Smith and Hyndman scale.

    Returns ``max(0, 1 - Var(remainder) / Var(seasonal + remainder))``.
    Values near one indicate strong seasonality; near zero, none.
    """
    from statsmodels.tsa.seasonal import STL

    stl = STL(series.dropna(), period=period, robust=True).fit()
    denom = np.var(stl.seasonal + stl.resid)

    if denom == 0:
        return 0.0

    return float(max(0.0, 1.0 - np.var(stl.resid) / denom))


def stationarity_report(
    series: pd.Series, period: int = 24, seasonal_period: int = 168
) -> pd.DataFrame:
    """Run the diagnostics on the level, first-differenced and seasonally
    differenced series so that the choice of ``d`` and ``D`` is evidence based.
    """
    variants = {
        "level": series,
        "first_difference": series.diff(),
        f"seasonal_difference_{period}": series.diff(period),
        f"both_differences_{period}": series.diff(period).diff(),
    }

    rows = []
    for name, values in variants.items():
        for fn in (adf_test, kpss_test):
            row = fn(values)
            row["series"] = name
            rows.append(row)

    report = pd.DataFrame(rows)
    return report[
        ["series", "test", "null", "statistic", "p_value",
         "crit_5pct", "lags_used", "reject_null_5pct"]
    ]
