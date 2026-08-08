"""Figures for the report.

The comparison figure deliberately shows a short window rather than the whole
fortnight: nine overlaid series across 336 hours is unreadable, and the faceted
panel carries the same information legibly.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def plot_forecast_window(
    forecast_df: pd.DataFrame,
    window_hours: int = 72,
    horizon: int = config.HORIZON,
):
    """Overlay the first few days of the test period with origin markers."""
    view = forecast_df.iloc[:window_hours]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.plot(view.index, view["actual"], color="black", lw=2.2, label="actual", zorder=5)

    for col in view.columns:
        if col != "actual":
            ax.plot(view.index, view[col], lw=1.2, alpha=0.85, label=col)

    for start in range(0, len(view), horizon):
        ax.axvline(view.index[start], color="grey", ls=":", lw=0.9, alpha=0.7)

    ax.set_title(f"Rolling-origin forecasts, first {window_hours} test hours "
                 f"(dotted lines mark forecast origins)")
    ax.set_ylabel("Appliance energy use (Wh per 10-min interval, hourly mean)")
    ax.set_xlabel("Date")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()

    return fig


def plot_forecast_panel(forecast_df: pd.DataFrame, ncols: int = 3):
    """One panel per model across the full test period."""
    models = [c for c in forecast_df.columns if c != "actual"]
    nrows = int(np.ceil(len(models) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 2.6 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, models):
        ax.plot(forecast_df.index, forecast_df["actual"], color="black", lw=1.0)
        ax.plot(forecast_df.index, forecast_df[name], color="tab:red", lw=1.0, alpha=0.85)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.suptitle("Model forecasts against actuals over the test fortnight", y=1.0)
    fig.tight_layout()

    return fig


def plot_error_by_lead_time(errors: pd.DataFrame):
    """MAE against lead time: how fast each model decays away from the origin."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for col in errors.columns:
        ax.plot(errors.index, errors[col], marker="o", ms=3, lw=1.3, label=col)

    ax.set_xlabel("Lead time (hours ahead of forecast origin)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Forecast error growth with lead time")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()

    return fig


def plot_residual_diagnostics(residuals: pd.Series, lags: int = 72):
    """Residual series, distribution and autocorrelation for the chosen model."""
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(residuals.index, residuals, lw=0.9)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Residuals over the test period")

    axes[1].hist(residuals.dropna(), bins=40, edgecolor="white")
    axes[1].set_title("Residual distribution")

    plot_acf(residuals.dropna(), lags=lags, ax=axes[2])
    axes[2].set_title("Residual ACF")

    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 20):
    """Permutation importance for the top features."""
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(top) + 1.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="tab:blue")
    ax.set_xlabel("Increase in MAE when the feature is permuted")
    ax.set_title(f"Permutation importance, top {len(top)} features")
    fig.tight_layout()

    return fig


def plot_series_overview(y: pd.Series):
    """Series, daily profile and weekly profile for the EDA section."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(y.index, y, lw=0.6)
    axes[0].set_title("Hourly appliance energy use")

    by_hour = y.groupby(y.index.hour)
    axes[1].plot(by_hour.mean(), marker="o", ms=3)
    axes[1].fill_between(
        range(24), by_hour.quantile(0.25), by_hour.quantile(0.75), alpha=0.25
    )
    axes[1].set_title("Mean profile by hour of day")
    axes[1].set_xlabel("Hour")

    by_dow = y.groupby(y.index.dayofweek).mean()
    axes[2].bar(range(7), by_dow)
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[2].set_title("Mean by day of week")

    fig.tight_layout()
    return fig
