"""End-to-end forecasting pipeline.

Usage
-----
    python scripts/run_pipeline.py                    # full run
    python scripts/run_pipeline.py --no-foundation    # skip Chronos
    python scripts/run_pipeline.py --secondary        # add 336-step experiment
    python scripts/run_pipeline.py --tune             # grid search the ML model

Protocol
--------
Primary experiment: rolling origin, fourteen origins spaced twenty-four hours
apart.  At each origin every model observes the series up to the origin and
issues a twenty-four hour forecast.  Concatenating the blocks yields 336
predictions per model on a common index and a common information set.

Secondary experiment: a single 336-step forecast from the first origin, which
shows how each model degrades over a long horizon.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy import config, data, evaluation, features, plotting, stationarity
from appliance_energy.models import benchmarks, feature_models, foundation, sarimax


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-foundation", action="store_true",
                        help="skip the Chronos forecast")
    parser.add_argument("--secondary", action="store_true",
                        help="also run the single 336-step experiment")
    parser.add_argument("--tune", action="store_true",
                        help="grid search the gradient boosting model")
    parser.add_argument("--device", default="cpu", help="device for Chronos")
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    args = parse_args()
    config.ensure_directories()
    np.random.seed(config.RANDOM_STATE)

    # ------------------------------------------------------------------
    section("1. Data")
    # ------------------------------------------------------------------

    frame = data.load_hourly()
    y = frame[config.TARGET]

    y_train, y_test = data.train_test_split(y)
    test_index = y_test.index

    print(f"Hourly observations : {len(y)}")
    print(f"Train : {y_train.index.min()} to {y_train.index.max()}  ({len(y_train)})")
    print(f"Test  : {test_index.min()} to {test_index.max()}  ({len(y_test)})")
    print(f"Target aggregation  : {config.TARGET_AGG}")

    # ------------------------------------------------------------------
    section("2. Stationarity diagnostics (training sample only)")
    # ------------------------------------------------------------------

    report = stationarity.stationarity_report(y_train, period=config.DAILY_PERIOD)
    print(report.to_string(index=False))
    report.to_csv(config.METRICS_DIR / "stationarity_tests.csv", index=False)

    strength_d = stationarity.seasonal_strength(y_train, config.DAILY_PERIOD)
    strength_w = stationarity.seasonal_strength(y_train, config.WEEKLY_PERIOD)
    print(f"\nSeasonal strength, daily  (period 24) : {strength_d:.3f}")
    print(f"Seasonal strength, weekly (period 168): {strength_w:.3f}")

    forecasts: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    section("3. Benchmarks")
    # ------------------------------------------------------------------

    for name, fn in benchmarks.benchmark_suite(
        config.DAILY_PERIOD, config.WEEKLY_PERIOD
    ).items():
        forecasts[name] = benchmarks.rolling_origin_forecast(
            y, test_index, config.HORIZON, fn
        )
        print(f"  {name:<24} done")

    # ------------------------------------------------------------------
    section("4. SARIMAX")
    # ------------------------------------------------------------------

    weather = [c for c in config.WEATHER_COLS if c in frame.columns]
    exog = frame[weather] if weather else None

    print(f"Order {config.SARIMAX_ORDER} x {config.SARIMAX_SEASONAL_ORDER}")
    print(f"Exogenous: {weather or 'none (target only)'}")

    t0 = time.time()
    fit = sarimax.fit_sarimax(
        y_train, exog_train=None if exog is None else exog.loc[y_train.index]
    )
    print(f"Estimated in {time.time() - t0:.1f}s   AIC {fit.aic:.1f}   BIC {fit.bic:.1f}")

    if exog is not None:
        # Conditional: realised weather over the forecast window.
        forecasts["sarimax_conditional"] = sarimax.rolling_origin_sarimax(
            fit, y, test_index, config.HORIZON, exog=exog
        )
        # Operational: weather persisted from the last pre-origin observation.
        forecasts["sarimax_operational"] = sarimax.rolling_origin_sarimax(
            fit, y, test_index, config.HORIZON,
            exog=sarimax.persisted_exog(exog, test_index, config.HORIZON),
        )
    else:
        forecasts["sarimax"] = sarimax.rolling_origin_sarimax(
            fit, y, test_index, config.HORIZON
        )

    with open(config.METRICS_DIR / "sarimax_summary.txt", "w") as handle:
        handle.write(str(fit.summary()))

    # ------------------------------------------------------------------
    section("5. Feature-based model")
    # ------------------------------------------------------------------

    indoor = [c for c in config.INDOOR_COLS if c in frame.columns]
    exog_origin = frame[indoor + weather] if (indoor or weather) else None

    design, feature_cols = feature_models.build_design(
        y, exog_origin=exog_origin, exog_future=None, max_horizon=config.HORIZON
    )

    train_rows = design.loc[design.index < test_index[0]]
    X_train = train_rows[feature_cols]
    y_train_ml = train_rows["target"]

    print(f"Design matrix: {design.shape[0]} rows x {len(feature_cols)} features")
    print(f"Training rows: {len(train_rows)}")

    if args.tune:
        best = feature_models.tune_feature_model(X_train, y_train_ml)
        print(f"Best CV params: {best['params']}  (MAE {best['cv_mae']:.2f})")
        model = feature_models.fit_feature_model(X_train, y_train_ml, **best["params"])
    else:
        model = feature_models.fit_feature_model(X_train, y_train_ml)

    forecasts["feature_model"] = feature_models.rolling_origin_predict(
        model, design, feature_cols, test_index, config.HORIZON
    )

    test_rows = features.select_rows(
        design, features.rolling_origin_pairs(test_index, config.HORIZON)
    )
    importance = feature_models.feature_importance(
        model, test_rows[feature_cols], test_rows["target"]
    )
    importance.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)
    print("\nTop features:")
    print(importance.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    section("6. Foundation model (Chronos-T5, zero shot)")
    # ------------------------------------------------------------------

    if args.no_foundation:
        print("Skipped (--no-foundation).")
    else:
        try:
            t0 = time.time()
            chronos = foundation.chronos_forecaster(device=args.device)
            forecasts["chronos_zeroshot"] = benchmarks.rolling_origin_forecast(
                y, test_index, config.HORIZON, chronos
            )
            print(f"Completed in {time.time() - t0:.1f}s "
                  f"(univariate, no covariates, {config.CHRONOS_SAMPLES} samples)")
        except ImportError as exc:
            print(f"Unavailable: {exc}")

    # ------------------------------------------------------------------
    section("7. Evaluation")
    # ------------------------------------------------------------------

    results = evaluation.evaluate_all(forecasts, y_test, y_train)
    print(results.round(3).to_string(index=False))

    lead_errors = evaluation.errors_by_horizon(forecasts, y_test, config.HORIZON)
    lead_errors.to_csv(config.METRICS_DIR / "error_by_lead_time.csv")

    forecast_df = pd.DataFrame({"actual": y_test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test_index)

    forecast_df.to_csv(config.FORECAST_DIR / "all_forecasts.csv")
    results.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    # ------------------------------------------------------------------
    section("8. Figures")
    # ------------------------------------------------------------------

    best_model = results["model"].iloc[0]
    residuals = forecast_df[best_model] - forecast_df["actual"]

    figures = {
        "series_overview.png": plotting.plot_series_overview(y),
        "forecast_comparison.png": plotting.plot_forecast_window(forecast_df),
        "forecast_panel.png": plotting.plot_forecast_panel(forecast_df),
        "error_by_lead_time.png": plotting.plot_error_by_lead_time(lead_errors),
        "residual_diagnostics.png": plotting.plot_residual_diagnostics(residuals),
        "feature_importance.png": plotting.plot_feature_importance(importance),
    }

    for name, fig in figures.items():
        fig.savefig(config.FIGURE_DIR / name, dpi=200, bbox_inches="tight")
        print(f"  saved {name}")

    print(f"\nBest model by MASE: {best_model}")

    # ------------------------------------------------------------------
    if args.secondary:
        section("9. Secondary experiment: single 336-step forecast")
        run_secondary(y, y_train, y_test, frame, weather, indoor)

    return results, forecast_df


def run_secondary(y, y_train, y_test, frame, weather, indoor):
    """One 336-step forecast from a single origin, for long-horizon behaviour.

    Contrast with the primary experiment isolates how much accuracy comes from
    reissuing the forecast daily rather than from model quality.
    """
    test_index = y_test.index
    horizon = len(y_test)
    out = {}

    for name, fn in benchmarks.benchmark_suite(
        config.DAILY_PERIOD, config.WEEKLY_PERIOD
    ).items():
        out[name] = pd.Series(fn(y_train, horizon), index=test_index)

    exog = frame[weather] if weather else None
    fit = sarimax.fit_sarimax(
        y_train, exog_train=None if exog is None else exog.loc[y_train.index]
    )
    forecast = fit.get_forecast(
        steps=horizon, exog=None if exog is None else exog.loc[test_index]
    )
    out["sarimax_conditional"] = pd.Series(
        np.asarray(forecast.predicted_mean), index=test_index
    )

    # The direct multi-horizon design is omitted here: at a 336-step horizon
    # it would require 3,290 x 336 rows, which is not tractable without
    # subsampling origins in a way that would make the comparison unequal.
    # Long-horizon behaviour of the benchmarks and SARIMAX carries the point.

    table = evaluation.evaluate_all(out, y_test, y_train)
    print(table.round(3).to_string(index=False))
    table.to_csv(config.METRICS_DIR / "model_comparison_secondary.csv", index=False)


if __name__ == "__main__":
    main()
