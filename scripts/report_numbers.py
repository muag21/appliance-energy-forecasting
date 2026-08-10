"""Print every number the report needs, computed from your own pipeline outputs.

Run this after ``run_pipeline.py``, then transfer the values into
``reports/report.md``.  Section headings below match the report.

    python scripts/report_numbers.py

Figures will differ slightly between machines: BLAS/LAPACK builds and library
versions change the SARIMAX optimiser path and the boosting fit.  The numbers
this prints are the correct ones for YOUR run, and those are what belong in
your report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from appliance_energy import config, data, evaluation, stationarity  # noqa: E402


def header(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def diebold_mariano(actual, f1, f2, h: int = 24):
    """Absolute-error loss differential with a Newey-West variance estimator."""
    d = ((actual - f1).abs() - (actual - f2).abs()).dropna()
    n = len(d)
    var = d.var(ddof=0)
    for lag in range(1, h):
        var += 2 * (1 - lag / h) * np.cov(d[lag:], d[:-lag])[0, 1]
    dm = d.mean() / np.sqrt(var / n)
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


def main() -> None:
    metrics = config.METRICS_DIR / "model_comparison.csv"
    forecasts = config.FORECAST_DIR / "all_forecasts.csv"

    if not metrics.exists():
        raise SystemExit("No outputs found. Run: python scripts/run_pipeline.py")

    res = pd.read_csv(metrics)
    fd = pd.read_csv(forecasts, index_col=0, parse_dates=True)

    frame = data.load_hourly()
    y = frame[config.TARGET]
    y_train, y_test = data.train_test_split(y)

    # ------------------------------------------------------------------
    header("SECTION 2 — Data and preprocessing")
    print(f"hourly observations : {len(y)}")
    print(f"train : {y_train.index.min()} to {y_train.index.max()}  n={len(y_train)}")
    print(f"test  : {y_test.index.min()} to {y_test.index.max()}  n={len(y_test)}")
    print(f"train mean {y_train.mean():.1f} sd {y_train.std():.1f} | "
          f"test mean {y_test.mean():.1f} sd {y_test.std():.1f}")

    # ------------------------------------------------------------------
    header("SECTION 3.1 — Distribution")
    for label, value in [
        ("mean", y.mean()), ("median", y.median()), ("std", y.std()),
        ("min", y.min()), ("max", y.max()), ("skewness", y.skew()),
        ("excess kurtosis", y.kurtosis()),
        ("p95/median", y.quantile(0.95) / y.median()),
    ]:
        print(f"  {label:<16} {value:.2f}")

    # ------------------------------------------------------------------
    header("SECTION 3.2 — Seasonal structure")
    profile = y.groupby(y.index.hour).mean()
    print(f"trough hour {profile.idxmin()} = {profile.min():.1f}")
    print(f"peak   hour {profile.idxmax()} = {profile.max():.1f}")
    print(f"peak/trough ratio {profile.max() / profile.min():.2f}")

    weekday = y[y.index.dayofweek < 5].mean()
    weekend = y[y.index.dayofweek >= 5].mean()
    print(f"weekday {weekday:.1f}  weekend {weekend:.1f}  ratio {weekend / weekday:.3f}")

    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = y.groupby(y.index.dayofweek).mean()
    print("by day: " + "  ".join(f"{names[d]} {v:.1f}" for d, v in by_dow.items()))

    print(f"\nseasonal strength daily  (24) : "
          f"{stationarity.seasonal_strength(y_train, config.DAILY_PERIOD):.3f}")
    print(f"seasonal strength weekly (168): "
          f"{stationarity.seasonal_strength(y_train, config.WEEKLY_PERIOD):.3f}")

    # ------------------------------------------------------------------
    header("SECTION 3.3 — Stationarity (training sample)")
    print(stationarity.stationarity_report(y_train).round(4).to_string(index=False))

    # ------------------------------------------------------------------
    header("SECTIONS 5, 7.4, 9.1 — Metrics table (paste as markdown)")
    print(res.round(3).to_markdown(index=False))

    # ------------------------------------------------------------------
    header("SECTION 9.1 — Diebold-Mariano tests")
    best = res["model"].iloc[0]
    print(f"Leading model: {best}\n")
    for other in res["model"].iloc[1:]:
        dm, p = diebold_mariano(fd["actual"], fd[best], fd[other])
        verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  vs {other:<24} DM {dm:+6.2f}  p {p:.4f}  {verdict}")

    pairs = [("sarimax_conditional", "sarimax_operational"),
             ("feature_model", "seasonal_naive_weekly"),
             ("sarimax_conditional", "seasonal_naive_weekly")]
    print()
    for a, b in pairs:
        if a in fd.columns and b in fd.columns:
            dm, p = diebold_mariano(fd["actual"], fd[a], fd[b])
            verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"  {a} vs {b}: DM {dm:+.2f}  p {p:.4f}  {verdict}")

    # ------------------------------------------------------------------
    header("SECTION 6.3 — Conditional vs operational SARIMAX")
    if {"sarimax_conditional", "sarimax_operational"} <= set(res["model"]):
        idx = res.set_index("model")
        cond = idx.loc["sarimax_conditional", "MASE"]
        oper = idx.loc["sarimax_operational", "MASE"]
        print(f"conditional {cond:.3f}  operational {oper:.3f}")
        print(f"gap {oper - cond:.3f} MASE, {100 * (oper - cond) / cond:.1f}% relative")

    # ------------------------------------------------------------------
    header("SECTION 9.2 — MAE by lead time")
    fc = {c: fd[c] for c in fd.columns if c != "actual"}
    lead = evaluation.errors_by_horizon(fc, fd["actual"], config.HORIZON)
    print(lead.loc[[1, 6, 12, 18, 24]].round(1).to_markdown())
    print("\ndegradation lead 1 -> lead 24:")
    for col in lead.columns:
        print(f"  {col:<24} {lead[col].iloc[0]:6.1f} -> {lead[col].iloc[-1]:6.1f}"
              f"  ({lead[col].iloc[-1] / lead[col].iloc[0]:.2f}x)")

    # ------------------------------------------------------------------
    header("SECTION 9.3 — Residual diagnostics (leading model)")
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    resid = fd[best] - fd["actual"]
    lb = acorr_ljungbox(resid, lags=[24, 48], return_df=True)
    a = acf(resid, nlags=48)

    print(f"Ljung-Box  24 lags : Q = {lb.lb_stat.iloc[0]:.1f}, p = {lb.lb_pvalue.iloc[0]:.2g}")
    print(f"Ljung-Box  48 lags : Q = {lb.lb_stat.iloc[1]:.1f}, p = {lb.lb_pvalue.iloc[1]:.2g}")
    print(f"residual ACF lag 1 : {a[1]:.3f}   lag 24 : {a[24]:.3f}")
    print(f"95% band           : +/- {1.96 / np.sqrt(len(resid)):.3f}")
    print(f"skewness           : {resid.skew():.2f}")
    print(f"excess kurtosis    : {resid.kurtosis():.2f}")
    print(f"corr(|resid|, act) : {resid.abs().corr(fd['actual']):.3f}")

    # ------------------------------------------------------------------
    header("SECTION 7.3 — Feature importance")
    imp_path = config.METRICS_DIR / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        print(imp.head(10).round(3).to_markdown(index=False))

        calendar = {"hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "dayofweek", "is_weekend", "horizon"}
        positive = imp["importance"].clip(lower=0).sum()
        shares = {
            "calendar": imp[imp.feature.isin(calendar)]["importance"].clip(lower=0).sum(),
            "target lags / rolling":
                imp[imp.feature.str.startswith(("lag_", "roll_"))]["importance"].clip(lower=0).sum(),
            "lagged exogenous":
                imp[imp.feature.str.endswith("_origin")]["importance"].clip(lower=0).sum(),
        }
        print()
        for label, value in shares.items():
            print(f"  {label:<24} {value / positive:.1%}")

    # ------------------------------------------------------------------
    header("SECTION 7.2 — Boosting iterations selected")
    try:
        from appliance_energy.models import feature_models
        indoor = [c for c in config.INDOOR_COLS if c in frame.columns]
        weather = [c for c in config.WEATHER_COLS if c in frame.columns]
        design, cols = feature_models.build_design(
            y, exog_origin=frame[indoor + weather], exog_future=None)
        rows = design.loc[design.index < y_test.index[0]]
        chosen = feature_models.select_n_iter(rows[cols], rows["target"])
        print(f"selected iterations : {chosen['n_iter']}")
        print(f"holdout MAE         : {chosen['holdout_mae']:.2f}")
        print(f"holdout cutoff      : {chosen['cutoff']}")
        curve = chosen["curve"]
        print("\nholdout MAE by iteration:")
        for i in (1, 25, 50, 100, 200, 400, 800, 1500):
            if i <= len(curve):
                print(f"  {i:>5} : {curve[i - 1]:.2f}")
    except Exception as exc:
        print(f"could not compute: {exc}")

    header("Transfer these into reports/report.md, then re-run verify.py")


if __name__ == "__main__":
    main()
