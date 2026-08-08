"""Direct multi-horizon gradient boosting.

A single learner is trained on the long ``(timestamp, horizon)`` table, with
``horizon`` supplied as a feature.  Because every target-derived feature is
anchored to the forecast origin, the model cannot see inside the forecast
window at any horizon, and the rolling-origin predictions are produced in one
vectorised pass rather than recursively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from .. import config, features


def make_estimator(**overrides) -> HistGradientBoostingRegressor:
    """Gradient boosting configured for a small tabular time-series problem.

    scikit-learn's built-in early stopping is DISABLED here, deliberately.

    ``HistGradientBoostingRegressor`` carves its validation set out at random.
    In the long ``(timestamp, horizon)`` table each timestamp contributes 24
    rows whose features are highly correlated, so a random validation split
    places near-duplicates of validation rows into the training partition.  The
    validation score then improves spuriously, early stopping never fires, and
    the model overfits: raising ``max_iter`` from 600 to 2000 left ``n_iter_``
    pinned at the ceiling while test MASE degraded from 0.732 to 0.756.

    This is the same failure mode as target leakage, arriving through the
    validation split rather than the feature matrix.  The iteration count is
    instead selected chronologically by :func:`select_n_iter`.
    """
    params = dict(
        max_iter=config.ML_MAX_ITER,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=config.RANDOM_STATE,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def select_n_iter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    holdout_fraction: float = 0.15,
    max_iter: int = config.ML_MAX_ITER,
    **overrides,
) -> dict:
    """Choose the boosting iteration count on a chronological holdout.

    The final ``holdout_fraction`` of *timestamps* — not of rows — is held out,
    so no timestamp contributes rows to both partitions.  ``staged_predict``
    then yields the holdout MAE at every iteration from a single fit, and the
    minimising iteration is returned.
    """
    from sklearn.metrics import mean_absolute_error

    stamps = X_train.index.unique().sort_values()
    cut = stamps[int(len(stamps) * (1 - holdout_fraction))]

    fit_mask = X_train.index < cut
    X_fit, y_fit = X_train.loc[fit_mask], y_train.loc[fit_mask]
    X_val, y_val = X_train.loc[~fit_mask], y_train.loc[~fit_mask]

    probe = make_estimator(max_iter=max_iter, **overrides)
    probe.fit(X_fit, y_fit)

    scores = [mean_absolute_error(y_val, p) for p in probe.staged_predict(X_val)]

    best = int(np.argmin(scores)) + 1
    return {"n_iter": best, "holdout_mae": float(scores[best - 1]),
            "curve": scores, "cutoff": cut}


def fit_feature_model(X_train: pd.DataFrame, y_train: pd.Series, **overrides):
    """Fit using an iteration count selected on a chronological holdout."""
    if "max_iter" not in overrides:
        chosen = select_n_iter(X_train, y_train, **overrides)
        overrides = dict(overrides, max_iter=chosen["n_iter"])

    model = make_estimator(**overrides)
    model.fit(X_train, y_train)
    model.selected_n_iter_ = overrides["max_iter"]
    return model


def tune_feature_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    grid: list[dict] | None = None,
    n_splits: int = 4,
) -> dict:
    """Small grid search under blocked time-series cross-validation.

    Splitting is chronological and confined to the training sample, so no test
    observation influences model selection.
    """
    from sklearn.metrics import mean_absolute_error

    grid = grid or [
        {"learning_rate": lr, "max_leaf_nodes": leaves}
        for lr in (0.03, 0.05, 0.10)
        for leaves in (15, 31, 63)
    ]

    splitter = TimeSeriesSplit(n_splits=n_splits)
    best, best_score = grid[0], np.inf

    for params in grid:
        scores = []
        for train_idx, valid_idx in splitter.split(X_train):
            model = make_estimator(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_train.iloc[valid_idx])
            scores.append(mean_absolute_error(y_train.iloc[valid_idx], pred))

        score = float(np.mean(scores))
        if score < best_score:
            best, best_score = params, score

    return {"params": best, "cv_mae": best_score}


def build_design(
    y: pd.Series,
    exog_origin: pd.DataFrame | None,
    exog_future: pd.DataFrame | None,
    max_horizon: int = config.HORIZON,
):
    """Build the supervised table and return it with its feature names."""
    frame = features.build_supervised_frame(
        y,
        max_horizon=max_horizon,
        exog_origin=exog_origin,
        exog_future=exog_future,
    )
    return frame, features.feature_columns(frame)


def rolling_origin_predict(
    model,
    frame: pd.DataFrame,
    feature_cols: list[str],
    test_index: pd.DatetimeIndex,
    horizon: int = config.HORIZON,
) -> pd.Series:
    """Predict the rolling-origin test path from the prebuilt design matrix."""
    pairs = features.rolling_origin_pairs(test_index, horizon)
    rows = features.select_rows(frame, pairs)
    preds = model.predict(rows[feature_cols])
    return pd.Series(preds, index=rows.index)


def feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance, computed on held-out rows.

    Permutation importance is preferred to split-count importance because the
    feature set contains many strongly collinear channels, among which split
    counts are arbitrarily divided.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring="neg_mean_absolute_error",
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
