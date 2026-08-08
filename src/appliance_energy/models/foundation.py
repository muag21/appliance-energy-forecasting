"""Chronos-T5 zero-shot forecasting.

Chronos tokenises a scaled, quantised version of the target series and samples
future trajectories from a T5 decoder.  It is univariate and, in this
configuration, has no access to covariates: only the target history enters the
context window.  The forecast reported is the pointwise median of the sampled
paths, which is the appropriate point summary for an absolute-error metric.

The 24-step rolling-origin horizon sits comfortably inside the model's native
prediction length, so no autoregressive stitching is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

_PIPELINE_CACHE: dict[str, object] = {}


def load_chronos(model_name: str = config.CHRONOS_MODEL, device: str = "cpu"):
    """Load and cache the Chronos pipeline.

    Import is deferred so that the rest of the pipeline runs without torch
    installed.
    """
    key = f"{model_name}:{device}"
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Chronos requires 'torch' and 'chronos-forecasting'. "
            "Install with: pip install torch chronos-forecasting"
        ) from exc

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,  # float32 for CPU inference
    )

    _PIPELINE_CACHE[key] = pipeline
    return pipeline


def chronos_forecaster(
    model_name: str = config.CHRONOS_MODEL,
    context_length: int = config.CHRONOS_CONTEXT,
    num_samples: int = config.CHRONOS_SAMPLES,
    device: str = "cpu",
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
):
    """Return a forecaster with the standard ``(history, horizon)`` signature.

    The returned callable also records the sampled quantiles on its
    ``.quantiles_`` attribute so prediction intervals can be plotted.
    """
    import torch

    pipeline = load_chronos(model_name, device=device)
    store: list[pd.DataFrame] = []

    def forecast(history: pd.Series, horizon: int) -> np.ndarray:
        context = torch.tensor(
            history.to_numpy(dtype="float32")[-context_length:]
        )

        samples = pipeline.predict(
            context=context,
            prediction_length=horizon,
            num_samples=num_samples,
        )

        arr = samples[0].numpy()  # (num_samples, horizon)
        qs = np.quantile(arr, quantiles, axis=0)
        store.append(pd.DataFrame(qs.T, columns=[f"q{q}" for q in quantiles]))

        return np.median(arr, axis=0)

    forecast.quantiles_ = store
    return forecast


def is_available(model_name: str = config.CHRONOS_MODEL) -> bool:
    """Whether Chronos can be loaded in the current environment."""
    try:
        load_chronos(model_name)
        return True
    except Exception:
        return False
