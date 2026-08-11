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


def resolve_device(device: str = "auto") -> str:
    """Return 'cuda' when a GPU is available and ``device`` is 'auto'."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_chronos(model_name: str = config.CHRONOS_MODEL, device: str = "auto"):
    """Load and cache the Chronos pipeline.

    Import is deferred so that the rest of the pipeline runs without torch
    installed.
    """
    device = resolve_device(device)
    key = f"{model_name}:{device}"
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    try:
        import torch
        try:
            from chronos import BaseChronosPipeline as ChronosPipeline
        except ImportError:
            from chronos import ChronosPipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Chronos requires 'torch' and 'chronos-forecasting'. "
            "Install with: pip install torch chronos-forecasting"
        ) from exc

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # `torch_dtype` was renamed to `dtype` in recent transformers releases.
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map=device, dtype=dtype,
        )
    except TypeError:
        pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=dtype,
        )

    _PIPELINE_CACHE[key] = pipeline
    return pipeline


def chronos_forecaster(
    model_name: str = config.CHRONOS_MODEL,
    context_length: int = config.CHRONOS_CONTEXT,
    num_samples: int = config.CHRONOS_SAMPLES,
    device: str = "auto",
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

        # The first argument has been named `context` and later `inputs` across
        # releases, so it is passed positionally.  `num_samples` exists only on
        # sample-based pipelines, not on quantile-based ones.
        import inspect

        kwargs = {"prediction_length": horizon}
        try:
            params = inspect.signature(pipeline.predict).parameters
            if "num_samples" in params:
                kwargs["num_samples"] = num_samples
        except (TypeError, ValueError):
            kwargs["num_samples"] = num_samples

        raw = pipeline.predict(context, **kwargs)

        # Output is a tensor, a list of tensors, or a (quantiles, mean) tuple
        # depending on version.  Normalise to a 2-D array of shape
        # (draws, horizon).
        if isinstance(raw, tuple):
            raw = raw[0]
        if isinstance(raw, (list, tuple)):
            raw = raw[0]

        arr = np.asarray(
            raw.float().cpu().numpy() if hasattr(raw, "cpu") else raw
        )
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] == horizon and arr.shape[1] != horizon:
            arr = arr.T

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
