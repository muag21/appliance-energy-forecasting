"""Loading, cleaning and resampling of the Appliances Energy Prediction data.

The raw file is cached under ``data/raw`` so that the pipeline is reproducible
from a fresh clone without repeated network access.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def download_raw(url: str = config.DATA_URL, dest: Path = config.RAW_CSV) -> Path:
    """Download and cache the raw 10-minute CSV.

    The UCI archive serves this dataset as a zip archive.  If the download
    fails the user is told to place the file manually, which keeps the
    pipeline usable in offline marking environments.
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return dest

    payload = None
    errors = []

    for source in (url, config.DATA_URL_MIRROR, config.DATA_URL_LEGACY):
        try:
            with urllib.request.urlopen(source, timeout=60) as response:
                payload = response.read()
            break
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{source}: {exc}")

    if payload is None:
        raise RuntimeError(
            "Could not download the dataset from any source:\n  "
            + "\n  ".join(errors)
            + f"\nDownload 'energydata_complete.csv' manually and place it at {dest}."
        )

    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".csv"))
            dest.write_bytes(archive.read(name))
    else:
        dest.write_bytes(payload)

    return dest


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Read the cached 10-minute CSV into a datetime-indexed frame."""
    if not path.exists():
        download_raw(dest=path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=[config.TARGET])


def to_hourly(
    df: pd.DataFrame,
    target_agg: str = config.TARGET_AGG,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Resample 10-minute observations to an hourly frame.

    Parameters
    ----------
    target_agg
        ``"mean"`` gives the average Wh per 10-minute interval within the hour;
        ``"sum"`` gives total Wh consumed within the hour.  The choice rescales
        MAE, RMSE and Bias but leaves MASE unchanged.
    train_end
        If given, interpolation of residual gaps is performed separately either
        side of this timestamp so that training-period gaps are never filled
        using post-split observations.

    Notes
    -----
    Sensor and weather channels are always averaged: they are instantaneous
    measurements, not accumulated quantities.
    """
    if target_agg not in {"mean", "sum"}:
        raise ValueError("target_agg must be 'mean' or 'sum'")

    sensor_cols = [c for c in df.columns if c != config.TARGET]

    target = getattr(df[config.TARGET].resample("h"), target_agg)()
    sensors = df[sensor_cols].resample("h").mean()

    hourly = pd.concat([target.rename(config.TARGET), sensors], axis=1)

    if train_end is None:
        hourly = hourly.interpolate("time")
    else:
        left = hourly.loc[:train_end].interpolate("time")
        right = hourly.loc[train_end:].iloc[1:].interpolate("time")
        hourly = pd.concat([left, right])

    hourly = hourly.drop(columns=config.EXCLUDED_COLS, errors="ignore")

    return hourly.dropna(subset=[config.TARGET])


def train_test_split(
    frame: pd.DataFrame | pd.Series,
    test_steps: int = config.TEST_STEPS,
):
    """Split chronologically, returning ``(train, test)``.

    Slicing is performed on timestamps rather than positions so the split
    survives any interior gap in the index.
    """
    if len(frame) <= test_steps:
        raise ValueError("Series is shorter than the requested test period.")

    boundary = frame.index[-test_steps]
    return frame.loc[frame.index < boundary], frame.loc[frame.index >= boundary]


def load_hourly(
    target_agg: str = config.TARGET_AGG,
    cache: Path = config.HOURLY_CSV,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the hourly analysis frame, using the processed cache if present."""
    if cache.exists() and not refresh:
        hourly = pd.read_csv(cache, index_col=0, parse_dates=True)
        return hourly

    raw = load_raw()

    # Determine the split boundary on the 10-minute index so that gap filling
    # respects the train/test boundary.
    hourly_index = raw[config.TARGET].resample("h").mean().dropna().index
    train_end = hourly_index[-config.TEST_STEPS - 1]

    hourly = to_hourly(raw, target_agg=target_agg, train_end=train_end)

    cache.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(cache)

    return hourly
