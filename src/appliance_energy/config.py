"""Central configuration for the appliance energy forecasting pipeline."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths.  Resolved relative to the package so the pipeline works from any cwd.
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"

RAW_CSV = RAW_DIR / "energydata_complete.csv"
HOURLY_CSV = PROCESSED_DIR / "appliance_hourly.csv"

#: Primary source (UCI) and a mirror maintained by the dataset's authors.
#: The mirror is tried when the archive is unreachable, which it periodically is.
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/374/"
    "appliances+energy+prediction.zip"
)
DATA_URL_MIRROR = (
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv"
)
DATA_URL_LEGACY = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)

# --------------------------------------------------------------------------
# Series definition
# --------------------------------------------------------------------------

TARGET = "Appliances"

#: Aggregation applied when resampling 10-minute data to hourly.
#: "mean" -> average Wh per 10-minute interval within the hour.
#: "sum"  -> total Wh consumed within the hour.
TARGET_AGG = "mean"

DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

# --------------------------------------------------------------------------
# Evaluation protocol
# --------------------------------------------------------------------------

TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * 24          # 336 hours
HORIZON = 24                         # rolling-origin forecast horizon
N_ORIGINS = TEST_STEPS // HORIZON    # 14 origins

# --------------------------------------------------------------------------
# Covariates
# --------------------------------------------------------------------------

#: Outdoor weather.  Not known at the forecast origin in operation; using the
#: realised test-set values yields a CONDITIONAL forecast.
WEATHER_COLS = ["T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]

#: Indoor sensors.  These respond to occupant and appliance activity, so their
#: contemporaneous values are partly a consequence of the target.  They are
#: only ever used at origin-anchored lags.
INDOOR_COLS = [f"T{i}" for i in range(1, 10)] + [f"RH_{i}" for i in range(1, 10)]

#: Contemporaneous and strongly collinear with the target.  Excluded outright.
EXCLUDED_COLS = ["lights"]

# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

#: Upper bound on boosting iterations. The working count is selected
#: chronologically by models.feature_models.select_n_iter.
ML_MAX_ITER = 1500

LAG_OFFSETS = (1, 2, 3, 6, 12, 24, 48, 168)
ROLL_WINDOWS = (3, 6, 24, 168)

# --------------------------------------------------------------------------
# SARIMAX
# --------------------------------------------------------------------------

SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 0, 1, 24)
SARIMAX_TREND = "c"

#: statsmodels defaults to 50, which terminates before convergence here.
SARIMAX_MAXITER = 500

# --------------------------------------------------------------------------
# Foundation model
# --------------------------------------------------------------------------

CHRONOS_MODEL = "amazon/chronos-t5-small"
CHRONOS_CONTEXT = 512
CHRONOS_SAMPLES = 20

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

RANDOM_STATE = 0


def ensure_directories() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (RAW_DIR, PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
