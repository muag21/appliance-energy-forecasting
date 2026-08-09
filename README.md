# Appliance Energy Forecasting

Short-term forecasting of household appliance electricity demand using the UCI
*Appliances Energy Prediction* dataset. The repository compares five benchmark
rules, a SARIMAX state-space model, a direct multi-horizon gradient boosting
model, and a zero-shot time-series foundation model under a single, shared
rolling-origin evaluation protocol.

---

## Forecasting task

| | |
|---|---|
| Target | `Appliances`, resampled to hourly |
| Units | mean Wh per 10-minute interval within the hour (see [Units](#units)) |
| Horizon | 24 hours ahead |
| Protocol | rolling origin, 14 origins spaced 24 h apart |
| Test period | final 14 days (336 hourly observations) |
| Metrics | MAE, RMSE, MASE (daily seasonal in-sample scale), Bias |

At each origin every model observes the series up to that origin — including
test observations released by earlier blocks — and issues a 24-hour forecast.
Concatenating the fourteen blocks yields 336 predictions per model on a common
index and a common information set. Aggregate metrics are computed over all
336 points; error is additionally decomposed by lead time.

A secondary experiment issues a single 336-step forecast from the first origin
to show how each model degrades over a long horizon.

---

## Documentation

| File | Purpose |
|---|---|
| `SETUP.md` | End-to-end setup: local, GitHub and Colab workflows |
| `CHECKLIST.md` | Pre-submission verification |
| `reports/report.md` | The report itself |

## Quick start

```bash
git clone <repository-url>
cd appliance-energy-forecasting

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest                             # 37 tests, ~3 s

python scripts/run_pipeline.py --no-foundation
```

The dataset is downloaded and cached to `data/raw/` on first run. If the UCI
archive is unreachable, download `energydata_complete.csv` manually and place
it at `data/raw/energydata_complete.csv`; the pipeline will use the cache.

To include the foundation model:

```bash
pip install torch chronos-forecasting
python scripts/run_pipeline.py
```

CPU inference is sufficient. Expect roughly 2–4 minutes for the 14 Chronos
calls at 20 samples each.

### Command-line options

| Flag | Effect |
|---|---|
| `--no-foundation` | Skip the Chronos forecast |
| `--secondary` | Additionally run the single 336-step experiment |
| `--tune` | Grid search the gradient boosting model under blocked time-series CV |
| `--device cuda` | Run Chronos on GPU |

---

## Running on Google Colab

Open `notebooks/00_colab_quickstart.ipynb` in Colab. This is the recommended way
to run the foundation model: Colab reaches the model host and offers a free GPU,
where a local or restricted environment may not.

```python
!git clone https://github.com/YOUR_USERNAME/appliance-energy-forecasting.git /content/project
%cd /content/project
!pip install -q chronos-forecasting
!python scripts/run_pipeline.py --device auto
```

Colab already ships numpy, pandas, scikit-learn, statsmodels and torch — only
Chronos needs installing, and upgrading the others forces a runtime restart.
`weasyprint` is unavailable there, so build the report PDF locally.

Colab's filesystem is wiped when the runtime disconnects. Download `outputs/`
or push back to GitHub before closing the tab.

## Repository layout

```
appliance-energy-forecasting/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/                       # cached UCI CSV (git-ignored)
│   ├── interim/
│   └── processed/                 # hourly analysis frame (git-ignored)
│
├── notebooks/
│   ├── 00_colab_quickstart.ipynb
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/appliance_energy/
│   ├── config.py                  # paths, protocol constants, covariate groups
│   ├── data.py                    # download, cache, resample, split
│   ├── features.py                # origin-anchored supervised table
│   ├── stationarity.py            # ADF, KPSS, STL seasonal strength
│   ├── evaluation.py              # MAE, RMSE, MASE, Bias, error by lead time
│   ├── plotting.py                # report figures
│   └── models/
│       ├── benchmarks.py          # five benchmarks + rolling-origin driver
│       ├── sarimax.py             # estimation, state extension, exog variants
│       ├── feature_models.py      # direct multi-horizon gradient boosting
│       └── foundation.py          # Chronos-T5 zero-shot wrapper
│
├── scripts/
│   └── run_pipeline.py            # single entry point, stages 1–8
│
├── outputs/
│   ├── figures/
│   ├── forecasts/all_forecasts.csv
│   ├── metrics/
│   │   ├── model_comparison.csv
│   │   ├── error_by_lead_time.csv
│   │   ├── stationarity_tests.csv
│   │   ├── feature_importance.csv
│   │   └── sarimax_summary.txt
│   └── model_objects/
│
├── reports/
│   ├── report.md
│   └── figures/
│
└── tests/
    ├── test_features.py           # leakage guarantees
    ├── test_evaluation.py         # metric correctness
    └── test_benchmarks.py         # benchmark and protocol correctness
```

Notebooks import from `src/` and are for exploration and exposition only. No
notebook defines analysis logic; every result in the report is reproducible
from `scripts/run_pipeline.py` alone.

---

## Design decisions

### Preventing leakage structurally

A row of the supervised table is a *(target timestamp `t`, horizon `h`)* pair.
The forecast origin is `o = t − h + 1`, so the most recent observable target
value is `y[o − 1] = y[t − h]`. Target-derived features are anchored to the
origin rather than the target timestamp:

```
lag_k        = y[o − k]                       = y.shift(h − 1 + k)
roll_mean_w  = mean of y over [o − w, o − 1]  = y.shift(h).rolling(w).mean()
```

Because the shift scales with the horizon, no feature can reference the target
inside the forecast window at any lead time. This is stronger than a coding
convention: it is enforced by `tests/test_features.py`, which perturbs every
test-period observation and asserts that no training-row feature responds.

The naive alternative — `y.shift(1)` regardless of horizon — silently converts
a 24-hour-ahead task into a one-step-ahead task and inflates the machine
learning result beyond comparability with the other models.

### Covariate availability

Covariates are partitioned by what an operator would actually know when the
forecast is issued.

| Group | Availability at origin | Treatment |
|---|---|---|
| Calendar (`hour`, `dayofweek`, cyclic encodings) | Known indefinitely ahead | Used unshifted |
| Indoor sensors (`T1`–`T9`, `RH_1`–`RH_9`) | Observed up to the origin only | Lagged by `h` |
| Outdoor weather (`T_out`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`) | Not observed; a forecast would be required | Two variants, see below |
| `lights` | Contemporaneous, not knowable ahead | Excluded outright |

Indoor temperature and humidity respond *to* occupant and appliance activity —
cooking raises kitchen temperature and humidity, occupancy raises RH. Their
contemporaneous values are partly a consequence of the target, so using them
unshifted would be regressing the target on its own effects while also assuming
information no operator possesses.

SARIMAX is run twice to quantify this directly:

- **`sarimax_conditional`** uses realised test-set weather. This is a
  conditional forecast and an optimistic upper bound.
- **`sarimax_operational`** persists the last pre-origin weather observation
  across the window, standing in for whatever a real deployment would have.

The gap between them is the empirical cost of covariate uncertainty.

### SARIMAX under rolling origins

Parameters are estimated once on the training sample. At each subsequent origin
the Kalman filter is extended with newly released observations via
`MLEResults.append(..., refit=False)`. Coefficients are therefore held at their
training values while the state conditions on all data up to the origin. Full
re-estimation fourteen times at seasonal period 24 costs substantial compute for
a negligible coefficient change over a two-week window. This is stated as an
assumption in the report rather than left implicit.

### Units

`Appliances` records Wh consumed per 10-minute interval. Hourly resampling uses
the **mean**, giving the average 10-minute consumption rate within each hour;
`config.TARGET_AGG = "sum"` switches to total Wh per hour. The choice rescales
MAE, RMSE and Bias by a factor of six and leaves MASE unchanged. Sensor and
weather channels are always averaged, being instantaneous measurements rather
than accumulated quantities.

### Two leakage failures, not one

The origin-anchored design above prevents leakage in the *feature matrix*. It
does not prevent it downstream. `HistGradientBoostingRegressor` selects its
internal validation set at random, and because each timestamp contributes 24
near-identical rows to the long table, a random split places near-duplicates of
validation rows into training. Early stopping then never fires and the model
overfits — raising `max_iter` from 600 to 2000 left `n_iter_` pinned at the
ceiling while test MASE degraded from 0.732 to 0.756.

Internal early stopping is therefore disabled, and `select_n_iter` chooses the
iteration count on a chronological holdout of the final 15% of *timestamps*,
using `staged_predict` to score every iteration from a single fit. It selects 28
iterations; test MASE improves to 0.709.

The lesson is worth stating plainly: structural guarantees on feature
construction do not extend to how a library partitions rows.

### SARIMAX convergence

statsmodels defaults to `maxiter=50`, which terminates before convergence on
this series — the seasonal AR coefficient of 0.987 is close to a unit root and
flattens the likelihood surface. `config.SARIMAX_MAXITER = 500` converges
reliably. A stability check across optimiser configurations shows ARMA terms
identical to four decimal places while weather coefficients vary by more than
their own magnitude, reflecting collinearity between `T_out` and `Tdewpoint`.

### Reproducibility

- Seeds fixed in `config.RANDOM_STATE` and applied to NumPy and the estimator.
- Raw data cached; the pipeline runs offline once the cache exists.
- Gap interpolation is performed separately either side of the train/test
  boundary so that training-period gaps are never filled using post-split data.
- Model selection uses `TimeSeriesSplit` on the training sample only.
- SARIMAX convergence warnings are captured and printed rather than suppressed.

---

## Outputs

`outputs/forecasts/all_forecasts.csv` — actuals and one column per model on the
336-hour test index.

`outputs/metrics/model_comparison.csv` — MAE, RMSE, MASE and Bias per model,
sorted by MASE.

`outputs/metrics/error_by_lead_time.csv` — MAE at each lead time 1–24,
exposing error growth that aggregate metrics conceal.

`outputs/figures/` — series overview, forecast comparison window, per-model
panel, error-by-lead-time curve, residual diagnostics, permutation importance.

---

## Verification

```bash
python scripts/verify.py    # 24 automated pre-submission checks
```

Checks repository structure, that tests pass, that outputs exist and are
internally consistent, that no draft placeholders remain, and that every metric
in `model_comparison.csv` appears in the report. `CHECKLIST.md` covers the
reproducibility and judgement checks that cannot be automated.

## Tests

```bash
pytest              # 37 tests
pytest -v           # per-test detail
```

Coverage:

- **`test_features.py`** — lag anchoring against a positional index; invariance
  of training features to perturbed test observations; rolling windows ending
  strictly before the origin; origin versus future exogenous handling; cyclic
  encodings on the unit circle; completeness of the design matrix.
- **`test_evaluation.py`** — MASE zero under a perfect forecast, exactly one
  when MAE equals the seasonal scale, invariant to rescaling; RMSE bounded
  below by MAE and more sensitive to isolated large errors; Bias sign
  convention; forecast length matching the test period.
- **`test_benchmarks.py`** — each benchmark against its closed form; blocks
  never observing their own actuals; later blocks reacting to released
  observations; ragged final block handling.

---

## Known limitations

- Weather is not forecast. `sarimax_operational` uses persistence as a proxy,
  which understates what a numerical weather prediction service would supply.
- The test period is a single fortnight. Metric differences between adjacent
  models should not be over-read without a longer backtest or a Diebold–Mariano
  test.
- The dataset covers one dwelling over roughly 4.5 months. Nothing here
  generalises across households without re-estimation.
- Chronos is used zero-shot and univariate; it never sees covariates.

---

## References

Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017) 'Data driven prediction
models of energy use of appliances in a low-energy house', *Energy and
Buildings*, 140, pp. 81–97.

Hyndman, R.J. and Koehler, A.B. (2006) 'Another look at measures of forecast
accuracy', *International Journal of Forecasting*, 22(4), pp. 679–688.

Ansari, A.F. et al. (2024) 'Chronos: Learning the Language of Time Series',
*Transactions on Machine Learning Research*.
