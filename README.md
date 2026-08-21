# Medium-Term Electricity Consumption Forecasting with GRU

A modular Python implementation of medium-term (1-month to 1-year horizon)
electricity consumption forecasting, built from the methodology described in:

> Klyuev, R.V.; Morgoev, I.D.; Morgoeva, A.D.; Gavrina, O.A.; Martyushev, N.V.;
> Efremenkov, E.A.; Mengxu, Q. **"Methods of Forecasting Electric Energy
> Consumption: A Literature Review."** *Energies* 2022, 15, 8919.
> https://doi.org/10.3390/en15238919

This is a **literature review**, not a paper proposing one specific novel
algorithm. Section 3.3 ("Medium-Term Forecasting") of the paper describes
the dominant recipe used across the reviewed studies for the 1-month–1-year
horizon:

1. **Decompose** the raw monthly series into trend / periodic / stochastic
   components with a **discrete wavelet transform** before modeling
   (paper's Ref. [6]: Meng, Niu, Sun, *Energies* 2011).
2. Feed the decomposed + lagged + calendar + weather features into a
   **neural network** (the paper uses ANN/FNN, Ref. [26]; this project uses
   a **GRU**, a more modern recurrent architecture, as requested).
3. Compare against a classical **ARIMA/SARIMA** baseline (Table 3 in the
   paper reports ARIMA at 15.92% MAPE vs. gradient boosting at 7.95% MAPE
   for a small industrial enterprise).
4. **Evaluate with MAPE** (the metric used throughout the paper's Table 3),
   plus MAE / RMSE / R² here for extra diagnostics.

This implementation follows that recipe end-to-end.

## Project structure

```
gru_power_forecasting/
├── config.py                          # All hyperparameters & paths
├── main.py                            # CLI entry point (run this)
├── train.py                           # GRU training loop (early stopping)
├── evaluate.py                        # GRU vs ARIMA vs naive comparison
├── data/
│   ├── synthetic_data_generator.py    # Synthetic monthly dataset (+ real-data loader)
│   └── monthly_power_consumption.csv  # Cached generated dataset (created on first run)
├── preprocessing/
│   ├── wavelet_decomposition.py       # DWT trend/detail/denoising (Section 3.3)
│   └── feature_engineering.py         # Lag windows, calendar, rolling min/max
├── models/
│   ├── gru_model.py                   # GRU architecture (PyTorch)
│   ├── gru_forecaster.py              # Recursive multi-step rollout
│   └── arima_baseline.py              # SARIMA baseline (Table 3 comparison)
├── evaluation/
│   └── metrics.py                     # MAPE / MAE / RMSE / R2 (paper's Table 3 metrics)
├── tests/
│   └── test_pipeline.py               # Unit + end-to-end sanity tests
├── outputs/                            # Model checkpoint, results CSV, metrics JSON
└── requirements.txt
```

## Why no real dataset is bundled

The paper is a review of methods across many studies/datasets, none of
which are distributed with the paper itself. `data/synthetic_data_generator.py`
generates a realistic synthetic monthly consumption series (trend + yearly
seasonality + weather-driven load + noise + occasional shock events) with a
correlated synthetic average-monthly-temperature exogenous series, matching
the structural properties Section 3.3 attributes to real monthly load data.

**To use your own data**, drop a CSV at `data/monthly_power_consumption.csv`
with columns `date, consumption[, avg_temperature]` — `data/synthetic_data_generator.py::load_real_data()`
will pick it up automatically (delete the cached CSV first if a synthetic
one was already generated) and the rest of the pipeline requires no changes.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline (generate data if needed → train GRU → evaluate vs. ARIMA & naive baseline):

```bash
python main.py --cpu
```

Options:
```
--cpu             Force CPU execution
--epochs N        Override number of training epochs (default: 200, early-stopped)
--horizon N        Override forecast horizon in months (default: 12)
--quiet           Suppress verbose logs
```

Run individual stages:

```bash
python data/synthetic_data_generator.py     # generate/inspect the dataset
python preprocessing/wavelet_decomposition.py  # inspect wavelet decomposition
python preprocessing/feature_engineering.py    # inspect windowed train/val/test tensors
python models/gru_model.py                  # GRU architecture shape check
python models/arima_baseline.py             # ARIMA baseline alone
python train.py                             # train the GRU only
python models/gru_forecaster.py             # recursive multi-step GRU forecast
python evaluate.py                          # full GRU vs ARIMA vs naive comparison
```

Run tests:

```bash
python tests/test_pipeline.py
```

## Method details

### Wavelet decomposition (`preprocessing/wavelet_decomposition.py`)
- Daubechies-4 (`db4`) wavelet, 2-level discrete wavelet transform (`pywt`).
- Produces: `wt_trend` (approximation-only reconstruction), `wt_detail_1..L`
  (individual detail bands), `wt_denoised` (soft-thresholded finest-scale
  detail, i.e. noise removed) — mirroring the paper's description of
  decomposing the series into "trend, seasonal, and random (noise)"
  components (Section 3.3).

### Feature set (`preprocessing/feature_engineering.py`)
Following the paper's Ref. [26] model inputs (power consumption for the
previous month, min/max monthly loads, average monthly temperature, day
length, calendar month):
- `consumption` (lagged, autoregressive window of `LOOKBACK_WINDOW=12` months)
- `wt_trend`, `wt_denoised` (wavelet-decomposed features)
- `avg_temperature` (weather driver)
- `month_sin`, `month_cos` (cyclical calendar / day-length proxy)
- `rolling_min`, `rolling_max` (proxy for "minimum and maximum monthly loads")

All features are `MinMaxScaler`-normalized, fit **only on the training
split** to avoid look-ahead leakage. Splits are chronological (70% train /
15% val / 15% test) — never shuffled — because this is a time series task.

### GRU model (`models/gru_model.py`)
- 2-layer GRU, hidden size 64, dropout 0.2, followed by a small MLP head.
- Trained for 1-step-ahead prediction (MSE loss, Adam, gradient clipping,
  `ReduceLROnPlateau`, early stopping on validation loss).
- Medium-term (multi-month) forecasts are produced by **recursive rollout**
  (`models/gru_forecaster.py`): each prediction is fed back into the input
  window and all engineered features (wavelet bands, rolling min/max,
  calendar) are recomputed at every step.

### ARIMA baseline (`models/arima_baseline.py`)
- `statsmodels` SARIMAX with a small grid search over `(p, d, q)` by AIC,
  seasonal order fixed to a 12-month yearly cycle — reproducing the
  paper's Table 3 "Autoregression of the integrated moving average (ARIMA)"
  comparison point.

### Evaluation (`evaluate.py`, `evaluation/metrics.py`)
- Last `FORECAST_HORIZON` (default 12) months held out entirely.
- GRU: recursive multi-step forecast over the holdout.
- ARIMA: native multi-step forecast over the holdout.
- Naive seasonal (persistence: same month, previous year) as a sanity floor.
- All three scored with **MAPE** (headline metric, per the paper's Table 3),
  plus MAE / RMSE / R².
- Results written to `outputs/forecast_results.csv` and `outputs/metrics.json`.

## Notes / limitations

- The bundled dataset is **synthetic**; MAPE numbers are illustrative of the
  pipeline's correctness, not a claim about performance on real utility data.
- Swap in real monthly billing data via `load_real_data()` for a genuine
  benchmark against the paper's reported medium-term MAPE figures
  (Table 3: ARIMA 15.92%, CatBoost gradient boosting 7.95%, perceptron
  13.29% on real industrial/residential datasets).
- The GRU is trained 1-step-ahead and rolled forward recursively for
  medium-term horizons; an alternative (not implemented here) is a
  direct multi-output GRU head that predicts all `FORECAST_HORIZON` months
  simultaneously, trading recursive-rollout error accumulation for reduced
  flexibility in horizon length.
