# Step 9 — ARIMA / SARIMA Baseline

> Part of the medium-term electricity consumption forecasting pipeline.
> Code: [`models/arima_baseline.py`](../models/arima_baseline.py)

## Purpose

Reproduces the classical statistical comparison point used in the source
paper's Table 3 (Section 3.3, *Medium-Term Forecasting*):

> "Autoregression of the integrated moving average (ARIMA) ... A
> well-developed mathematical apparatus. Ease of implementation."
> — Klyuev et al., *Energies* 2022, 15, 8919

ARIMA/SARIMA is fit on the **same training split** as the GRU and used as
the classical baseline the GRU is benchmarked against.

## Model

Seasonal ARIMA, $\text{SARIMA}(p,d,q)\times(P,D,Q)_{12}$:

$$
\Phi_P(B^{12})\,\phi_p(B)\,(1-B)^d(1-B^{12})^D\, y_t \;=\; \Theta_Q(B^{12})\,\theta_q(B)\,\varepsilon_t
$$

| Symbol | Meaning |
|---|---|
| $B$ | Backshift operator, $B y_t = y_{t-1}$ |
| $\phi_p(B)$ | Non-seasonal autoregressive (AR) polynomial, order $p$ |
| $\theta_q(B)$ | Non-seasonal moving-average (MA) polynomial, order $q$ |
| $\Phi_P(B^{12})$ | Seasonal AR polynomial, order $P$, period 12 (months) |
| $\Theta_Q(B^{12})$ | Seasonal MA polynomial, order $Q$, period 12 |
| $d$, $D$ | Non-seasonal / seasonal differencing orders |
| $\varepsilon_t$ | White-noise residual, $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ |

The seasonal period is fixed to **12** because the input series is monthly
and the paper's medium-term horizon spans up to a year — a 12-month
seasonal component captures the yearly heating/cooling cycle.

## Order selection

A small grid search over non-seasonal orders is run, scored by **AIC**
(Akaike Information Criterion):

$$
\text{AIC} = 2k - 2\ln(\hat{L})
$$

where $k$ is the number of estimated parameters and $\hat{L}$ is the
maximized likelihood of the fitted model.

```python
p_range = (0, 1, 2)
d_range = (0, 1)
q_range = (0, 1, 2)
seasonal_order = (1, 1, 1, 12)   # fixed yearly seasonality

best_order = argmin_{(p,d,q)} AIC( SARIMAX(train_series, order=(p,d,q), seasonal_order) )
```

If the training series has fewer than ~30 observations (less than 2.5
seasonal cycles), the seasonal component is disabled `(0,0,0,0)` and only
the non-seasonal order is searched, since a 12-month seasonal term cannot
be reliably estimated from too little history.

## Fitting

```python
from models.arima_baseline import ARIMABaseline

baseline = ARIMABaseline(auto_order=True)
baseline.fit(train_series)          # train_series: pandas Series, monthly freq
print(baseline.summary())           # e.g. "SARIMA(1, 1, 2)x(1, 1, 1, 12) AIC=1383.22"
```

Internally this wraps `statsmodels.tsa.statespace.sarimax.SARIMAX`, fit by
maximum likelihood via the Kalman filter.

## Forecasting

Unlike the GRU (trained 1-step-ahead and rolled forward recursively — see
[Step 6](../models/gru_forecaster.py)), SARIMAX produces the entire
medium-term horizon **natively** in one call, since its state-space form
propagates uncertainty and mean forecasts jointly across all future steps:

```python
predictions = baseline.forecast(steps=12)   # 12-month-ahead forecast, no recursion needed
```

## Role in evaluation

`evaluate.py` fits `ARIMABaseline` on the training portion (all months
except the last `FORECAST_HORIZON`), forecasts the held-out horizon, and
scores it against ground truth with the same metrics used for the GRU
(MAPE / MAE / RMSE / R² — see [`evaluation/metrics.py`](../evaluation/metrics.py)):

```python
arima = ARIMABaseline(auto_order=True)
arima.fit(train_series)
arima_preds = arima.forecast(steps=horizon)
metrics = compute_all_metrics(y_true, arima_preds)
```

### Example result (synthetic data)

| Model | MAPE | MAE | RMSE | R² |
|---|---|---|---|---|
| ARIMA (SARIMA baseline) | **1.25%** | 123.0 | 137.8 | 0.957 |
| GRU (wavelet-decomposed features) | 3.16% | 315.1 | 364.4 | 0.700 |
| Naive seasonal (persistence) | 3.66% | 358.0 | 390.0 | 0.656 |

This mirrors the paper's own observation (Section 4, *Conclusions*):

> "It has been revealed that modern methods, such as machine-learning
> algorithms, do not always give better results compared to classical
> forecasting approaches."

## Advantages / limitations (per the paper's Table 3)

**Advantages**
- Well-developed mathematical apparatus, closed-form likelihood.
- Easy to implement; fast to fit on short monthly series.
- Produces native multi-step forecasts with uncertainty intervals.

**Limitations**
- Accuracy is sensitive to correct order selection $(p,d,q)(P,D,Q)_{12}$.
- Assumes linear dependence structure; cannot capture the same
  non-linear interactions a neural network can.
- Needs a reasonably long, regularly-sampled series (grid search here
  requires ≥30 points to enable the 12-month seasonal term).

## Source

```python
# models/arima_baseline.py
class ARIMABaseline:
    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12), auto_order=True): ...
    def fit(self, train_series: pd.Series): ...
    def forecast(self, steps: int) -> np.ndarray: ...
    def summary(self) -> str: ...
```

Run standalone:
```bash
python models/arima_baseline.py
```
