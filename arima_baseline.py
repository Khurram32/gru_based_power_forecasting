"""
ARIMA / SARIMA baseline for medium-term electricity consumption forecasting.

"Autoregression of the integrated moving average (ARIMA)" against a perceptron and CatBoost
gradient boosting for monthly forecasting of a small industrial enterprise,
reporting a MAPE of 15.92%. The paper notes ARIMA's advantages ("a
well-developed mathematical apparatus", "ease of implementation") and its
main drawback ("insufficient level of model accuracy due to the complexity
of the selection of model parameters").

This module implements a seasonal ARIMA (SARIMA) baseline used as the
classical-methods comparison point for the GRU model, matching the paper's
own comparison methodology (statistical model vs. deep-learning model,
both scored with MAPE).
"""

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")


def grid_search_sarima(
    train_series: pd.Series,
    p_range=(0, 1, 2),
    d_range=(0, 1),
    q_range=(0, 1, 2),
    seasonal_order=(1, 1, 1, 12),
):
    """Small grid search over (p, d, q) using AIC, seasonal order fixed to
    a yearly cycle (12 months) as appropriate for monthly data.

    Returns the best (p, d, q) order found.
    """
    best_aic = np.inf
    best_order = config.ARIMA_ORDER

    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = SARIMAX(
                train_series,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order, best_aic


class ARIMABaseline:
    """Thin wrapper around statsmodels SARIMAX providing a scikit-learn-like
    fit/predict interface consistent with the rest of this project."""

    def __init__(
        self,
        order=config.ARIMA_ORDER,
        seasonal_order=config.ARIMA_SEASONAL_ORDER,
        auto_order: bool = True,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self.fitted_model = None
        self.train_series = None

    def fit(self, train_series: pd.Series):
        self.train_series = train_series

        if self.auto_order:
            n = len(train_series)
            # Use a light seasonal order search only if there is enough
            # history for at least ~2 full seasonal cycles; otherwise fall
            # back to the non-seasonal component search.
            seasonal = self.seasonal_order if n >= 30 else (0, 0, 0, 0)
            self.order, best_aic = grid_search_sarima(
                train_series, seasonal_order=seasonal
            )
            self.seasonal_order = seasonal

        model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_model = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """Multi-step-ahead forecast (native to ARIMA, unlike the GRU which
        needs a recursive rollout)."""
        if self.fitted_model is None:
            raise RuntimeError("Call .fit() before .forecast().")
        result = self.fitted_model.get_forecast(steps=steps)
        return result.predicted_mean.values

    def summary(self) -> str:
        if self.fitted_model is None:
            return "Model not fitted."
        return (
            f"SARIMA{self.order}x{self.seasonal_order} "
            f"AIC={self.fitted_model.aic:.2f}"
        )


if __name__ == "__main__":
    from data.synthetic_data_generator import get_dataset

    data = get_dataset()
    series = pd.Series(
        data["consumption"].values, index=pd.DatetimeIndex(data["date"])
    )
    series = series.asfreq("MS")

    n_test = config.FORECAST_HORIZON
    train_series, test_series = series[:-n_test], series[-n_test:]

    baseline = ARIMABaseline(auto_order=True)
    baseline.fit(train_series)
    print(baseline.summary())

    preds = baseline.forecast(steps=n_test)
    mape = np.mean(np.abs((test_series.values - preds) / test_series.values)) * 100
    print(f"\nHold-out MAPE over last {n_test} months: {mape:.2f}%")
    assert len(preds) == n_test
    print("ARIMA baseline sanity check passed.")
