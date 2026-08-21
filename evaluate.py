"""
Evaluate the GRU medium-term forecaster against the ARIMA baseline.

Reproduces the paper's Table 3 comparison methodology (Section 3.3):
different forecasting methods scored on the same held-out horizon using
MAPE (plus MAE / RMSE / R^2 here for extra diagnostic detail).

Test protocol:
  - The last FORECAST_HORIZON months of the dataset are held out entirely.
  - GRU: seeded with all data up to the holdout, then rolled forward
    recursively for FORECAST_HORIZON months (models/gru_forecaster.py).
  - ARIMA: fit on the same training portion (with auto (p,d,q) grid
    search), then forecast FORECAST_HORIZON steps natively.
  - Both are compared against the true held-out consumption values.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from train import train_model
from models.gru_forecaster import recursive_forecast
from models.arima_baseline import ARIMABaseline
from evaluation.metrics import compute_all_metrics, print_metrics_table


def evaluate_all(verbose: bool = True):
    config.set_seed()

    # ---- Train / load the GRU model -----------------------------------
    model, history, data, device = train_model(verbose=verbose)

    raw_df = data["raw_df"]
    horizon = config.FORECAST_HORIZON
    seed_history = raw_df.iloc[:-horizon].copy()
    holdout = raw_df.iloc[-horizon:].copy().reset_index(drop=True)
    y_true = holdout["consumption"].values

    # ---- GRU: recursive multi-step forecast ----------------------------
    gru_forecast_df = recursive_forecast(
        model, seed_history, data["feature_scaler"], data["target_scaler"], device,
        horizon=horizon,
    )
    gru_preds = gru_forecast_df["forecast"].values

    # ---- ARIMA baseline: native multi-step forecast --------------------
    series = pd.Series(
        raw_df["consumption"].values, index=pd.DatetimeIndex(raw_df["date"])
    ).asfreq("MS")
    train_series = series.iloc[:-horizon]

    arima = ARIMABaseline(auto_order=True)
    arima.fit(train_series)
    arima_preds = arima.forecast(steps=horizon)
    if verbose:
        print(f"\nARIMA baseline order: {arima.summary()}")

    # ---- Naive seasonal baseline (same month, previous year) -----------
    # A minimal sanity baseline: is the model better than "same as last year"?
    if len(raw_df) >= 2 * horizon:
        naive_preds = raw_df["consumption"].values[-2 * horizon: -horizon]
    else:
        naive_preds = np.full(horizon, train_series.mean())

    # ---- Metrics ---------------------------------------------------------
    results = {
        "GRU (wavelet-decomposed features)": compute_all_metrics(y_true, gru_preds),
        "ARIMA (SARIMA baseline)": compute_all_metrics(y_true, arima_preds),
        "Naive seasonal (persistence)": compute_all_metrics(y_true, naive_preds),
    }

    if verbose:
        print_metrics_table(results, title="Medium-Term Forecast Comparison "
                                            f"(last {horizon} months held out)")

    # ---- Save results ------------------------------------------------
    results_df = pd.DataFrame(
        {
            "date": holdout["date"].values,
            "actual": y_true,
            "gru_forecast": gru_preds,
            "arima_forecast": arima_preds,
            "naive_forecast": naive_preds,
        }
    )
    results_df.to_csv(config.RESULTS_CSV_PATH, index=False)

    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nPer-forecast results saved to: {config.RESULTS_CSV_PATH}")
        print(f"Metrics summary saved to:      {config.METRICS_JSON_PATH}")

    return results, results_df


if __name__ == "__main__":
    results, results_df = evaluate_all(verbose=True)

    gru_mape = results["GRU (wavelet-decomposed features)"]["MAPE"]
    arima_mape = results["ARIMA (SARIMA baseline)"]["MAPE"]
    naive_mape = results["Naive seasonal (persistence)"]["MAPE"]

    print(f"\nSummary: GRU MAPE={gru_mape:.2f}% | "
          f"ARIMA MAPE={arima_mape:.2f}% | Naive MAPE={naive_mape:.2f}%")

    assert not np.isnan(gru_mape) and not np.isnan(arima_mape)
    print("Evaluation script sanity check passed.")
