"""
Multi-step (medium-term, up to FORECAST_HORIZON months) recursive rollout
for the GRU model.

The GRU is trained for 1-step-ahead prediction (train.py). To produce a
medium-term forecast (the paper's horizon: "from a month to a year",
Section 2), we recursively feed each prediction back into the input window,
re-deriving the engineered features (wavelet bands, rolling min/max,
calendar encoding, temperature) at each step -- the standard recursive
multi-step strategy for RNN-based load forecasters.

Note on temperature: since future avg_temperature is not known at
inference time in a real deployment, we use the historical monthly
seasonal-mean temperature as a naive forecast input (a common, simple
proxy consistent with the paper's Ref. [26] approach of supplying
"forecast values of temperature ... for the forecast month").
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.wavelet_decomposition import wavelet_decompose
from preprocessing.feature_engineering import FEATURE_COLUMNS


def _seasonal_mean_temperature(history_df: pd.DataFrame) -> dict:
    """Map month-of-year (1-12) -> historical mean avg_temperature, used as
    a naive future-temperature forecast during recursive rollout."""
    return history_df.groupby(history_df["date"].dt.month)["avg_temperature"].mean().to_dict()


def recursive_forecast(
    model: torch.nn.Module,
    history_df: pd.DataFrame,
    feature_scaler,
    target_scaler,
    device: torch.device,
    horizon: int = config.FORECAST_HORIZON,
    lookback: int = config.LOOKBACK_WINDOW,
) -> pd.DataFrame:
    """Produce a `horizon`-month-ahead recursive forecast.

    Parameters
    ----------
    history_df : DataFrame with at least the last `lookback` rows containing
        ['date', 'consumption', 'avg_temperature'] (already chronologically
        sorted), used as the seed window.

    Returns
    -------
    DataFrame with columns ['date', 'forecast'] for the `horizon` future
    months immediately following the last date in `history_df`.
    """
    model.eval()
    working_df = history_df.copy().sort_values("date").reset_index(drop=True)
    temp_lookup = _seasonal_mean_temperature(working_df)

    last_date = working_df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )

    forecasts = []

    for step, forecast_date in enumerate(future_dates):
        # Recompute engineered features on the *current* working series
        # (which includes previously forecast months), so wavelet /
        # rolling-min-max features stay consistent as we roll forward.
        decomposition = wavelet_decompose(working_df["consumption"].values)
        trend = decomposition["trend"]
        denoised = decomposition["denoised"]

        month_of_year = working_df["date"].dt.month.values
        month_sin = np.sin(2 * np.pi * (month_of_year - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month_of_year - 1) / 12)

        window = lookback
        rolling_min = (
            working_df["consumption"].shift(1).rolling(window=window, min_periods=1).min().bfill()
        )
        rolling_max = (
            working_df["consumption"].shift(1).rolling(window=window, min_periods=1).max().bfill()
        )

        feat_df = pd.DataFrame(
            {
                "consumption": working_df["consumption"].values,
                "wt_trend": trend,
                "wt_denoised": denoised,
                "avg_temperature": working_df["avg_temperature"].values,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "rolling_min": rolling_min.values,
                "rolling_max": rolling_max.values,
            }
        )

        last_window = feat_df[FEATURE_COLUMNS].values[-lookback:]
        last_window_scaled = feature_scaler.transform(last_window)
        X = torch.from_numpy(last_window_scaled.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(X).cpu().numpy().flatten()[0]

        pred = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        forecasts.append(pred)

        # Append the prediction (with a naive seasonal-mean temperature) to
        # the working series so the next step's window includes it.
        next_temp = temp_lookup.get(forecast_date.month, working_df["avg_temperature"].mean())
        working_df = pd.concat(
            [
                working_df,
                pd.DataFrame(
                    {
                        "date": [forecast_date],
                        "consumption": [pred],
                        "avg_temperature": [next_temp],
                    }
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame({"date": future_dates, "forecast": forecasts})


if __name__ == "__main__":
    from train import train_model

    model, history, data, device = train_model(verbose=False)

    seed_history = data["raw_df"].iloc[: -config.FORECAST_HORIZON].copy()
    ground_truth = data["raw_df"].iloc[-config.FORECAST_HORIZON:].copy()

    forecast_df = recursive_forecast(
        model, seed_history, data["feature_scaler"], data["target_scaler"], device
    )

    print(forecast_df)
    assert len(forecast_df) == config.FORECAST_HORIZON
    assert forecast_df["forecast"].notna().all()
    print("\nRecursive multi-step GRU forecaster sanity check passed.")
