"""
Feature engineering + windowing for the GRU medium-term forecaster.

The following data are fed to the model inputs: power consumption for
the previous month, minimum and maximum monthly loads, average monthly
temperature, day length, the number of holidays in the month preceding
the forecast, ... and forecast values of temperature ... for the
forecast month."

We reproduce the spirit of this feature set with what is derivable from a
monthly consumption + temperature series:
  - lagged consumption (autoregressive window, LOOKBACK_WINDOW months)
  - wavelet trend / denoised / detail bands (Section 3.3 preprocessing)
  - rolling min/max of the lookback window (proxy for "min/max monthly load")
  - calendar month encoded as sin/cos (proxy for day-length / seasonality)
  - average monthly temperature (weather driver)

All continuous features are scaled with a MinMaxScaler fit only on the
training split to avoid look-ahead leakage.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.wavelet_decomposition import add_wavelet_features


FEATURE_COLUMNS = [
    "consumption",
    "wt_trend",
    "wt_denoised",
    "avg_temperature",
    "month_sin",
    "month_cos",
    "rolling_min",
    "rolling_max",
]
TARGET_COLUMN = "consumption"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, rolling min/max, and wavelet features to the raw df."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Wavelet-decomposed features (Section 3.3 preprocessing step)
    df = add_wavelet_features(df, target_col="consumption")

    # Calendar encoding as a smooth cyclical proxy for "day length" /
    # seasonality mentioned in Ref. [26]
    month_of_year = df["date"].dt.month.values
    df["month_sin"] = np.sin(2 * np.pi * (month_of_year - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month_of_year - 1) / 12)

    # Rolling min/max monthly load over the lookback window (proxy for
    # "minimum and maximum monthly loads" input in Ref. [26])
    window = config.LOOKBACK_WINDOW
    df["rolling_min"] = (
        df["consumption"].shift(1).rolling(window=window, min_periods=1).min()
    )
    df["rolling_max"] = (
        df["consumption"].shift(1).rolling(window=window, min_periods=1).max()
    )
    df["rolling_min"] = df["rolling_min"].bfill()
    df["rolling_max"] = df["rolling_max"].bfill()

    return df


def chronological_split(df: pd.DataFrame):
    """Split a time-ordered dataframe into train/val/test without shuffling,
    respecting the time series nature of medium-term forecasting."""
    n = len(df)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train: n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train_df, val_df, test_df


def fit_scalers(train_df: pd.DataFrame):
    """Fit one MinMaxScaler over the feature columns and one over the target,
    using ONLY the training split (no leakage)."""
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURE_COLUMNS].values)

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[TARGET_COLUMN]].values)

    return feature_scaler, target_scaler


def make_windows(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    lookback: int = config.LOOKBACK_WINDOW,
):
    """Turn a (scaled) dataframe into supervised (X, y) sliding windows.

    X[i] shape: (lookback, n_features)
    y[i] shape: scalar -> next month's consumption (1-step ahead target,
                 used for the standard GRU training loop; multi-step
                 12-month-ahead forecasts are produced by iterative /
                 recursive rollout at inference time, see gru_forecaster.py)
    """
    features_scaled = feature_scaler.transform(df[FEATURE_COLUMNS].values)
    target_scaled = target_scaler.transform(df[[TARGET_COLUMN]].values).flatten()

    X, y, dates = [], [], []
    for i in range(lookback, len(df)):
        X.append(features_scaled[i - lookback:i])
        y.append(target_scaled[i])
        dates.append(df["date"].iloc[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, dates


def prepare_data(csv_path: str = config.RAW_DATA_PATH):
    """Full pipeline: load -> engineer features -> split -> scale -> window.

    Returns a dict with numpy arrays for train/val/test plus the fitted
    scalers (needed to invert predictions back to physical units, i.e. MWh).
    """
    from data.synthetic_data_generator import get_dataset

    raw_df = get_dataset(csv_path)
    feat_df = build_feature_frame(raw_df)

    train_df, val_df, test_df = chronological_split(feat_df)
    feature_scaler, target_scaler = fit_scalers(train_df)

    # For val/test windows we need `lookback` months of history from the
    # preceding split so the first window isn't wasted; we simply window
    # over the full feature frame and then slice indices back into splits.
    X_all, y_all, dates_all = make_windows(feat_df, feature_scaler, target_scaler)

    n = len(feat_df)
    lookback = config.LOOKBACK_WINDOW
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    # Window index i corresponds to feat_df row (i + lookback)
    train_end_idx = max(0, n_train - lookback)
    val_end_idx = max(0, n_train + n_val - lookback)

    X_train, y_train = X_all[:train_end_idx], y_all[:train_end_idx]
    X_val, y_val = X_all[train_end_idx:val_end_idx], y_all[train_end_idx:val_end_idx]
    X_test, y_test = X_all[val_end_idx:], y_all[val_end_idx:]
    dates_test = dates_all[val_end_idx:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "dates_test": dates_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feat_df": feat_df,
        "raw_df": raw_df,
        "n_features": len(FEATURE_COLUMNS),
    }


if __name__ == "__main__":
    data = prepare_data()
    print("n_features:", data["n_features"])
    print("X_train:", data["X_train"].shape, "y_train:", data["y_train"].shape)
    print("X_val:  ", data["X_val"].shape, "  y_val:  ", data["y_val"].shape)
    print("X_test: ", data["X_test"].shape, "  y_test: ", data["y_test"].shape)
    assert data["X_train"].shape[1] == config.LOOKBACK_WINDOW
    assert data["X_train"].shape[2] == data["n_features"]
    print("Feature engineering sanity checks passed.")
