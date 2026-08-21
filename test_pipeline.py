"""
Lightweight test suite (no pytest dependency required -- run directly with
`python tests/test_pipeline.py`).

Covers:
  1. Synthetic data generation shape/columns
  2. Wavelet decomposition reduces variance of first differences (trend
     should be smoother than raw)
  3. Feature engineering produces correctly-shaped train/val/test windows
  4. GRU forward pass shape correctness
  5. Metrics functions correctness against hand-computed values
  6. End-to-end: training runs and produces a finite validation MAPE
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["FORCE_CPU"] = "1"

import config
config.NUM_EPOCHS = 15          # keep tests fast
config.EARLY_STOPPING_PATIENCE = 15

from data.synthetic_data_generator import generate_synthetic_monthly_series
from preprocessing.wavelet_decomposition import wavelet_decompose
from preprocessing.feature_engineering import prepare_data
from models.gru_model import GRUForecaster
from evaluation.metrics import mape, mae, rmse, r2_score


def test_synthetic_data():
    df = generate_synthetic_monthly_series(n_years=5)
    assert len(df) == 60
    assert set(["date", "consumption", "avg_temperature"]).issubset(df.columns)
    assert df["consumption"].min() > 0
    print("[PASS] test_synthetic_data")


def test_wavelet_decomposition():
    df = generate_synthetic_monthly_series(n_years=5)
    decomp = wavelet_decompose(df["consumption"].values)
    assert len(decomp["trend"]) == len(df)
    assert len(decomp["denoised"]) == len(df)
    raw_var = np.var(np.diff(df["consumption"].values))
    trend_var = np.var(np.diff(decomp["trend"]))
    assert trend_var < raw_var
    print("[PASS] test_wavelet_decomposition")


def test_feature_engineering():
    data = prepare_data()
    assert data["X_train"].shape[1] == config.LOOKBACK_WINDOW
    assert data["X_train"].shape[2] == data["n_features"]
    assert data["X_train"].shape[0] == data["y_train"].shape[0]
    assert data["X_val"].shape[0] == data["y_val"].shape[0]
    assert data["X_test"].shape[0] == data["y_test"].shape[0]
    print("[PASS] test_feature_engineering")


def test_gru_forward_pass():
    import torch
    model = GRUForecaster(n_features=8, hidden_size=32, num_layers=1, dropout=0.0)
    x = torch.randn(5, 12, 8)
    out = model(x)
    assert out.shape == (5,)
    print("[PASS] test_gru_forward_pass")


def test_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert mape(y_true, y_pred) == 0.0
    assert mae(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0
    assert r2_score(y_true, y_pred) == 1.0
    print("[PASS] test_metrics")


def test_end_to_end_training():
    from train import train_model
    model, history, data, device = train_model(verbose=False)
    assert len(history["val_loss"]) > 0
    assert np.isfinite(history["val_loss"][-1])
    print("[PASS] test_end_to_end_training")


if __name__ == "__main__":
    test_synthetic_data()
    test_wavelet_decomposition()
    test_feature_engineering()
    test_gru_forward_pass()
    test_metrics()
    test_end_to_end_training()
    print("\nAll tests passed.")
