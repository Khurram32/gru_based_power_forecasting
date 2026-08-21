"""
Configuration for medium-term electricity consumption forecasting.

Design basis: Klyuev et al., "Methods of Forecasting Electric Energy
Consumption: A Literature Review", Energies 2022, 15, 8919.

Medium-term forecasting is defined as a
forecast horizon "from a month to a year".(Medium-Term
Forecasting) recommends:
  - Decomposing the raw monthly series into trend / periodic / stochastic
    (noise) components via wavelet transform before modeling (Ref. [6]
    in the paper: Meng, Niu, Sun, "Forecasting Monthly Electric Energy
    Consumption Using Feature Extraction", Energies 2011).
  - Using neural-network based models (ANN / FNN) fed with lagged
    consumption, min/max monthly load, temperature, day length, and
    calendar features (Ref. [26]).
  - Comparing against classical statistical baselines such as ARIMA
    (Table 3 reports ARIMA vs. gradient boosting vs. perceptron for
    monthly forecasting).
  - Evaluating forecast accuracy with MAPE (used throughout Table 3).

This implementation follows that recipe: wavelet decomposition -> GRU
neural network (a natural, more modern substitute for the paper's ANN/FNN
recurrent-style models, consistent with the paper's broader observation
that deep learning / RNN-family models are the dominant modern approach)
-> comparison against an ARIMA baseline -> MAPE-based evaluation.
"""

import os
import random
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "monthly_power_consumption.csv")
MODEL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "gru_model.pt")
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "forecast_results.csv")
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Data / horizon settings (Section 2: medium-term = 1 month .. 1 year)
# ---------------------------------------------------------------------------
FREQUENCY = "M"                 # monthly series (medium-term horizon)
N_SYNTHETIC_YEARS = 12          # length of synthetic dataset if no CSV given
FORECAST_HORIZON = 12           # predict up to 12 months ahead (medium-term)
LOOKBACK_WINDOW = 12            # months of history fed to the model
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# remaining ~0.15 goes to test

# ---------------------------------------------------------------------------
# Wavelet decomposition (Section 3.3, Ref. [6], [65])
# ---------------------------------------------------------------------------
WAVELET_NAME = "db4"            # Daubechies-4, standard choice for load series
WAVELET_LEVEL = 2               # decomposition level -> trend + 2 detail bands
WAVELET_MODE = "periodization"

# ---------------------------------------------------------------------------
# Feature engineering (Ref. [26]: previous month load, min/max monthly load,
# average monthly temperature, day length, holiday counts, calendar month)
# ---------------------------------------------------------------------------
USE_CALENDAR_FEATURES = True    # month-of-year sin/cos encoding
USE_TEMPERATURE_FEATURE = True  # synthetic/exogenous avg monthly temperature
USE_LAGGED_MINMAX = True        # previous month min/max load

# ---------------------------------------------------------------------------
# GRU model hyperparameters
# ---------------------------------------------------------------------------
GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.2
GRU_BIDIRECTIONAL = False

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
GRAD_CLIP_NORM = 1.0
DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"

# ---------------------------------------------------------------------------
# ARIMA baseline (Section 3.3 / Table 3)
# ---------------------------------------------------------------------------
ARIMA_ORDER = (2, 1, 2)         # (p, d, q); grid-searched in arima_baseline.py
ARIMA_SEASONAL_ORDER = (1, 1, 1, 12)  # yearly seasonality for monthly data

# ---------------------------------------------------------------------------
# Evaluation (MAPE is used throughout Table 3 of the paper)
# ---------------------------------------------------------------------------
METRICS = ["MAPE", "MAE", "RMSE", "R2"]
