"""
Wavelet decomposition of the monthly power-consumption series.

    "This approach was based on the decomposition of the original power
    consumption series into a trend, a periodic component, and a
    stochastic series (noise) by applying a discrete wavelet transform.
    After removing the noise, the trend and harmonic series were modeled
    ... As a result of the preprocessing of the series, the generalizing
    ability of the model was significantly improved."

    "for medium-term forecasting, periodic components must be considered.
    To do this, the literature recommends using wavelet analysis ...
    to decompose the original series into additive components: trend,
    seasonal, and random (noise). Such analysis allows extracting
    features, which are then fed to the input of the models."

We use a multi-level discrete wavelet transform (DWT) via PyWavelets:
  - approximation coefficients (level `WAVELET_LEVEL`) -> trend component
  - detail coefficients at each level -> periodic / stochastic components
  - a denoised reconstruction (soft-thresholding the finest detail level,
    which carries most of the noise energy) is used as the model's main
    input series, with the individual band reconstructions supplied as
    extra features.
"""

import os
import sys
import numpy as np
import pandas as pd
import pywt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _pad_to_even(signal: np.ndarray) -> tuple:
    """Pad a 1D signal so its length is compatible with 'periodization' mode
    at the requested decomposition level (needs length divisible by 2**level).
    Returns (padded_signal, original_length)."""
    n = len(signal)
    factor = 2 ** config.WAVELET_LEVEL
    remainder = n % factor
    if remainder == 0:
        return signal, n
    pad_len = factor - remainder
    # reflect-pad at the end to preserve local statistics
    padded = np.pad(signal, (0, pad_len), mode="reflect")
    return padded, n


def wavelet_decompose(
    signal: np.ndarray,
    wavelet: str = config.WAVELET_NAME,
    level: int = config.WAVELET_LEVEL,
    mode: str = config.WAVELET_MODE,
) -> dict:
    """Decompose a 1D signal into trend (approximation) and detail bands.

    Returns a dict with:
      'trend'    : reconstructed approximation-only signal (same length as input)
      'details'  : list of reconstructed single-band detail signals, level 1..L
      'denoised' : trend + all details reconstructed after soft-thresholding
                   the finest-scale detail coefficients (noise removal)
      'coeffs'   : raw pywt coefficient list [cA_L, cD_L, ..., cD_1]
    """
    padded_signal, orig_len = _pad_to_even(np.array(signal, dtype=float, copy=True))

    coeffs = pywt.wavedec(padded_signal, wavelet=wavelet, level=level, mode=mode)
    # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]

    # --- Trend: reconstruct using ONLY the approximation coefficients ---
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(trend_coeffs, wavelet=wavelet, mode=mode)[:len(padded_signal)]

    # --- Individual detail bands (periodic / stochastic components) ---
    details = []
    for i in range(1, len(coeffs)):
        band_coeffs = [np.zeros_like(c) for c in coeffs]
        band_coeffs[i] = coeffs[i]
        band = pywt.waverec(band_coeffs, wavelet=wavelet, mode=mode)[:len(padded_signal)]
        details.append(band)

    # --- Denoising: soft-threshold the finest detail level (highest freq,
    #     carries most measurement/stochastic noise), following the paper's
    #     "after removing the noise, the trend and harmonic series were
    #     modeled" strategy ---
    denoise_coeffs = [c.copy() for c in coeffs]
    finest = denoise_coeffs[-1]
    sigma = np.median(np.abs(finest - np.median(finest))) / 0.6745 + 1e-8
    universal_thresh = sigma * np.sqrt(2 * np.log(max(len(finest), 2)))
    denoise_coeffs[-1] = pywt.threshold(finest, universal_thresh, mode="soft")
    denoised = pywt.waverec(denoise_coeffs, wavelet=wavelet, mode=mode)[:len(padded_signal)]

    # Trim back to original length
    trend = trend[:orig_len]
    details = [d[:orig_len] for d in details]
    denoised = denoised[:orig_len]

    return {
        "trend": trend,
        "details": details,
        "denoised": denoised,
        "coeffs": coeffs,
    }


def add_wavelet_features(df: pd.DataFrame, target_col: str = "consumption") -> pd.DataFrame:
    """Augment a dataframe with wavelet-decomposed features of `target_col`.

    Adds columns: 'wt_trend', 'wt_detail_1' .. 'wt_detail_L', 'wt_denoised'.
    """
    df = df.copy()
    decomposition = wavelet_decompose(df[target_col].values)

    df["wt_trend"] = decomposition["trend"]
    for i, band in enumerate(decomposition["details"], start=1):
        df[f"wt_detail_{i}"] = band
    df["wt_denoised"] = decomposition["denoised"]

    return df


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.synthetic_data_generator import get_dataset

    data = get_dataset()
    decomposed = add_wavelet_features(data)
    print(decomposed[["date", "consumption", "wt_trend", "wt_denoised"]].head(12))
    print("\nColumns added:", [c for c in decomposed.columns if c.startswith("wt_")])

    # Sanity check: denoised MAPE vs raw should be small (denoising shouldn't
    # destroy the signal), and trend should be smoother (lower variance of
    # first differences) than the raw series.
    raw_diff_var = np.var(np.diff(data["consumption"].values))
    trend_diff_var = np.var(np.diff(decomposed["wt_trend"].values))
    print(f"\nRaw first-difference variance:   {raw_diff_var:.2f}")
    print(f"Trend first-difference variance: {trend_diff_var:.2f}")
    assert trend_diff_var < raw_diff_var, "Trend component should be smoother than raw signal"
    print("Sanity check passed: trend is smoother than raw signal.")
