"""
Synthetic monthly electricity consumption generator.

No public dataset was supplied with this project, so this module generates
a realistic synthetic monthly power-consumption series that reproduces the
structural properties the paper says a medium-term series has (Section 3.3):

    "monthly data on electricity consumption are a complex aggregate
    indicator ... [decomposable into] trend, seasonal, and random (noise)
    [components]"

The series is built as:  trend + yearly-seasonal + weather-driven + noise
plus a correlated synthetic average-monthly-temperature exogenous series
(used as a feature, following Ref. [26] in the paper, which feeds average
monthly temperature into the medium-term model).

Swap `load_real_data()` for your own loader (e.g. read a utility's monthly
billing CSV) -- the rest of the pipeline only requires a DataFrame with
columns ['date', 'consumption', 'avg_temperature'].
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_synthetic_monthly_series(
    n_years: int = config.N_SYNTHETIC_YEARS,
    base_load: float = 5000.0,
    trend_slope: float = 35.0,
    yearly_amplitude: float = 900.0,
    noise_std: float = 120.0,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Generate a synthetic monthly power-consumption dataset.

    Components:
      - trend: slow linear + mild quadratic growth (load growth over years)
      - seasonal: yearly cycle (higher in winter/summer due to heating/cooling)
      - temperature: synthetic exogenous series, anti-correlated with heating
        season and correlated with the seasonal load bump (as in real grids)
      - noise: Gaussian white noise (stochastic residual, Section 3.3)

    Returns
    -------
    pd.DataFrame with columns ['date', 'consumption', 'avg_temperature']
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * 12
    dates = pd.date_range(start="2013-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)

    # Trend component (Section 3.3: "trend" component of the decomposition)
    trend = base_load + trend_slope * t + 0.015 * t ** 1.5

    # Seasonal / periodic component (yearly cycle, two peaks: winter heating +
    # summer cooling), matches "periodic component" in Section 3.3
    month_of_year = dates.month.values
    seasonal = (
        yearly_amplitude * np.cos(2 * np.pi * (month_of_year - 1) / 12)
        + 0.4 * yearly_amplitude * np.cos(4 * np.pi * (month_of_year - 1) / 12)
    )

    # Weather / temperature exogenous series (Ref. [26]: avg. monthly temp.
    # is a model input). Roughly sinusoidal, warmest in July, coldest in Jan
    # (northern-hemisphere convention), plus small inter-annual variability.
    avg_temperature = (
        12.0
        - 15.0 * np.cos(2 * np.pi * (month_of_year - 1) / 12)
        + rng.normal(0, 1.5, size=n_months)
    )
    # Extra load contribution from temperature extremes (U-shaped: heating
    # AND cooling both raise consumption)
    temp_effect = 6.0 * (avg_temperature - 12.0) ** 2 / 15.0

    # Stochastic / noise component
    noise = rng.normal(0, noise_std, size=n_months)

    # A handful of irregular events (e.g. an industrial shutdown / demand
    # spike) to make the series more realistic and harder to model perfectly
    n_events = max(1, n_years // 4)
    event_idx = rng.choice(n_months, size=n_events, replace=False)
    events = np.zeros(n_months)
    events[event_idx] = rng.normal(0, 3 * noise_std, size=n_events)

    consumption = trend + seasonal + temp_effect + noise + events
    consumption = np.clip(consumption, a_min=base_load * 0.3, a_max=None)

    df = pd.DataFrame(
        {
            "date": dates,
            "consumption": consumption,
            "avg_temperature": avg_temperature,
        }
    )
    return df


def load_real_data(csv_path: str) -> pd.DataFrame:
    """Load a real monthly consumption CSV.

    Expected columns: 'date' (parseable), 'consumption', and optionally
    'avg_temperature'. If temperature is missing, it is filled with a
    smooth synthetic seasonal proxy so the pipeline still runs end-to-end.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "avg_temperature" not in df.columns:
        month_of_year = df["date"].dt.month.values
        df["avg_temperature"] = 12.0 - 15.0 * np.cos(
            2 * np.pi * (month_of_year - 1) / 12
        )
    return df


def get_dataset(csv_path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Entry point used by the rest of the pipeline.

    If a real CSV exists at `csv_path`, it is loaded; otherwise a synthetic
    dataset is generated and cached to disk for reproducibility.
    """
    if os.path.exists(csv_path):
        return load_real_data(csv_path)

    df = generate_synthetic_monthly_series()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    data = get_dataset()
    print(data.head(12))
    print(f"\nGenerated {len(data)} monthly records "
          f"({data['date'].min().date()} to {data['date'].max().date()})")
    print(f"Saved to: {config.RAW_DATA_PATH}")
