"""
Main entry point: run the full medium-term electricity-consumption
forecasting pipeline end-to-end.

    python main.py               # generate data (if missing) -> train GRU
                                  # -> evaluate GRU vs ARIMA vs naive

Usage as a library:
    from evaluate import evaluate_all
    results, results_df = evaluate_all()
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    parser = argparse.ArgumentParser(
        description="Medium-term electricity consumption forecasting "
                     "(GRU + wavelet decomposition, vs. ARIMA baseline). "
                     "Based on: Klyuev et al., Energies 2022, 15, 8919."
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU execution even if a GPU is available.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Override number of training epochs (default: {config.NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help=f"Override forecast horizon in months (default: {config.FORECAST_HORIZON}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress training/eval logs.",
    )
    args = parser.parse_args()

    if args.cpu:
        os.environ["FORCE_CPU"] = "1"
        config.DEVICE = "cpu"
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.horizon is not None:
        config.FORECAST_HORIZON = args.horizon

    from evaluate import evaluate_all

    print("=" * 80)
    print("Medium-Term Electricity Consumption Forecasting")
    print("GRU (wavelet-decomposed features) vs. ARIMA baseline vs. naive seasonal")
    print("Reference: Klyuev, R.V. et al. 'Methods of Forecasting Electric Energy")
    print("Consumption: A Literature Review.' Energies 2022, 15, 8919.")
    print("=" * 80)

    results, results_df = evaluate_all(verbose=not args.quiet)

    print("\nDone. Outputs written to:", config.OUTPUT_DIR)
    return results, results_df


if __name__ == "__main__":
    main()
