"""
Forecast accuracy metrics.

(and the review generally) evaluates all medium-term
methods using "the magnitude of the average absolute error in percent
(MAPE)" (see the stated objectives in Section 1, item 2). We report MAPE
as the headline metric, plus MAE / RMSE / R^2 as supporting diagnostics.
"""

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error, in percent. Matches the metric used
    throughout the paper's Table 3 comparisons."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAPE": mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics_table(results: dict, title: str = "Model Comparison") -> None:
    """Pretty-print a {model_name: {metric: value}} dict as an aligned table."""
    if not results:
        print("No results to display.")
        return
    metric_names = list(next(iter(results.values())).keys())
    model_names = list(results.keys())

    col_width = max(12, max(len(m) for m in model_names) + 2)
    header = f"{'Model':<{col_width}}" + "".join(f"{m:>12}" for m in metric_names)
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name in model_names:
        row = f"{model_name:<{col_width}}"
        for metric_name in metric_names:
            val = results[model_name][metric_name]
            row += f"{val:>12.3f}"
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 320.0, 380.0])

    m = compute_all_metrics(y_true, y_pred)
    print("Metrics:", m)
    assert abs(m["MAPE"] - np.mean([10, 5, 6.6667, 5]) ) < 0.5
    print_metrics_table({"dummy_model": m})
    print("Metrics module sanity check passed.")
