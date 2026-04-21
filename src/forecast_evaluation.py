"""
forecast_evaluation.py — Forecast Evaluation & Backtesting Module

Reusable functions for computing MASE and running expanding-window
backtests on time series forecasting models.

Author: [Yun Deng]
Course: ECON 5200, Lab 21
"""

import numpy as np
import pandas as pd
from typing import Callable


def compute_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    insample: np.ndarray,
    m: int = 1
) -> float:
    """Compute Mean Absolute Scaled Error.
    
    MASE < 1: model beats naive seasonal benchmark.
    MASE > 1: naive benchmark is better.
    
    Args:
        actual: True out-of-sample values
        forecast: Model predictions (same length as actual)
        insample: In-sample (training) data for naive baseline
        m: Seasonal period (1=random walk, 12=monthly seasonal)
    
    Returns:
        MASE score (float)
    """
    # YOUR IMPLEMENTATION HERE
    # Hint:
    # mae_forecast = np.mean(np.abs(actual - forecast))
    # naive_errors = insample[m:] - insample[:-m]
    # mae_naive = np.mean(np.abs(naive_errors))
    # return mae_forecast / mae_naive
    actual   = np.asarray(actual)
    forecast = np.asarray(forecast)
    insample = np.asarray(insample)

    mae_forecast = np.mean(np.abs(actual - forecast))
    naive_errors = insample[m:] - insample[:-m]
    mae_naive    = np.mean(np.abs(naive_errors))

    if mae_naive == 0:
        raise ValueError("Naive MAE is zero — check insample data or m parameter.")

    return mae_forecast / mae_naive


def backtest_expanding_window(
    series: pd.Series,
    model_fn: Callable,
    min_train: int = 120,
    horizon: int = 12,
    step: int = 12
) -> pd.DataFrame:
    """Expanding-window time series backtest.
    
    Args:
        series: Full series with DatetimeIndex
        model_fn: Callable(train) -> np.ndarray of length horizon
        min_train: Minimum training observations
        horizon: Forecast horizon per iteration
        step: Observations added per iteration
    
    Returns:
        DataFrame with backtest results
    """
    # YOUR IMPLEMENTATION HERE
    # Hint: loop from min_train to len(series)-horizon, stepping by step
    # For each origin:
    #   train = series[:origin]
    #   actual = series[origin:origin+horizon].values
    #   forecast = model_fn(train)
    #   compute errors and MASE
    results = []

    for origin in range(min_train, len(series) - horizon + 1, step):
        train  = series.iloc[:origin]
        actual = series.iloc[origin:origin + horizon].values

        try:
            forecast = model_fn(train)
        except Exception as e:
            print(f"Model failed at origin={origin}: {e}")
            continue

        errors     = actual - forecast
        abs_errors = np.abs(errors)

        try:
            mase_val = compute_mase(actual, forecast, train.values, m=12)
        except Exception:
            mase_val = np.nan

        for h in range(horizon):
            results.append({
                'origin':    origin,
                'horizon':   h + 1,
                'actual':    actual[h],
                'forecast':  forecast[h],
                'error':     errors[h],
                'abs_error': abs_errors[h],
                'mase':      mase_val
            })

    return pd.DataFrame(results)


# --- Quick self-test ---
if __name__ == '__main__':
    print('forecast_evaluation.py loaded successfully.')
    actual   = np.array([100, 102, 104, 106], dtype=float)
    forecast = np.array([101, 103, 103, 107], dtype=float)
    insample = np.arange(1, 101, dtype=float)
    mase = compute_mase(actual, forecast, insample, m=1)
    print(f'Test MASE: {mase:.4f}')
    print('Self-test passed.')
