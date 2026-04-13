"""
ARIMA evaluation helpers: reference forecasts with statsmodels (local, in-process).

Used only for benchmarking / validation against TSLib linear and parallel ARIMA.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from ..metrics.evaluation import ForecastMetrics


def statsmodels_arima_forecast(
    train: np.ndarray,
    order: Tuple[int, int, int],
    horizon: int,
) -> np.ndarray:
    """
    One-step reference: fit statsmodels ARIMA on train and forecast ``horizon`` steps.

    Parameters
    ----------
    train : array
        Training series (1-D, no NaN).
    order : (p, d, q)
        Fixed ARIMA order.
    horizon : int
        Number of out-of-sample steps.

    Returns
    -------
    ndarray
        Forecast vector of length ``horizon`` (or shorter if model fails partially).
    """
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(train, dtype=float).ravel()
    p, d, q = order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y, order=(p, d, q)).fit()
        fc = res.forecast(horizon)
    out = np.asarray(fc, dtype=float).ravel()
    return out


def statsmodels_arima_fit_only(train: np.ndarray, order: Tuple[int, int, int]) -> None:
    """Fit statsmodels ARIMA on full array (for timing parity with TSLib ``.fit``)."""
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(train, dtype=float).ravel()
    p, d, q = order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ARIMA(y, order=(p, d, q)).fit()


def holdout_error_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """RMSE, MAE, MAPE for aligned actual vs predicted."""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    m = min(len(a), len(p))
    if m == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    a, p = a[:m], p[:m]
    return {
        "rmse": float(ForecastMetrics.rmse(a, p)),
        "mae": float(ForecastMetrics.mae(a, p)),
        "mape": float(ForecastMetrics.mape(a, p)),
    }
