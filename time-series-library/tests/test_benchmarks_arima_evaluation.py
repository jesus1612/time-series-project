"""Tests for evaluation-only statsmodels reference helpers."""
import numpy as np
import pytest

from tslib.benchmarks.arima_evaluation import (
    holdout_error_metrics,
    statsmodels_arima_fit_only,
    statsmodels_arima_forecast,
)

try:
    import statsmodels  # noqa: F401

    HAS_SM = True
except ImportError:
    HAS_SM = False


@pytest.mark.skipif(not HAS_SM, reason="statsmodels not installed")
def test_statsmodels_forecast_shape():
    rng = np.random.default_rng(7)
    train = np.cumsum(rng.standard_normal(80)) + 20.0
    fc = statsmodels_arima_forecast(train, (1, 1, 1), horizon=15)
    assert fc.shape == (15,)
    assert np.all(np.isfinite(fc))


@pytest.mark.skipif(not HAS_SM, reason="statsmodels not installed")
def test_statsmodels_fit_only_runs():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(60)) + 10.0
    statsmodels_arima_fit_only(y, (1, 0, 1))


def test_holdout_error_metrics():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.0, 2.9])
    m = holdout_error_metrics(a, p)
    assert "rmse" in m and "mae" in m and "mape" in m
    assert m["mae"] < 0.2
