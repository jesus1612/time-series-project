"""Unit tests for ACF/PACF-based (p,q) order suggestion."""

import numpy as np
import pytest

from tslib.core.arima_order_suggestion import suggest_p_q_orders


def test_white_noise_small_orders():
    rng = np.random.default_rng(42)
    y = rng.standard_normal(400)
    mp, mq, meta = suggest_p_q_orders(y, d=0, alpha=0.05)
    assert 1 <= mp <= 5
    assert 1 <= mq <= 5
    assert "significant_acf_lags" in meta


def test_ar2_process_runs_and_populates_pacf():
    """Synthetic AR(2): identification metadata must be well-formed."""
    rng = np.random.default_rng(7)
    n = 600
    eps = rng.standard_normal(n + 2)
    y = np.zeros(n)
    phi1, phi2 = 0.5, 0.35
    y[0] = eps[2]
    y[1] = phi1 * y[0] + eps[3]
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t + 2]
    mp, mq, meta = suggest_p_q_orders(y, d=0, alpha=0.05)
    assert mp >= 1 and mq >= 1
    assert "pacf_tail" in meta and len(meta["pacf_tail"]) > 2


def test_short_series_returns_bounds():
    y = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    mp, mq, meta = suggest_p_q_orders(y, d=0)
    assert mp >= 1 and mq >= 1
    assert meta.get("note") == "short_series_after_diff"
