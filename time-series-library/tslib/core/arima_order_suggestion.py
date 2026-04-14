"""
ACF/PACF-based suggestion of AR and MA orders for ARIMA grid search.

Uses Bartlett's approximate standard errors for ACF and the usual 1/sqrt(n)
PACF bound at lag k (Box–Jenkins style identification).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats

from .acf_pacf import ACFCalculator, PACFCalculator


def apply_differencing(y: np.ndarray, d: int) -> np.ndarray:
    """Return *y* differenced *d* times (same convention as ARIMA integration order)."""
    x = np.asarray(y, dtype=float)
    for _ in range(max(0, int(d))):
        x = np.diff(x)
    return x


def _bartlett_acf_threshold(acf_values: np.ndarray, n: int, k: int, z: float) -> float:
    """Approximate two-sided critical value for ACF at lag *k* (Bartlett)."""
    if k < 1 or n < 1:
        return float("inf")
    # Var(r_k) ≈ (1 + 2 * sum_{j=1}^{k-1} rho_j^2) / n
    tail = float(np.sum(acf_values[1:k] ** 2)) if k > 1 else 0.0
    var = (1.0 + 2.0 * tail) / float(n)
    return z * float(np.sqrt(max(var, 1e-15)))


def suggest_p_q_orders(
    y: np.ndarray,
    d: int,
    *,
    max_lag: Optional[int] = None,
    alpha: float = 0.05,
    min_order: int = 1,
    max_p_bound: Optional[int] = None,
    max_q_bound: Optional[int] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Suggest maximum *p* and *q* for a grid search from ACF/PACF on the
    *d*-times differenced series (stationary for identification).

    Parameters
    ----------
    y
        Level series (e.g. after log); *d* differences are applied inside.
    d
        ARIMA integration order already chosen (number of differences).
    max_lag
        Maximum lag for ACF/PACF; default ``min(n//4, 40)`` after differencing.
    alpha
        Two-sided significance level for approximate bounds.
    min_order
        Floor when no significant spikes are found (keeps grid non-degenerate).
    max_p_bound, max_q_bound
        Hard caps (e.g. from ``_determine_parameter_ranges``); applied last.

    Returns
    -------
    max_p_suggested, max_q_suggested, metadata
        Integer ceilings for the (p, q) grid and diagnostics (lags, bounds).
    """
    yd = apply_differencing(y, d)
    n = len(yd)
    z = float(stats.norm.ppf(1 - alpha / 2))

    if n < 10:
        mp = max(min_order, 1)
        mq = max(min_order, 1)
        if max_p_bound is not None:
            mp = min(mp, max_p_bound)
        if max_q_bound is not None:
            mq = min(mq, max_q_bound)
        return mp, mq, {
            "note": "short_series_after_diff",
            "n_after_diff": n,
            "significant_acf_lags": [],
            "significant_pacf_lags": [],
        }

    ml = max_lag if max_lag is not None else min(n // 4, 40)
    ml = max(1, min(ml, n - 1))

    acf_calc = ACFCalculator(max_lags=ml)
    _, acf = acf_calc.calculate(yd)
    _, pacf = PACFCalculator(max_lags=ml).calculate(yd)

    se_pacf = z / np.sqrt(float(n))

    sig_acf: list[int] = []
    sig_pacf: list[int] = []
    for k in range(1, min(len(acf), ml + 1)):
        thr = _bartlett_acf_threshold(acf, n, k, z)
        if abs(acf[k]) > thr:
            sig_acf.append(k)
    for k in range(1, min(len(pacf), ml + 1)):
        if abs(pacf[k]) > se_pacf:
            sig_pacf.append(k)

    max_p_raw = max(sig_pacf) if sig_pacf else min_order
    max_q_raw = max(sig_acf) if sig_acf else min_order

    # Structural ceiling: sample-size rule, optional safety cap at 5 when no explicit bound
    structural = max(1, n // 10)
    if max_p_bound is not None:
        structural_p = min(structural, int(max_p_bound))
    else:
        structural_p = min(structural, 5)
    if max_q_bound is not None:
        structural_q = min(structural, int(max_q_bound))
    else:
        structural_q = min(structural, 5)

    mp = int(min(max_p_raw, structural_p))
    mq = int(min(max_q_raw, structural_q))
    mp = max(min_order, mp)
    mq = max(min_order, mq)

    if max_p_bound is not None:
        mp = min(mp, int(max_p_bound))
    if max_q_bound is not None:
        mq = min(mq, int(max_q_bound))

    meta: Dict[str, Any] = {
        "n_after_diff": n,
        "max_lag_used": ml,
        "alpha": alpha,
        "z": z,
        "pacf_se": float(se_pacf),
        "significant_acf_lags": sig_acf,
        "significant_pacf_lags": sig_pacf,
        "acf_tail_for_bartlett": acf[: ml + 1].tolist(),
        "pacf_tail": pacf[: ml + 1].tolist(),
    }
    return mp, mq, meta


# Backward-compatible name from methodology roadmap
suggest_arima_orders_from_acf_pacf = suggest_p_q_orders
