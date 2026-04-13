"""
Shared helpers for benchmark tests (timing, synthetic series).
"""

import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def timed_call(fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def ar_series_dict(n_series: int, n_obs: int, ar_coef: float = 0.5, seed: int = 42) -> Dict[str, np.ndarray]:
    """Stable AR(1)-like series per id for benchmarking."""
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for i in range(n_series):
        eps = rng.normal(0, 0.5, n_obs)
        y = np.zeros(n_obs)
        phi = ar_coef * (1.0 - 0.1 * i)
        for t in range(1, n_obs):
            y[t] = phi * y[t - 1] + eps[t]
        out[f"s{i}"] = y
    return out


def ma_series_dict(n_series: int, n_obs: int, seed: int = 43) -> Dict[str, np.ndarray]:
    """MA(2) synthetic: y_t = eps_t + 0.4 eps_{t-1} + 0.3 eps_{t-2}."""
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for i in range(n_series):
        e = rng.normal(0, 0.8, n_obs + 2)
        y = e[2:] + 0.4 * e[1:-1] + 0.3 * e[:-2]
        out[f"s{i}"] = y.astype(float)
    return out


def arma_series_dict(n_series: int, n_obs: int, seed: int = 44) -> Dict[str, np.ndarray]:
    """Causal ARMA(1,1): y_t = 0.5 y_{t-1} + eps_t + 0.4 eps_{t-1}."""
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for i in range(n_series):
        e = rng.normal(0, 0.5, n_obs)
        y = np.zeros(n_obs)
        y[0] = e[0]
        for t in range(1, n_obs):
            y[t] = 0.5 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        out[f"s{i}"] = y
    return out


def arima_like_series_dict(n_series: int, n_obs: int, seed: int = 45) -> Dict[str, np.ndarray]:
    """Random walk with drift (needs d>=1): y_t = y_{t-1} + 0.05 + noise."""
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for i in range(n_series):
        y = np.cumsum(0.05 + rng.normal(0, 0.4, n_obs))
        out[f"s{i}"] = y
    return out


def forecasts_from_sequential(
    series_dict: Dict[str, np.ndarray],
    model_type: str,
    order: Any,
    steps: int,
    n_jobs: int,
) -> Dict[str, np.ndarray]:
    """Fit each series locally; return series_id -> forecast vector."""
    from tslib.spark.parallel_processor import GenericParallelProcessor

    res = GenericParallelProcessor.fit_multiple_sequential(
        series_dict, model_type=model_type, order=order, steps=steps, n_jobs=n_jobs
    )
    return {k: np.asarray(v["forecast"], dtype=float) for k, v in res.items()}


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _row_field(row: Any, name: str) -> Any:
    if hasattr(row, name):
        return getattr(row, name)
    return row[name]


def forecasts_from_spark_collected(collected: List[Any]) -> Dict[str, np.ndarray]:
    """
    Build series_id -> forecast vector from GenericParallelProcessor.fit_multiple rows
    (fields: series_id, step, forecast, status).
    """
    buckets: Dict[str, List[Tuple[int, float]]] = {}
    for r in collected:
        sid = str(_row_field(r, "series_id"))
        step = int(_row_field(r, "step"))
        fc = float(_row_field(r, "forecast"))
        buckets.setdefault(sid, []).append((step, fc))
    out: Dict[str, np.ndarray] = {}
    for sid, pairs in buckets.items():
        pairs.sort(key=lambda x: x[0])
        out[sid] = np.array([p[1] for p in pairs], dtype=float)
    return out


def mean_parallel_agreement_pct(
    fc_reference: Dict[str, np.ndarray],
    fc_experimental: Dict[str, np.ndarray],
) -> float:
    """
    How close the experimental (Spark) forecasts are to the sequential baseline, in [0, 100].

    Per series: score = 100 * max(0, 1 - rel_mae) with rel_mae = mean(|ref-exp|) / (mean(|ref|)+eps),
    then averaged over series that align in shape. 100% means identical; lower means more drift.
    """
    scores: List[float] = []
    for sid, ref in fc_reference.items():
        if sid not in fc_experimental:
            continue
        exp = fc_experimental[sid]
        if ref.shape != exp.shape or ref.size == 0:
            continue
        denom = max(float(np.mean(np.abs(ref))), 1e-12)
        rel_mae = float(np.mean(np.abs(ref - exp)) / denom)
        scores.append(100.0 * max(0.0, min(1.0, 1.0 - rel_mae)))
    if not scores:
        return float("nan")
    return float(np.mean(scores))
