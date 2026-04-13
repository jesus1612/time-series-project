"""
Forecast agreement metrics: Spark / parallel vs sequential (normal) baselines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import numpy as np


def per_series_forecast_agreement_pct(ref: np.ndarray, exp: np.ndarray) -> float:
    """Score in [0, 100]: 100 = identical forecasts; lower = more drift vs reference."""
    ref = np.asarray(ref, dtype=float).ravel()
    exp = np.asarray(exp, dtype=float).ravel()
    if ref.shape != exp.shape or ref.size == 0:
        return float("nan")
    denom = max(float(np.mean(np.abs(ref))), 1e-12)
    rel_mae = float(np.mean(np.abs(ref - exp)) / denom)
    return 100.0 * max(0.0, min(1.0, 1.0 - rel_mae))


def predictions_frame_id_column(predictions_pandas: Any) -> str:
    """Parallel ARIMA emits ``group_id``; some tests use ``series_id``."""
    cols = set(predictions_pandas.columns)
    if "group_id" in cols:
        return "group_id"
    if "series_id" in cols:
        return "series_id"
    raise KeyError(
        "predictions DataFrame needs 'group_id' or 'series_id'; "
        f"got columns={list(predictions_pandas.columns)}"
    )


def suite_arima_agreement_pct(
    normal_results: List[MutableMapping[str, Any]],
    predictions_pandas: Any,
    series_id_col: Optional[str] = None,
    pred_col: str = "predictions",
) -> tuple[float, int]:
    """
    Mean agreement % over all series that exist in both normal and Spark dataframes.
    ``normal_results`` entries: ``series_id``, ``predictions``, ``success``.
    """
    id_col = series_id_col or predictions_frame_id_column(predictions_pandas)
    normal_map = {
        r["series_id"]: r
        for r in normal_results
        if r.get("success") and r.get("predictions") is not None
    }
    scores: List[float] = []
    for _, row in predictions_pandas.iterrows():
        sid = row[id_col]
        if sid not in normal_map:
            continue
        ref = np.asarray(normal_map[sid]["predictions"], dtype=float)
        exp = np.asarray(row[pred_col], dtype=float)
        s = per_series_forecast_agreement_pct(ref, exp)
        if not np.isnan(s):
            scores.append(s)
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


def benchmark_pair_agreement_pct(
    normal_results: Mapping[str, Any], spark_results: Mapping[str, Any]
) -> tuple[float, int]:
    """
    Agreement for ``PerformanceBenchmark``-style dicts with ``results`` list
    (``series_id``, ``predictions``, ``success``).
    """
    n_list = normal_results.get("results") or []
    s_list = spark_results.get("results") or []
    n_map = {
        r["series_id"]: r
        for r in n_list
        if r.get("success") and r.get("predictions") is not None
    }
    scores: List[float] = []
    for r in s_list:
        if not r.get("success") or r.get("predictions") is None:
            continue
        sid = r["series_id"]
        if sid not in n_map:
            continue
        ref = np.asarray(n_map[sid]["predictions"], dtype=float)
        exp = np.asarray(r["predictions"], dtype=float)
        val = per_series_forecast_agreement_pct(ref, exp)
        if not np.isnan(val):
            scores.append(val)
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


def print_parallel_vs_sequential_accuracy(
    label: str,
    agreement_pct: float,
    n_compared: int,
    extra: str = "",
) -> None:
    suffix = f" ({extra})" if extra else ""
    if n_compared <= 0 or np.isnan(agreement_pct):
        print(f"  [{label}] precisión Spark vs secuencial: n/d (sin pares comparables){suffix}")
    else:
        print(
            f"  [{label}] precisión Spark vs secuencial (acuerdo de pronósticos): "
            f"{agreement_pct:.2f}%  (series comparadas: {n_compared}){suffix}"
        )
