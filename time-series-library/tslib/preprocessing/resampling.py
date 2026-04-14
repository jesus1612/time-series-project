"""
Time-based resampling / granularity alignment for series with a DatetimeIndex.

Diagram step 1 (frequency / aggregation) is handled here, not inside
``ParallelARIMAWorkflow`` STEP 1 (which selects *d* and transformations).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


Agg = Literal["mean", "sum"]


def resample_series(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    rule: str,
    agg: Agg = "mean",
    dropna: bool = True,
) -> pd.Series:
    """
    Resample a column to a new frequency using sum or mean aggregation.

    Parameters
    ----------
    df
        Input frame with at least *datetime_col* and *value_col*.
    datetime_col
        Column parsed to pandas datetime (coerced if needed).
    value_col
        Numeric series to aggregate.
    rule
        Pandas offset alias, e.g. ``'1D'``, ``'1H'``, ``'15min'``.
    agg
        ``'mean'`` (typical for levels / rates) or ``'sum'`` (counts / flows).
    dropna
        Drop rows with NaN in *value_col* before resampling.

    Returns
    -------
    pd.Series
        Resampled series indexed by the resampled DatetimeIndex.
    """
    work = df[[datetime_col, value_col]].copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if dropna:
        work = work.dropna(subset=[datetime_col, value_col])
    work = work.sort_values(datetime_col)
    work = work.set_index(datetime_col)
    s = work[value_col]
    if agg == "mean":
        out = s.resample(rule).mean()
    elif agg == "sum":
        out = s.resample(rule).sum()
    else:
        raise ValueError("agg must be 'mean' or 'sum'")
    return out


def resample_numpy_with_index(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    rule: str,
    agg: Agg = "mean",
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Convenience wrapper: build a minimal DataFrame and return numpy values + new index.
    """
    df = pd.DataFrame({"ts": timestamps, "y": np.asarray(values, dtype=float)})
    ser = resample_series(df, "ts", "y", rule=rule, agg=agg)
    return ser.values.astype(float), pd.DatetimeIndex(ser.index)
