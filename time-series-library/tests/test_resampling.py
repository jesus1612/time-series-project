"""Tests for time-based resampling helpers."""

import numpy as np
import pandas as pd

from tslib.preprocessing import resample_series


def test_resample_series_mean():
    idx = pd.date_range("2020-01-01", periods=48, freq="h")
    df = pd.DataFrame(
        {
            "ts": idx,
            "y": np.arange(48, dtype=float),
        }
    )
    out = resample_series(df, "ts", "y", "1D", agg="mean")
    assert len(out) == 2
    assert np.allclose(out.iloc[0], np.mean(np.arange(24)))


def test_resample_series_sum():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"ts": idx, "y": np.ones(10)})
    out = resample_series(df, "ts", "y", "5D", agg="sum")
    assert len(out) == 2
