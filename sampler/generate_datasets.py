#!/usr/bin/env python3
"""
Generate all sampler CSVs under sampler/datasets/.
Run from anywhere: python path/to/sampler/generate_datasets.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "datasets"


def _ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def _write(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def try_statsmodels_classics() -> None:
    try:
        from statsmodels.datasets import get_rdataset
    except ImportError:
        print("statsmodels not installed: skipping AirPassengers / sunspots from package.")
        return

    try:
        ap = get_rdataset("AirPassengers", "datasets").data
        # Typical columns: value or 'AirPassengers'
        col = [c for c in ap.columns if c.lower() not in ("time", "month")][0]
        dates = pd.date_range("1949-01-01", periods=len(ap), freq="MS")
        _write(
            pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "passengers": ap[col].astype(float)}),
            "airline_passengers.csv",
        )
    except Exception as e:
        print(f"AirPassengers export failed: {e}")

    try:
        ss = get_rdataset("sunspot.year", "sunspots").data
        val_col = [c for c in ss.columns if ss[c].dtype.kind in "fiu"][0]
        _write(
            pd.DataFrame({"year": np.arange(len(ss), dtype=int) + 1700, "sunspots": ss[val_col].astype(float)}),
            "sunspot_yearly.csv",
        )
    except Exception as e:
        print(f"sunspot export failed: {e}")


def synthetic_daily_temperatures(n: int = 3650, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 18.0 + 4.0 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 1.2, size=n)
    y = seasonal + noise
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "temperature": y})


def synthetic_sp500_like(n: int = 2500, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, size=n)
    logp = np.cumsum(r) + 7.5
    price = np.exp(logp) * 1000.0
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": price})


def synthetic_arima_like_211(n: int = 500, seed: int = 3) -> pd.DataFrame:
    """
    Integrated AR-like process: (1-B)y_t = z_t, z_t = 0.4 z_{t-1} + 0.25 z_{t-2} + eps + 0.35 eps_{t-1}
    (informal ARIMA(2,1,1)-style simulation, not exact MLE target).
    """
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    eps = rng.standard_normal(n + 1)
    for t in range(2, n):
        z[t] = 0.4 * z[t - 1] + 0.25 * z[t - 2] + eps[t] + 0.35 * eps[t - 1]
    y = np.cumsum(z)
    return pd.DataFrame({"index": np.arange(n, dtype=int), "value": y})


def large_synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        y[t] = 0.85 * y[t - 1] + eps[t]
    return pd.DataFrame({"index": np.arange(n, dtype=int), "value": y})


def synthetic_arima_eval_benchmark(
    n: int = 960,
    seed: int = 42,
    *,
    phi: float = 0.65,
    theta: float = 0.40,
    sigma: float = 1.0,
    level: float = 100.0,
    burn: int = 400,
) -> pd.DataFrame:
    """
    Integrated ARMA path designed for ARIMA evaluation (complete series, no NaN).

    DGP (informal ARIMA(1,1,1)-style): let w be stationary ARMA(1,1),
        w_t = phi * w_{t-1} + eps_t + theta * eps_{t-1}, eps ~ N(0, sigma^2).
    Then y_t = level + sum_{s=1..t} w_s (integration). Sensible for checking
    forecasts, residuals, and parameter recovery on medium/long horizons.
    """
    rng = np.random.default_rng(seed)
    total = n + burn
    w = np.zeros(total + 1)
    eps = rng.standard_normal(total + 1) * sigma
    for t in range(1, total):
        w[t] = phi * w[t - 1] + eps[t] + theta * eps[t - 1]
    w = w[burn : burn + n]
    y = level + np.cumsum(w)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "y": y.astype(float)})


def main() -> None:
    _ensure_out()
    try_statsmodels_classics()
    _write(synthetic_daily_temperatures(), "daily_temperatures.csv")
    _write(synthetic_sp500_like(), "sp500_daily.csv")
    _write(synthetic_arima_like_211(), "synthetic_arima_211.csv")
    _write(large_synthetic(10_000, seed=4), "large_synthetic_10k.csv")
    _write(large_synthetic(50_000, seed=5), "large_synthetic_50k.csv")
    _write(synthetic_arima_eval_benchmark(), "arima_eval_benchmark.csv")

    # Fallback if statsmodels classics missing
    ap_path = OUT / "airline_passengers.csv"
    if not ap_path.exists():
        rng = np.random.default_rng(99)
        n = 144
        t = np.arange(n)
        y = 100 + 0.3 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 4, size=n)
        dates = pd.date_range("2000-01-01", periods=n, freq="MS")
        _write(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "passengers": y}), "airline_passengers.csv")

    sy_path = OUT / "sunspot_yearly.csv"
    if not sy_path.exists():
        rng = np.random.default_rng(100)
        n = 289
        t = np.linspace(0, 40, n)
        y = 40 + 30 * np.sin(t) + rng.normal(0, 8, size=n)
        _write(
            pd.DataFrame({"year": np.arange(1700, 1700 + n), "sunspots": np.maximum(y, 0)}),
            "sunspot_yearly.csv",
        )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
