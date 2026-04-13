"""
Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) calculations.

Implements ACF and PACF from scratch using pure NumPy — no external statistical
libraries required. These are essential for identifying ARIMA model orders (p, q).

Mathematical Background
-----------------------
ACF at lag k:

    ρ_k = Σ_{t=k+1}^{n} (y_t - ȳ)(y_{t-k} - ȳ) / Σ_{t=1}^{n} (y_t - ȳ)²

PACF at lag k (Durbin-Levinson algorithm):

    φ_{k,k} = [ρ_k - Σ_{j=1}^{k-1} φ_{k-1,j} · ρ_{k-j}]
               / [1 - Σ_{j=1}^{k-1} φ_{k-1,j} · ρ_j]

    φ_{k,j} = φ_{k-1,j} - φ_{k,k} · φ_{k-1,k-j},  j = 1, …, k-1

Spark-based variants (SparkACFCalculator, SparkPACFCalculator, SparkACFPACFAnalyzer)
live in tslib.spark.acf_pacf and require PySpark.
"""

import numpy as np
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from .base import BaseTest, SparkEnabled


class ACFCalculator:
    """
    Calculate Autocorrelation Function (ACF) from scratch using NumPy.

    The ACF measures the linear correlation between a time series and its
    lagged values:

        ρ_k = Σ(y_t - ȳ)(y_{t-k} - ȳ) / Σ(y_t - ȳ)²

    Parameters
    ----------
    max_lags : int, optional
        Maximum number of lags. Defaults to min(n//4, 40).
    n_jobs : int
        Parallel workers for large datasets (-1 = all cores).
    """

    def __init__(self, max_lags: Optional[int] = None, n_jobs: int = -1):
        self.max_lags = max_lags
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self._acf_values: Optional[np.ndarray] = None
        self._lags: Optional[np.ndarray] = None
        # Vectorised NumPy is faster up to this size; parallelise beyond it
        self.parallel_threshold = 1000

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ACF values for all lags up to max_lags.

        Parameters
        ----------
        data : np.ndarray
            1-D time series (at least 2 observations).

        Returns
        -------
        lags : np.ndarray
            Lag indices [0, 1, …, max_lags].
        acf_values : np.ndarray
            Autocorrelation at each lag (ρ_0 = 1 by definition).
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n < 2:
            raise ValueError("Data must have at least 2 observations.")

        max_lags = (
            min(n // 4, 40) if self.max_lags is None else min(self.max_lags, n - 1)
        )

        mean = np.mean(data)
        centered = data - mean
        variance = np.sum(centered ** 2)

        lags = np.arange(max_lags + 1)

        if variance == 0:
            acf_values = np.ones(max_lags + 1)
        elif n > self.parallel_threshold and self.n_jobs > 1:
            acf_values = self._calculate_parallel(centered, variance, max_lags)
        else:
            acf_values = self._calculate_vectorised(centered, variance, max_lags)

        self._lags = lags
        self._acf_values = acf_values
        return lags, acf_values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_vectorised(
        self, centered: np.ndarray, variance: float, max_lags: int
    ) -> np.ndarray:
        """Vectorised NumPy implementation — fast for small/medium series."""
        acf_values = np.empty(max_lags + 1)
        acf_values[0] = 1.0
        for k in range(1, max_lags + 1):
            acf_values[k] = np.sum(centered[k:] * centered[:-k]) / variance
        return acf_values

    def _calculate_parallel(
        self, centered: np.ndarray, variance: float, max_lags: int
    ) -> np.ndarray:
        """Thread-parallel implementation for large series."""

        def _lag(k: int) -> float:
            if k == 0:
                return 1.0
            return float(np.sum(centered[k:] * centered[:-k]) / variance)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            results = list(pool.map(_lag, range(max_lags + 1)))
        return np.array(results)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_acf_values(self) -> Optional[np.ndarray]:
        """Return the last computed ACF values (None if not yet computed)."""
        return self._acf_values

    def get_lags(self) -> Optional[np.ndarray]:
        """Return the last computed lag indices."""
        return self._lags


class PACFCalculator:
    """
    Calculate Partial Autocorrelation Function (PACF) from scratch.

    PACF measures the correlation between y_t and y_{t-k} after controlling
    for the effects of intermediate lags. Computed via the Durbin-Levinson
    recursive algorithm:

        φ_{1,1} = ρ_1

    For k ≥ 2:

        φ_{k,k} = [ρ_k - Σ_{j=1}^{k-1} φ_{k-1,j} ρ_{k-j}]
                   / [1 - Σ_{j=1}^{k-1} φ_{k-1,j} ρ_j]

        φ_{k,j} = φ_{k-1,j} - φ_{k,k} · φ_{k-1,k-j}

    Parameters
    ----------
    max_lags : int, optional
        Maximum number of lags. Defaults to min(n//4, 40).
    n_jobs : int
        Passed to the underlying ACFCalculator.
    """

    def __init__(self, max_lags: Optional[int] = None, n_jobs: int = -1):
        self.max_lags = max_lags
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self._pacf_values: Optional[np.ndarray] = None
        self._lags: Optional[np.ndarray] = None

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate PACF values using the Durbin-Levinson algorithm.

        Parameters
        ----------
        data : np.ndarray
            1-D time series (at least 2 observations).

        Returns
        -------
        lags : np.ndarray
            Lag indices [0, 1, …, max_lags].
        pacf_values : np.ndarray
            Partial autocorrelation at each lag (φ_{0,0} = 1).
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n < 2:
            raise ValueError("Data must have at least 2 observations.")

        max_lags = (
            min(n // 4, 40) if self.max_lags is None else min(self.max_lags, n - 1)
        )

        # Step 1 — compute ACF (needed by Durbin-Levinson)
        acf_calc = ACFCalculator(max_lags, n_jobs=self.n_jobs)
        _, acf_values = acf_calc.calculate(data)

        lags = np.arange(max_lags + 1)
        pacf_values = np.zeros(max_lags + 1)
        pacf_values[0] = 1.0

        if max_lags == 0:
            self._lags, self._pacf_values = lags, pacf_values
            return lags, pacf_values

        # Step 2 — initialise with lag-1
        pacf_values[1] = acf_values[1]

        # φ[k, j]: AR(k) coefficients; only lower-triangle is used
        phi = np.zeros((max_lags + 1, max_lags + 1))
        phi[1, 1] = acf_values[1]

        # Step 3 — Durbin-Levinson recursion for k ≥ 2
        for k in range(2, max_lags + 1):
            num = acf_values[k] - np.dot(phi[k - 1, 1:k], acf_values[k - 1:0:-1])
            den = 1.0 - np.dot(phi[k - 1, 1:k], acf_values[1:k])

            phi[k, k] = 0.0 if abs(den) < 1e-10 else num / den
            pacf_values[k] = phi[k, k]

            # Update coefficients for smaller lags
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

        self._lags = lags
        self._pacf_values = pacf_values
        return lags, pacf_values

    def get_pacf_values(self) -> Optional[np.ndarray]:
        """Return the last computed PACF values."""
        return self._pacf_values

    def get_lags(self) -> Optional[np.ndarray]:
        """Return the last computed lag indices."""
        return self._lags


class ACFPACFAnalyzer:
    """
    Combined ACF/PACF analyzer for ARIMA model identification.

    Computes both ACF and PACF and infers preliminary AR/MA orders by
    detecting the first lag at which each function becomes non-significant
    (approximate 95 % bound = 1.96 / √n).

    Parameters
    ----------
    max_lags : int, optional
        Maximum lags to analyse.
    """

    def __init__(self, max_lags: Optional[int] = None):
        self.max_lags = max_lags
        self.acf_calc = ACFCalculator(max_lags)
        self.pacf_calc = PACFCalculator(max_lags)

    def analyze(self, data: np.ndarray) -> dict:
        """
        Perform a complete ACF/PACF analysis.

        Returns
        -------
        dict with keys:
            acf_lags, acf_values, pacf_lags, pacf_values, suggested_orders
        """
        acf_lags, acf_values = self.acf_calc.calculate(data)
        pacf_lags, pacf_values = self.pacf_calc.calculate(data)
        suggested_orders = self._suggest_orders(acf_values, pacf_values)
        return {
            "acf_lags": acf_lags,
            "acf_values": acf_values,
            "pacf_lags": pacf_lags,
            "pacf_values": pacf_values,
            "suggested_orders": suggested_orders,
        }

    def _suggest_orders(
        self, acf_values: np.ndarray, pacf_values: np.ndarray
    ) -> dict:
        """
        Suggest p and q based on ACF/PACF cutoff heuristic.

        Identification rules:
        - AR(p): PACF cuts off at lag p (first insignificant lag beyond p).
        - MA(q): ACF  cuts off at lag q.
        - ARMA : both decay without clear cutoff.
        """
        n = len(acf_values)
        bound = 1.96 / np.sqrt(n)  # ~95 % Bartlett confidence bound

        sig_acf = np.where(np.abs(acf_values[1:]) > bound)[0] + 1
        sig_pacf = np.where(np.abs(pacf_values[1:]) > bound)[0] + 1

        suggested_p = int(sig_pacf[0]) if len(sig_pacf) > 0 else 0
        suggested_q = int(sig_acf[0]) if len(sig_acf) > 0 else 0

        return {
            "suggested_p": suggested_p,
            "suggested_q": suggested_q,
            "suggested_d": 0,  # determined by stationarity tests
            "significant_acf_lags": sig_acf.tolist(),
            "significant_pacf_lags": sig_pacf.tolist(),
            "significance_bound": float(bound),
        }
