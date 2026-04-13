"""
Spark-distributed ACF and PACF calculations.

Provides Spark-native implementations of ACFCalculator, PACFCalculator,
and ACFPACFAnalyzer for large-scale distributed workloads.

These classes mirror the pure-NumPy API from ``tslib.core.acf_pacf`` but
distribute lag computations across a Spark cluster:

* **SparkACFCalculator** — parallelises each lag covariance computation
  across Spark executors using ``parallelize + map``.
* **SparkPACFCalculator** — distributes the ACF step; the Durbin-Levinson
  recursion itself is sequential by nature but still benefits from the
  distributed ACF.
* **SparkACFPACFAnalyzer** — high-level wrapper that calls both and returns
  model-order suggestions.

Requires PySpark ≥ 3.4.0 and ``tslib[spark]`` extras (``pyspark``,
``pyarrow``).  Import this module only when a SparkSession is available;
``tslib.core.acf_pacf`` is always safe to import without PySpark.
"""

import numpy as np
from typing import Tuple, Optional

from pyspark.sql.functions import col, lit
from ..core.base import SparkEnabled
from ..core.acf_pacf import ACFCalculator


class SparkACFCalculator(SparkEnabled):
    """
    Calculate ACF using Spark for distributed lag computation.

    Each lag covariance is computed on a separate Spark partition, giving
    near-linear speedup for very long time series (n >> 10 000) where
    ``tslib.core.acf_pacf.ACFCalculator`` becomes memory-bound.

    Parameters
    ----------
    max_lags : int, optional
        Maximum lags to calculate. Defaults to min(n//4, 40).
    spark_session : SparkSession, optional
        Existing session to use; a new one is created if None.
    spark_config : dict, optional
        Spark configuration overrides.
    """

    def __init__(
        self,
        max_lags: Optional[int] = None,
        spark_session=None,
        spark_config: Optional[dict] = None,
    ):
        super().__init__(spark_session, spark_config)
        self.max_lags = max_lags
        self._acf_values: Optional[np.ndarray] = None
        self._lags: Optional[np.ndarray] = None

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ACF via Spark.

        Each lag k is mapped to one Spark task that computes:

            ρ_k = Σ_{t=k}^{n-1} (y_t - ȳ)(y_{t-k} - ȳ) / Σ(y_t - ȳ)²

        Parameters
        ----------
        data : np.ndarray
            1-D time series.

        Returns
        -------
        lags : np.ndarray
        acf_values : np.ndarray
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n < 2:
            raise ValueError("Data must have at least 2 observations.")

        max_lags = (
            min(n // 4, 40) if self.max_lags is None else min(self.max_lags, n - 1)
        )

        # Use Spark to compute mean and variance via distributed aggregation
        df_spark = self.converter.to_spark_dataframe(data, cache=True)
        mean = self.math_ops.vector_mean(df_spark)
        variance = self.math_ops.vector_variance(df_spark, mean=mean)

        lags = list(range(max_lags + 1))

        if variance == 0:
            self._lags = np.array(lags)
            self._acf_values = np.ones(max_lags + 1)
            return self._lags, self._acf_values

        # Distribute lag computations across Spark executors
        lags_rdd = self.spark.sparkContext.parallelize(lags)

        def _acf_for_lag(k: int) -> float:
            if k == 0:
                return 1.0
            df_lagged = df_spark.alias("orig").join(
                df_spark.select(
                    (col("index") + lit(k)).alias("index_lag"),
                    col("value").alias("value_lag"),
                ).alias("lag"),
                col("orig.index") == col("lag.index_lag"),
            )
            cov_df = df_lagged.select(
                ((col("orig.value") - lit(mean)) * (col("lag.value_lag") - lit(mean))).alias("cov")
            )
            covariance = cov_df.agg({"cov": "mean"}).collect()[0][0]
            return covariance / variance if covariance is not None else 0.0

        acf_values = lags_rdd.map(_acf_for_lag).collect()

        self._lags = np.array(lags)
        self._acf_values = np.array(acf_values)
        return self._lags, self._acf_values

    def get_acf_values(self) -> Optional[np.ndarray]:
        """Return the last computed ACF values."""
        return self._acf_values

    def get_lags(self) -> Optional[np.ndarray]:
        """Return the last computed lag indices."""
        return self._lags


class SparkPACFCalculator(SparkEnabled):
    """
    Calculate PACF using Spark — distributed ACF computation + sequential
    Durbin-Levinson recursion on the collected results.

    The ACF step is distributed (see SparkACFCalculator); the recursion
    itself is sequential because each step depends on the previous one.

    Parameters
    ----------
    max_lags : int, optional
    spark_session : SparkSession, optional
    spark_config : dict, optional
    """

    def __init__(
        self,
        max_lags: Optional[int] = None,
        spark_session=None,
        spark_config: Optional[dict] = None,
    ):
        super().__init__(spark_session, spark_config)
        self.max_lags = max_lags
        self._pacf_values: Optional[np.ndarray] = None
        self._lags: Optional[np.ndarray] = None

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate PACF via distributed ACF + Durbin-Levinson.

        Parameters
        ----------
        data : np.ndarray
            1-D time series.

        Returns
        -------
        lags : np.ndarray
        pacf_values : np.ndarray
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n < 2:
            raise ValueError("Data must have at least 2 observations.")

        max_lags = (
            min(n // 4, 40) if self.max_lags is None else min(self.max_lags, n - 1)
        )

        # Distributed ACF computation
        acf_calc = SparkACFCalculator(max_lags, self.spark)
        _, acf_values = acf_calc.calculate(data)

        # Sequential Durbin-Levinson on collected ACF
        lags = np.arange(max_lags + 1)
        pacf_values = np.zeros(max_lags + 1)
        pacf_values[0] = 1.0

        if max_lags == 0:
            self._lags, self._pacf_values = lags, pacf_values
            return lags, pacf_values

        pacf_values[1] = acf_values[1]
        phi = np.zeros((max_lags + 1, max_lags + 1))
        phi[1, 1] = acf_values[1]

        for k in range(2, max_lags + 1):
            num = acf_values[k] - np.dot(phi[k - 1, 1:k], acf_values[k - 1:0:-1])
            den = 1.0 - np.dot(phi[k - 1, 1:k], acf_values[1:k])
            phi[k, k] = 0.0 if abs(den) < 1e-10 else num / den
            pacf_values[k] = phi[k, k]
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


class SparkACFPACFAnalyzer(SparkEnabled):
    """
    Combined ACF/PACF analyzer using Spark for distributed computing.

    High-level wrapper that runs SparkACFCalculator + SparkPACFCalculator
    and applies the same ACF/PACF cutoff heuristic as ACFPACFAnalyzer to
    suggest preliminary ARIMA orders.

    Parameters
    ----------
    max_lags : int, optional
    spark_session : SparkSession, optional
    spark_config : dict, optional
    """

    def __init__(
        self,
        max_lags: Optional[int] = None,
        spark_session=None,
        spark_config: Optional[dict] = None,
    ):
        super().__init__(spark_session, spark_config)
        self.max_lags = max_lags
        self.acf_calc = SparkACFCalculator(max_lags, self.spark)
        self.pacf_calc = SparkPACFCalculator(max_lags, self.spark)

    def analyze(self, data: np.ndarray) -> dict:
        """
        Perform complete ACF/PACF analysis using Spark.

        Returns
        -------
        dict with keys: acf_lags, acf_values, pacf_lags, pacf_values,
        suggested_orders
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
        """95 % Bartlett-bound heuristic for p and q identification."""
        n = len(acf_values)
        bound = 1.96 / np.sqrt(n)
        sig_acf = np.where(np.abs(acf_values[1:]) > bound)[0] + 1
        sig_pacf = np.where(np.abs(pacf_values[1:]) > bound)[0] + 1

        return {
            "suggested_p": int(sig_pacf[0]) if len(sig_pacf) > 0 else 0,
            "suggested_q": int(sig_acf[0]) if len(sig_acf) > 0 else 0,
            "suggested_d": 0,
            "significant_acf_lags": sig_acf.tolist(),
            "significant_pacf_lags": sig_pacf.tolist(),
            "significance_bound": float(bound),
        }
