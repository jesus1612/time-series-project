"""
PySpark integration module for distributed time series analysis.

This module provides Spark-native implementations of core TSLib algorithms
for large-scale workloads across a Spark cluster. All classes in this
package require PySpark >= 3.4.0 and Java 17+.

Available components (when PySpark is installed)
-------------------------------------------------
Autocorrelation:
    SparkACFCalculator   — distributed ACF computation
    SparkPACFCalculator  — distributed PACF (Durbin-Levinson on Spark ACF)
    SparkACFPACFAnalyzer — combined analyser with order suggestions

Optimisation:
    SparkMLEOptimizer    — distributed MLE parameter estimation

Linear algebra / statistics:
    SparkLinearAlgebra   — distributed matrix operations
    SparkStatistics      — distributed summary statistics

Parallel workflows:
    GenericParallelProcessor — AR, MA, ARMA, and ARIMA on many series (Pandas UDF)
    ParallelARIMAProcessor — ARIMA-focused fit+predict via Pandas UDF
    ParallelARIMAWorkflow  — full 11-step parallel batch ARIMA workflow
    ParallelARWorkflow     — parallel classic AR(p) workflow (same methodology, q=0)
    ParallelMAWorkflow     — parallel classic MA(q) workflow (same methodology, p=0)
    ParallelARMAWorkflow   — parallel classic ARMA(p,q) on stationary data (d=0 in model)

Session policy:
    ensure_spark_session — verify PySpark, validate or start SparkSession (master, spark_config)

To check whether PySpark is ready:

    from tslib.spark import check_spark_availability
    if check_spark_availability():
        from tslib.spark.acf_pacf import SparkACFCalculator
"""

from .utils import check_spark_availability
from .ensure import DISTRIBUTED_REQUIRES_SPARK, ensure_spark_session

# Guard everything behind a runtime availability check so the rest of TSLib
# remains importable without PySpark installed.
if check_spark_availability():
    from .parallel_arima import fit_predict_arima_udf, ParallelARIMAProcessor
    from .parallel_arima_workflow import ParallelARIMAWorkflow
    from .parallel_ar_workflow import ParallelARWorkflow
    from .parallel_ma_workflow import ParallelMAWorkflow
    from .parallel_arma_workflow import ParallelARMAWorkflow
    from .parallel_processor import GenericParallelProcessor
    from .acf_pacf import SparkACFCalculator, SparkPACFCalculator, SparkACFPACFAnalyzer
    from .optimization import SparkMLEOptimizer
    from .math_operations import SparkLinearAlgebra, SparkStatistics

    __all__ = [
        "check_spark_availability",
        "ensure_spark_session",
        "DISTRIBUTED_REQUIRES_SPARK",
        # ACF / PACF
        "SparkACFCalculator",
        "SparkPACFCalculator",
        "SparkACFPACFAnalyzer",
        # Optimisation
        "SparkMLEOptimizer",
        # Math
        "SparkLinearAlgebra",
        "SparkStatistics",
        # Parallel workflow
        "fit_predict_arima_udf",
        "ParallelARIMAProcessor",
        "ParallelARIMAWorkflow",
        "ParallelARWorkflow",
        "ParallelMAWorkflow",
        "ParallelARMAWorkflow",
        # Generic parallel processor (all model types)
        "GenericParallelProcessor",
    ]
else:
    __all__ = [
        "check_spark_availability",
        "ensure_spark_session",
        "DISTRIBUTED_REQUIRES_SPARK",
    ]
