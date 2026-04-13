"""
Core algorithms module.

Contains the fundamental mathematical implementations for ARIMA time series
analysis, all built from scratch without wrapping external statistical
libraries:

- ``arima``:        AR, MA, ARMA, ARIMA process classes
- ``acf_pacf``:     ACF / PACF calculators and combined analyser
- ``stationarity``: ADF and KPSS stationarity tests
- ``optimization``: Maximum Likelihood Estimation (MLE) optimizer
- ``base``:         Abstract base classes shared across the library

Spark-based variants (SparkACFCalculator, etc.) live in tslib.spark and
require PySpark — they are intentionally excluded here so that the core
package remains importable in environments without PySpark.
"""

from .arima import ARProcess, MAProcess, ARMAProcess, ARIMAProcess
from .acf_pacf import ACFCalculator, PACFCalculator, ACFPACFAnalyzer
from .stationarity import ADFTest, KPSSTest
from .optimization import MLEOptimizer

__all__ = [
    # ARIMA processes
    "ARProcess",
    "MAProcess",
    "ARMAProcess",
    "ARIMAProcess",
    # ACF / PACF
    "ACFCalculator",
    "PACFCalculator",
    "ACFPACFAnalyzer",
    # Stationarity tests
    "ADFTest",
    "KPSSTest",
    # Optimisation
    "MLEOptimizer",
]
