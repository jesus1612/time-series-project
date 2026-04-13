"""
TSLib - Time Series Library

A comprehensive time series analysis library with ARIMA implementation from scratch,
designed with object-oriented principles and optional PySpark support for distributed computing.

Main Components:
- Core algorithms: ARIMA, ACF/PACF, stationarity tests
- High-level models: User-friendly API
- Preprocessing: Data transformations and validation
- Metrics: Model evaluation and diagnostics
- Spark integration: Distributed processing with PySpark
"""

__version__ = "0.1.0"
__author__ = "Genaro Melgar"

# Core imports
from .models import ARModel, MAModel, ARMAModel, ARIMAModel

# Utility imports
from .utils.checks import check_spark_availability

__all__ = [
    "ARModel",
    "MAModel",
    "ARMAModel",
    "ARIMAModel",
    "check_spark_availability",
]

