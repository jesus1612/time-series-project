"""
Utility functions and helpers

Contains general-purpose utilities:
- Dependency checks (PySpark availability)
- Data type conversions
- Common helper functions
"""

from .checks import check_spark_availability

__all__ = [
    "check_spark_availability",
]




