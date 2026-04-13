"""
Pytest hooks: ensure Java is visible to PySpark when running `make test`.
"""

from tests.spark_test_utils import ensure_java_for_pytest
from tslib.spark.python_env import apply_pyspark_python_env


def pytest_configure(config):
    ensure_java_for_pytest()
    apply_pyspark_python_env()
