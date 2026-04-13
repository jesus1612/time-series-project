"""
Benchmark fixtures: optional Spark session for distributed paths.
"""

import pytest

from tslib.utils.checks import check_spark_availability


@pytest.fixture(scope="session")
def spark_session_benchmark():
    """
    Single Spark session for all benchmark files; local[2] keeps overhead moderate.
    """
    if not check_spark_availability():
        pytest.skip("PySpark not installed")
    from tslib.spark.ensure import ensure_spark_session

    try:
        spark = ensure_spark_session(
            spark_session=None,
            master="local[2]",
            app_name="TSLib-benchmark-suite",
            register_global=True,
        )
    except Exception as e:
        pytest.skip(
            f"Spark session could not start (install JDK 17+ / set JAVA_HOME): {e!r}"
        )
    yield spark
