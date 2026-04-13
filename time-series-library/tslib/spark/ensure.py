"""
Spark session verification and startup for distributed TSLib APIs.

Distributed entry points should use ensure_spark_session() so that:
- PySpark is required (clear error otherwise; use tslib.models for linear-only).
- An existing session is validated as active before use.
- A new session is started with configurable master (cores) and spark_config.
"""

from typing import Any, Dict, Optional

# Message reused by distributed components
DISTRIBUTED_REQUIRES_SPARK = (
    "Distributed TSLib requires PySpark and Java 17+. "
    "Install: pip install -r requirements-spark.txt and configure JAVA_HOME. "
    "For single-series / linear use without Spark, use tslib.models (ARModel, MAModel, "
    "ARMAModel, ARIMAModel) with numpy or pandas data only."
)


def ensure_spark_session(
    spark_session: Optional[Any] = None,
    spark_config: Optional[Dict[str, str]] = None,
    master: Optional[str] = None,
    app_name: Optional[str] = None,
    register_global: bool = True,
) -> Any:
    """
    Verify PySpark is available; return an active SparkSession or start one.

    Parameters
    ----------
    spark_session : SparkSession, optional
        Reuse this session if it is still active.
    spark_config : dict, optional
        Extra Spark configuration keys (e.g. spark.executor.cores, memory).
    master : str, optional
        Spark master URL, e.g. ``local[*]`` or ``local[8]`` for eight local cores.
    app_name : str, optional
        Application name for a newly created session.
    register_global : bool
        If True, store the resolved session in SparkSessionManager for reuse.

    Returns
    -------
    SparkSession

    Raises
    ------
    ImportError
        If PySpark is not installed.
    RuntimeError
        If a non-None spark_session was passed but is stopped or invalid.
    """
    from ..utils.checks import check_spark_availability

    if not check_spark_availability():
        raise ImportError(DISTRIBUTED_REQUIRES_SPARK)

    from .python_env import apply_pyspark_python_env

    apply_pyspark_python_env()

    from .core import (
        SparkSessionManager,
        _spark_session_is_active,
        get_or_create_spark_session,
    )

    if spark_session is not None:
        if _spark_session_is_active(spark_session):
            if register_global:
                SparkSessionManager.set_global_spark(spark_session)
            return spark_session
        raise RuntimeError(
            "The provided SparkSession is stopped or unusable. "
            "Pass spark_session=None to create or recover a session via master/spark_config."
        )

    # Prefer existing active session in this JVM
    try:
        from pyspark.sql import SparkSession as SS

        active = SS.getActiveSession()
        if active is not None and _spark_session_is_active(active):
            if register_global:
                SparkSessionManager.set_global_spark(active)
            return active
    except Exception:
        pass

    spark = get_or_create_spark_session(
        None,
        spark_config,
        master=master,
        app_name=app_name,
    )
    if register_global:
        SparkSessionManager.set_global_spark(spark)
    return spark
