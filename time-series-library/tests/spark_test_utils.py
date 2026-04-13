"""
Spark test helpers: Java autodetection for pytest and SparkSession with graceful skip.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent


def _java_home_candidates() -> list[Path]:
    system = platform.system()
    if system == "Darwin":
        return [
            Path("/opt/homebrew/opt/openjdk@17"),
            Path("/opt/homebrew/opt/openjdk@21"),
            Path("/usr/local/opt/openjdk@17"),
            Path("/usr/local/opt/openjdk@21"),
        ]
    if system == "Linux":
        return [
            Path("/usr/lib/jvm/java-21-openjdk-amd64"),
            Path("/usr/lib/jvm/java-17-openjdk-amd64"),
            Path("/usr/lib/jvm/java-21-openjdk"),
            Path("/usr/lib/jvm/java-17-openjdk"),
        ]
    return []


def _check_java_ok(env: Optional[Dict[str, str]] = None) -> bool:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_java_spark.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    return proc.returncode == 0


def ensure_java_for_pytest() -> None:
    """
    If `java` is not on PATH, try typical JDK locations and set JAVA_HOME / PATH
    so `make test` can start PySpark without a manual export (same idea as lib_java_spark.sh).
    """
    if os.environ.get("SPARK_SKIP_JAVA_AUTODETECT"):
        return
    if _check_java_ok():
        return
    for jhome in _java_home_candidates():
        java_bin = jhome / "bin" / "java"
        if not java_bin.is_file():
            continue
        env = os.environ.copy()
        env["JAVA_HOME"] = str(jhome)
        env["PATH"] = str(jhome / "bin") + os.pathsep + env.get("PATH", "")
        if _check_java_ok(env):
            os.environ["JAVA_HOME"] = str(jhome)
            os.environ["PATH"] = env["PATH"]
            return


def get_spark_session_or_skip(
    app_name: str = "tslib-test",
    master: str = "local[1]",
    extra_config: Optional[Dict[str, str]] = None,
):
    """
    Return an active SparkSession or skip the current test if the JVM gateway cannot start.
    """
    import pytest

    from tslib.utils.checks import check_spark_availability

    if not check_spark_availability():
        pytest.skip("PySpark not installed")

    from tslib.spark.python_env import apply_pyspark_python_env

    apply_pyspark_python_env()

    from pyspark.sql import SparkSession

    try:
        from pyspark.errors.exceptions.base import PySparkRuntimeError
    except ImportError:  # pragma: no cover
        PySparkRuntimeError = RuntimeError  # type: ignore[misc,assignment]

    b = SparkSession.builder.appName(app_name).master(master)
    if extra_config:
        for k, v in extra_config.items():
            b = b.config(k, v)
    try:
        return b.getOrCreate()
    except (PySparkRuntimeError, RuntimeError, OSError) as e:  # pragma: no cover
        lowered = str(e).lower()
        if any(s in lowered for s in ("java", "gateway", "jvm", "runtime")):
            pytest.skip(
                "Java 17+ required for Spark (set JAVA_HOME or run: make check-java). "
                f"Detail: {e!r}"
            )
        raise
