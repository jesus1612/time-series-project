"""
Ensure PySpark worker processes use the same Python as the driver.

Spark defaults to ``python3`` on PATH; that interpreter often lacks the venv
packages (numpy, pandas, tslib). Set PYSPARK_* before the first SparkContext starts.
"""

from __future__ import annotations

import os
import sys


def apply_pyspark_python_env() -> None:
    """Force executor and driver workers to use ``sys.executable`` (e.g. venv/bin/python)."""
    exe = sys.executable
    if not exe:
        return
    os.environ["PYSPARK_PYTHON"] = exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = exe
