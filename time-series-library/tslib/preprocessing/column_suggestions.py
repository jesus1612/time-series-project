"""
Heuristics to suggest time-index and numeric value columns from a tabular frame.

TSLib does not read files; callers pass a pandas DataFrame after load. These
functions only inspect dtypes and names—no modeling.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def suggest_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Suggest a column for time axis: name keywords, datetime dtype, or parseable values.

    Returns the first matching column name, or None.
    """
    datetime_keywords = (
        "date",
        "time",
        "timestamp",
        "fecha",
        "tiempo",
        "datetime",
    )
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in datetime_keywords):
            return str(col)
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
        try:
            pd.to_datetime(df[col].head(10), errors="raise")
            return str(col)
        except (ValueError, TypeError, OverflowError):
            continue
    return None


def suggest_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    List columns usable as numeric series: native numeric dtypes plus object columns
    that convert after stripping common currency/percent formatting.
    """
    numeric_cols: List[str] = []
    numeric_cols.extend(df.select_dtypes(include=[np.number]).columns.tolist())

    for col in df.select_dtypes(include=["object"]).columns:
        if col in numeric_cols:
            continue
        try:
            sample = df[col].dropna().head(10)
            if len(sample) == 0:
                continue
            test_values = sample.astype(str).str.replace("$", "", regex=False)
            test_values = test_values.str.replace(",", "", regex=False)
            test_values = test_values.str.replace("%", "", regex=False)
            test_values = test_values.str.strip()
            pd.to_numeric(test_values, errors="raise")
            numeric_cols.append(str(col))
        except (ValueError, TypeError, AttributeError):
            continue

    return numeric_cols
