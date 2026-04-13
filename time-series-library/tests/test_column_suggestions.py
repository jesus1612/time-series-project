"""Tests for column suggestion helpers."""
import pandas as pd

from tslib.preprocessing.column_suggestions import suggest_datetime_column, suggest_numeric_columns


def test_suggest_datetime_by_name():
    df = pd.DataFrame({"fecha": [1, 2], "y": [0.1, 0.2]})
    assert suggest_datetime_column(df) == "fecha"


def test_suggest_numeric_mixed():
    df = pd.DataFrame({"a": [1, 2], "b": ["$1", "$2"]})
    cols = suggest_numeric_columns(df)
    assert "a" in cols and "b" in cols
