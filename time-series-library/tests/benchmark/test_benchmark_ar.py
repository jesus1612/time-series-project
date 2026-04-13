"""
AR model: sequential (n_jobs variants), Spark GenericParallelProcessor, statsmodels reference.
"""

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    ar_series_dict,
    forecasts_from_sequential,
    forecasts_from_spark_collected,
    mean_abs_diff,
    mean_parallel_agreement_pct,
    timed_call,
)

pytestmark = pytest.mark.benchmark


@pytest.fixture
def ar_data():
    return ar_series_dict(n_series=5, n_obs=80, seed=101)


def test_ar_sequential_n_jobs_variants(ar_data):
    """Linear path: compare wall time n_jobs=1 vs n_jobs=-1 (within-process parallelism)."""
    order = 2
    steps = 5
    _, t1 = timed_call(
        forecasts_from_sequential, ar_data, "AR", order, steps, n_jobs=1
    )
    fc2, t2 = timed_call(
        forecasts_from_sequential, ar_data, "AR", order, steps, n_jobs=-1
    )
    assert len(fc2) == len(ar_data)
    for sid in ar_data:
        assert np.all(np.isfinite(fc2[sid]))
    # Smoke: both paths complete; timing is informational (do not assert Spark-like speedup)
    print(f"[AR sequential] n_jobs=1: {t1:.4f}s, n_jobs=-1: {t2:.4f}s")


@pytest.mark.spark
def test_ar_spark_vs_sequential(ar_data, spark_session_benchmark):
    """Distributed path vs sequential baseline (same orders)."""
    from tslib.spark.parallel_processor import GenericParallelProcessor

    order = 2
    steps = 5
    fc_seq, t_seq = timed_call(
        forecasts_from_sequential, ar_data, "AR", order, steps, n_jobs=1
    )

    rows = []
    for sid, y in ar_data.items():
        for v in y:
            rows.append((sid, float(v)))
    pdf = pd.DataFrame(rows, columns=["series_id", "y"])
    sdf = spark_session_benchmark.createDataFrame(pdf)

    proc = GenericParallelProcessor(
        model_type="AR",
        spark=spark_session_benchmark,
        n_jobs=1,
        master="local[2]",
        app_name="bench-AR",
    )
    out, t_spark = timed_call(
        proc.fit_multiple,
        sdf,
        "series_id",
        "y",
        order,
        steps,
    )
    collected = out.collect()
    assert len(collected) == len(ar_data) * steps
    assert all(r.status == "ok" for r in collected)
    fc_spark = forecasts_from_spark_collected(collected)
    agree = mean_parallel_agreement_pct(fc_seq, fc_spark)
    print(
        f"[AR] sequential: {t_seq:.4f}s, spark: {t_spark:.4f}s, "
        f"parallel vs sequential agreement: {agree:.2f}% (experimental Spark path)"
    )


def test_ar_statsmodels_reference_alignment(ar_data):
    """Optional: tslib AR(2) forecast vs statsmodels ARIMA(2,0,0) on one series."""
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.arima.model import ARIMA

    y = ar_data["s0"]
    steps = 5
    order = 2

    from tslib.models.ar_model import ARModel

    m = ARModel(order=order, auto_select=False, validation=False, n_jobs=1)
    m.fit(y)
    fc_ts = m.predict(steps=steps)

    sm = ARIMA(y, order=(order, 0, 0), trend="c").fit()
    fc_sm = np.asarray(sm.forecast(steps=steps), dtype=float)

    mad = mean_abs_diff(fc_ts, fc_sm)
    # Implementations differ (MLE vs statespace); loose gate for regression detection
    assert mad < max(0.5, 0.15 * np.std(y)), f"AR reference misaligned, MAD={mad}"
