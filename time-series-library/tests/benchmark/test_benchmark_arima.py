"""
ARIMA model: sequential n_jobs, Spark, optional statsmodels reference.
"""

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    arima_like_series_dict,
    forecasts_from_sequential,
    forecasts_from_spark_collected,
    mean_abs_diff,
    mean_parallel_agreement_pct,
    timed_call,
)

pytestmark = pytest.mark.benchmark


@pytest.fixture
def arima_data():
    return arima_like_series_dict(n_series=5, n_obs=90, seed=404)


def test_arima_sequential_n_jobs_variants(arima_data):
    order = (1, 1, 1)
    steps = 5
    _, t1 = timed_call(
        forecasts_from_sequential, arima_data, "ARIMA", order, steps, n_jobs=1
    )
    fc2, t2 = timed_call(
        forecasts_from_sequential, arima_data, "ARIMA", order, steps, n_jobs=-1
    )
    assert len(fc2) == len(arima_data)
    for sid in arima_data:
        assert np.all(np.isfinite(fc2[sid]))
    print(f"[ARIMA sequential] n_jobs=1: {t1:.4f}s, n_jobs=-1: {t2:.4f}s")


@pytest.mark.spark
def test_arima_spark_vs_sequential(arima_data, spark_session_benchmark):
    from tslib.spark.parallel_processor import GenericParallelProcessor

    order = (1, 1, 1)
    steps = 5
    fc_seq, t_seq = timed_call(
        forecasts_from_sequential, arima_data, "ARIMA", order, steps, n_jobs=1
    )

    rows = [(sid, float(v)) for sid, y in arima_data.items() for v in y]
    pdf = pd.DataFrame(rows, columns=["series_id", "y"])
    sdf = spark_session_benchmark.createDataFrame(pdf)

    proc = GenericParallelProcessor(
        model_type="ARIMA",
        spark=spark_session_benchmark,
        n_jobs=1,
        master="local[2]",
        app_name="bench-ARIMA",
    )
    out, t_spark = timed_call(
        proc.fit_multiple, sdf, "series_id", "y", order, steps
    )
    collected = out.collect()
    assert len(collected) == len(arima_data) * steps
    assert all(r.status == "ok" for r in collected)
    fc_spark = forecasts_from_spark_collected(collected)
    agree = mean_parallel_agreement_pct(fc_seq, fc_spark)
    print(
        f"[ARIMA] sequential: {t_seq:.4f}s, spark: {t_spark:.4f}s, "
        f"parallel vs sequential agreement: {agree:.2f}% (experimental Spark path)"
    )


def test_arima_statsmodels_reference_alignment(arima_data):
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.arima.model import ARIMA

    y = arima_data["s0"]
    steps = 5
    order = (1, 1, 1)

    from tslib.models.arima_model import ARIMAModel

    m = ARIMAModel(order=order, auto_select=False, validation=False, n_jobs=1)
    m.fit(y)
    fc_ts = m.predict(steps=steps)

    # statsmodels disallows trend="c" when d>0 (constant removed by differencing)
    sm = ARIMA(y, order=order, trend="n").fit()
    fc_sm = np.asarray(sm.forecast(steps=steps), dtype=float)

    mad = mean_abs_diff(fc_ts, fc_sm)
    assert mad < max(1.5, 0.3 * np.std(y)), f"ARIMA reference misaligned, MAD={mad}"
