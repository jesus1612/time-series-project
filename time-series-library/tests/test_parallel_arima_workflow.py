"""
Regression tests for ParallelARIMAWorkflow (requires PySpark + Java).

Skip entirely when Spark is not available.
"""

import numpy as np
import pytest

from tslib.utils.checks import check_spark_availability

pytestmark = pytest.mark.skipif(
    not check_spark_availability(),
    reason="PySpark / Java not available",
)


def test_workflow_fit_smoke_and_results_keys():
    from tslib.spark import ParallelARIMAWorkflow

    rng = np.random.default_rng(99)
    n = 220
    eps = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.75 * y[t - 1] + eps[t]

    wf = ParallelARIMAWorkflow(verbose=False, grid_mode="auto_n", d_max=2)
    wf.fit(y)
    assert wf.fitted_
    assert wf.order_ is not None
    res = wf.get_results()
    assert "order" in res
    assert "config" in res
    assert res["config"].get("grid_mode") == "auto_n"
    assert "step1_differencing" in res["step_results"]
    step1 = res["step_results"]["step1_differencing"]
    assert "stationarity_results" in step1
    assert "iterations" in step1["stationarity_results"]


def test_grid_mode_acf_pacf_sets_identification():
    from tslib.spark import ParallelARIMAWorkflow

    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(280))

    wf = ParallelARIMAWorkflow(verbose=False, grid_mode="acf_pacf", d_max=2)
    wf.fit(y)
    assert "acf_pacf_identification" in wf.results_
    meta = wf.results_["acf_pacf_identification"]
    assert "significant_acf_lags" in meta


def test_manual_grid_mode_requires_bounds():
    from tslib.spark import ParallelARIMAWorkflow

    rng = np.random.default_rng(2)
    y = rng.standard_normal(150)

    with pytest.raises(ValueError, match="manual"):
        wf = ParallelARIMAWorkflow(
            verbose=False,
            grid_mode="manual",
            manual_max_p=None,
            manual_max_q=2,
        )
        wf.fit(y)
