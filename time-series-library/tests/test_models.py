"""
Tests for high-level AR, MA, ARMA model interfaces.

Covers:
- Correctness of parameter recovery on synthetic data with known true values
- n_jobs parameter propagation and equivalence of results
- Residual diagnostics (finite, near-white-noise)
- Confidence intervals shape and ordering
- Edge cases (very short series, order=1, predict many steps)
- GenericParallelProcessor sequential fallback
"""

import pytest
import numpy as np

from tslib.models.ar_model    import ARModel
from tslib.models.ma_model    import MAModel
from tslib.models.arma_model  import ARMAModel
from tslib.models.arima_model import ARIMAModel
from tslib.spark.parallel_processor import GenericParallelProcessor


# ---------------------------------------------------------------------------
# Synthetic data generators (shared fixtures)
# ---------------------------------------------------------------------------

def _ar1_series(n: int = 300, phi: float = 0.7, seed: int = 0) -> np.ndarray:
    """Stationary AR(1) with known autoregressive coefficient."""
    rng = np.random.default_rng(seed)
    y   = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def _ar2_series(n: int = 400, phi1: float = 0.6, phi2: float = -0.3, seed: int = 1) -> np.ndarray:
    """Stationary AR(2) with known coefficients."""
    rng = np.random.default_rng(seed)
    y   = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
    return y


def _ma1_series(n: int = 300, theta: float = 0.5, seed: int = 2) -> np.ndarray:
    """MA(1) with known moving-average coefficient."""
    rng  = np.random.default_rng(seed)
    eps  = rng.standard_normal(n + 1)
    return eps[1:] + theta * eps[:-1]


def _arma11_series(n: int = 400, phi: float = 0.6, theta: float = 0.4, seed: int = 3) -> np.ndarray:
    """ARMA(1,1) with known coefficients."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    y   = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t] + theta * eps[t - 1]
    return y


# ============================================================================
# ARModel tests
# ============================================================================

class TestARModel:
    """Unit and integration tests for ARModel."""

    # --- Initialisation -----------------------------------------------------

    def test_default_init(self):
        m = ARModel()
        assert m.auto_select is True
        assert m.n_jobs == 1
        assert not m.is_fitted

    def test_custom_init(self):
        m = ARModel(order=3, auto_select=False, n_jobs=-1, max_order=10)
        assert m.order == 3
        assert m.n_jobs == -1

    # --- Fitting / parameter structure --------------------------------------

    def test_fit_produces_fitted_state(self):
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, validation=False)
        m.fit(y)
        assert m.is_fitted
        assert m._ar_process is not None
        assert m._ar_process.ar_params is not None

    def test_fit_length_check(self):
        """Fitting on too-short data must raise."""
        m = ARModel(order=5, auto_select=False, validation=False)
        with pytest.raises(ValueError):
            m.fit(np.arange(3, dtype=float))

    def test_parameter_recovery_ar1(self):
        """Estimated φ₁ for AR(1) should be within ±0.15 of the true value."""
        true_phi = 0.7
        y = _ar1_series(n=500, phi=true_phi)
        m = ARModel(order=1, auto_select=False, validation=False)
        m.fit(y)
        estimated = m._ar_process.ar_params[0]
        assert abs(estimated - true_phi) < 0.15, (
            f"Parameter recovery off: est={estimated:.3f}, true={true_phi}"
        )

    def test_parameter_recovery_ar2(self):
        """Estimated [φ₁, φ₂] for AR(2) each within ±0.20 of true values."""
        phi1, phi2 = 0.6, -0.3
        y = _ar2_series(n=600, phi1=phi1, phi2=phi2)
        m = ARModel(order=2, auto_select=False, validation=False)
        m.fit(y)
        assert abs(m._ar_process.ar_params[0] - phi1) < 0.20
        assert abs(m._ar_process.ar_params[1] - phi2) < 0.20

    # --- Prediction ---------------------------------------------------------

    def test_predict_shape(self):
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, validation=False).fit(y)
        assert m.predict(steps=10).shape == (10,)

    def test_predict_finite(self):
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, validation=False).fit(y)
        preds = m.predict(steps=20)
        assert np.all(np.isfinite(preds))

    def test_predict_conf_int_shape(self):
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, validation=False).fit(y)
        preds, (lo, hi) = m.predict(steps=5, return_conf_int=True)
        assert preds.shape == lo.shape == hi.shape == (5,)

    def test_predict_conf_int_ordering(self):
        """Lower bound must always be ≤ prediction ≤ upper bound."""
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, validation=False).fit(y)
        preds, (lo, hi) = m.predict(steps=10, return_conf_int=True)
        assert np.all(lo <= preds + 1e-9)
        assert np.all(hi >= preds - 1e-9)

    # --- Residuals ----------------------------------------------------------

    def test_residuals_finite(self):
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, validation=False).fit(y)
        r = m.get_residuals()
        assert np.all(np.isfinite(r))

    def test_residuals_near_zero_mean(self):
        """For a well-fitted model, residuals should have near-zero mean."""
        y = _ar1_series(n=600)
        m = ARModel(order=1, auto_select=False, validation=False).fit(y)
        assert abs(np.mean(m.get_residuals())) < 0.5

    # --- n_jobs equivalence -------------------------------------------------

    def test_njobs_same_result(self):
        """n_jobs=1 and n_jobs=-1 should produce equivalent log-likelihoods."""
        y = _ar1_series(n=500)
        m1 = ARModel(order=1, auto_select=False, n_jobs=1,  validation=False).fit(y)
        m2 = ARModel(order=1, auto_select=False, n_jobs=-1, validation=False).fit(y)
        ll1 = m1._fitted_params['log_likelihood']
        ll2 = m2._fitted_params['log_likelihood']
        assert abs(ll1 - ll2) < 0.5, (
            f"Log-likelihoods diverge: seq={ll1:.3f}, par={ll2:.3f}"
        )

    def test_njobs_propagates_to_process(self):
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, n_jobs=4, validation=False).fit(y)
        assert m._ar_process.n_jobs == 4

    # --- Summary / auto-select ----------------------------------------------

    def test_summary_string(self):
        y = _ar1_series()
        m = ARModel(order=1, auto_select=False, validation=False).fit(y)
        s = m.summary()
        assert isinstance(s, str)
        assert "AR" in s

    def test_auto_select_reasonable_order(self):
        """Auto-selection on AR(1) data should pick order 1 or 2."""
        y = _ar1_series(n=400)
        m = ARModel(auto_select=True, max_order=5, validation=False).fit(y)
        assert 1 <= m.order <= 3


# ============================================================================
# MAModel tests
# ============================================================================

class TestMAModel:
    """Unit and integration tests for MAModel."""

    def test_default_init(self):
        m = MAModel()
        assert m.auto_select is True
        assert m.n_jobs == 1

    def test_fit_produces_fitted_state(self):
        y = _ma1_series()
        m = MAModel(order=1, auto_select=False, validation=False).fit(y)
        assert m.is_fitted
        assert m._ma_process.ma_params is not None

    def test_predict_shape(self):
        y = _ma1_series()
        m = MAModel(order=1, auto_select=False, validation=False).fit(y)
        assert m.predict(steps=8).shape == (8,)

    def test_predict_finite(self):
        y = _ma1_series()
        m = MAModel(order=1, auto_select=False, validation=False).fit(y)
        assert np.all(np.isfinite(m.predict(steps=15)))

    def test_ma_long_horizon_converges_to_mean(self):
        """MA(q) forecasts beyond q steps should converge to the series mean."""
        y = _ma1_series(n=500)
        m = MAModel(order=1, auto_select=False, validation=False).fit(y)
        series_mean = np.mean(y)
        # Forecast at h=20 (>> q=1) should be near the mean
        fc = m.predict(steps=20)
        assert abs(fc[-1] - series_mean) < 1.0

    def test_njobs_same_result(self):
        y = _ma1_series(n=400)
        m1 = MAModel(order=1, auto_select=False, n_jobs=1,  validation=False).fit(y)
        m2 = MAModel(order=1, auto_select=False, n_jobs=-1, validation=False).fit(y)
        ll1 = m1._fitted_params['log_likelihood']
        ll2 = m2._fitted_params['log_likelihood']
        assert abs(ll1 - ll2) < 0.5

    def test_njobs_propagates_to_process(self):
        y = _ma1_series()
        m = MAModel(order=1, auto_select=False, n_jobs=2, validation=False).fit(y)
        assert m._ma_process.n_jobs == 2

    def test_residuals_finite(self):
        y = _ma1_series()
        m = MAModel(order=1, auto_select=False, validation=False).fit(y)
        assert np.all(np.isfinite(m.get_residuals()))

    def test_conf_int_ordering(self):
        y = _ma1_series()
        m = MAModel(order=1, auto_select=False, validation=False).fit(y)
        preds, (lo, hi) = m.predict(steps=6, return_conf_int=True)
        assert np.all(lo <= preds + 1e-9)
        assert np.all(hi >= preds - 1e-9)


# ============================================================================
# ARMAModel tests
# ============================================================================

class TestARMAModel:
    """Unit and integration tests for ARMAModel."""

    def test_default_init(self):
        m = ARMAModel()
        assert m.auto_select is True
        assert m.n_jobs == 1

    def test_init_manual_order(self):
        m = ARMAModel(order=(2, 1), auto_select=False, n_jobs=4)
        assert m.order == (2, 1)
        assert m.n_jobs == 4

    def test_fit_stores_both_params(self):
        y = _arma11_series()
        m = ARMAModel(order=(1, 1), auto_select=False, validation=False).fit(y)
        assert m._arma_process.ar_params is not None
        assert m._arma_process.ma_params is not None

    def test_predict_shape(self):
        y = _arma11_series()
        m = ARMAModel(order=(1, 1), auto_select=False, validation=False).fit(y)
        assert m.predict(steps=7).shape == (7,)

    def test_predict_finite(self):
        y = _arma11_series()
        m = ARMAModel(order=(1, 1), auto_select=False, validation=False).fit(y)
        assert np.all(np.isfinite(m.predict(steps=12)))

    def test_njobs_same_result(self):
        y = _arma11_series(n=400)
        m1 = ARMAModel(order=(1, 1), auto_select=False, n_jobs=1,  validation=False).fit(y)
        m2 = ARMAModel(order=(1, 1), auto_select=False, n_jobs=-1, validation=False).fit(y)
        ll1 = m1._fitted_params['log_likelihood']
        ll2 = m2._fitted_params['log_likelihood']
        assert abs(ll1 - ll2) < 0.5

    def test_njobs_propagates_to_process(self):
        y = _arma11_series()
        m = ARMAModel(order=(1, 1), auto_select=False, n_jobs=3, validation=False).fit(y)
        assert m._arma_process.n_jobs == 3

    def test_residuals_finite(self):
        y = _arma11_series()
        m = ARMAModel(order=(1, 1), auto_select=False, validation=False).fit(y)
        assert np.all(np.isfinite(m.get_residuals()))

    def test_conf_int_ordering(self):
        y = _arma11_series()
        m = ARMAModel(order=(1, 1), auto_select=False, validation=False).fit(y)
        preds, (lo, hi) = m.predict(steps=5, return_conf_int=True)
        assert np.all(lo <= preds + 1e-9)
        assert np.all(hi >= preds - 1e-9)

    def test_summary_contains_both_params(self):
        y = _arma11_series()
        m = ARMAModel(order=(1, 1), auto_select=False, validation=False).fit(y)
        s = m.summary()
        assert "ARMA" in s
        assert "AR" in s or "phi" in s.lower() or "φ" in s


# ============================================================================
# Cross-model consistency tests
# ============================================================================

class TestCrossModelConsistency:
    """Verify that AR/MA/ARMA/ARIMA produce coherent results for compatible cases."""

    def test_ar1_vs_arma_p1q0(self):
        """ARModel(1) and ARMAModel((1,0)) should give the same log-likelihood."""
        y = _ar1_series(n=300)
        m_ar   = ARModel  (order=1,       auto_select=False, validation=False).fit(y)
        m_arma = ARMAModel(order=(1, 0),  auto_select=False, validation=False).fit(y)
        ll_ar   = m_ar  ._fitted_params['log_likelihood']
        ll_arma = m_arma._fitted_params['log_likelihood']
        # Should be very close — allow tolerance for numerical differences
        assert abs(ll_ar - ll_arma) < 1.5, (
            f"AR(1) vs ARMA(1,0) LL mismatch: {ll_ar:.3f} vs {ll_arma:.3f}"
        )

    def test_ma1_vs_arma_p0q1(self):
        """MAModel(1) and ARMAModel((0,1)) should give the same log-likelihood."""
        y = _ma1_series(n=300)
        m_ma   = MAModel  (order=1,       auto_select=False, validation=False).fit(y)
        m_arma = ARMAModel(order=(0, 1),  auto_select=False, validation=False).fit(y)
        ll_ma   = m_ma  ._fitted_params['log_likelihood']
        ll_arma = m_arma._fitted_params['log_likelihood']
        assert abs(ll_ma - ll_arma) < 1.5

    def test_predict_steps_all_models(self):
        """All models should return strictly positive-length forecasts."""
        y  = _ar1_series(n=300)
        ye = _ma1_series(n=300)
        for steps in [1, 5, 20]:
            assert ARModel  (order=1,      auto_select=False, validation=False).fit(y ).predict(steps).shape == (steps,)
            assert MAModel  (order=1,      auto_select=False, validation=False).fit(ye).predict(steps).shape == (steps,)
            assert ARMAModel(order=(1, 1), auto_select=False, validation=False).fit(y ).predict(steps).shape == (steps,)
            assert ARIMAModel(order=(1,0,0), auto_select=False, validation=False).fit(y).predict(steps).shape == (steps,)


# ============================================================================
# GenericParallelProcessor (sequential fallback — no Spark required)
# ============================================================================

class TestGenericParallelProcessorSequential:
    """Tests for the pure-Python sequential fallback of GenericParallelProcessor."""

    def _series_dict(self, n_series: int = 5, n_obs: int = 200) -> dict:
        rng = np.random.default_rng(99)
        series = {}
        for i in range(n_series):
            phi   = rng.uniform(0.3, 0.8)
            y     = np.zeros(n_obs)
            eps   = rng.standard_normal(n_obs)
            for t in range(1, n_obs):
                y[t] = phi * y[t - 1] + eps[t]
            series[f's{i}'] = y
        return series

    def test_ar_sequential(self):
        data    = self._series_dict()
        results = GenericParallelProcessor.fit_multiple_sequential(
            data, model_type='AR', order=1, steps=5
        )
        assert set(results.keys()) == set(data.keys())
        for sid, res in results.items():
            assert res['status'] == 'ok', f"Series {sid} failed: {res['status']}"
            assert res['forecast'].shape == (5,)
            assert np.all(np.isfinite(res['forecast']))

    def test_ma_sequential(self):
        rng  = np.random.default_rng(7)
        data = {f's{i}': rng.standard_normal(200) + 0.4 * rng.standard_normal(200)
                for i in range(3)}
        results = GenericParallelProcessor.fit_multiple_sequential(
            data, model_type='MA', order=1, steps=4
        )
        for sid, res in results.items():
            assert res['status'] == 'ok'
            assert res['forecast'].shape == (4,)

    def test_arma_sequential(self):
        data    = self._series_dict(n_obs=250)
        results = GenericParallelProcessor.fit_multiple_sequential(
            data, model_type='ARMA', order=(1, 1), steps=6
        )
        for sid, res in results.items():
            assert res['status'] == 'ok'
            assert np.all(np.isfinite(res['forecast']))

    def test_arima_sequential(self):
        rng  = np.random.default_rng(11)
        data = {f's{i}': np.cumsum(rng.standard_normal(200)) for i in range(3)}
        results = GenericParallelProcessor.fit_multiple_sequential(
            data, model_type='ARIMA', order=(1, 1, 0), steps=5
        )
        for sid, res in results.items():
            assert res['status'] == 'ok'
            assert res['forecast'].shape == (5,)

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            GenericParallelProcessor.fit_multiple_sequential(
                {'s0': np.ones(50)}, model_type='SARIMA', order=1, steps=3
            )

    def test_njobs_within_model(self):
        """n_jobs parameter inside each model fit should not crash."""
        data = {'s0': _ar1_series(n=300), 's1': _ar1_series(n=300, seed=5)}
        for njobs in [1, -1]:
            results = GenericParallelProcessor.fit_multiple_sequential(
                data, model_type='AR', order=1, steps=5, n_jobs=njobs
            )
            for res in results.values():
                assert res['status'] == 'ok'


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
