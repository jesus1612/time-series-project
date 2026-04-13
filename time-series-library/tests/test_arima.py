"""
Tests for ARIMA model implementation

Tests the core ARIMA functionality including AR, MA, ARMA, and ARIMA processes.
"""

import pytest
import numpy as np
import pandas as pd
from tslib.core.arima import ARProcess, MAProcess, ARMAProcess, ARIMAProcess
from tslib.models.arima_model import ARIMAModel


class TestARProcess:
    """Test AR process implementation"""
    
    def test_ar_process_initialization(self):
        """Test AR process initialization"""
        ar = ARProcess(order=2, trend='c')
        assert ar.order == 2
        assert ar.trend == 'c'
        assert not ar.is_fitted
    
    def test_ar_process_fit(self):
        """Test AR process fitting"""
        # Generate synthetic AR(1) data
        np.random.seed(42)
        n = 100
        phi = 0.7
        sigma = 1.0
        epsilon = np.random.normal(0, sigma, n)
        
        y = np.zeros(n)
        y[0] = epsilon[0]
        for t in range(1, n):
            y[t] = phi * y[t-1] + epsilon[t]
        
        # Fit AR(1) model
        ar = ARProcess(order=1, trend='c')
        ar.fit(y)
        
        assert ar.is_fitted
        assert ar.ar_params is not None
        assert len(ar.ar_params) == 1
        assert abs(ar.ar_params[0] - phi) < 0.2  # Should be close to true parameter
    
    def test_ar_process_predict(self):
        """Test AR process prediction"""
        # Generate synthetic data
        np.random.seed(42)
        n = 50
        phi = 0.5
        epsilon = np.random.normal(0, 1, n)
        
        y = np.zeros(n)
        y[0] = epsilon[0]
        for t in range(1, n):
            y[t] = phi * y[t-1] + epsilon[t]
        
        # Fit and predict
        ar = ARProcess(order=1, trend='c')
        ar.fit(y)
        
        predictions = ar.predict(steps=5)
        assert len(predictions) == 5
        assert not np.any(np.isnan(predictions))
    
    def test_ar_process_residuals(self):
        """Test AR process residuals"""
        # Generate synthetic data
        np.random.seed(42)
        n = 50
        phi = 0.5
        epsilon = np.random.normal(0, 1, n)
        
        y = np.zeros(n)
        y[0] = epsilon[0]
        for t in range(1, n):
            y[t] = phi * y[t-1] + epsilon[t]
        
        # Fit model
        ar = ARProcess(order=1, trend='c')
        ar.fit(y)
        
        residuals = ar.get_residuals()
        assert len(residuals) > 0
        assert not np.any(np.isnan(residuals))


class TestMAProcess:
    """Test MA process implementation"""
    
    def test_ma_process_initialization(self):
        """Test MA process initialization"""
        ma = MAProcess(order=2)
        assert ma.order == 2
        assert not ma.is_fitted
    
    def test_ma_process_fit(self):
        """Test MA process fitting"""
        # Generate synthetic MA(1) data
        np.random.seed(42)
        n = 100
        theta = 0.5
        sigma = 1.0
        epsilon = np.random.normal(0, sigma, n)
        
        y = np.zeros(n)
        for t in range(n):
            if t == 0:
                y[t] = epsilon[t]
            else:
                y[t] = epsilon[t] + theta * epsilon[t-1]
        
        # Fit MA(1) model
        ma = MAProcess(order=1)
        ma.fit(y)
        
        assert ma.is_fitted
        assert ma.ma_params is not None
        assert len(ma.ma_params) == 1
    
    def test_ma_process_predict(self):
        """Test MA process prediction"""
        # Generate synthetic data
        np.random.seed(42)
        n = 50
        theta = 0.5
        epsilon = np.random.normal(0, 1, n)
        
        y = np.zeros(n)
        for t in range(n):
            if t == 0:
                y[t] = epsilon[t]
            else:
                y[t] = epsilon[t] + theta * epsilon[t-1]
        
        # Fit and predict
        ma = MAProcess(order=1)
        ma.fit(y)
        
        predictions = ma.predict(steps=5)
        assert len(predictions) == 5
        assert not np.any(np.isnan(predictions))


class TestARMAProcess:
    """Test ARMA process implementation"""
    
    def test_arma_process_initialization(self):
        """Test ARMA process initialization"""
        arma = ARMAProcess(ar_order=1, ma_order=1, trend='c')
        assert arma.ar_order == 1
        assert arma.ma_order == 1
        assert arma.trend == 'c'
        assert not arma.is_fitted
    
    def test_arma_process_fit(self):
        """Test ARMA process fitting"""
        # Generate synthetic ARMA(1,1) data
        np.random.seed(42)
        n = 100
        phi = 0.7
        theta = 0.5
        sigma = 1.0
        epsilon = np.random.normal(0, sigma, n)
        
        y = np.zeros(n)
        y[0] = epsilon[0]
        for t in range(1, n):
            y[t] = phi * y[t-1] + epsilon[t] + theta * epsilon[t-1]
        
        # Fit ARMA(1,1) model
        arma = ARMAProcess(ar_order=1, ma_order=1, trend='c')
        arma.fit(y)
        
        assert arma.is_fitted
        assert arma.ar_params is not None
        assert arma.ma_params is not None
        assert len(arma.ar_params) == 1
        assert len(arma.ma_params) == 1


class TestARIMAProcess:
    """Test ARIMA process implementation"""
    
    def test_arima_process_initialization(self):
        """Test ARIMA process initialization"""
        arima = ARIMAProcess(ar_order=1, diff_order=1, ma_order=1, trend='c')
        assert arima.ar_order == 1
        assert arima.diff_order == 1
        assert arima.ma_order == 1
        assert arima.trend == 'c'
        assert not arima.is_fitted
    
    def test_arima_process_fit(self):
        """Test ARIMA process fitting"""
        # Generate synthetic ARIMA(1,1,1) data
        np.random.seed(42)
        n = 100
        
        # Generate non-stationary data with trend
        t = np.arange(n)
        trend = 0.1 * t
        noise = np.random.normal(0, 1, n)
        y = trend + noise
        
        # Fit ARIMA(1,1,1) model
        arima = ARIMAProcess(ar_order=1, diff_order=1, ma_order=1, trend='c')
        arima.fit(y)
        
        assert arima.is_fitted
        assert arima.ar_params is not None
        assert arima.ma_params is not None
    
    def test_arima_process_predict(self):
        """Test ARIMA process prediction"""
        # Generate synthetic data with trend
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        trend = 0.05 * t
        noise = np.random.normal(0, 1, n)
        y = trend + noise
        
        # Fit and predict
        arima = ARIMAProcess(ar_order=1, diff_order=1, ma_order=1, trend='c')
        arima.fit(y)
        
        predictions = arima.predict(steps=5)
        assert len(predictions) == 5
        assert not np.any(np.isnan(predictions))


class TestARIMAModel:
    """Test high-level ARIMA model interface"""
    
    def test_arima_model_initialization(self):
        """Test ARIMA model initialization"""
        model = ARIMAModel(order=(1, 1, 1), auto_select=False)
        assert model.order == (1, 1, 1)
        assert model.auto_select is False
        assert not model.is_fitted
    
    def test_arima_model_fit(self):
        """Test ARIMA model fitting"""
        # Generate synthetic data
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        trend = 0.02 * t
        noise = np.random.normal(0, 1, n)
        y = trend + noise
        
        # Fit model
        model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
        model.fit(y)
        
        assert model.is_fitted
        assert model._arima_process is not None
    
    def test_arima_model_auto_select(self):
        """Test automatic model selection"""
        # Generate synthetic data
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        trend = 0.02 * t
        noise = np.random.normal(0, 1, n)
        y = trend + noise
        
        # Fit model with auto selection
        model = ARIMAModel(auto_select=True, validation=False)
        model.fit(y)
        
        assert model.is_fitted
        assert model.order is not None
        assert len(model.order) == 3
    
    def test_arima_model_predict(self):
        """Test ARIMA model prediction"""
        # Generate synthetic data
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        trend = 0.02 * t
        noise = np.random.normal(0, 1, n)
        y = trend + noise
        
        # Fit and predict
        model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
        model.fit(y)
        
        predictions = model.predict(steps=5)
        assert len(predictions) == 5
        assert not np.any(np.isnan(predictions))
        
        # Test with confidence intervals
        predictions, conf_int = model.predict(steps=5, return_conf_int=True)
        assert len(predictions) == 5
        assert len(conf_int) == 2
        assert len(conf_int[0]) == 5
        assert len(conf_int[1]) == 5
    
    def test_arima_model_summary(self):
        """Test model summary generation"""
        # Generate synthetic data
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        trend = 0.02 * t
        noise = np.random.normal(0, 1, n)
        y = trend + noise
        
        # Fit model
        model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
        model.fit(y)
        
        summary = model.summary()
        assert isinstance(summary, str)
        assert "ARIMA" in summary
        assert "Model Summary" in summary
    
    def test_arima_model_residuals(self):
        """Test residual analysis"""
        # Generate synthetic data
        np.random.seed(42)
        n = 50
        t = np.arange(n)
        trend = 0.02 * t
        noise = np.random.normal(0, 1, n)
        y = trend + noise
        
        # Fit model
        model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
        model.fit(y)
        
        residuals = model.get_residuals()
        fitted_values = model.get_fitted_values()
        
        assert len(residuals) > 0
        assert len(fitted_values) > 0
        assert not np.any(np.isnan(residuals))
        assert not np.any(np.isnan(fitted_values))


if __name__ == "__main__":
    pytest.main([__file__])




