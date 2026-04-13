"""
ARIMA model implementation from scratch

Implements AR, MA, ARMA, and ARIMA processes with mathematical rigor.
Each process is implemented as a separate class following object-oriented principles.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from .base import TimeSeriesModel, SparkEnabled
from .optimization import MLEOptimizer
from .acf_pacf import ACFCalculator, PACFCalculator


class ARProcess(TimeSeriesModel):
    """
    AutoRegressive (AR) process implementation
    
    AR(p): y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
    
    where ε_t ~ N(0, σ²) and φ₁, φ₂, ..., φₚ are AR parameters.
    """
    
    def __init__(self, order: int, trend: str = 'c', n_jobs: int = -1):
        """
        Initialize AR process
        
        Parameters:
        -----------
        order : int
            AR order (p)
        trend : str
            'c' (constant), 'nc' (no constant)
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        super().__init__()
        self.order = order
        self.trend = trend
        self.n_jobs = n_jobs
        self.ar_params = None
        self.constant = None
        self.variance = None
        self.optimizer = MLEOptimizer(n_jobs=n_jobs)
        self.acf_calculator = ACFCalculator(n_jobs=n_jobs)
        self.pacf_calculator = PACFCalculator(n_jobs=n_jobs)
    
    def fit(self, data: Union[np.ndarray, list], **kwargs) -> 'ARProcess':
        """
        Fit AR model to data using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        data : array-like
            Time series data
        **kwargs
            Additional fitting parameters
            
        Returns:
        --------
        self : ARProcess
            Fitted AR model
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        
        if n < self.order + 1:
            raise ValueError(f"Data must have at least {self.order + 1} observations")
        
        # Estimate parameters using MLE
        results = self.optimizer.estimate(
            data, 
            model_type='AR',
            p=self.order,
            d=0,
            q=0,
            **kwargs
        )
        
        # Extract parameters
        self.ar_params = np.array([results['parameters'][f'phi_{i+1}'] for i in range(self.order)])
        self.variance = results['parameters']['sigma2']
        
        # Calculate constant if trend='c'
        if self.trend == 'c':
            self.constant = np.mean(data) * (1 - np.sum(self.ar_params))
        else:
            self.constant = 0.0
        
        # Store fitted data and results
        self._data = data
        self._fitted_params = results
        self._fitted = True
        
        return self
    
    def predict(self, steps: int = 1, return_conf_int: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions from the fitted AR model
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to predict
        return_conf_int : bool
            Whether to return confidence intervals
        **kwargs
            Additional prediction parameters
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        conf_int : tuple, optional
            Confidence intervals (lower, upper) if return_conf_int=True
        """
        self._validate_fitted()
        
        if steps <= 0:
            raise ValueError("Steps must be positive")
        
        # Get last p observations for prediction
        last_obs = self._data[-self.order:]
        predictions = np.zeros(steps)
        
        # Generate predictions step by step
        for h in range(steps):
            if h == 0:
                # First prediction
                pred = self.constant
                for i in range(self.order):
                    pred += self.ar_params[i] * last_obs[-(i+1)]
            else:
                # Subsequent predictions use previous predictions
                pred = self.constant
                for i in range(min(self.order, h)):
                    pred += self.ar_params[i] * predictions[h-1-i]
                for i in range(h, self.order):
                    pred += self.ar_params[i] * last_obs[-(i-h+1)]
            
            predictions[h] = pred
        
        if return_conf_int:
            conf_int = self._calculate_confidence_intervals(predictions, steps)
            return predictions, conf_int
        
        return predictions
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for predictions
        
        Parameters:
        -----------
        predictions : np.ndarray
            Predicted values
        steps : int
            Number of prediction steps
            
        Returns:
        --------
        lower : np.ndarray
            Lower confidence bounds
        upper : np.ndarray
            Upper confidence bounds
        """
        # Approximate confidence intervals
        # For AR models, the variance increases with forecast horizon
        std_error = np.sqrt(self.variance)
        
        # Calculate forecast variance for each step
        forecast_var = np.zeros(steps)
        for h in range(steps):
            # Simplified calculation - in practice, you'd use the proper formula
            forecast_var[h] = self.variance * (1 + h * 0.1)  # Approximate
        
        # 95% confidence intervals
        z_score = 1.96
        margin = z_score * np.sqrt(forecast_var)
        
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals"""
        self._validate_fitted()
        
        n = len(self._data)
        residuals = np.zeros(n)
        
        for t in range(self.order, n):
            predicted = self.constant
            for i in range(self.order):
                predicted += self.ar_params[i] * self._data[t - i - 1]
            residuals[t] = self._data[t] - predicted
        
        return residuals[self.order:]
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values"""
        self._validate_fitted()
        
        n = len(self._data)
        fitted = np.zeros(n)
        
        for t in range(self.order, n):
            fitted[t] = self.constant
            for i in range(self.order):
                fitted[t] += self.ar_params[i] * self._data[t - i - 1]
        
        return fitted[self.order:]
    
    def summary(self) -> str:
        """Generate model summary"""
        self._validate_fitted()
        
        results = self._fitted_params
        summary = f"AR({self.order}) Model Summary\n"
        summary += "=" * 40 + "\n"
        summary += f"Model: AR({self.order})\n"
        summary += f"Trend: {self.trend}\n"
        summary += f"Constant: {self.constant:.6f}\n\n"
        
        summary += "AR Parameters:\n"
        for i in range(self.order):
            summary += f"  φ{i+1}: {self.ar_params[i]:.6f}\n"
        
        summary += f"\nVariance: {self.variance:.6f}\n"
        summary += f"Log-Likelihood: {results['log_likelihood']:.6f}\n"
        summary += f"AIC: {results['aic']:.6f}\n"
        summary += f"BIC: {results['bic']:.6f}\n"
        
        return summary


class MAProcess(TimeSeriesModel):
    """
    Moving Average (MA) process implementation
    
    MA(q): y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}
    
    where ε_t ~ N(0, σ²) and θ₁, θ₂, ..., θ_q are MA parameters.
    """
    
    def __init__(self, order: int, n_jobs: int = -1):
        """
        Initialize MA process
        
        Parameters:
        -----------
        order : int
            MA order (q)
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        super().__init__()
        self.order = order
        self.n_jobs = n_jobs
        self.ma_params = None
        self.mean = None
        self.variance = None
        self.optimizer = MLEOptimizer(n_jobs=n_jobs)
        self.acf_calculator = ACFCalculator(n_jobs=n_jobs)
        self.pacf_calculator = PACFCalculator(n_jobs=n_jobs)
    
    def fit(self, data: Union[np.ndarray, list], **kwargs) -> 'MAProcess':
        """
        Fit MA model to data using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        data : array-like
            Time series data
        **kwargs
            Additional fitting parameters
            
        Returns:
        --------
        self : MAProcess
            Fitted MA model
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        
        if n < self.order + 1:
            raise ValueError(f"Data must have at least {self.order + 1} observations")
        
        # Estimate parameters using MLE
        results = self.optimizer.estimate(
            data, 
            model_type='MA',
            p=0,
            d=0,
            q=self.order,
            **kwargs
        )
        
        # Extract parameters
        self.ma_params = np.array([results['parameters'][f'theta_{i+1}'] for i in range(self.order)])
        self.variance = results['parameters']['sigma2']
        self.mean = np.mean(data)
        
        # Store fitted data and results
        self._data = data
        self._fitted_params = results
        self._fitted = True
        
        return self
    
    def predict(self, steps: int = 1, return_conf_int: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions from the fitted MA model
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to predict
        return_conf_int : bool
            Whether to return confidence intervals
        **kwargs
            Additional prediction parameters
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        conf_int : tuple, optional
            Confidence intervals (lower, upper) if return_conf_int=True
        """
        self._validate_fitted()
        
        if steps <= 0:
            raise ValueError("Steps must be positive")
        
        # For MA models, predictions beyond q steps are just the mean
        predictions = np.full(steps, self.mean)
        
        if return_conf_int:
            conf_int = self._calculate_confidence_intervals(predictions, steps)
            return predictions, conf_int
        
        return predictions
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        std_error = np.sqrt(self.variance)
        z_score = 1.96
        
        # For MA models, forecast variance is constant after q steps
        margin = z_score * std_error
        
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals"""
        self._validate_fitted()
        
        # Calculate residuals using innovation algorithm
        n = len(self._data)
        residuals = np.zeros(n)
        
        # Simplified residual calculation
        for t in range(self.order, n):
            predicted = self.mean
            for i in range(self.order):
                predicted += self.ma_params[i] * residuals[t - i - 1]
            residuals[t] = self._data[t] - predicted
        
        return residuals[self.order:]
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values"""
        self._validate_fitted()
        
        # Calculate full residuals (not truncated)
        n = len(self._data)
        residuals_full = np.zeros(n)
        
        for t in range(self.order, n):
            predicted = self.mean
            for i in range(self.order):
                predicted += self.ma_params[i] * residuals_full[t - i - 1]
            residuals_full[t] = self._data[t] - predicted
        
        # Calculate fitted values using residuals
        fitted = np.zeros(n)
        for t in range(self.order, n):
            fitted[t] = self.mean
            for i in range(self.order):
                fitted[t] += self.ma_params[i] * residuals_full[t - i - 1]
        
        return fitted[self.order:]
    
    def summary(self) -> str:
        """Generate model summary"""
        self._validate_fitted()
        
        results = self._fitted_params
        summary = f"MA({self.order}) Model Summary\n"
        summary += "=" * 40 + "\n"
        summary += f"Model: MA({self.order})\n"
        summary += f"Mean: {self.mean:.6f}\n\n"
        
        summary += "MA Parameters:\n"
        for i in range(self.order):
            summary += f"  θ{i+1}: {self.ma_params[i]:.6f}\n"
        
        summary += f"\nVariance: {self.variance:.6f}\n"
        summary += f"Log-Likelihood: {results['log_likelihood']:.6f}\n"
        summary += f"AIC: {results['aic']:.6f}\n"
        summary += f"BIC: {results['bic']:.6f}\n"
        
        return summary


class ARMAProcess(TimeSeriesModel):
    """
    AutoRegressive Moving Average (ARMA) process implementation
    
    ARMA(p,q): y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
    
    Combines AR and MA components.
    """
    
    def __init__(self, ar_order: int, ma_order: int, trend: str = 'c', n_jobs: int = -1):
        """
        Initialize ARMA process
        
        Parameters:
        -----------
        ar_order : int
            AR order (p)
        ma_order : int
            MA order (q)
        trend : str
            'c' (constant), 'nc' (no constant)
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        super().__init__()
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.trend = trend
        self.n_jobs = n_jobs
        self.ar_params = None
        self.ma_params = None
        self.constant = None
        self.variance = None
        self.optimizer = MLEOptimizer(n_jobs=n_jobs)
        self.acf_calculator = ACFCalculator(n_jobs=n_jobs)
        self.pacf_calculator = PACFCalculator(n_jobs=n_jobs)
    
    def fit(self, data: Union[np.ndarray, list], **kwargs) -> 'ARMAProcess':
        """
        Fit ARMA model to data using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        data : array-like
            Time series data
        **kwargs
            Additional fitting parameters
            
        Returns:
        --------
        self : ARMAProcess
            Fitted ARMA model
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        
        if n < max(self.ar_order, self.ma_order) + 1:
            raise ValueError(f"Data must have at least {max(self.ar_order, self.ma_order) + 1} observations")
        
        # Estimate parameters using MLE
        results = self.optimizer.estimate(
            data, 
            model_type='ARMA',
            p=self.ar_order,
            d=0,
            q=self.ma_order,
            **kwargs
        )
        
        # Extract parameters
        if self.ar_order > 0:
            self.ar_params = np.array([results['parameters'][f'phi_{i+1}'] for i in range(self.ar_order)])
        else:
            self.ar_params = np.array([])
        
        if self.ma_order > 0:
            self.ma_params = np.array([results['parameters'][f'theta_{i+1}'] for i in range(self.ma_order)])
        else:
            self.ma_params = np.array([])
        
        self.variance = results['parameters']['sigma2']
        
        # Calculate constant if trend='c'
        if self.trend == 'c':
            self.constant = np.mean(data) * (1 - np.sum(self.ar_params))
        else:
            self.constant = 0.0
        
        # Store fitted data and results
        self._data = data
        self._fitted_params = results
        self._fitted = True
        
        return self
    
    def predict(self, steps: int = 1, return_conf_int: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions from the fitted ARMA model
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to predict
        return_conf_int : bool
            Whether to return confidence intervals
        **kwargs
            Additional prediction parameters
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        conf_int : tuple, optional
            Confidence intervals (lower, upper) if return_conf_int=True
        """
        self._validate_fitted()
        
        if steps <= 0:
            raise ValueError("Steps must be positive")
        
        # Get residuals for MA component
        residuals = self.get_residuals()
        
        # Get last observations for AR component
        last_obs = self._data[-self.ar_order:] if self.ar_order > 0 else np.array([])
        
        predictions = np.zeros(steps)
        
        # Generate predictions step by step
        for h in range(steps):
            pred = self.constant
            
            # AR component
            for i in range(self.ar_order):
                if h - i - 1 >= 0:
                    pred += self.ar_params[i] * predictions[h - i - 1]
                else:
                    pred += self.ar_params[i] * last_obs[-(i-h+1)]
            
            # MA component (only for first q steps)
            for i in range(self.ma_order):
                if h - i - 1 >= 0:
                    # Use previous prediction residuals (approximate)
                    pred += self.ma_params[i] * 0  # Simplified
                else:
                    pred += self.ma_params[i] * residuals[-(i-h+1)]
            
            predictions[h] = pred
        
        if return_conf_int:
            conf_int = self._calculate_confidence_intervals(predictions, steps)
            return predictions, conf_int
        
        return predictions
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        std_error = np.sqrt(self.variance)
        
        # Calculate forecast variance for each step
        forecast_var = np.zeros(steps)
        for h in range(steps):
            # Simplified calculation
            forecast_var[h] = self.variance * (1 + h * 0.1)
        
        z_score = 1.96
        margin = z_score * np.sqrt(forecast_var)
        
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals"""
        self._validate_fitted()
        
        # Calculate residuals using innovation algorithm
        n = len(self._data)
        residuals = np.zeros(n)
        
        max_order = max(self.ar_order, self.ma_order)
        
        for t in range(max_order, n):
            predicted = self.constant
            
            # AR component
            for i in range(self.ar_order):
                predicted += self.ar_params[i] * self._data[t - i - 1]
            
            # MA component
            for i in range(self.ma_order):
                predicted += self.ma_params[i] * residuals[t - i - 1]
            
            residuals[t] = self._data[t] - predicted
        
        return residuals[max_order:]
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values"""
        self._validate_fitted()
        
        # Calculate full residuals (not truncated)
        n = len(self._data)
        max_order = max(self.ar_order, self.ma_order)
        residuals_full = np.zeros(n)
        
        for t in range(max_order, n):
            predicted = self.constant
            
            # AR component
            for i in range(self.ar_order):
                predicted += self.ar_params[i] * self._data[t - i - 1]
            
            # MA component
            for i in range(self.ma_order):
                predicted += self.ma_params[i] * residuals_full[t - i - 1]
            
            residuals_full[t] = self._data[t] - predicted
        
        # Calculate fitted values using full residuals
        fitted = np.zeros(n)
        for t in range(max_order, n):
            fitted[t] = self.constant
            
            # AR component
            for i in range(self.ar_order):
                fitted[t] += self.ar_params[i] * self._data[t - i - 1]
            
            # MA component
            for i in range(self.ma_order):
                fitted[t] += self.ma_params[i] * residuals_full[t - i - 1]
        
        return fitted[max_order:]
    
    def summary(self) -> str:
        """Generate model summary"""
        self._validate_fitted()
        
        results = self._fitted_params
        summary = f"ARMA({self.ar_order},{self.ma_order}) Model Summary\n"
        summary += "=" * 50 + "\n"
        summary += f"Model: ARMA({self.ar_order},{self.ma_order})\n"
        summary += f"Trend: {self.trend}\n"
        summary += f"Constant: {self.constant:.6f}\n\n"
        
        if self.ar_order > 0:
            summary += "AR Parameters:\n"
            for i in range(self.ar_order):
                summary += f"  φ{i+1}: {self.ar_params[i]:.6f}\n"
            summary += "\n"
        
        if self.ma_order > 0:
            summary += "MA Parameters:\n"
            for i in range(self.ma_order):
                summary += f"  θ{i+1}: {self.ma_params[i]:.6f}\n"
            summary += "\n"
        
        summary += f"Variance: {self.variance:.6f}\n"
        summary += f"Log-Likelihood: {results['log_likelihood']:.6f}\n"
        summary += f"AIC: {results['aic']:.6f}\n"
        summary += f"BIC: {results['bic']:.6f}\n"
        
        return summary


class ARIMAProcess(ARMAProcess):
    """
    AutoRegressive Integrated Moving Average (ARIMA) process implementation
    
    ARIMA(p,d,q): (1-φ₁B-...-φₚB^p)(1-B)^d y_t = c + (1+θ₁B+...+θ_qB^q)ε_t
    
    where B is the backshift operator and d is the order of differencing.
    """
    
    def __init__(self, ar_order: int, diff_order: int, ma_order: int, trend: str = 'c', n_jobs: int = -1):
        """
        Initialize ARIMA process
        
        Parameters:
        -----------
        ar_order : int
            AR order (p)
        diff_order : int
            Differencing order (d)
        ma_order : int
            MA order (q)
        trend : str
            'c' (constant), 'nc' (no constant)
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        super().__init__(ar_order, ma_order, trend, n_jobs=n_jobs)
        self.diff_order = diff_order
        self.original_data = None
        self.differenced_data = None
    
    def fit(self, data: Union[np.ndarray, list], **kwargs) -> 'ARIMAProcess':
        """
        Fit ARIMA model to data using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        data : array-like
            Time series data
        **kwargs
            Additional fitting parameters
            
        Returns:
        --------
        self : ARIMAProcess
            Fitted ARIMA model
        """
        data = np.asarray(data, dtype=float)
        self.original_data = data.copy()
        
        # Apply differencing
        if self.diff_order > 0:
            self.differenced_data = self._apply_differencing(data, self.diff_order)
        else:
            self.differenced_data = data.copy()
        
        # Fit ARMA model to differenced data
        super().fit(self.differenced_data, **kwargs)
        
        return self
    
    def _apply_differencing(self, data: np.ndarray, d: int) -> np.ndarray:
        """
        Apply differencing of order d
        
        Parameters:
        -----------
        data : np.ndarray
            Original time series
        d : int
            Order of differencing
            
        Returns:
        --------
        diff_data : np.ndarray
            Differenced time series
        """
        diff_data = data.copy()
        
        for _ in range(d):
            diff_data = np.diff(diff_data)
        
        return diff_data
    
    def predict(self, steps: int = 1, return_conf_int: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions from the fitted ARIMA model
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to predict
        return_conf_int : bool
            Whether to return confidence intervals
        **kwargs
            Additional prediction parameters
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values in original scale
        conf_int : tuple, optional
            Confidence intervals (lower, upper) if return_conf_int=True
        """
        self._validate_fitted()
        
        if steps <= 0:
            raise ValueError("Steps must be positive")
        
        # Get predictions from ARMA model (in differenced scale)
        if return_conf_int:
            diff_predictions, diff_conf_int = super().predict(steps, return_conf_int=True, **kwargs)
        else:
            diff_predictions = super().predict(steps, return_conf_int=False, **kwargs)
        
        # Convert back to original scale
        predictions = self._integrate_predictions(diff_predictions, steps)
        
        if return_conf_int:
            # Convert confidence intervals back to original scale
            lower_conf = self._integrate_predictions(diff_conf_int[0], steps)
            upper_conf = self._integrate_predictions(diff_conf_int[1], steps)
            return predictions, (lower_conf, upper_conf)
        
        return predictions
    
    def _integrate_predictions(self, diff_predictions: np.ndarray, steps: int) -> np.ndarray:
        """
        Convert differenced predictions back to original scale
        
        Parameters:
        -----------
        diff_predictions : np.ndarray
            Predictions in differenced scale
        steps : int
            Number of prediction steps
            
        Returns:
        --------
        predictions : np.ndarray
            Predictions in original scale
        """
        if self.diff_order == 0:
            return diff_predictions
        
        # Start with the last value of original data
        predictions = np.zeros(steps)
        last_value = self.original_data[-1]
        
        # Apply inverse differencing
        for i in range(steps):
            if i == 0:
                predictions[i] = last_value + diff_predictions[i]
            else:
                predictions[i] = predictions[i-1] + diff_predictions[i]
        
        return predictions
    
    def summary(self) -> str:
        """Generate model summary"""
        self._validate_fitted()
        
        results = self._fitted_params
        summary = f"ARIMA({self.ar_order},{self.diff_order},{self.ma_order}) Model Summary\n"
        summary += "=" * 60 + "\n"
        summary += f"Model: ARIMA({self.ar_order},{self.diff_order},{self.ma_order})\n"
        summary += f"Trend: {self.trend}\n"
        summary += f"Constant: {self.constant:.6f}\n\n"
        
        if self.ar_order > 0:
            summary += "AR Parameters:\n"
            for i in range(self.ar_order):
                summary += f"  φ{i+1}: {self.ar_params[i]:.6f}\n"
            summary += "\n"
        
        if self.ma_order > 0:
            summary += "MA Parameters:\n"
            for i in range(self.ma_order):
                summary += f"  θ{i+1}: {self.ma_params[i]:.6f}\n"
            summary += "\n"
        
        summary += f"Differencing Order: {self.diff_order}\n"
        summary += f"Variance: {self.variance:.6f}\n"
        summary += f"Log-Likelihood: {results['log_likelihood']:.6f}\n"
        summary += f"AIC: {results['aic']:.6f}\n"
        summary += f"BIC: {results['bic']:.6f}\n"
        
        return summary
