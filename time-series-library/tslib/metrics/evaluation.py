"""
Model evaluation metrics for time series analysis

Implements comprehensive metrics for model assessment:
- Information criteria (AIC, BIC)
- Forecast accuracy metrics (RMSE, MAE, MAPE)
- Residual analysis and diagnostic tests
"""

import warnings

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from scipy import stats
from ..core.base import BaseTest


class InformationCriteria:
    """
    Information criteria for model selection and comparison
    
    Implements AIC, BIC, and other information criteria for model evaluation
    """
    
    @staticmethod
    def aic(log_likelihood: float, n_params: int) -> float:
        """
        Calculate Akaike Information Criterion (AIC)
        
        AIC = 2k - 2ln(L)
        
        Parameters:
        -----------
        log_likelihood : float
            Log-likelihood of the model
        n_params : int
            Number of parameters
            
        Returns:
        --------
        aic : float
            AIC value
        """
        return 2 * n_params - 2 * log_likelihood
    
    @staticmethod
    def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
        """
        Calculate Bayesian Information Criterion (BIC)
        
        BIC = k*ln(n) - 2ln(L)
        
        Parameters:
        -----------
        log_likelihood : float
            Log-likelihood of the model
        n_params : int
            Number of parameters
        n_obs : int
            Number of observations
            
        Returns:
        --------
        bic : float
            BIC value
        """
        return n_params * np.log(n_obs) - 2 * log_likelihood
    
    @staticmethod
    def aicc(log_likelihood: float, n_params: int, n_obs: int) -> float:
        """
        Calculate Akaike Information Criterion with correction (AICc)
        
        AICc = AIC + 2k(k+1)/(n-k-1)
        
        Parameters:
        -----------
        log_likelihood : float
            Log-likelihood of the model
        n_params : int
            Number of parameters
        n_obs : int
            Number of observations
            
        Returns:
        --------
        aicc : float
            AICc value
        """
        aic = InformationCriteria.aic(log_likelihood, n_params)
        correction = 2 * n_params * (n_params + 1) / (n_obs - n_params - 1)
        return aic + correction
    
    @staticmethod
    def hqic(log_likelihood: float, n_params: int, n_obs: int) -> float:
        """
        Calculate Hannan-Quinn Information Criterion (HQIC)
        
        HQIC = 2k*ln(ln(n)) - 2ln(L)
        
        Parameters:
        -----------
        log_likelihood : float
            Log-likelihood of the model
        n_params : int
            Number of parameters
        n_obs : int
            Number of observations
            
        Returns:
        --------
        hqic : float
            HQIC value
        """
        return 2 * n_params * np.log(np.log(n_obs)) - 2 * log_likelihood


class ForecastMetrics:
    """
    Forecast accuracy metrics for time series predictions
    
    Implements various metrics to evaluate forecast quality
    """
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error (RMSE)
        
        RMSE = sqrt(mean((actual - predicted)²))
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        rmse : float
            RMSE value
        """
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE)
        
        MAE = mean(|actual - predicted|)
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        mae : float
            MAE value
        """
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        MAPE = mean(|actual - predicted| / |actual|) * 100
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        mape : float
            MAPE value (in percentage)
        """
        # Avoid division by zero
        mask = actual != 0
        if np.sum(mask) == 0:
            return np.inf
        
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    @staticmethod
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
        
        sMAPE = mean(2|actual - predicted| / (|actual| + |predicted|)) * 100
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        smape : float
            sMAPE value (in percentage)
        """
        denominator = np.abs(actual) + np.abs(predicted)
        mask = denominator != 0
        
        if np.sum(mask) == 0:
            return np.inf
        
        return np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def mase(actual: np.ndarray, predicted: np.ndarray, 
             seasonal_naive: Optional[np.ndarray] = None) -> float:
        """
        Calculate Mean Absolute Scaled Error (MASE)
        
        MASE = MAE / MAE_naive
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
        seasonal_naive : np.ndarray, optional
            Seasonal naive forecasts. If None, uses naive forecast
            
        Returns:
        --------
        mase : float
            MASE value
        """
        mae = ForecastMetrics.mae(actual, predicted)
        
        if seasonal_naive is None:
            # Use naive forecast (previous value)
            naive_forecast = np.roll(actual, 1)
            naive_forecast[0] = actual[0]  # First value
        else:
            naive_forecast = seasonal_naive
        
        mae_naive = ForecastMetrics.mae(actual, naive_forecast)
        
        if mae_naive == 0:
            return np.inf if mae > 0 else 0
        
        return mae / mae_naive
    
    @staticmethod
    def theil_u(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Theil's U statistic
        
        U = sqrt(mean((actual - predicted)²)) / sqrt(mean(actual²)) + sqrt(mean(predicted²))
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        theil_u : float
            Theil's U statistic
        """
        numerator = np.sqrt(np.mean((actual - predicted) ** 2))
        denominator = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(predicted ** 2))
        
        if denominator == 0:
            return np.inf
        
        return numerator / denominator
    
    @staticmethod
    def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction changes)
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
            predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        accuracy : float
            Directional accuracy (0-1)
        """
        if len(actual) < 2:
            return 0.0
        
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        
        return np.mean(actual_direction == predicted_direction)


def _lag_correlation(residuals: np.ndarray, lag: int, n: int) -> float:
    """Pearson ACF at lag; avoids corrcoef when either segment is (near-)constant."""
    if lag >= n or n < 2:
        return 0.0
    r = np.asarray(residuals, dtype=float).ravel()
    a, b = r[:-lag], r[lag:]
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


class ResidualAnalyzer:
    """
    Residual analysis for model diagnostics
    
    Implements various tests and statistics for residual analysis
    """
    
    def __init__(self):
        self._residuals = None
        self._fitted_values = None
    
    def analyze(self, residuals: np.ndarray, fitted_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis
        
        Parameters:
        -----------
        residuals : np.ndarray
            Model residuals
        fitted_values : np.ndarray, optional
            Fitted values
            
        Returns:
        --------
        analysis : dict
            Residual analysis results
        """
        self._residuals = residuals
        self._fitted_values = fitted_values
        
        analysis = {
            'basic_stats': self._calculate_basic_stats(residuals),
            'normality_tests': self._test_normality(residuals),
            'autocorrelation_tests': self._test_autocorrelation(residuals),
            'heteroscedasticity_tests': self._test_heteroscedasticity(residuals, fitted_values),
            'ljung_box_test': self._ljung_box_test(residuals)
        }
        
        return analysis
    
    def _calculate_basic_stats(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calculate basic residual statistics"""
        r = np.asarray(residuals, dtype=float).ravel()
        n = len(r)
        std_r = float(np.std(r)) if n else 0.0
        mean_r = float(np.mean(r)) if n else 0.0
        if n < 2 or std_r < 1e-15:
            return {
                'mean': mean_r,
                'std': std_r,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'jarque_bera_stat': float('nan'),
                'jarque_bera_pvalue': float('nan'),
            }
        jb_stat, jb_pvalue = stats.jarque_bera(r)
        return {
            'mean': mean_r,
            'std': std_r,
            'skewness': float(stats.skew(r)),
            'kurtosis': float(stats.kurtosis(r)),
            'jarque_bera_stat': float(jb_stat),
            'jarque_bera_pvalue': float(jb_pvalue),
        }
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residual normality"""
        r = np.asarray(residuals, dtype=float).ravel()
        n = len(r)
        std_r = float(np.std(r)) if n else 0.0
        mean_r = float(np.mean(r)) if n else 0.0
        degenerate = n < 3 or std_r < 1e-15 or (float(np.max(r)) - float(np.min(r))) < 1e-15

        shapiro_stat = shapiro_pvalue = np.nan
        if not degenerate and n <= 5000:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                shapiro_stat, shapiro_pvalue = stats.shapiro(r)

        ks_stat = ks_pvalue = np.nan
        if not degenerate and std_r >= 1e-15:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                ks_stat, ks_pvalue = stats.kstest(r, "norm", args=(mean_r, std_r))

        jb_stat = jb_pvalue = np.nan
        if not degenerate and n >= 2:
            jb_stat, jb_pvalue = stats.jarque_bera(r)

        return {
            'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_pvalue},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_pvalue},
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue},
        }
    
    def _test_autocorrelation(self, residuals: np.ndarray, max_lags: int = 10) -> Dict[str, Any]:
        """Test residual autocorrelation"""
        n = len(residuals)
        max_lags = min(max_lags, n // 4)
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, max_lags + 1):
            autocorrs.append(_lag_correlation(residuals, lag, n))
        
        # Q-statistic (Ljung-Box test)
        q_stat = n * (n + 2) * np.sum([ac**2 / (n - lag) for lag, ac in enumerate(autocorrs, 1)])
        q_pvalue = 1 - stats.chi2.cdf(q_stat, max_lags)
        
        return {
            'autocorrelations': autocorrs,
            'q_statistic': q_stat,
            'q_pvalue': q_pvalue,
            'max_lag': max_lags
        }
    
    def _test_heteroscedasticity(self, residuals: np.ndarray, 
                                fitted_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Test for heteroscedasticity"""
        if fitted_values is None:
            return {'breusch_pagan': None, 'white': None}
        
        # Breusch-Pagan test (simplified)
        # Regress squared residuals on fitted values
        try:
            from sklearn.linear_model import LinearRegression
            
            X = fitted_values.reshape(-1, 1)
            y = residuals ** 2
            
            model = LinearRegression().fit(X, y)
            r_squared = model.score(X, y)
            
            # LM statistic
            n = len(residuals)
            lm_stat = n * r_squared
            lm_pvalue = 1 - stats.chi2.cdf(lm_stat, 1)
            
            return {
                'breusch_pagan': {
                    'lm_statistic': lm_stat,
                    'p_value': lm_pvalue,
                    'r_squared': r_squared
                }
            }
        except ImportError:
            return {'breusch_pagan': None}
    
    def _ljung_box_test(self, residuals: np.ndarray, max_lags: int = 10) -> Dict[str, Any]:
        """Ljung-Box test for residual autocorrelation"""
        n = len(residuals)
        max_lags = min(max_lags, n // 4)
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, max_lags + 1):
            autocorrs.append(_lag_correlation(residuals, lag, n))
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum([ac**2 / (n - lag) for lag, ac in enumerate(autocorrs, 1)])
        lb_pvalue = 1 - stats.chi2.cdf(lb_stat, max_lags)
        
        return {
            'statistic': lb_stat,
            'p_value': lb_pvalue,
            'max_lags': max_lags
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    
    Combines all evaluation metrics and provides model comparison capabilities
    """
    
    def __init__(self):
        self.info_criteria = InformationCriteria()
        self.forecast_metrics = ForecastMetrics()
        self.residual_analyzer = ResidualAnalyzer()
    
    def evaluate_model(self, 
                      model_results: Dict[str, Any],
                      actual: Optional[np.ndarray] = None,
                      predicted: Optional[np.ndarray] = None,
                      residuals: Optional[np.ndarray] = None,
                      fitted_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Parameters:
        -----------
        model_results : dict
            Model fitting results (from MLE optimization)
        actual : np.ndarray, optional
            Actual values for forecast evaluation
        predicted : np.ndarray, optional
            Predicted values for forecast evaluation
        residuals : np.ndarray, optional
            Model residuals for diagnostic analysis
        fitted_values : np.ndarray, optional
            Fitted values for diagnostic analysis
            
        Returns:
        --------
        evaluation : dict
            Comprehensive evaluation results
        """
        evaluation = {
            'information_criteria': {},
            'forecast_metrics': {},
            'residual_analysis': {},
            'model_summary': {}
        }
        
        # Information criteria
        if 'log_likelihood' in model_results and 'aic' in model_results:
            evaluation['information_criteria'] = {
                'aic': model_results['aic'],
                'bic': model_results['bic'],
                'log_likelihood': model_results['log_likelihood']
            }
        
        # Forecast metrics
        if actual is not None and predicted is not None:
            evaluation['forecast_metrics'] = {
                'rmse': self.forecast_metrics.rmse(actual, predicted),
                'mae': self.forecast_metrics.mae(actual, predicted),
                'mape': self.forecast_metrics.mape(actual, predicted),
                'smape': self.forecast_metrics.smape(actual, predicted),
                'mase': self.forecast_metrics.mase(actual, predicted),
                'theil_u': self.forecast_metrics.theil_u(actual, predicted),
                'directional_accuracy': self.forecast_metrics.directional_accuracy(actual, predicted)
            }
        
        # Residual analysis
        if residuals is not None:
            evaluation['residual_analysis'] = self.residual_analyzer.analyze(residuals, fitted_values)
        
        # Model summary
        evaluation['model_summary'] = {
            'model_type': model_results.get('model_type', 'Unknown'),
            'orders': model_results.get('orders', {}),
            'n_parameters': len(model_results.get('parameters', {})),
            'n_observations': model_results.get('n_observations', len(residuals) if residuals is not None else 0)
        }
        
        return evaluation
    
    def compare_models(self, evaluations: list) -> Dict[str, Any]:
        """
        Compare multiple model evaluations
        
        Parameters:
        -----------
        evaluations : list
            List of model evaluation dictionaries
            
        Returns:
        --------
        comparison : dict
            Model comparison results
        """
        if len(evaluations) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        comparison = {
            'model_rankings': {},
            'best_model': {},
            'summary_table': {}
        }
        
        # Extract metrics for comparison
        aic_values = []
        bic_values = []
        rmse_values = []
        
        for i, eval_result in enumerate(evaluations):
            if 'information_criteria' in eval_result:
                aic_values.append(eval_result['information_criteria'].get('aic', np.inf))
                bic_values.append(eval_result['information_criteria'].get('bic', np.inf))
            
            if 'forecast_metrics' in eval_result:
                rmse_values.append(eval_result['forecast_metrics'].get('rmse', np.inf))
        
        # Rank models by different criteria
        if aic_values:
            aic_ranks = np.argsort(aic_values)
            comparison['model_rankings']['aic'] = aic_ranks.tolist()
        
        if bic_values:
            bic_ranks = np.argsort(bic_values)
            comparison['model_rankings']['bic'] = bic_ranks.tolist()
        
        if rmse_values:
            rmse_ranks = np.argsort(rmse_values)
            comparison['model_rankings']['rmse'] = rmse_ranks.tolist()
        
        # Determine best model (lowest AIC)
        if aic_values:
            best_idx = np.argmin(aic_values)
            comparison['best_model'] = {
                'index': best_idx,
                'aic': aic_values[best_idx],
                'model_type': evaluations[best_idx]['model_summary'].get('model_type', 'Unknown')
            }
        
        return comparison




