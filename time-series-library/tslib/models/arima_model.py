"""
High-level ARIMA model interface

Provides a user-friendly API for ARIMA modeling with automatic model selection,
comprehensive diagnostics, and easy-to-use methods following scikit-learn conventions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, Any, Tuple, List
from ..core.base import TimeSeriesModel
from ..core.arima import ARIMAProcess
from ..core.acf_pacf import ACFPACFAnalyzer
from ..core.stationarity import StationarityAnalyzer
from ..preprocessing.transformations import DifferencingTransformer, LogTransformer
from ..preprocessing.validation import DataValidator, DataQualityReport
from ..metrics.evaluation import ModelEvaluator
from .selection import ARIMAOrderSelector


class ARIMAModel(TimeSeriesModel):
    """
    High-level ARIMA model with automatic model selection and comprehensive diagnostics
    
    Provides an easy-to-use interface for ARIMA modeling with automatic parameter
    selection, data validation, and comprehensive model evaluation.
    """
    
    def __init__(self, 
                 order: Optional[Tuple[int, int, int]] = None,
                 trend: str = 'c',
                 auto_select: bool = True,
                 max_p: int = 5,
                 max_d: int = 2,
                 max_q: int = 5,
                 seasonal: bool = False,
                 seasonal_periods: Optional[int] = None,
                 validation: bool = True,
                 n_jobs: int = -1):
        """
        Initialize ARIMA model
        
        Parameters:
        -----------
        order : tuple (p, d, q), optional
            ARIMA order. If None and auto_select=True, will be determined automatically
        trend : str
            'c' (constant), 'nc' (no constant)
        auto_select : bool
            Whether to automatically select optimal order
        max_p : int
            Maximum AR order for automatic selection
        max_d : int
            Maximum differencing order for automatic selection
        max_q : int
            Maximum MA order for automatic selection
        seasonal : bool
            Whether to include seasonal components (not implemented yet)
        seasonal_periods : int, optional
            Number of seasonal periods
        validation : bool
            Whether to validate input data
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        super().__init__()
        self.order = order
        self.trend = trend
        self.auto_select = auto_select
        self.max_p = max_p
        self.max_d = max_d
        self.n_jobs = n_jobs
        self.max_q = max_q
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.validation = validation
        
        self._arima_process = None
        self._data_validator = DataValidator() if validation else None
        self._model_evaluator = ModelEvaluator()
        self._acf_pacf_analyzer = ACFPACFAnalyzer()
        self._stationarity_analyzer = StationarityAnalyzer()
        self._order_selector = ARIMAOrderSelector(
            max_ar=max_p,
            max_ma=max_q,
            max_d=max_d,
            criterion='aic'
        )
        
        self._acf_pacf_results = None
        self._stationarity_results = None
        self._model_selection_results = None
        self._data_quality_report = None
    
    def fit(self, data: Union[np.ndarray, pd.Series, list], **kwargs) -> 'ARIMAModel':
        """
        Fit ARIMA model to data
        
        Parameters:
        -----------
        data : array-like
            Time series data
        **kwargs
            Additional fitting parameters
            
        Returns:
        --------
        self : ARIMAModel
            Fitted model
        """
        if isinstance(data, pd.Series):
            data = data.values
        else:
            data = np.asarray(data, dtype=float)
        
        if self.validation and self._data_validator:
            validation_results = self._data_validator.validate(data)
            if not validation_results['is_valid']:
                raise ValueError(f"Data validation failed: {validation_results['issues']}")
            
            # Generate data quality report
            quality_report = DataQualityReport(self._data_validator)
            self._data_quality_report = quality_report.generate_report(data)
        
        # Store original data
        self._data = data
        
        # Perform exploratory analysis
        self._perform_exploratory_analysis(data)
        
        # Determine model order
        if self.auto_select or self.order is None:
            # Use new ARIMAOrderSelector
            p, d, q = self._order_selector.select(data)
            self.order = (p, d, q)
            self._model_selection_results = {
                'best_order': self.order,
                'selection_method': 'ARIMAOrderSelector',
                'selector_results': self._order_selector.selection_results
            }
        
        # Fit the ARIMA model
        self._arima_process = ARIMAProcess(
            ar_order=self.order[0],
            diff_order=self.order[1],
            ma_order=self.order[2],
            trend=self.trend,
            n_jobs=self.n_jobs
        )
        
        self._arima_process.fit(data, **kwargs)
        
        # Store fitted parameters
        self._fitted_params = self._arima_process._fitted_params
        self._fitted = True
        
        return self
    
    def predict(self, 
                steps: int = 1, 
                return_conf_int: bool = False,
                alpha: float = 0.05,
                **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions from the fitted model
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to predict
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals
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
        
        if return_conf_int:
            predictions, conf_int = self._arima_process.predict(steps, return_conf_int=True, **kwargs)
            return predictions, conf_int
        else:
            return self._arima_process.predict(steps, return_conf_int=False, **kwargs)
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals"""
        self._validate_fitted()
        return self._arima_process.get_residuals()
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values"""
        self._validate_fitted()
        return self._arima_process.get_fitted_values()
    
    def summary(self) -> str:
        """Generate comprehensive model summary"""
        self._validate_fitted()
        
        summary = "ARIMA Model Summary\n"
        summary += "=" * 60 + "\n\n"
        
        # Model information
        summary += f"Model: ARIMA{self.order}\n"
        summary += f"Trend: {self.trend}\n"
        summary += f"Auto-selection: {self.auto_select}\n\n"
        
        # Data information
        summary += f"Data Information:\n"
        summary += f"  Length: {len(self._data)}\n"
        summary += f"  Mean: {np.mean(self._data):.6f}\n"
        summary += f"  Std: {np.std(self._data):.6f}\n\n"
        
        # Model parameters
        if self._fitted_params:
            summary += "Model Parameters:\n"
            for param_name, param_value in self._fitted_params['parameters'].items():
                summary += f"  {param_name}: {param_value:.6f}\n"
            summary += "\n"
        
        # Model statistics
        if self._fitted_params:
            summary += "Model Statistics:\n"
            summary += f"  Log-Likelihood: {self._fitted_params['log_likelihood']:.6f}\n"
            summary += f"  AIC: {self._fitted_params['aic']:.6f}\n"
            summary += f"  BIC: {self._fitted_params['bic']:.6f}\n\n"
        
        # Add exploratory analysis results
        if self._acf_pacf_results:
            summary += "ACF/PACF Analysis:\n"
            suggested = self._acf_pacf_results['suggested_orders']
            summary += f"  Suggested p: {suggested['suggested_p']}\n"
            summary += f"  Suggested q: {suggested['suggested_q']}\n\n"
        
        if self._stationarity_results:
            summary += "Stationarity Analysis:\n"
            summary += f"  Is Stationary: {self._stationarity_results['is_stationary']}\n"
            summary += f"  Suggested d: {self._stationarity_results['suggested_differencing_order']}\n\n"
        
        return summary
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot diagnostic plots for the model
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        self._validate_fitted()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'ARIMA{self.order} Model Diagnostics', fontsize=16)
        
        # Get residuals and fitted values
        residuals = self.get_residuals()
        fitted_values = self.get_fitted_values()
        
        # 1. Residuals vs Time
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals vs Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        
        # 2. Residuals vs Fitted Values
        axes[0, 1].scatter(fitted_values, residuals, alpha=0.6)
        axes[0, 1].set_title('Residuals vs Fitted Values')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        
        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # 4. ACF of Residuals
        if len(residuals) > 10:
            from ..core.acf_pacf import ACFCalculator
            acf_calc = ACFCalculator(max_lags=min(20, len(residuals)//4))
            lags, acf_values = acf_calc.calculate(residuals)
            
            axes[1, 1].bar(lags, acf_values, width=0.8)
            axes[1, 1].set_title('ACF of Residuals')
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('ACF')
            axes[1, 1].axhline(y=0, color='k', linestyle='-')
            
            # Add confidence bounds
            n = len(residuals)
            conf_bound = 1.96 / np.sqrt(n)
            axes[1, 1].axhline(y=conf_bound, color='r', linestyle='--', alpha=0.7)
            axes[1, 1].axhline(y=-conf_bound, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(self, 
                     steps: int = 10, 
                     figsize: Tuple[int, int] = (12, 6),
                     include_data: bool = True):
        """
        Plot forecast with confidence intervals
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        figsize : tuple
            Figure size
        include_data : bool
            Whether to include historical data in the plot
        """
        self._validate_fitted()
        
        # Generate forecast
        forecast, conf_int = self.predict(steps, return_conf_int=True)
        
        plt.figure(figsize=figsize)
        
        # Plot historical data
        if include_data:
            plt.plot(self._data, label='Historical Data', color='blue')
        
        # Plot forecast
        forecast_index = np.arange(len(self._data), len(self._data) + steps)
        plt.plot(forecast_index, forecast, label='Forecast', color='red', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(forecast_index, conf_int[0], conf_int[1], 
                        alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.title(f'ARIMA{self.order} Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _perform_exploratory_analysis(self, data: np.ndarray):
        """Perform exploratory data analysis"""
        # ACF/PACF analysis
        self._acf_pacf_results = self._acf_pacf_analyzer.analyze(data)
        
        # Stationarity analysis
        self._stationarity_results = self._stationarity_analyzer.analyze(data)
    
    def _select_optimal_order(self, data: np.ndarray) -> Tuple[int, int, int]:
        """
        Select optimal ARIMA order using information criteria
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        optimal_order : tuple
            Optimal (p, d, q) order
        """
        best_aic = np.inf
        best_order = (0, 0, 0)
        results = []
        
        # Get suggested differencing order
        suggested_d = self._stationarity_results['suggested_differencing_order']
        
        # Search over parameter space
        for p in range(self.max_p + 1):
            for d in range(min(suggested_d + 1, self.max_d + 1)):
                for q in range(self.max_q + 1):
                    # Skip (0,0,0) model
                    if p == 0 and q == 0:
                        continue
                    
                    try:
                        # Fit model
                        model = ARIMAProcess(p, d, q, self.trend, n_jobs=self.n_jobs)
                        model.fit(data)
                        
                        # Get AIC
                        aic = model._fitted_params['aic']
                        results.append((p, d, q, aic))
                        
                        # Update best model
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    
                    except Exception:
                        # Skip models that fail to fit
                        continue
        
        # Store model selection results
        self._model_selection_results = {
            'best_order': best_order,
            'best_aic': best_aic,
            'all_results': results
        }
        
        return best_order
    
    def get_model_selection_results(self) -> Optional[Dict[str, Any]]:
        """Get model selection results"""
        return self._model_selection_results
    
    def get_exploratory_analysis(self) -> Dict[str, Any]:
        """Get exploratory analysis results"""
        return {
            'acf_pacf': self._acf_pacf_results,
            'stationarity': self._stationarity_results,
            'data_quality': self._data_quality_report
        }
    
    def evaluate_forecast(self, 
                         actual: np.ndarray, 
                         predicted: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecast accuracy
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
            
        Returns:
        --------
        metrics : dict
            Forecast evaluation metrics
        """
        evaluation = self._model_evaluator.evaluate_model(
            self._fitted_params,
            actual=actual,
            predicted=predicted
        )
        
        return evaluation['forecast_metrics']
    
    def get_residual_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive residual diagnostics"""
        self._validate_fitted()
        
        residuals = self.get_residuals()
        fitted_values = self.get_fitted_values()
        
        evaluation = self._model_evaluator.evaluate_model(
            self._fitted_params,
            residuals=residuals,
            fitted_values=fitted_values
        )
        
        return evaluation['residual_analysis']

