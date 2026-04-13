"""
High-level ARMA model interface

Provides a user-friendly API for ARMA modeling with automatic order selection,
comprehensive diagnostics, and easy-to-use methods following scikit-learn conventions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, Any, Tuple
from ..core.base import TimeSeriesModel
from ..core.arima import ARMAProcess
from ..core.acf_pacf import ACFPACFAnalyzer
from ..core.stationarity import StationarityAnalyzer
from ..preprocessing.validation import DataValidator, DataQualityReport
from ..metrics.evaluation import ModelEvaluator
from .selection import ARMAOrderSelector


class ARMAModel(TimeSeriesModel):
    """
    High-level ARMA model with automatic order selection and comprehensive diagnostics
    
    Autoregressive Moving Average (ARMA) models combine AR and MA components
    to model stationary time series with both autocorrelation and moving average
    structure. The model automatically identifies optimal orders using ACF/PACF 
    analysis and grid search.
    
    The ARMA(p,q) model is defined as:
        y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
    
    where ε_t is white noise, φ₁, ..., φₚ are AR parameters, and θ₁, ..., θ_q are MA parameters.
    """
    
    def __init__(self,
                 order: Optional[Tuple[int, int]] = None,
                 trend: str = 'c',
                 auto_select: bool = True,
                 max_ar: int = 5,
                 max_ma: int = 5,
                 criterion: str = 'aic',
                 validation: bool = True,
                 n_jobs: int = 1):
        """
        Initialize ARMA model
        
        Parameters:
        -----------
        order : tuple (p, q), optional
            ARMA order. If None and auto_select=True, will be determined automatically
        trend : str
            'c' (constant/intercept), 'nc' (no constant)
        auto_select : bool
            Whether to automatically select optimal order
        max_ar : int
            Maximum AR order to consider for automatic selection
        max_ma : int
            Maximum MA order to consider for automatic selection
        criterion : str
            Information criterion for model selection: 'aic' or 'bic'
        validation : bool
            Whether to validate input data
        n_jobs : int
            Number of parallel threads for ACF/PACF and MLE computation.
            1 = sequential (default), -1 = all available cores.
            Parallelism pays off for n_obs > ~2 000.
        """
        super().__init__()
        self.order = order
        self.trend = trend
        self.auto_select = auto_select
        self.max_ar = max_ar
        self.max_ma = max_ma
        self.criterion = criterion
        self.validation = validation
        self.n_jobs = n_jobs
        
        # Initialize components
        self._arma_process = None
        self._data_validator = DataValidator() if validation else None
        self._model_evaluator = ModelEvaluator()
        self._acf_pacf_analyzer = ACFPACFAnalyzer()
        self._stationarity_analyzer = StationarityAnalyzer()
        self._order_selector = ARMAOrderSelector(max_ar, max_ma, criterion)
        
        # Store analysis results
        self._acf_pacf_results = None
        self._stationarity_results = None
        self._order_selection_results = None
        self._data_quality_report = None
    
    def fit(self, data: Union[np.ndarray, pd.Series, list], **kwargs) -> 'ARMAModel':
        """
        Fit ARMA model to data
        
        Parameters:
        -----------
        data : array-like
            Time series data (should be stationary or will be checked)
        **kwargs
            Additional fitting parameters
            
        Returns:
        --------
        self : ARMAModel
            Fitted model
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        else:
            data = np.asarray(data, dtype=float)
        
        # Validate data if requested
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
        
        # Check stationarity
        if not self._stationarity_results['is_stationary']:
            import warnings
            warnings.warn(
                "Data appears to be non-stationary. ARMA models require stationary data. "
                "Consider differencing the data first or using ARIMA model instead.",
                UserWarning
            )
        
        # Determine model order
        if self.auto_select or self.order is None:
            p, q = self._order_selector.select(data)
            self.order = (p, q)
            self._order_selection_results = self._order_selector.selection_results
            
            # Ensure at least (1,1) if selection gives (0,0)
            if self.order == (0, 0):
                self.order = (1, 1)
        
        # Fit the ARMA process (n_jobs controls ACF/PACF + MLE parallelism)
        self._arma_process = ARMAProcess(
            ar_order=self.order[0],
            ma_order=self.order[1],
            trend=self.trend,
            n_jobs=self.n_jobs
        )
        
        self._arma_process.fit(data, **kwargs)
        
        # Store fitted parameters
        self._fitted_params = self._arma_process._fitted_params
        self._fitted = True
        
        return self
    
    def predict(self, 
                steps: int = 1, 
                return_conf_int: bool = False,
                alpha: float = 0.05,
                **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate predictions from the fitted model
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to predict
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals (default: 0.05 for 95% CI)
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
            predictions, conf_int = self._arma_process.predict(steps, return_conf_int=True, **kwargs)
            return predictions, conf_int
        else:
            return self._arma_process.predict(steps, return_conf_int=False, **kwargs)
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals"""
        self._validate_fitted()
        return self._arma_process.get_residuals()
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values"""
        self._validate_fitted()
        return self._arma_process.get_fitted_values()
    
    def summary(self) -> str:
        """Generate comprehensive model summary"""
        self._validate_fitted()
        
        summary = "ARMA Model Summary\n"
        summary += "=" * 60 + "\n\n"
        
        # Model information
        summary += f"Model: ARMA({self.order[0]},{self.order[1]})\n"
        summary += f"Trend: {self.trend}\n"
        summary += f"Auto-selection: {self.auto_select}\n"
        if self.auto_select:
            summary += f"Selection criterion: {self.criterion}\n"
        summary += "\n"
        
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
        
        # Add ACF/PACF analysis results
        if self._acf_pacf_results:
            summary += "ACF/PACF Analysis:\n"
            suggested = self._acf_pacf_results['suggested_orders']
            summary += f"  Suggested p (from PACF): {suggested['suggested_p']}\n"
            summary += f"  Suggested q (from ACF): {suggested['suggested_q']}\n"
            summary += f"  Note: Both ACF and PACF decay gradually for ARMA\n\n"
        
        # Stationarity analysis
        if self._stationarity_results:
            summary += "Stationarity Analysis:\n"
            summary += f"  Is Stationary: {self._stationarity_results['is_stationary']}\n"
            if not self._stationarity_results['is_stationary']:
                summary += f"  Warning: ARMA requires stationary data!\n"
            summary += "\n"
        
        return summary
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot diagnostic plots for the model
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        self._validate_fitted()
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'ARMA({self.order[0]},{self.order[1]}) Model Diagnostics', fontsize=16)
        
        # Get residuals and fitted values
        residuals = self.get_residuals()
        fitted_values = self.get_fitted_values()
        
        # 1. Residuals vs Time
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals vs Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Fitted Values
        axes[0, 1].scatter(fitted_values, residuals, alpha=0.6)
        axes[0, 1].set_title('Residuals vs Fitted Values')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Histogram of Residuals
        axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Histogram of Residuals')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. ACF Plot
        if self._acf_pacf_results:
            acf_values = self._acf_pacf_results['acf_values']
            acf_lags = self._acf_pacf_results['acf_lags']
            
            axes[2, 0].bar(acf_lags, acf_values, width=0.8)
            axes[2, 0].set_title('ACF (Gradual Decay for ARMA)')
            axes[2, 0].set_xlabel('Lag')
            axes[2, 0].set_ylabel('ACF')
            axes[2, 0].axhline(y=0, color='k', linestyle='-')
            
            # Add confidence bounds
            n = len(self._data)
            conf_bound = 1.96 / np.sqrt(n)
            axes[2, 0].axhline(y=conf_bound, color='r', linestyle='--', alpha=0.7, label='95% CI')
            axes[2, 0].axhline(y=-conf_bound, color='r', linestyle='--', alpha=0.7)
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. PACF Plot
        if self._acf_pacf_results:
            pacf_values = self._acf_pacf_results['pacf_values']
            pacf_lags = self._acf_pacf_results['pacf_lags']
            
            axes[2, 1].bar(pacf_lags, pacf_values, width=0.8)
            axes[2, 1].set_title('PACF (Gradual Decay for ARMA)')
            axes[2, 1].set_xlabel('Lag')
            axes[2, 1].set_ylabel('PACF')
            axes[2, 1].axhline(y=0, color='k', linestyle='-')
            axes[2, 1].axhline(y=conf_bound, color='r', linestyle='--', alpha=0.7, label='95% CI')
            axes[2, 1].axhline(y=-conf_bound, color='r', linestyle='--', alpha=0.7)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
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
        
        # Add mean line
        mean_value = np.mean(self._data)
        plt.axhline(y=mean_value, color='green', linestyle=':', alpha=0.7,
                   label=f'Mean ({mean_value:.2f})')
        
        plt.title(f'ARMA({self.order[0]},{self.order[1]}) Forecast')
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
    
    def get_order_selection_results(self) -> Optional[Dict[str, Any]]:
        """Get order selection results"""
        return self._order_selection_results
    
    def get_exploratory_analysis(self) -> Dict[str, Any]:
        """Get exploratory analysis results"""
        return {
            'acf_pacf': self._acf_pacf_results,
            'stationarity': self._stationarity_results,
            'data_quality': self._data_quality_report,
            'order_selection': self._order_selection_results
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

