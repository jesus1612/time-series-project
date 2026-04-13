"""
Order selection utilities for time series models

Provides model-specific order selectors using ACF/PACF analysis and
information criteria for automatic model identification.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
from ..core.acf_pacf import ACFCalculator, PACFCalculator, ACFPACFAnalyzer
from ..core.stationarity import StationarityAnalyzer, ADFTest, KPSSTest


class OrderSelector(ABC):
    """
    Abstract base class for order selection
    
    Defines the interface for model-specific order selection algorithms
    """
    
    def __init__(self, max_order: int = 5, alpha: float = 0.05):
        """
        Initialize order selector
        
        Parameters:
        -----------
        max_order : int
            Maximum order to consider
        alpha : float
            Significance level for statistical tests
        """
        self.max_order = max_order
        self.alpha = alpha
        self._selection_results = None
    
    @abstractmethod
    def select(self, data: np.ndarray, **kwargs) -> int:
        """
        Select optimal order for the model
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        **kwargs
            Additional parameters
            
        Returns:
        --------
        order : int
            Optimal model order
        """
        pass
    
    @property
    def selection_results(self) -> Optional[Dict[str, Any]]:
        """Get detailed selection results"""
        return self._selection_results


class AROrderSelector(OrderSelector):
    """
    AR order selector using PACF analysis
    
    The PACF of an AR(p) process cuts off after lag p, which is the
    key characteristic used for identification.
    """
    
    def __init__(self, max_order: int = 5, alpha: float = 0.05, method: str = 'pacf'):
        """
        Initialize AR order selector
        
        Parameters:
        -----------
        max_order : int
            Maximum AR order to consider
        alpha : float
            Significance level for PACF cutoff
        method : str
            Selection method: 'pacf' (PACF cutoff) or 'aic' (information criterion)
        """
        super().__init__(max_order, alpha)
        self.method = method
        self.pacf_calculator = PACFCalculator(max_lags=max_order)
    
    def select(self, data: np.ndarray, **kwargs) -> int:
        """
        Select optimal AR order
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data (must be stationary)
        **kwargs
            Additional parameters
            
        Returns:
        --------
        order : int
            Optimal AR order
        """
        if self.method == 'pacf':
            return self._select_by_pacf(data)
        elif self.method == 'aic':
            return self._select_by_aic(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _select_by_pacf(self, data: np.ndarray) -> int:
        """
        Select order by PACF cutoff
        
        The PACF should be significant up to lag p and then cut off.
        """
        n = len(data)
        lags, pacf_values = self.pacf_calculator.calculate(data)
        
        # Calculate confidence bounds
        conf_bound = 1.96 / np.sqrt(n)
        
        # Find last significant lag
        last_significant_lag = 0
        for lag_idx, lag in enumerate(lags[1:], start=1):  # Skip lag 0
            if abs(pacf_values[lag_idx]) > conf_bound:
                last_significant_lag = lag
        
        # Store results
        self._selection_results = {
            'method': 'pacf',
            'pacf_values': pacf_values.tolist(),
            'lags': lags.tolist(),
            'confidence_bound': conf_bound,
            'last_significant_lag': last_significant_lag,
            'selected_order': min(last_significant_lag, self.max_order)
        }
        
        return min(last_significant_lag, self.max_order)
    
    def _select_by_aic(self, data: np.ndarray) -> int:
        """
        Select order by minimizing AIC
        
        Fits AR models of different orders and selects the one with lowest AIC.
        """
        from ..core.arima import ARProcess
        
        best_aic = np.inf
        best_order = 1
        aic_values = []
        
        for p in range(1, self.max_order + 1):
            try:
                model = ARProcess(order=p, trend='c', n_jobs=1)
                model.fit(data)
                aic = model._fitted_params['aic']
                aic_values.append((p, aic))
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = p
            except Exception:
                # Skip orders that fail to fit
                continue
        
        self._selection_results = {
            'method': 'aic',
            'aic_values': aic_values,
            'best_aic': best_aic,
            'selected_order': best_order
        }
        
        return best_order


class MAOrderSelector(OrderSelector):
    """
    MA order selector using ACF analysis
    
    The ACF of an MA(q) process cuts off after lag q, which is the
    key characteristic used for identification.
    """
    
    def __init__(self, max_order: int = 5, alpha: float = 0.05, method: str = 'acf'):
        """
        Initialize MA order selector
        
        Parameters:
        -----------
        max_order : int
            Maximum MA order to consider
        alpha : float
            Significance level for ACF cutoff
        method : str
            Selection method: 'acf' (ACF cutoff) or 'aic' (information criterion)
        """
        super().__init__(max_order, alpha)
        self.method = method
        self.acf_calculator = ACFCalculator(max_lags=max_order)
    
    def select(self, data: np.ndarray, **kwargs) -> int:
        """
        Select optimal MA order
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data (must be stationary)
        **kwargs
            Additional parameters
            
        Returns:
        --------
        order : int
            Optimal MA order
        """
        if self.method == 'acf':
            return self._select_by_acf(data)
        elif self.method == 'aic':
            return self._select_by_aic(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _select_by_acf(self, data: np.ndarray) -> int:
        """
        Select order by ACF cutoff
        
        The ACF should be significant up to lag q and then cut off.
        """
        n = len(data)
        lags, acf_values = self.acf_calculator.calculate(data)
        
        # Calculate confidence bounds
        conf_bound = 1.96 / np.sqrt(n)
        
        # Find last significant lag (skip lag 0 which is always 1)
        last_significant_lag = 0
        for lag_idx, lag in enumerate(lags[1:], start=1):
            if abs(acf_values[lag_idx]) > conf_bound:
                last_significant_lag = lag
        
        # Store results
        self._selection_results = {
            'method': 'acf',
            'acf_values': acf_values.tolist(),
            'lags': lags.tolist(),
            'confidence_bound': conf_bound,
            'last_significant_lag': last_significant_lag,
            'selected_order': min(last_significant_lag, self.max_order)
        }
        
        return min(last_significant_lag, self.max_order)
    
    def _select_by_aic(self, data: np.ndarray) -> int:
        """
        Select order by minimizing AIC
        
        Fits MA models of different orders and selects the one with lowest AIC.
        """
        from ..core.arima import MAProcess
        
        best_aic = np.inf
        best_order = 1
        aic_values = []
        
        for q in range(1, self.max_order + 1):
            try:
                model = MAProcess(order=q, n_jobs=1)
                model.fit(data)
                aic = model._fitted_params['aic']
                aic_values.append((q, aic))
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = q
            except Exception:
                # Skip orders that fail to fit
                continue
        
        self._selection_results = {
            'method': 'aic',
            'aic_values': aic_values,
            'best_aic': best_aic,
            'selected_order': best_order
        }
        
        return best_order


class ARMAOrderSelector(OrderSelector):
    """
    ARMA order selector using combined ACF/PACF analysis and grid search
    
    For ARMA models, both ACF and PACF decay gradually, making identification
    more challenging. Uses grid search with AIC/BIC to find optimal (p,q).
    """
    
    def __init__(self, max_ar: int = 5, max_ma: int = 5, 
                 criterion: str = 'aic', alpha: float = 0.05):
        """
        Initialize ARMA order selector
        
        Parameters:
        -----------
        max_ar : int
            Maximum AR order to consider
        max_ma : int
            Maximum MA order to consider
        criterion : str
            Information criterion: 'aic' or 'bic'
        alpha : float
            Significance level
        """
        super().__init__(max_order=max(max_ar, max_ma), alpha=alpha)
        self.max_ar = max_ar
        self.max_ma = max_ma
        self.criterion = criterion
        self.acf_pacf_analyzer = ACFPACFAnalyzer()
    
    def select(self, data: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Select optimal ARMA order
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data (must be stationary)
        **kwargs
            Additional parameters
            
        Returns:
        --------
        order : tuple (p, q)
            Optimal ARMA order
        """
        # First, get suggestions from ACF/PACF
        acf_pacf_results = self.acf_pacf_analyzer.analyze(data)
        suggested_p = acf_pacf_results['suggested_orders']['suggested_p']
        suggested_q = acf_pacf_results['suggested_orders']['suggested_q']
        
        # Perform grid search around suggested values
        return self._grid_search(data, suggested_p, suggested_q)
    
    def _grid_search(self, data: np.ndarray, 
                    center_p: int, center_q: int) -> Tuple[int, int]:
        """
        Grid search for optimal (p,q) using information criterion
        
        Searches around suggested values from ACF/PACF analysis.
        """
        from ..core.arima import ARMAProcess
        
        best_criterion = np.inf
        best_order = (1, 1)
        results = []
        
        # Search grid
        for p in range(self.max_ar + 1):
            for q in range(self.max_ma + 1):
                # Skip (0,0)
                if p == 0 and q == 0:
                    continue
                
                try:
                    model = ARMAProcess(ar_order=p, ma_order=q, trend='c', n_jobs=1)
                    model.fit(data)
                    
                    criterion_value = (model._fitted_params[self.criterion]
                                      if self.criterion in model._fitted_params
                                      else model._fitted_params['aic'])
                    
                    results.append({
                        'p': p,
                        'q': q,
                        self.criterion: criterion_value
                    })
                    
                    if criterion_value < best_criterion:
                        best_criterion = criterion_value
                        best_order = (p, q)
                
                except Exception:
                    # Skip models that fail to fit
                    continue
        
        # Store results
        self._selection_results = {
            'method': f'grid_search_{self.criterion}',
            'suggested_from_acf_pacf': (center_p, center_q),
            'all_results': results,
            f'best_{self.criterion}': best_criterion,
            'selected_order': best_order
        }
        
        return best_order


class ARIMAOrderSelector(OrderSelector):
    """
    ARIMA order selector adding stationarity tests to determine d
    
    Combines stationarity testing (ADF, KPSS) to determine differencing order d,
    then uses ARMA order selection for (p,q).
    """
    
    def __init__(self, max_ar: int = 5, max_ma: int = 5, max_d: int = 2,
                 criterion: str = 'aic', alpha: float = 0.05):
        """
        Initialize ARIMA order selector
        
        Parameters:
        -----------
        max_ar : int
            Maximum AR order to consider
        max_ma : int
            Maximum MA order to consider
        max_d : int
            Maximum differencing order to consider
        criterion : str
            Information criterion: 'aic' or 'bic'
        alpha : float
            Significance level for stationarity tests
        """
        super().__init__(max_order=max(max_ar, max_ma), alpha=alpha)
        self.max_ar = max_ar
        self.max_ma = max_ma
        self.max_d = max_d
        self.criterion = criterion
        self.stationarity_analyzer = StationarityAnalyzer()
        self.arma_selector = ARMAOrderSelector(max_ar, max_ma, criterion, alpha)
    
    def select(self, data: np.ndarray, **kwargs) -> Tuple[int, int, int]:
        """
        Select optimal ARIMA order
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data (can be non-stationary)
        **kwargs
            Additional parameters
            
        Returns:
        --------
        order : tuple (p, d, q)
            Optimal ARIMA order
        """
        # Step 1: Determine differencing order d
        d = self._select_differencing_order(data)
        
        # Step 2: Apply differencing
        differenced_data = data.copy()
        for _ in range(d):
            differenced_data = np.diff(differenced_data)
        
        # Step 3: Select ARMA order on differenced data
        p, q = self.arma_selector.select(differenced_data)
        
        # Store results
        self._selection_results = {
            'differencing_order': d,
            'ar_order': p,
            'ma_order': q,
            'stationarity_results': self._stationarity_results,
            'arma_selection_results': self.arma_selector.selection_results
        }
        
        return (p, d, q)
    
    def _select_differencing_order(self, data: np.ndarray) -> int:
        """
        Determine optimal differencing order using stationarity tests
        
        Uses ADF and KPSS tests to determine if differencing is needed.
        """
        # Analyze stationarity
        stationarity_results = self.stationarity_analyzer.analyze(data)
        self._stationarity_results = stationarity_results
        
        suggested_d = stationarity_results['suggested_differencing_order']
        
        # Additional verification: try different d values and check AIC
        if suggested_d > 0:
            best_d = self._verify_d_with_model_selection(data, suggested_d)
            return min(best_d, self.max_d)
        
        return 0
    
    def _verify_d_with_model_selection(self, data: np.ndarray, suggested_d: int) -> int:
        """
        Verify differencing order by fitting models with different d values
        
        Sometimes stationarity tests can be ambiguous, so we verify by
        comparing model fit.
        """
        from ..core.arima import ARIMAProcess
        
        best_criterion = np.inf
        best_d = suggested_d
        
        # Try d and d-1 (if d > 0)
        d_candidates = [suggested_d]
        if suggested_d > 0:
            d_candidates.append(suggested_d - 1)
        if suggested_d < self.max_d:
            d_candidates.append(suggested_d + 1)
        
        for d in d_candidates:
            if d < 0 or d > self.max_d:
                continue
            
            try:
                # Fit a simple ARIMA(1,d,1) to compare
                model = ARIMAProcess(ar_order=1, diff_order=d, ma_order=1, 
                                    trend='c', n_jobs=1)
                model.fit(data)
                
                criterion_value = model._fitted_params.get(self.criterion, 
                                                          model._fitted_params.get('aic', np.inf))
                
                if criterion_value < best_criterion:
                    best_criterion = criterion_value
                    best_d = d
            
            except Exception:
                # Skip if model fails to fit
                continue
        
        return best_d


class AutoOrderSelector:
    """
    Automatic model and order selector
    
    Determines whether AR, MA, ARMA, or ARIMA is most appropriate,
    and selects optimal orders.
    """
    
    def __init__(self, max_ar: int = 5, max_ma: int = 5, max_d: int = 2,
                 criterion: str = 'aic', alpha: float = 0.05):
        """
        Initialize automatic selector
        
        Parameters:
        -----------
        max_ar : int
            Maximum AR order
        max_ma : int
            Maximum MA order
        max_d : int
            Maximum differencing order
        criterion : str
            Information criterion for model selection
        alpha : float
            Significance level
        """
        self.max_ar = max_ar
        self.max_ma = max_ma
        self.max_d = max_d
        self.criterion = criterion
        self.alpha = alpha
        
        self.arima_selector = ARIMAOrderSelector(max_ar, max_ma, max_d, 
                                                 criterion, alpha)
    
    def select(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Automatically select best model type and order
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        selection : dict
            Selected model type and order
        """
        # Use ARIMA selector which handles all cases
        p, d, q = self.arima_selector.select(data)
        
        # Determine model type based on orders
        if d == 0:
            if p > 0 and q == 0:
                model_type = 'AR'
                order = p
            elif p == 0 and q > 0:
                model_type = 'MA'
                order = q
            elif p > 0 and q > 0:
                model_type = 'ARMA'
                order = (p, q)
            else:
                # Default to AR(1) if all zeros
                model_type = 'AR'
                order = 1
        else:
            model_type = 'ARIMA'
            order = (p, d, q)
        
        return {
            'model_type': model_type,
            'order': order,
            'full_order': (p, d, q),
            'selection_results': self.arima_selector.selection_results
        }

