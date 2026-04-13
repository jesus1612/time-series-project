"""
Stationarity tests implementation from scratch

Implements Augmented Dickey-Fuller (ADF) and KPSS tests for testing
the stationarity of time series data. These tests are crucial for
determining the order of differencing (d) in ARIMA models.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats
from .base import BaseTest


class ADFTest(BaseTest):
    """
    Augmented Dickey-Fuller test implementation from scratch
    
    Tests the null hypothesis that a unit root is present in a time series.
    If the test statistic is less than the critical value, we reject the null
    hypothesis and conclude the series is stationary.
    """
    
    def __init__(self, max_lags: Optional[int] = None, regression: str = 'c'):
        """
        Initialize ADF test
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags to include in the test. If None, uses AIC
        regression : str
            Type of regression: 'c' (constant), 'ct' (constant and trend), 'n' (none)
        """
        self.max_lags = max_lags
        self.regression = regression
        self._test_statistic = None
        self._p_value = None
        self._critical_values = None
        self._used_lags = None
    
    def test(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data to test
        **kwargs
            Additional test parameters
            
        Returns:
        --------
        results : Dict[str, Any]
            Test results including statistic, p-value, critical values
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 4:
            raise ValueError("Data must have at least 4 observations")
        
        # Determine optimal number of lags if not specified
        if self.max_lags is None:
            optimal_lags = self._select_lags(data)
        else:
            optimal_lags = min(self.max_lags, n // 4)
        
        # Perform the test
        test_stat, p_value, critical_values = self._perform_adf_test(data, optimal_lags)
        
        # Store results
        self._test_statistic = test_stat
        self._p_value = p_value
        self._critical_values = critical_values
        self._used_lags = optimal_lags
        
        # Determine stationarity
        is_stationary = test_stat < critical_values['5%']
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'used_lags': optimal_lags,
            'is_stationary': is_stationary,
            'regression': self.regression
        }
    
    def _select_lags(self, data: np.ndarray) -> int:
        """
        Select optimal number of lags using AIC
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        optimal_lags : int
            Optimal number of lags
        """
        n = len(data)
        max_lags = min(n // 4, 12)  # Standard practice
        
        best_aic = np.inf
        optimal_lags = 0
        
        for lags in range(max_lags + 1):
            try:
                aic = self._calculate_aic(data, lags)
                if aic < best_aic:
                    best_aic = aic
                    optimal_lags = lags
            except Exception:
                continue
        
        return optimal_lags
    
    def _calculate_aic(self, data: np.ndarray, lags: int) -> float:
        """
        Calculate AIC for given number of lags
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        lags : int
            Number of lags
            
        Returns:
        --------
        float
            Akaike Information Criterion, or ``inf`` if undefined.
        """
        # Create lagged differences
        diff_data = np.diff(data)
        n = len(diff_data)
        
        if n <= lags:
            return np.inf
        
        # Create design matrix
        X, y = self._create_design_matrix(data, lags)
        
        if X.shape[0] == 0:
            return np.inf
        
        # Calculate OLS
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            mse = float(np.mean(residuals ** 2))
            # log(mse) is undefined for non-positive MSE (perfect/invalid fit)
            if not np.isfinite(mse) or mse <= 0:
                return np.inf
            k = X.shape[1]
            aic = n * np.log(mse) + 2 * k
            return float(aic) if np.isfinite(aic) else np.inf
        except Exception:
            return np.inf
    
    def _create_design_matrix(self, data: np.ndarray, lags: int) -> tuple:
        """
        Create design matrix for ADF regression
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        lags : int
            Number of lags
            
        Returns:
        --------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Dependent variable (first differences)
        """
        n = len(data)
        diff_data = np.diff(data)
        
        # Start from lag + 1 to have enough observations
        start_idx = lags + 1
        end_idx = n - 1
        
        if end_idx <= start_idx:
            return np.array([]).reshape(0, 0), np.array([])
        
        y = diff_data[start_idx:end_idx]
        
        # Create lagged differences
        X = np.zeros((len(y), lags))
        for i in range(lags):
            X[:, i] = diff_data[start_idx - i - 1:end_idx - i - 1]
        
        # Add constant and/or trend based on regression type
        if self.regression == 'c':
            X = np.column_stack([np.ones(len(y)), X])
        elif self.regression == 'ct':
            trend = np.arange(len(y))
            X = np.column_stack([np.ones(len(y)), trend, X])
        # For 'n' (none), no additional terms
        
        return X, y
    
    def _perform_adf_test(self, data: np.ndarray, lags: int) -> tuple:
        """
        Perform the actual ADF test
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        lags : int
            Number of lags
            
        Returns:
        --------
        test_stat : float
            ADF test statistic
        p_value : float
            P-value (approximate)
        critical_values : dict
            Critical values at different significance levels
        """
        # Create design matrix
        X, y = self._create_design_matrix(data, lags)
        
        if X.shape[0] == 0:
            raise ValueError("Insufficient data for ADF test")
        
        # Add lagged level (y_{t-1})
        lagged_level = data[lags:len(data)-1]
        # Ensure lagged_level has the same length as y
        if len(lagged_level) != len(y):
            lagged_level = lagged_level[:len(y)]
        X = np.column_stack([lagged_level, X])
        
        # Perform OLS regression
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            mse = np.mean(residuals ** 2)
            
            # Test statistic is the coefficient on lagged level
            test_stat = beta[0]
            
            # Standard error of the coefficient
            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(mse * XtX_inv[0, 0])
            
            # Calculate t-statistic
            t_stat = test_stat / se
            
            # Approximate p-value using t-distribution
            df = len(y) - X.shape[1]
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            test_stat = 0.0
            p_value = 1.0
        
        # Critical values (approximate for ADF test)
        n = len(data)
        critical_values = self._get_critical_values(n)
        
        return test_stat, p_value, critical_values
    
    def _get_critical_values(self, n: int) -> Dict[str, float]:
        """
        Get approximate critical values for ADF test
        
        Parameters:
        -----------
        n : int
            Sample size
            
        Returns:
        --------
        critical_values : dict
            Critical values at different significance levels
        """
        # These are approximate critical values for ADF test
        # In practice, you might want to use more precise values
        if self.regression == 'c':
            return {
                '1%': -3.43,
                '5%': -2.86,
                '10%': -2.57
            }
        elif self.regression == 'ct':
            return {
                '1%': -3.96,
                '5%': -3.41,
                '10%': -3.12
            }
        else:  # 'n'
            return {
                '1%': -2.58,
                '5%': -1.95,
                '10%': -1.62
            }


class KPSSTest(BaseTest):
    """
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test implementation from scratch
    
    Tests the null hypothesis that the series is stationary around a deterministic trend.
    If the test statistic is greater than the critical value, we reject the null
    hypothesis and conclude the series is non-stationary.
    """
    
    def __init__(self, regression: str = 'c'):
        """
        Initialize KPSS test
        
        Parameters:
        -----------
        regression : str
            Type of regression: 'c' (constant), 'ct' (constant and trend)
        """
        self.regression = regression
        self._test_statistic = None
        self._p_value = None
        self._critical_values = None
    
    def test(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform KPSS test
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data to test
        **kwargs
            Additional test parameters
            
        Returns:
        --------
        results : Dict[str, Any]
            Test results including statistic, p-value, critical values
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 4:
            raise ValueError("Data must have at least 4 observations")
        
        # Perform the test
        test_stat, p_value, critical_values = self._perform_kpss_test(data)
        
        # Store results
        self._test_statistic = test_stat
        self._p_value = p_value
        self._critical_values = critical_values
        
        # Determine stationarity (opposite of ADF)
        is_stationary = test_stat < critical_values['5%']
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'regression': self.regression
        }
    
    def _perform_kpss_test(self, data: np.ndarray) -> tuple:
        """
        Perform the actual KPSS test
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        test_stat : float
            KPSS test statistic
        p_value : float
            P-value (approximate)
        critical_values : dict
            Critical values at different significance levels
        """
        n = len(data)
        
        # Detrend the data
        if self.regression == 'c':
            # Remove constant (demean)
            detrended = data - np.mean(data)
        elif self.regression == 'ct':
            # Remove trend using OLS
            X = np.column_stack([np.ones(n), np.arange(n)])
            beta = np.linalg.lstsq(X, data, rcond=None)[0]
            detrended = data - X @ beta
        else:
            detrended = data
        
        # Calculate cumulative sum of residuals
        cumsum = np.cumsum(detrended)
        
        # Calculate long-run variance (using Newey-West estimator)
        lrv = self._calculate_long_run_variance(detrended)
        
        if lrv == 0:
            test_stat = 0.0
        else:
            # KPSS statistic
            test_stat = np.sum(cumsum ** 2) / (n ** 2 * lrv)
        
        # Approximate p-value (this is a simplified approach)
        p_value = self._approximate_p_value(test_stat, n)
        
        # Critical values
        critical_values = self._get_critical_values()
        
        return test_stat, p_value, critical_values
    
    def _calculate_long_run_variance(self, data: np.ndarray) -> float:
        """
        Calculate long-run variance using Newey-West estimator
        
        Parameters:
        -----------
        data : np.ndarray
            Detrended data
            
        Returns:
        --------
        lrv : float
            Long-run variance
        """
        n = len(data)
        
        # Calculate autocovariances
        gamma_0 = np.var(data)
        
        # Use a simple truncation lag (can be improved)
        max_lag = min(int(4 * (n / 100) ** (2/9)), n // 4)
        
        lrv = gamma_0
        
        for lag in range(1, max_lag + 1):
            if lag < n:
                gamma_k = np.cov(data[:-lag], data[lag:])[0, 1]
                weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
                lrv += 2 * weight * gamma_k
        
        return max(lrv, 1e-10)  # Ensure positive
    
    def _approximate_p_value(self, test_stat: float, n: int) -> float:
        """
        Approximate p-value for KPSS test
        
        Parameters:
        -----------
        test_stat : float
            KPSS test statistic
        n : int
            Sample size
            
        Returns:
        --------
        p_value : float
            Approximate p-value
        """
        # This is a simplified approximation
        # In practice, you'd use more sophisticated methods
        if self.regression == 'c':
            if test_stat < 0.347:
                return 0.1
            elif test_stat < 0.463:
                return 0.05
            elif test_stat < 0.739:
                return 0.01
            else:
                return 0.001
        else:  # 'ct'
            if test_stat < 0.119:
                return 0.1
            elif test_stat < 0.146:
                return 0.05
            elif test_stat < 0.216:
                return 0.01
            else:
                return 0.001
    
    def _get_critical_values(self) -> Dict[str, float]:
        """
        Get critical values for KPSS test
        
        Returns:
        --------
        critical_values : dict
            Critical values at different significance levels
        """
        if self.regression == 'c':
            return {
                '1%': 0.739,
                '5%': 0.463,
                '10%': 0.347
            }
        else:  # 'ct'
            return {
                '1%': 0.216,
                '5%': 0.146,
                '10%': 0.119
            }


class StationarityAnalyzer:
    """
    Combined stationarity analyzer
    
    Performs both ADF and KPSS tests to provide comprehensive
    stationarity assessment for determining differencing order.
    """
    
    def __init__(self, max_lags: Optional[int] = None):
        """
        Initialize stationarity analyzer
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags for ADF test
        """
        self.max_lags = max_lags
        self.adf_test = ADFTest(max_lags)
        self.kpss_test = KPSSTest()
    
    def analyze(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive stationarity analysis
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        results : dict
            Combined results from ADF and KPSS tests
        """
        # Perform both tests
        adf_results = self.adf_test.test(data)
        kpss_results = self.kpss_test.test(data)
        
        # Determine overall stationarity
        is_stationary = self._determine_stationarity(adf_results, kpss_results)
        
        # Suggest differencing order
        suggested_d = self._suggest_differencing_order(data, adf_results, kpss_results)
        
        return {
            'adf_test': adf_results,
            'kpss_test': kpss_results,
            'is_stationary': is_stationary,
            'suggested_differencing_order': suggested_d
        }
    
    def _determine_stationarity(self, adf_results: Dict, kpss_results: Dict) -> bool:
        """
        Determine overall stationarity based on both tests
        
        Parameters:
        -----------
        adf_results : dict
            ADF test results
        kpss_results : dict
            KPSS test results
            
        Returns:
        --------
        is_stationary : bool
            Overall stationarity assessment
        """
        # If both tests agree, use that result
        if adf_results['is_stationary'] == kpss_results['is_stationary']:
            return adf_results['is_stationary']
        
        # If tests disagree, be conservative and assume non-stationary
        # This is a common approach in practice
        return False
    
    def _suggest_differencing_order(self, data: np.ndarray, adf_results: Dict, kpss_results: Dict) -> int:
        """
        Suggest differencing order based on test results
        
        Parameters:
        -----------
        data : np.ndarray
            Original time series data
        adf_results : dict
            ADF test results
        kpss_results : dict
            KPSS test results
            
        Returns:
        --------
        suggested_d : int
            Suggested differencing order
        """
        if adf_results['is_stationary'] and kpss_results['is_stationary']:
            return 0  # No differencing needed
        
        # Test first difference
        diff_data = np.diff(data)
        if len(diff_data) < 4:
            return 1  # Can't test further
        
        diff_adf = self.adf_test.test(diff_data)
        diff_kpss = self.kpss_test.test(diff_data)
        
        if diff_adf['is_stationary'] and diff_kpss['is_stationary']:
            return 1  # First difference is sufficient
        
        # Test second difference if needed
        diff2_data = np.diff(diff_data)
        if len(diff2_data) < 4:
            return 2  # Can't test further
        
        diff2_adf = self.adf_test.test(diff2_data)
        diff2_kpss = self.kpss_test.test(diff2_data)
        
        if diff2_adf['is_stationary'] and diff2_kpss['is_stationary']:
            return 2  # Second difference is sufficient
        
        return 2  # Maximum recommended differencing

