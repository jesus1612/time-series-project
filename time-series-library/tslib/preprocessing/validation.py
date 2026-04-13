"""
Data validation utilities for time series preprocessing

Provides comprehensive data validation and cleaning functions
for time series data quality assurance.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, List, Optional, Tuple

from .constants import DEFAULT_MAX_MISSING_RATIO, DEFAULT_MIN_SERIES_LENGTH


class DataValidator:
    """
    Comprehensive data validator for time series data
    
    Validates data quality and provides cleaning recommendations
    """
    
    def __init__(self, 
                 min_length: int = DEFAULT_MIN_SERIES_LENGTH,
                 max_missing_ratio: float = DEFAULT_MAX_MISSING_RATIO,
                 outlier_method: str = 'iqr'):
        """
        Initialize data validator
        
        Parameters:
        -----------
        min_length : int
            Minimum required data length
        max_missing_ratio : float
            Maximum allowed ratio of missing values
        outlier_method : str
            Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
        """
        self.min_length = min_length
        self.max_missing_ratio = max_missing_ratio
        self.outlier_method = outlier_method
        self._validation_results = None
    
    def validate(self, data: Union[np.ndarray, pd.Series, list]) -> Dict[str, Any]:
        """
        Perform comprehensive data validation
        
        Parameters:
        -----------
        data : array-like
            Time series data to validate
            
        Returns:
        --------
        results : dict
            Validation results including issues and recommendations
        """
        data = np.asarray(data, dtype=float)
        
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'data_info': self._get_data_info(data),
            'quality_metrics': self._calculate_quality_metrics(data),
            'diagnostics': {}
        }
        
        if len(data) < self.min_length:
            results['is_valid'] = False
            results['issues'].append(f"Data too short: {len(data)} < {self.min_length}")
        
        missing_info = self._check_missing_values(data)
        if missing_info['has_missing']:
            if missing_info['missing_ratio'] > self.max_missing_ratio:
                results['is_valid'] = False
                results['issues'].append(f"Too many missing values: {missing_info['missing_ratio']:.2%} > {self.max_missing_ratio:.2%}")
            else:
                results['warnings'].append(f"Missing values detected: {missing_info['missing_ratio']:.2%}")
        
        if np.any(np.isinf(data)):
            results['is_valid'] = False
            results['issues'].append("Infinite values detected")
        
        if np.var(data) == 0:
            results['warnings'].append("Constant data detected")
            results['recommendations'].append("Consider if this is appropriate for time series analysis")
        
        outlier_info = self._detect_outliers(data)
        if outlier_info['has_outliers']:
            results['warnings'].append(f"Outliers detected: {outlier_info['outlier_count']} ({outlier_info['outlier_ratio']:.2%})")
            results['recommendations'].append("Consider outlier treatment before modeling")
        
        seasonality_info = self._check_seasonality(data)
        results['diagnostics']['seasonality'] = seasonality_info
        if seasonality_info['has_seasonality']:
            results['recommendations'].append("Seasonal patterns detected - consider seasonal ARIMA")
        
        trend_info = self._check_trend(data)
        results['diagnostics']['trend'] = trend_info
        if trend_info['has_trend']:
            results['recommendations'].append("Trend detected - consider differencing")
        
        self._validation_results = results
        return results
    
    def _get_data_info(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Get basic data information
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        info : dict
            Basic data information
        """
        return {
            'length': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'data_type': str(data.dtype)
        }
    
    def _calculate_quality_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate data quality metrics
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        metrics : dict
            Quality metrics
        """
        # Remove missing values for calculations
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return {'completeness': 0.0, 'consistency': 0.0, 'validity': 0.0}
        
        # Completeness: ratio of non-missing values
        completeness = len(clean_data) / len(data)
        
        # Consistency: inverse of coefficient of variation
        if np.mean(clean_data) != 0:
            consistency = 1.0 / (np.std(clean_data) / abs(np.mean(clean_data)))
        else:
            consistency = 1.0 if np.std(clean_data) == 0 else 0.0
        
        # Validity: ratio of finite values
        validity = np.sum(np.isfinite(clean_data)) / len(data)
        
        return {
            'completeness': completeness,
            'consistency': consistency,
            'validity': validity
        }
    
    def _check_missing_values(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Check for missing values
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        missing_info : dict
            Missing value information
        """
        missing_mask = np.isnan(data)
        missing_count = np.sum(missing_mask)
        missing_ratio = missing_count / len(data)
        
        return {
            'has_missing': missing_count > 0,
            'missing_count': missing_count,
            'missing_ratio': missing_ratio,
            'missing_indices': np.where(missing_mask)[0].tolist()
        }
    
    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect outliers using specified method
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        outlier_info : dict
            Outlier detection results
        """
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 4:
            return {'has_outliers': False, 'outlier_count': 0, 'outlier_ratio': 0.0}
        
        if self.outlier_method == 'iqr':
            outliers = self._detect_outliers_iqr(clean_data)
        elif self.outlier_method == 'zscore':
            outliers = self._detect_outliers_zscore(clean_data)
        elif self.outlier_method == 'modified_zscore':
            outliers = self._detect_outliers_modified_zscore(clean_data)
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
        
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / len(clean_data)
        
        return {
            'has_outliers': outlier_count > 0,
            'outlier_count': outlier_count,
            'outlier_ratio': outlier_ratio,
            'outlier_indices': outliers.tolist()
        }
    
    def _detect_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return np.where((data < lower_bound) | (data > upper_bound))[0]
    
    def _detect_outliers_zscore(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method"""
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return np.where(z_scores > threshold)[0]
    
    def _detect_outliers_modified_zscore(self, data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """Detect outliers using modified Z-score method"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.where(np.abs(modified_z_scores) > threshold)[0]
    
    def _check_seasonality(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Check for seasonal patterns
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        seasonality_info : dict
            Seasonality detection results
        """
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 12:
            return {'has_seasonality': False, 'seasonal_period': None}
        
        # Simple seasonality detection using autocorrelation
        # Look for peaks in ACF at seasonal lags
        max_lag = min(len(clean_data) // 4, 24)
        acf_values = self._calculate_acf(clean_data, max_lag)
        
        # Check for significant peaks at common seasonal periods
        seasonal_periods = [4, 7, 12, 24]  # Quarterly, weekly, monthly, daily
        significant_periods = []
        
        for period in seasonal_periods:
            if period < len(acf_values):
                if abs(acf_values[period]) > 0.3:  # Threshold for significance
                    significant_periods.append(period)
        
        return {
            'has_seasonality': len(significant_periods) > 0,
            'seasonal_periods': significant_periods,
            'acf_values': acf_values.tolist()
        }
    
    def _check_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Check for trend in the data
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        trend_info : dict
            Trend detection results
        """
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 3:
            return {'has_trend': False, 'trend_direction': None}
        
        # Simple trend detection using linear regression
        x = np.arange(len(clean_data))
        slope, intercept = np.polyfit(x, clean_data, 1)
        
        # Calculate R-squared to measure trend strength
        y_pred = slope * x + intercept
        ss_res = np.sum((clean_data - y_pred) ** 2)
        ss_tot = np.sum((clean_data - np.mean(clean_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        has_trend = r_squared > 0.1  # Threshold for trend significance
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'none'
        
        return {
            'has_trend': has_trend,
            'trend_direction': trend_direction,
            'slope': slope,
            'r_squared': r_squared
        }
    
    def _calculate_acf(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Calculate autocorrelation function
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        max_lag : int
            Maximum lag to calculate
            
        Returns:
        --------
        acf_values : np.ndarray
            ACF values
        """
        n = len(data)
        mean = np.mean(data)
        variance = np.sum((data - mean) ** 2)
        
        if variance == 0:
            return np.zeros(max_lag + 1)
        
        acf_values = np.zeros(max_lag + 1)
        
        for k in range(max_lag + 1):
            if k == 0:
                acf_values[k] = 1.0
            else:
                numerator = 0.0
                for t in range(k, n):
                    numerator += (data[t] - mean) * (data[t - k] - mean)
                acf_values[k] = numerator / variance
        
        return acf_values
    
    def clean_data(self, data: Union[np.ndarray, pd.Series, list], 
                   method: str = 'interpolate') -> np.ndarray:
        """
        Clean data based on validation results
        
        Parameters:
        -----------
        data : array-like
            Time series data to clean
        method : str
            Cleaning method ('interpolate', 'forward_fill', 'backward_fill', 'drop')
            
        Returns:
        --------
        cleaned_data : np.ndarray
            Cleaned time series data
        """
        data = np.asarray(data, dtype=float)
        
        if method == 'interpolate':
            mask = ~np.isnan(data)
            if np.sum(mask) > 1:
                data = np.interp(np.arange(len(data)), np.arange(len(data))[mask], data[mask])
            else:
                data = np.full_like(data, np.nanmean(data))
        
        elif method == 'forward_fill':
            mask = ~np.isnan(data)
            data = np.where(mask, data, np.nan)
            data = pd.Series(data).fillna(method='ffill').values
        
        elif method == 'backward_fill':
            mask = ~np.isnan(data)
            data = np.where(mask, data, np.nan)
            data = pd.Series(data).fillna(method='bfill').values
        
        elif method == 'drop':
            data = data[~np.isnan(data)]
        
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
        
        data = np.where(np.isfinite(data), data, np.nanmedian(data))
        
        return data
    
    @property
    def validation_results(self) -> Optional[Dict[str, Any]]:
        """Get the last validation results"""
        return self._validation_results


class DataQualityReport:
    """
    Generate comprehensive data quality reports
    """
    
    def __init__(self, validator: DataValidator):
        """
        Initialize data quality report generator
        
        Parameters:
        -----------
        validator : DataValidator
            Data validator instance
        """
        self.validator = validator
    
    def generate_report(self, data: Union[np.ndarray, pd.Series, list]) -> str:
        """
        Generate a comprehensive data quality report
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        report : str
            Formatted quality report
        """
        results = self.validator.validate(data)
        
        report = "Data Quality Report\n"
        report += "=" * 50 + "\n\n"
        
        # Data information
        info = results['data_info']
        report += "Data Information:\n"
        report += f"  Length: {info['length']}\n"
        report += f"  Mean: {info['mean']:.6f}\n"
        report += f"  Std: {info['std']:.6f}\n"
        report += f"  Min: {info['min']:.6f}\n"
        report += f"  Max: {info['max']:.6f}\n"
        report += f"  Median: {info['median']:.6f}\n\n"
        
        # Quality metrics
        metrics = results['quality_metrics']
        report += "Quality Metrics:\n"
        report += f"  Completeness: {metrics['completeness']:.2%}\n"
        report += f"  Consistency: {metrics['consistency']:.4f}\n"
        report += f"  Validity: {metrics['validity']:.2%}\n\n"
        
        # Issues
        if results['issues']:
            report += "Issues Found:\n"
            for issue in results['issues']:
                report += f"  ❌ {issue}\n"
            report += "\n"
        
        # Warnings
        if results['warnings']:
            report += "Warnings:\n"
            for warning in results['warnings']:
                report += f"  ⚠️  {warning}\n"
            report += "\n"
        
        # Recommendations
        if results['recommendations']:
            report += "Recommendations:\n"
            for rec in results['recommendations']:
                report += f"  💡 {rec}\n"
            report += "\n"
        
        # Overall status
        status = "✅ VALID" if results['is_valid'] else "❌ INVALID"
        report += f"Overall Status: {status}\n"
        
        return report




