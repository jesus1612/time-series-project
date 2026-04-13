"""
Data transformation classes for time series preprocessing

Implements various transformations following the transformer pattern:
- Differencing for stationarity
- Log transformations for variance stabilization
- Box-Cox transformations for normality
"""

import numpy as np
from typing import Union, Optional, Tuple
from scipy import stats
from ..core.base import BaseTransformer


class DifferencingTransformer(BaseTransformer):
    """
    Differencing transformer for achieving stationarity
    
    Applies differencing of order d: ∇^d y_t = y_t - y_{t-1} (applied d times)
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize differencing transformer
        
        Parameters:
        -----------
        order : int
            Order of differencing (d)
        """
        super().__init__()
        self.order = order
        self._original_data = None
        self._differenced_data = None
    
    def fit(self, data: Union[np.ndarray, list]) -> 'DifferencingTransformer':
        """
        Fit the differencing transformer
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        self : DifferencingTransformer
            Fitted transformer
        """
        data = np.asarray(data, dtype=float)
        
        if len(data) < self.order + 1:
            raise ValueError(f"Data must have at least {self.order + 1} observations")
        
        self._original_data = data.copy()
        self._differenced_data = self._apply_differencing(data, self.order)
        self._fitted = True
        
        return self
    
    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply differencing transformation
        
        Parameters:
        -----------
        data : array-like
            Data to transform
            
        Returns:
        --------
        transformed_data : np.ndarray
            Differenced data
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        data = np.asarray(data, dtype=float)
        return self._apply_differencing(data, self.order)
    
    def inverse_transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply inverse differencing transformation
        
        Parameters:
        -----------
        data : array-like
            Differenced data to inverse transform
            
        Returns:
        --------
        original_data : np.ndarray
            Data in original scale
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        data = np.asarray(data, dtype=float)
        return self._apply_inverse_differencing(data, self.order)
    
    def _apply_differencing(self, data: np.ndarray, order: int) -> np.ndarray:
        """
        Apply differencing of specified order
        
        Parameters:
        -----------
        data : np.ndarray
            Original data
        order : int
            Order of differencing
            
        Returns:
        --------
        diff_data : np.ndarray
            Differenced data
        """
        diff_data = data.copy()
        
        for _ in range(order):
            diff_data = np.diff(diff_data)
        
        return diff_data
    
    def _apply_inverse_differencing(self, data: np.ndarray, order: int) -> np.ndarray:
        """
        Apply inverse differencing
        
        Parameters:
        -----------
        data : np.ndarray
            Differenced data
        order : int
            Order of differencing
            
        Returns:
        --------
        original_data : np.ndarray
            Data in original scale
        """
        if order == 0:
            return data
        
        # Start with the last value of original data
        original_data = np.zeros(len(data))
        last_value = self._original_data[-1]
        
        # Apply inverse differencing
        for i in range(len(data)):
            if i == 0:
                original_data[i] = last_value + data[i]
            else:
                original_data[i] = original_data[i-1] + data[i]
        
        return original_data


class LogTransformer(BaseTransformer):
    """
    Log transformation for variance stabilization
    
    Applies natural logarithm: y' = log(y) or y' = log(y + c) if y has negative values
    """
    
    def __init__(self, constant: Optional[float] = None):
        """
        Initialize log transformer
        
        Parameters:
        -----------
        constant : float, optional
            Constant to add before taking log. If None, will be determined automatically
        """
        super().__init__()
        self.constant = constant
        self._fitted_constant = None
    
    def fit(self, data: Union[np.ndarray, list]) -> 'LogTransformer':
        """
        Fit the log transformer
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        self : LogTransformer
            Fitted transformer
        """
        data = np.asarray(data, dtype=float)
        
        if np.any(data <= 0):
            if self.constant is None:
                # Determine constant to make all values positive
                self._fitted_constant = -np.min(data) + 1e-6
            else:
                self._fitted_constant = self.constant
        else:
            self._fitted_constant = 0.0
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply log transformation
        
        Parameters:
        -----------
        data : array-like
            Data to transform
            
        Returns:
        --------
        transformed_data : np.ndarray
            Log-transformed data
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        data = np.asarray(data, dtype=float)
        return np.log(data + self._fitted_constant)
    
    def inverse_transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply inverse log transformation
        
        Parameters:
        -----------
        data : array-like
            Log-transformed data
            
        Returns:
        --------
        original_data : np.ndarray
            Data in original scale
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        data = np.asarray(data, dtype=float)
        return np.exp(data) - self._fitted_constant


class BoxCoxTransformer(BaseTransformer):
    """
    Box-Cox transformation for achieving normality and variance stabilization
    
    Applies Box-Cox transformation: y' = (y^λ - 1) / λ if λ ≠ 0, else y' = log(y)
    """
    
    def __init__(self, lambda_param: Optional[float] = None):
        """
        Initialize Box-Cox transformer
        
        Parameters:
        -----------
        lambda_param : float, optional
            Box-Cox parameter. If None, will be estimated from data
        """
        super().__init__()
        self.lambda_param = lambda_param
        self._fitted_lambda = None
    
    def fit(self, data: Union[np.ndarray, list]) -> 'BoxCoxTransformer':
        """
        Fit the Box-Cox transformer
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        self : BoxCoxTransformer
            Fitted transformer
        """
        data = np.asarray(data, dtype=float)
        
        if np.any(data <= 0):
            raise ValueError("Box-Cox transformation requires positive data")
        
        if self.lambda_param is None:
            # Estimate optimal lambda using scipy
            self._fitted_lambda, _ = stats.boxcox(data)
        else:
            self._fitted_lambda = self.lambda_param
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply Box-Cox transformation
        
        Parameters:
        -----------
        data : array-like
            Data to transform
            
        Returns:
        --------
        transformed_data : np.ndarray
            Box-Cox transformed data
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        data = np.asarray(data, dtype=float)
        
        if np.any(data <= 0):
            raise ValueError("Box-Cox transformation requires positive data")
        
        return stats.boxcox(data, lmbda=self._fitted_lambda)
    
    def inverse_transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply inverse Box-Cox transformation
        
        Parameters:
        -----------
        data : array-like
            Box-Cox transformed data
            
        Returns:
        --------
        original_data : np.ndarray
            Data in original scale
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        data = np.asarray(data, dtype=float)
        return stats.inv_boxcox(data, self._fitted_lambda)


class StandardScaler(BaseTransformer):
    """
    Standard scaler for normalizing time series data
    
    Applies standardization: y' = (y - μ) / σ
    """
    
    def __init__(self):
        super().__init__()
        self._mean = None
        self._std = None
    
    def fit(self, data: Union[np.ndarray, list]) -> 'StandardScaler':
        """
        Fit the standard scaler
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        self : StandardScaler
            Fitted transformer
        """
        data = np.asarray(data, dtype=float)
        
        self._mean = np.mean(data)
        self._std = np.std(data)
        
        if self._std == 0:
            raise ValueError("Cannot standardize constant data")
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply standardization
        
        Parameters:
        -----------
        data : array-like
            Data to transform
            
        Returns:
        --------
        transformed_data : np.ndarray
            Standardized data
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        data = np.asarray(data, dtype=float)
        return (data - self._mean) / self._std
    
    def inverse_transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply inverse standardization
        
        Parameters:
        -----------
        data : array-like
            Standardized data
            
        Returns:
        --------
        original_data : np.ndarray
            Data in original scale
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        data = np.asarray(data, dtype=float)
        return data * self._std + self._mean


class MinMaxScaler(BaseTransformer):
    """
    Min-Max scaler for normalizing time series data to [0, 1] range
    
    Applies min-max scaling: y' = (y - min) / (max - min)
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize min-max scaler
        
        Parameters:
        -----------
        feature_range : tuple
            Desired range of transformed data
        """
        super().__init__()
        self.feature_range = feature_range
        self._min = None
        self._max = None
        self._scale = None
        self._min_scale = None
    
    def fit(self, data: Union[np.ndarray, list]) -> 'MinMaxScaler':
        """
        Fit the min-max scaler
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        self : MinMaxScaler
            Fitted transformer
        """
        data = np.asarray(data, dtype=float)
        
        self._min = np.min(data)
        self._max = np.max(data)
        
        if self._max == self._min:
            raise ValueError("Cannot scale constant data")
        
        self._scale = (self.feature_range[1] - self.feature_range[0]) / (self._max - self._min)
        self._min_scale = self.feature_range[0] - self._min * self._scale
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply min-max scaling
        
        Parameters:
        -----------
        data : array-like
            Data to transform
            
        Returns:
        --------
        transformed_data : np.ndarray
            Min-max scaled data
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        data = np.asarray(data, dtype=float)
        return data * self._scale + self._min_scale
    
    def inverse_transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply inverse min-max scaling
        
        Parameters:
        -----------
        data : array-like
            Min-max scaled data
            
        Returns:
        --------
        original_data : np.ndarray
            Data in original scale
        """
        if not self._fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        data = np.asarray(data, dtype=float)
        return (data - self._min_scale) / self._scale




