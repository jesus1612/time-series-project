"""
Base classes for the time series library architecture

Implements the foundation classes following SOLID principles:
- Single Responsibility Principle
- Open/Closed Principle  
- Dependency Inversion Principle
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all time series models
    
    Defines the common interface that all models must implement
    """
    
    def __init__(self):
        self._fitted = False
        self._fitted_params = None
        self._data = None
        
    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs) -> 'BaseModel':
        """
        Fit the model to the provided data
        
        Parameters:
        -----------
        data : array-like
            Time series data to fit the model to
        **kwargs
            Additional fitting parameters
            
        Returns:
        --------
        self : BaseModel
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int = 1, **kwargs) -> np.ndarray:
        """
        Generate predictions from the fitted model
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to predict
        **kwargs
            Additional prediction parameters
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted"""
        return self._fitted
    
    @property
    def fitted_params(self) -> Optional[Dict[str, Any]]:
        """Get the fitted parameters"""
        return self._fitted_params
    
    def _validate_fitted(self):
        """Validate that the model has been fitted"""
        if not self._fitted:
            raise ValueError("Model must be fitted before making predictions")


class BaseEstimator(ABC):
    """
    Abstract base class for parameter estimation methods
    
    Defines the interface for different estimation algorithms
    """
    
    @abstractmethod
    def estimate(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Estimate model parameters from data
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        **kwargs
            Estimation parameters
            
        Returns:
        --------
        params : Dict[str, Any]
            Estimated parameters
        """
        pass


class BaseTransformer(ABC):
    """
    Abstract base class for data transformations
    
    Defines the interface for preprocessing transformations
    """
    
    def __init__(self):
        self._fitted = False
        self._fitted_params = None
    
    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.Series]) -> 'BaseTransformer':
        """
        Fit the transformer to the data
        
        Parameters:
        -----------
        data : array-like
            Data to fit the transformer to
            
        Returns:
        --------
        self : BaseTransformer
            Fitted transformer
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Apply the transformation to the data
        
        Parameters:
        -----------
        data : array-like
            Data to transform
            
        Returns:
        --------
        transformed_data : np.ndarray
            Transformed data
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Apply the inverse transformation
        
        Parameters:
        -----------
        data : array-like
            Transformed data to inverse transform
            
        Returns:
        --------
        original_data : np.ndarray
            Data in original scale
        """
        pass
    
    def fit_transform(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Fit the transformer and transform the data in one step
        
        Parameters:
        -----------
        data : array-like
            Data to fit and transform
            
        Returns:
        --------
        transformed_data : np.ndarray
            Transformed data
        """
        return self.fit(data).transform(data)
    
    @property
    def is_fitted(self) -> bool:
        """Check if the transformer has been fitted"""
        return self._fitted


class BaseTest(ABC):
    """
    Abstract base class for statistical tests
    
    Defines the interface for hypothesis testing
    """
    
    @abstractmethod
    def test(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform the statistical test
        
        Parameters:
        -----------
        data : np.ndarray
            Data to test
        **kwargs
            Test parameters
            
        Returns:
        --------
        results : Dict[str, Any]
            Test results including statistic, p-value, etc.
        """
        pass


class TimeSeriesModel(BaseModel):
    """
    Abstract base class specifically for time series models
    
    Extends BaseModel with time series specific functionality
    """
    
    def __init__(self):
        super().__init__()
        self._residuals = None
        self._fitted_values = None
    
    @abstractmethod
    def get_residuals(self) -> np.ndarray:
        """
        Get the model residuals
        
        Returns:
        --------
        residuals : np.ndarray
            Model residuals
        """
        pass
    
    @abstractmethod
    def get_fitted_values(self) -> np.ndarray:
        """
        Get the fitted values
        
        Returns:
        --------
        fitted_values : np.ndarray
            Fitted values from the model
        """
        pass
    
    @abstractmethod
    def summary(self) -> str:
        """
        Generate a summary of the model
        
        Returns:
        --------
        summary : str
            Model summary string
        """
        pass
    
    def plot_diagnostics(self):
        """
        Plot diagnostic plots for the model
        
        This method should be implemented by subclasses
        to provide model-specific diagnostic plots
        """
        raise NotImplementedError("Diagnostic plots not implemented for this model")


class SparkEnabled:
    """
    Mixin class for components that can use Spark
    
    Provides common functionality for classes that need to work with Spark
    for distributed computing operations.
    """
    
    def __init__(self, spark_session=None, spark_config=None):
        """
        Initialize Spark-enabled component
        
        Parameters:
        -----------
        spark_session : SparkSession, optional
            Spark session to use
        spark_config : dict, optional
            Configuration for Spark session
        """
        self._spark_session = spark_session
        self._spark_config = spark_config
        self._spark_owner = False
        self._converter = None
        self._math_ops = None
    
    def _get_or_create_spark(self):
        """Get or create Spark session"""
        if self._spark_session is not None:
            return self._spark_session
        
        # Import here to avoid circular imports
        from ..spark.core import get_or_create_spark_session, get_optimized_spark_config
        
        # Get optimized config if not provided
        if self._spark_config is None:
            self._spark_config = get_optimized_spark_config(1000)  # Default size
        
        self._spark_session = get_or_create_spark_session(
            spark_session=None,
            spark_config=self._spark_config
        )
        self._spark_owner = True
        
        return self._spark_session
    
    @property
    def spark(self):
        """Get Spark session"""
        if self._spark_session is None:
            self._get_or_create_spark()
        return self._spark_session
    
    @property
    def converter(self):
        """Get Spark data converter"""
        if self._converter is None:
            from ..spark.core import SparkDataConverter
            self._converter = SparkDataConverter(self.spark)
        return self._converter
    
    @property
    def math_ops(self):
        """Get Spark math operations"""
        if self._math_ops is None:
            from ..spark.core import SparkMathOperations
            self._math_ops = SparkMathOperations(self.spark)
        return self._math_ops
    
    def cleanup_spark(self):
        """Clean up Spark resources"""
        if self._spark_owner and self._spark_session is not None:
            self._spark_session.stop()
            self._spark_session = None
            self._spark_owner = False
        
        if self._converter is not None:
            self._converter.clear_cache()
            self._converter = None
        
        self._math_ops = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup_spark()
        except:
            pass  # Ignore errors during cleanup
