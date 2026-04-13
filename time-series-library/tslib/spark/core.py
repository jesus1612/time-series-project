"""
Spark Core Module - Central Spark Management

Provides core Spark functionality for TSLib including:
- SparkSession management (hybrid approach)
- Data conversion between Pandas/NumPy and Spark
- Global SparkSession singleton
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, isnan, isnull
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark import RDD

from .python_env import apply_pyspark_python_env


class SparkSessionManager:
    """
    Singleton manager for global SparkSession
    
    Manages a global SparkSession that can be shared across multiple
    ARIMA model instances to avoid overhead of creating multiple sessions.
    """
    
    _instance = None
    _global_spark = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_global_spark(cls) -> Optional[SparkSession]:
        """Get the global SparkSession if it exists"""
        return cls._global_spark
    
    @classmethod
    def set_global_spark(cls, spark_session: SparkSession):
        """Set the global SparkSession"""
        cls._global_spark = spark_session
    
    @classmethod
    def clear_global_spark(cls):
        """Clear the global SparkSession"""
        if cls._global_spark is not None:
            cls._global_spark.stop()
            cls._global_spark = None


def get_or_create_spark_session(spark_session: Optional[SparkSession] = None,
                               spark_config: Optional[Dict[str, str]] = None,
                               master: Optional[str] = None,
                               app_name: Optional[str] = None) -> SparkSession:
    """
    Get or create SparkSession following hybrid approach
    
    Hierarchy:
    1. Use provided spark_session
    2. Use global SparkSession if exists and still active
    3. Create new SparkSession with default config
    
    Parameters:
    -----------
    spark_session : SparkSession, optional
        Explicit SparkSession to use
    spark_config : dict, optional
        Configuration for new SparkSession
    master : str, optional
        Spark master URL (e.g. local[4], local[*]). Passed to the session builder.
    app_name : str, optional
        Application name (also may be set via spark.app.name in spark_config)
        
    Returns:
    --------
    spark : SparkSession
        SparkSession to use
    """
    if spark_session is not None:
        return spark_session
    
    # Check for global SparkSession (drop reference if stopped)
    global_spark = SparkSessionManager.get_global_spark()
    if global_spark is not None:
        if _spark_session_is_active(global_spark):
            return global_spark
        SparkSessionManager.clear_global_spark()
    
    # Create new SparkSession
    return create_spark_session(spark_config, master=master, app_name=app_name)


def _spark_session_is_active(spark: SparkSession) -> bool:
    """Return True if the SparkSession has a live SparkContext."""
    if spark is None:
        return False
    try:
        sc = spark.sparkContext
        return sc is not None and not sc._jsc.sc().isStopped()
    except Exception:
        return False


def create_spark_session(spark_config: Optional[Dict[str, str]] = None,
                         master: Optional[str] = None,
                         app_name: Optional[str] = None) -> SparkSession:
    """
    Create a new SparkSession with optimized configuration
    
    Parameters:
    -----------
    spark_config : dict, optional
        Additional configuration parameters
    master : str, optional
        Spark master (e.g. local[8] for eight local cores). If None, Spark
        uses its default (often local[*] from environment).
    app_name : str, optional
        Application display name. Overrides spark.app.name in default_config if both set.
        
    Returns:
    --------
    spark : SparkSession
        New SparkSession
    """
    apply_pyspark_python_env()

    # Default optimized configuration
    default_config = {
        'spark.app.name': 'TSLib-ARIMA',
        'spark.sql.execution.arrow.pyspark.enabled': 'true',
        'spark.sql.adaptive.enabled': 'true',
        'spark.sql.adaptive.coalescePartitions.enabled': 'true',
        'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
        'spark.driver.memory': '4g',
        'spark.executor.memory': '4g',
        'spark.sql.execution.arrow.maxRecordsPerBatch': '10000',
        'spark.sql.adaptive.skewJoin.enabled': 'true'
    }
    
    # Merge with user config
    if spark_config:
        default_config.update(spark_config)
    
    if app_name:
        default_config['spark.app.name'] = app_name

    master_from_config = default_config.pop('spark.master', None)
    effective_master = master or master_from_config

    # Create SparkSession
    builder = SparkSession.builder
    if effective_master:
        builder = builder.master(effective_master)

    for key, value in default_config.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def get_optimized_spark_config(data_size: int) -> Dict[str, str]:
    """
    Get optimized Spark configuration based on data size
    
    Parameters:
    -----------
    data_size : int
        Size of the dataset
        
    Returns:
    --------
    config : dict
        Optimized configuration
    """
    config = {
        'spark.app.name': 'TSLib-ARIMA',
        'spark.sql.execution.arrow.pyspark.enabled': 'true',
        'spark.sql.adaptive.enabled': 'true',
        'spark.serializer': 'org.apache.spark.serializer.KryoSerializer'
    }
    
    # Adjust memory based on data size
    if data_size > 1000000:  # Large dataset
        config.update({
            'spark.driver.memory': '8g',
            'spark.executor.memory': '8g',
            'spark.sql.adaptive.coalescePartitions.enabled': 'true',
            'spark.sql.adaptive.skewJoin.enabled': 'true'
        })
    elif data_size > 100000:  # Medium dataset
        config.update({
            'spark.driver.memory': '6g',
            'spark.executor.memory': '6g'
        })
    else:  # Small dataset
        config.update({
            'spark.driver.memory': '2g',
            'spark.executor.memory': '2g'
        })
    
    return config


class SparkDataConverter:
    """
    Efficient data conversion between Pandas/NumPy and Spark formats
    
    Provides optimized conversion methods with proper caching and
    memory management for time series data.
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize converter with SparkSession
        
        Parameters:
        -----------
        spark_session : SparkSession
            Spark session to use for conversions
        """
        self.spark = spark_session
        self._cached_dataframes = {}  # Cache for frequently used DataFrames
    
    def to_spark_dataframe(self, 
                          data: Union[np.ndarray, pd.Series, list],
                          cache: bool = True,
                          cache_key: Optional[str] = None) -> DataFrame:
        """
        Convert NumPy/Pandas data to Spark DataFrame
        
        Parameters:
        -----------
        data : array-like
            Time series data
        cache : bool
            Whether to cache the DataFrame
        cache_key : str, optional
            Key for caching (if None, auto-generated)
            
        Returns:
        --------
        df : DataFrame
            Spark DataFrame with time series data
        """
        # Convert to pandas if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                df_pandas = pd.DataFrame({
                    'index': range(len(data)),
                    'value': data
                })
            else:
                raise ValueError("Only 1D arrays supported")
        elif isinstance(data, pd.Series):
            df_pandas = pd.DataFrame({
                'index': data.index if hasattr(data.index, '__len__') else range(len(data)),
                'value': data.values
            })
        elif isinstance(data, list):
            df_pandas = pd.DataFrame({
                'index': range(len(data)),
                'value': data
            })
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Convert to Spark DataFrame
        df_spark = self.spark.createDataFrame(df_pandas)
        
        # Cache if requested
        if cache:
            if cache_key is None:
                cache_key = f"data_{id(data)}"
            df_spark.cache()
            df_spark.count()  # Materialize cache
            self._cached_dataframes[cache_key] = df_spark
        
        return df_spark
    
    def to_spark_rdd(self, data: np.ndarray) -> RDD:
        """
        Convert NumPy array to Spark RDD for low-level operations
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        rdd : RDD
            Spark RDD with indexed data
        """
        # Create indexed data
        indexed_data = [(i, float(val)) for i, val in enumerate(data)]
        
        # Convert to RDD
        return self.spark.sparkContext.parallelize(indexed_data)
    
    def to_numpy(self, df: DataFrame, column: str = 'value') -> np.ndarray:
        """
        Convert Spark DataFrame to NumPy array
        
        Parameters:
        -----------
        df : DataFrame
            Spark DataFrame
        column : str
            Column name to extract
            
        Returns:
        --------
        array : np.ndarray
            NumPy array
        """
        # Collect data and sort by index
        data = df.select('index', column).orderBy('index').collect()
        
        # Extract values
        values = [row[column] for row in data]
        
        return np.array(values)
    
    def broadcast_params(self, params: np.ndarray) -> Any:
        """
        Broadcast parameters to all workers
        
        Parameters:
        -----------
        params : np.ndarray
            Parameters to broadcast
            
        Returns:
        --------
        broadcast : Broadcast
            Broadcasted parameters
        """
        return self.spark.sparkContext.broadcast(params.tolist())
    
    def create_matrix_dataframe(self, 
                               matrix: np.ndarray,
                               row_col_names: Optional[Tuple[str, str]] = None) -> DataFrame:
        """
        Convert NumPy matrix to Spark DataFrame
        
        Parameters:
        -----------
        matrix : np.ndarray
            2D matrix
        row_col_names : tuple, optional
            Names for row and column indices
            
        Returns:
        --------
        df : DataFrame
            Spark DataFrame representing the matrix
        """
        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2D")
        
        row_name, col_name = row_col_names or ('row', 'col')
        
        # Create matrix data
        data = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                data.append((i, j, float(matrix[i, j])))
        
        # Create DataFrame
        schema = StructType([
            StructField(row_name, IntegerType(), True),
            StructField(col_name, IntegerType(), True),
            StructField('value', DoubleType(), True)
        ])
        
        return self.spark.createDataFrame(data, schema)
    
    def matrix_from_dataframe(self, 
                             df: DataFrame,
                             row_col_names: Tuple[str, str],
                             value_col: str = 'value') -> np.ndarray:
        """
        Convert Spark DataFrame back to NumPy matrix
        
        Parameters:
        -----------
        df : DataFrame
            Spark DataFrame with matrix data
        row_col_names : tuple
            Names of row and column index columns
        value_col : str
            Name of value column
            
        Returns:
        --------
        matrix : np.ndarray
            NumPy matrix
        """
        row_name, col_name = row_col_names
        
        # Get dimensions
        max_row = df.agg({row_name: 'max'}).collect()[0][0]
        max_col = df.agg({col_name: 'max'}).collect()[0][0]
        
        # Create matrix
        matrix = np.zeros((max_row + 1, max_col + 1))
        
        # Fill matrix
        data = df.collect()
        for row in data:
            matrix[row[row_name], row[col_name]] = row[value_col]
        
        return matrix
    
    def clear_cache(self):
        """Clear all cached DataFrames"""
        for df in self._cached_dataframes.values():
            df.unpersist()
        self._cached_dataframes.clear()
    
    def get_cached_dataframe(self, cache_key: str) -> Optional[DataFrame]:
        """Get cached DataFrame by key"""
        return self._cached_dataframes.get(cache_key)


class SparkMathOperations:
    """
    Distributed mathematical operations using Spark
    
    Provides optimized mathematical operations that leverage
    Spark's distributed computing capabilities.
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize with SparkSession
        
        Parameters:
        -----------
        spark_session : SparkSession
            Spark session to use
        """
        self.spark = spark_session
        self.converter = SparkDataConverter(spark_session)
    
    def vector_dot_product(self, 
                          df1: DataFrame, 
                          df2: DataFrame,
                          value_col: str = 'value') -> float:
        """
        Calculate dot product of two vectors using Spark
        
        Parameters:
        -----------
        df1, df2 : DataFrame
            DataFrames with vector data
        value_col : str
            Column name containing values
            
        Returns:
        --------
        dot_product : float
            Dot product result
        """
        # Join on index and calculate product
        joined = df1.alias('a').join(df2.alias('b'), 'index')
        product_df = joined.select(
            (col(f'a.{value_col}') * col(f'b.{value_col}')).alias('product')
        )
        
        # Sum products
        result = product_df.agg({'product': 'sum'}).collect()[0][0]
        
        return float(result) if result is not None else 0.0
    
    def vector_sum_squares(self, 
                          df: DataFrame,
                          value_col: str = 'value') -> float:
        """
        Calculate sum of squares using Spark
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with vector data
        value_col : str
            Column name containing values
            
        Returns:
        --------
        sum_squares : float
            Sum of squares
        """
        squared_df = df.select((col(value_col) ** 2).alias('squared'))
        result = squared_df.agg({'squared': 'sum'}).collect()[0][0]
        
        return float(result) if result is not None else 0.0
    
    def vector_mean(self, 
                   df: DataFrame,
                   value_col: str = 'value') -> float:
        """
        Calculate mean using Spark aggregation
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with vector data
        value_col : str
            Column name containing values
            
        Returns:
        --------
        mean : float
            Mean value
        """
        result = df.agg({value_col: 'mean'}).collect()[0][0]
        return float(result) if result is not None else 0.0
    
    def vector_variance(self, 
                       df: DataFrame,
                       value_col: str = 'value',
                       mean: Optional[float] = None) -> float:
        """
        Calculate variance using Spark
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with vector data
        value_col : str
            Column name containing values
        mean : float, optional
            Pre-calculated mean (if None, will calculate)
            
        Returns:
        --------
        variance : float
            Variance
        """
        if mean is None:
            mean = self.vector_mean(df, value_col)
        
        # Calculate squared deviations
        squared_dev_df = df.select(
            ((col(value_col) - lit(mean)) ** 2).alias('squared_dev')
        )
        
        # Calculate variance
        result = squared_dev_df.agg({'squared_dev': 'mean'}).collect()[0][0]
        
        return float(result) if result is not None else 0.0
    
    def matrix_vector_multiply(self, 
                              matrix_df: DataFrame,
                              vector_df: DataFrame,
                              matrix_row_col: Tuple[str, str] = ('row', 'col'),
                              matrix_value_col: str = 'value',
                              vector_value_col: str = 'value') -> DataFrame:
        """
        Multiply matrix by vector using Spark
        
        Parameters:
        -----------
        matrix_df : DataFrame
            DataFrame representing matrix
        vector_df : DataFrame
            DataFrame representing vector
        matrix_row_col : tuple
            Row and column column names for matrix
        matrix_value_col : str
            Value column name for matrix
        vector_value_col : str
            Value column name for vector
            
        Returns:
        --------
        result_df : DataFrame
            DataFrame with result vector
        """
        row_name, col_name = matrix_row_col
        
        # Join matrix with vector and multiply
        joined = matrix_df.alias('m').join(
            vector_df.alias('v'),
            col(f'm.{col_name}') == col(f'v.index')
        )
        
        # Calculate products
        product_df = joined.select(
            col(f'm.{row_name}').alias('index'),
            (col(f'm.{matrix_value_col}') * col(f'v.{vector_value_col}')).alias('product')
        )
        
        # Sum products by row
        result_df = product_df.groupBy('index').agg({'product': 'sum'}).orderBy('index')
        
        return result_df
    
    def parallel_map_operation(self, 
                              data: list,
                              operation: callable,
                              num_partitions: Optional[int] = None) -> list:
        """
        Apply operation to data in parallel using Spark
        
        Parameters:
        -----------
        data : list
            Data to process
        operation : callable
            Function to apply to each element
        num_partitions : int, optional
            Number of partitions
            
        Returns:
        --------
        results : list
            Results of applying operation
        """
        # Create RDD
        rdd = self.spark.sparkContext.parallelize(data, num_partitions)
        
        # Apply operation
        result_rdd = rdd.map(operation)
        
        # Collect results
        return result_rdd.collect()


# Global functions for easy access
def set_global_spark_session(spark_session: SparkSession):
    """Set global SparkSession for the application"""
    SparkSessionManager.set_global_spark(spark_session)


def get_global_spark_session() -> Optional[SparkSession]:
    """Get global SparkSession"""
    return SparkSessionManager.get_global_spark()


def clear_global_spark_session():
    """Clear global SparkSession"""
    SparkSessionManager.clear_global_spark()




