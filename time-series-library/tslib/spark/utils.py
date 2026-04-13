"""
PySpark utility functions for distributed time series analysis

Provides helper functions for working with PySpark DataFrames
and distributed time series processing.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
from ..utils.checks import check_spark_availability


def check_spark_availability() -> bool:
    """
    Check if PySpark is available in the system
    
    Returns:
    --------
    is_available : bool
        True if PySpark is available, False otherwise
    """
    try:
        import pyspark
        return True
    except ImportError:
        return False


def create_spark_session(app_name: str = "TimeSeriesAnalysis",
                        master: str = "local[*]",
                        **kwargs) -> Optional[Any]:
    """
    Create a Spark session for distributed processing
    
    Parameters:
    -----------
    app_name : str
        Application name
    master : str
        Spark master URL
    **kwargs
        Additional Spark configuration parameters
        
    Returns:
    --------
    spark : SparkSession or None
        Spark session if PySpark is available, None otherwise
    """
    if not check_spark_availability():
        print("PySpark not available. Install with: pip install pyspark")
        return None
    
    try:
        from pyspark.sql import SparkSession

        from .python_env import apply_pyspark_python_env

        apply_pyspark_python_env()

        # Default configuration
        config = {
            'spark.sql.adaptive.enabled': 'true',
            'spark.sql.adaptive.coalescePartitions.enabled': 'true',
            'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
            'spark.sql.execution.arrow.pyspark.enabled': 'true'
        }
        
        # Update with user-provided configuration
        config.update(kwargs)
        
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config(**config) \
            .getOrCreate()
        
        return spark
    
    except Exception as e:
        print(f"Failed to create Spark session: {e}")
        return None


def prepare_time_series_dataframe(spark_session: Any,
                                 data: Union[pd.DataFrame, np.ndarray, list],
                                 time_column: str = 'timestamp',
                                 value_column: str = 'value',
                                 group_column: Optional[str] = None) -> Any:
    """
    Prepare time series data for distributed processing
    
    Parameters:
    -----------
    spark_session : SparkSession
        Spark session
    data : array-like or DataFrame
        Time series data
    time_column : str
        Name of the time column
    value_column : str
        Name of the value column
    group_column : str, optional
        Name of the group column for multiple time series
        
    Returns:
    --------
    df : DataFrame
        Spark DataFrame with time series data
    """
    if not check_spark_availability():
        raise ImportError("PySpark not available")
    
    # Convert data to pandas DataFrame if needed
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            df_pandas = pd.DataFrame({
                time_column: range(len(data)),
                value_column: data
            })
        else:
            raise ValueError("2D numpy arrays not supported. Use pandas DataFrame instead.")
    
    elif isinstance(data, list):
        df_pandas = pd.DataFrame({
            time_column: range(len(data)),
            value_column: data
        })
    
    elif isinstance(data, pd.DataFrame):
        df_pandas = data.copy()
    
    else:
        raise ValueError("Unsupported data type")
    
    # Convert to Spark DataFrame
    df_spark = spark_session.createDataFrame(df_pandas)
    
    return df_spark


def validate_spark_dataframe(df: Any, 
                           required_columns: List[str],
                           value_column: str = 'value') -> Dict[str, Any]:
    """
    Validate Spark DataFrame for time series analysis
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame to validate
    required_columns : list
        List of required column names
    value_column : str
        Name of the value column
        
    Returns:
    --------
    validation : dict
        Validation results
    """
    if not check_spark_availability():
        raise ImportError("PySpark not available")
    
    validation = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'info': {}
    }
    
    # Check if DataFrame exists
    if df is None:
        validation['is_valid'] = False
        validation['issues'].append("DataFrame is None")
        return validation
    
    # Get DataFrame schema
    schema = df.schema
    column_names = [field.name for field in schema.fields]
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in column_names]
    if missing_columns:
        validation['is_valid'] = False
        validation['issues'].append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if value_column in column_names:
        value_field = next(field for field in schema.fields if field.name == value_column)
        if value_field.dataType.typeName() not in ['double', 'float', 'integer', 'long']:
            validation['warnings'].append(f"Value column '{value_column}' should be numeric")
    
    # Get basic statistics
    try:
        count = df.count()
        validation['info']['row_count'] = count
        
        if count == 0:
            validation['is_valid'] = False
            validation['issues'].append("DataFrame is empty")
        
    except Exception as e:
        validation['warnings'].append(f"Could not count rows: {e}")
    
    return validation


def optimize_spark_configuration(spark_session: Any,
                               data_size: int,
                               num_partitions: Optional[int] = None) -> Dict[str, Any]:
    """
    Optimize Spark configuration for time series processing
    
    Parameters:
    -----------
    spark_session : SparkSession
        Spark session
    data_size : int
        Approximate size of the dataset
    num_partitions : int, optional
        Number of partitions to use
        
    Returns:
    --------
    config : dict
        Optimized configuration
    """
    if not check_spark_availability():
        raise ImportError("PySpark not available")
    
    config = {}
    
    # Estimate optimal number of partitions
    if num_partitions is None:
        # Rule of thumb: 2-4 partitions per CPU core
        num_cores = spark_session.sparkContext.defaultParallelism
        estimated_partitions = min(max(data_size // 1000, num_cores * 2), num_cores * 4)
        config['spark.sql.shuffle.partitions'] = estimated_partitions
    else:
        config['spark.sql.shuffle.partitions'] = num_partitions
    
    # Memory optimization
    if data_size > 1000000:  # Large dataset
        config.update({
            'spark.sql.adaptive.enabled': 'true',
            'spark.sql.adaptive.coalescePartitions.enabled': 'true',
            'spark.sql.adaptive.skewJoin.enabled': 'true',
            'spark.serializer': 'org.apache.spark.serializer.KryoSerializer'
        })
    
    # Arrow optimization for pandas UDFs
    config.update({
        'spark.sql.execution.arrow.pyspark.enabled': 'true',
        'spark.sql.execution.arrow.maxRecordsPerBatch': '10000'
    })
    
    return config


def collect_results_safely(df: Any, max_rows: int = 1000000) -> pd.DataFrame:
    """
    Safely collect results from Spark DataFrame to pandas
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame
    max_rows : int
        Maximum number of rows to collect
        
    Returns:
    --------
    result : DataFrame
        Pandas DataFrame with results
    """
    if not check_spark_availability():
        raise ImportError("PySpark not available")
    
    # Check row count
    try:
        row_count = df.count()
        if row_count > max_rows:
            print(f"Warning: DataFrame has {row_count} rows, but max_rows is {max_rows}")
            print("Consider using sampling or filtering before collecting")
    except Exception as e:
        print(f"Could not count rows: {e}")
    
    # Collect results
    try:
        return df.toPandas()
    except Exception as e:
        print(f"Failed to collect results: {e}")
        return pd.DataFrame()


def create_time_series_groups(df: Any,
                            group_column: str,
                            time_column: str = 'timestamp',
                            value_column: str = 'value') -> Any:
    """
    Group time series data by group column and sort by time
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame with time series data
    group_column : str
        Column to group by
    time_column : str
        Time column name
    value_column : str
        Value column name
        
    Returns:
    --------
    grouped_df : DataFrame
        Grouped and sorted DataFrame
    """
    if not check_spark_availability():
        raise ImportError("PySpark not available")
    
    from pyspark.sql.functions import col
    
    # Group by group_column and sort by time within each group
    grouped_df = df.orderBy(col(group_column), col(time_column))
    
    return grouped_df


def sample_data_for_testing(df: Any,
                          sample_fraction: float = 0.1,
                          seed: int = 42) -> Any:
    """
    Sample data for testing purposes
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame
    sample_fraction : float
        Fraction of data to sample
    seed : int
        Random seed
        
    Returns:
    --------
    sampled_df : DataFrame
        Sampled DataFrame
    """
    if not check_spark_availability():
        raise ImportError("PySpark not available")
    
    return df.sample(fraction=sample_fraction, seed=seed)


def get_dataframe_info(df: Any) -> Dict[str, Any]:
    """
    Get information about Spark DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame
        
    Returns:
    --------
    info : dict
        DataFrame information
    """
    if not check_spark_availability():
        raise ImportError("PySpark not available")
    
    info = {
        'schema': str(df.schema),
        'columns': df.columns,
        'dtypes': dict(df.dtypes),
        'row_count': None,
        'partitions': None
    }
    
    try:
        info['row_count'] = df.count()
    except Exception:
        pass
    
    try:
        info['partitions'] = df.rdd.getNumPartitions()
    except Exception:
        pass
    
    return info
