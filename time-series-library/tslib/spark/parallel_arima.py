"""
Parallel ARIMA processing with PySpark and Pandas UDF

Implements distributed ARIMA fitting and forecasting using PySpark's
Pandas UDF functionality for processing multiple time series in parallel.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
from ..utils.checks import check_spark_availability
from ..models.arima_model import ARIMAModel


def _group_values_time_ordered(
    group_df: pd.DataFrame, value_column: str, time_column: str
) -> np.ndarray:
    """Values in chronological order (sort by ``time_column`` when available)."""
    if time_column in group_df.columns:
        group_df = group_df.sort_values(time_column)
    return group_df[value_column].to_numpy(dtype=float, copy=False)


def fit_predict_arima_udf(series: pd.Series, 
                         order: Tuple[int, int, int] = (1, 1, 1),
                         steps: int = 1,
                         return_conf_int: bool = False,
                         auto_select: bool = False,
                         **kwargs) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Pandas UDF for fitting ARIMA model and generating predictions
    
    This function is designed to work with PySpark's groupBy().applyInPandas()
    for distributed time series processing.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    order : tuple
        ARIMA order (p, d, q)
    steps : int
        Number of steps to forecast
    return_conf_int : bool
        Whether to return confidence intervals
    auto_select : bool
        Whether to automatically select optimal order
    **kwargs
        Additional parameters for ARIMA model
        
    Returns:
    --------
    predictions : pd.Series
        Forecasted values
    conf_int : tuple, optional
        Confidence intervals if return_conf_int=True
    """
    try:
        # Remove any NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:  # Minimum data requirement
            # Return NaN predictions if insufficient data
            if return_conf_int:
                nan_pred = pd.Series([np.nan] * steps)
                return nan_pred, (nan_pred, nan_pred)
            else:
                return pd.Series([np.nan] * steps)
        
        # Create and fit ARIMA model
        model = ARIMAModel(
            order=order if not auto_select else None,
            auto_select=auto_select,
            validation=False  # Skip validation for performance
        )
        
        model.fit(clean_series.values, **kwargs)
        
        # Generate predictions
        if return_conf_int:
            predictions, conf_int = model.predict(steps, return_conf_int=True)
            return pd.Series(predictions), (pd.Series(conf_int[0]), pd.Series(conf_int[1]))
        else:
            predictions = model.predict(steps)
            return pd.Series(predictions)
    
    except Exception as e:
        # Return NaN predictions if model fitting fails
        print(f"ARIMA fitting failed: {e}")
        if return_conf_int:
            nan_pred = pd.Series([np.nan] * steps)
            return nan_pred, (nan_pred, nan_pred)
        else:
            return pd.Series([np.nan] * steps)


def fit_arima_udf(series: pd.Series,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 auto_select: bool = False,
                 **kwargs) -> Dict[str, Any]:
    """
    Pandas UDF for fitting ARIMA model and returning model information
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    order : tuple
        ARIMA order (p, d, q)
    auto_select : bool
        Whether to automatically select optimal order
    **kwargs
        Additional parameters for ARIMA model
        
    Returns:
    --------
    model_info : dict
        Model information including parameters and statistics
    """
    try:
        # Remove any NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:  # Minimum data requirement
            return {
                'success': False,
                'error': 'Insufficient data',
                'order': order,
                'aic': np.nan,
                'bic': np.nan,
                'log_likelihood': np.nan
            }
        
        # Create and fit ARIMA model
        model = ARIMAModel(
            order=order if not auto_select else None,
            auto_select=auto_select,
            validation=False  # Skip validation for performance
        )
        
        model.fit(clean_series.values, **kwargs)
        
        # Extract model information
        fitted_params = model._fitted_params
        
        return {
            'success': True,
            'order': model.order,
            'aic': fitted_params['aic'],
            'bic': fitted_params['bic'],
            'log_likelihood': fitted_params['log_likelihood'],
            'parameters': fitted_params['parameters']
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'order': order,
            'aic': np.nan,
            'bic': np.nan,
            'log_likelihood': np.nan
        }


def predict_arima_udf(series: pd.Series,
                     model_params: Dict[str, Any],
                     steps: int = 1,
                     return_conf_int: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Pandas UDF for generating predictions from pre-fitted ARIMA model
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    model_params : dict
        Pre-fitted model parameters
    steps : int
        Number of steps to forecast
    return_conf_int : bool
        Whether to return confidence intervals
        
    Returns:
    --------
    predictions : pd.Series
        Forecasted values
    conf_int : tuple, optional
        Confidence intervals if return_conf_int=True
    """
    try:
        # Remove any NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:  # Minimum data requirement
            if return_conf_int:
                nan_pred = pd.Series([np.nan] * steps)
                return nan_pred, (nan_pred, nan_pred)
            else:
                return pd.Series([np.nan] * steps)
        
        # Create ARIMA model with pre-fitted parameters
        model = ARIMAModel(
            order=model_params['order'],
            validation=False
        )
        
        # Manually set fitted parameters (simplified approach)
        model._fitted = True
        model._fitted_params = model_params
        
        # Generate predictions
        if return_conf_int:
            predictions, conf_int = model.predict(steps, return_conf_int=True)
            return pd.Series(predictions), (pd.Series(conf_int[0]), pd.Series(conf_int[1]))
        else:
            predictions = model.predict(steps)
            return pd.Series(predictions)
    
    except Exception as e:
        # Return NaN predictions if prediction fails
        print(f"ARIMA prediction failed: {e}")
        if return_conf_int:
            nan_pred = pd.Series([np.nan] * steps)
            return nan_pred, (nan_pred, nan_pred)
        else:
            return pd.Series([np.nan] * steps)


class ParallelARIMAProcessor:
    """
    High-level interface for parallel ARIMA processing with PySpark
    """
    
    def __init__(
        self,
        spark_session: Optional[Any] = None,
        spark_config: Optional[Dict[str, str]] = None,
        master: str = "local[*]",
        app_name: str = "TSLib-ParallelARIMA",
    ):
        """
        Initialize parallel ARIMA processor
        
        Parameters:
        -----------
        spark_session : SparkSession, optional
            Spark session. If None, verifies PySpark and starts a session
            (see tslib.spark.ensure.ensure_spark_session).
        spark_config : dict, optional
            Extra Spark configuration (memory, cores, etc.).
        master : str
            Spark master URL, e.g. local[*] or local[8] for eight local cores.
        app_name : str
            Application name for a newly created session.
        """
        if not check_spark_availability():
            from .ensure import DISTRIBUTED_REQUIRES_SPARK

            raise ImportError(DISTRIBUTED_REQUIRES_SPARK)

        from .ensure import ensure_spark_session

        cfg = dict(spark_config or {})
        cfg.setdefault("spark.sql.execution.arrow.pyspark.enabled", "true")
        self.spark = ensure_spark_session(
            spark_session=spark_session,
            spark_config=cfg,
            master=master,
            app_name=app_name,
            register_global=True,
        )
    
    def fit_multiple_arima(self,
                          df: Any,
                          group_column: str,
                          value_column: str = 'value',
                          time_column: str = 'timestamp',
                          order: Tuple[int, int, int] = (1, 1, 1),
                          auto_select: bool = False,
                          **kwargs) -> Any:
        """
        Fit ARIMA models for multiple time series in parallel
        
        Parameters:
        -----------
        df : DataFrame
            Spark DataFrame with time series data
        group_column : str
            Column to group by (identifies different time series)
        value_column : str
            Name of the value column
        time_column : str
            Name of the time column
        order : tuple
            ARIMA order (p, d, q)
        auto_select : bool
            Whether to automatically select optimal order
        **kwargs
            Additional parameters for ARIMA model
            
        Returns:
        --------
        results_df : DataFrame
            DataFrame with model fitting results
        """
        from pyspark.sql.functions import col, collect_list, struct
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
        
        # Define schema for the result
        result_schema = StructType([
            StructField("group_id", StringType(), True),
            StructField("success", StringType(), True),
            StructField("order_p", IntegerType(), True),
            StructField("order_d", IntegerType(), True),
            StructField("order_q", IntegerType(), True),
            StructField("aic", DoubleType(), True),
            StructField("bic", DoubleType(), True),
            StructField("log_likelihood", DoubleType(), True),
            StructField("error", StringType(), True)
        ])
        
        # Group data and apply ARIMA fitting
        def fit_arima_group(group_df):
            group_id = group_df[group_column].iloc[0]
            values = _group_values_time_ordered(group_df, value_column, time_column)
            series = pd.Series(values)
            # Fit ARIMA model
            result = fit_arima_udf(series, order, auto_select, **kwargs)
            
            # Format result
            if result['success']:
                return pd.DataFrame([{
                    'group_id': group_id,
                    'success': 'True',
                    'order_p': result['order'][0],
                    'order_d': result['order'][1],
                    'order_q': result['order'][2],
                    'aic': result['aic'],
                    'bic': result['bic'],
                    'log_likelihood': result['log_likelihood'],
                    'error': None
                }])
            else:
                return pd.DataFrame([{
                    'group_id': group_id,
                    'success': 'False',
                    'order_p': result['order'][0] if isinstance(result['order'], tuple) else None,
                    'order_d': result['order'][1] if isinstance(result['order'], tuple) else None,
                    'order_q': result['order'][2] if isinstance(result['order'], tuple) else None,
                    'aic': result['aic'],
                    'bic': result['bic'],
                    'log_likelihood': result['log_likelihood'],
                    'error': result['error']
                }])
        
        # Apply the function to each group
        results_df = df.groupBy(group_column).applyInPandas(fit_arima_group, result_schema)
        
        return results_df
    
    def predict_multiple_arima(self,
                              df: Any,
                              group_column: str,
                              value_column: str = 'value',
                              time_column: str = 'timestamp',
                              order: Tuple[int, int, int] = (1, 1, 1),
                              steps: int = 1,
                              return_conf_int: bool = False,
                              **kwargs) -> Any:
        """
        Generate predictions for multiple time series in parallel
        
        Parameters:
        -----------
        df : DataFrame
            Spark DataFrame with time series data
        group_column : str
            Column to group by
        value_column : str
            Name of the value column
        time_column : str
            Name of the time column
        order : tuple
            ARIMA order (p, d, q)
        steps : int
            Number of steps to forecast
        return_conf_int : bool
            Whether to return confidence intervals
        **kwargs
            Additional parameters for ARIMA model
            
        Returns:
        --------
        predictions_df : DataFrame
            DataFrame with predictions
        """
        from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
        
        # Define schema for predictions
        if return_conf_int:
            prediction_schema = StructType([
                StructField("group_id", StringType(), True),
                StructField("predictions", ArrayType(DoubleType()), True),
                StructField("lower_bound", ArrayType(DoubleType()), True),
                StructField("upper_bound", ArrayType(DoubleType()), True)
            ])
        else:
            prediction_schema = StructType([
                StructField("group_id", StringType(), True),
                StructField("predictions", ArrayType(DoubleType()), True)
            ])
        
        def predict_arima_group(group_df):
            group_id = group_df[group_column].iloc[0]
            values = _group_values_time_ordered(group_df, value_column, time_column)
            series = pd.Series(values)
            # Generate predictions
            if return_conf_int:
                predictions, conf_int = fit_predict_arima_udf(
                    series, order, steps, return_conf_int=True, **kwargs
                )
                return pd.DataFrame([{
                    'group_id': group_id,
                    'predictions': predictions.tolist(),
                    'lower_bound': conf_int[0].tolist(),
                    'upper_bound': conf_int[1].tolist()
                }])
            else:
                predictions = fit_predict_arima_udf(
                    series, order, steps, return_conf_int=False, **kwargs
                )
                return pd.DataFrame([{
                    'group_id': group_id,
                    'predictions': predictions.tolist()
                }])
        
        # Apply the function to each group
        predictions_df = df.groupBy(group_column).applyInPandas(predict_arima_group, prediction_schema)
        
        return predictions_df
    
    def close(self):
        """Close the Spark session"""
        if hasattr(self, 'spark'):
            self.spark.stop()




