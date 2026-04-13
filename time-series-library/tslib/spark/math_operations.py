"""
Spark Mathematical Operations Module

Provides distributed mathematical operations optimized for time series analysis
using PySpark's distributed computing capabilities.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, isnan, isnull, sum as spark_sum, mean as spark_mean, stddev as spark_stddev
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from .core import SparkDataConverter, SparkMathOperations


class SparkLinearAlgebra:
    """
    Distributed linear algebra operations using Spark
    
    Provides matrix operations, eigenvalue decomposition, and other
    linear algebra operations optimized for distributed computing.
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
        self.math_ops = SparkMathOperations(spark_session)
    
    def matrix_multiply(self, 
                       matrix_a: np.ndarray, 
                       matrix_b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using Spark
        
        Parameters:
        -----------
        matrix_a, matrix_b : np.ndarray
            Matrices to multiply
            
        Returns:
        --------
        result : np.ndarray
            Result of matrix multiplication
        """
        # Convert matrices to Spark DataFrames
        df_a = self.converter.create_matrix_dataframe(matrix_a, ('row_a', 'col_a'))
        df_b = self.converter.create_matrix_dataframe(matrix_b, ('row_b', 'col_b'))
        
        # Perform matrix multiplication
        # For each element (i,j) in result, sum over k: A[i,k] * B[k,j]
        joined = df_a.alias('a').join(
            df_b.alias('b'),
            col('a.col_a') == col('b.row_b')
        )
        
        # Calculate products
        product_df = joined.select(
            col('a.row_a').alias('row'),
            col('b.col_b').alias('col'),
            (col('a.value') * col('b.value')).alias('product')
        )
        
        # Sum products by (row, col)
        result_df = product_df.groupBy('row', 'col').agg({'product': 'sum'}).orderBy('row', 'col')
        
        # Convert back to NumPy
        result_matrix = self.converter.matrix_from_dataframe(result_df, ('row', 'col'), 'sum(product)')
        
        return result_matrix
    
    def matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Matrix inverse using Spark (for small matrices)
        
        Note: For large matrices, this uses NumPy as Spark doesn't have
        efficient matrix inverse operations.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Square matrix to invert
            
        Returns:
        --------
        inverse : np.ndarray
            Inverse matrix
        """
        # For small matrices, use NumPy (more efficient)
        if matrix.shape[0] < 1000:
            return np.linalg.inv(matrix)
        
        # For large matrices, use distributed approach
        # This is a simplified implementation - in practice, you might want
        # to use specialized distributed linear algebra libraries
        
        # Convert to Spark DataFrame
        df = self.converter.create_matrix_dataframe(matrix, ('row', 'col'))
        
        # Use Spark MLlib for matrix operations if available
        # For now, fall back to NumPy for large matrices too
        return np.linalg.inv(matrix)
    
    def solve_linear_system(self, 
                           A: np.ndarray, 
                           b: np.ndarray) -> np.ndarray:
        """
        Solve linear system Ax = b using Spark
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
            
        Returns:
        --------
        x : np.ndarray
            Solution vector
        """
        # For small systems, use NumPy
        if A.shape[0] < 1000:
            return np.linalg.solve(A, b)
        
        # For large systems, use distributed approach
        # Convert to Spark format
        df_A = self.converter.create_matrix_dataframe(A, ('row', 'col'))
        df_b = self.converter.to_spark_dataframe(b.reshape(-1, 1))
        
        # Use iterative methods or specialized distributed solvers
        # For now, fall back to NumPy
        return np.linalg.solve(A, b)
    
    def eigenvalue_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalue decomposition using Spark
        
        Parameters:
        -----------
        matrix : np.ndarray
            Square matrix
            
        Returns:
        --------
        eigenvalues : np.ndarray
            Eigenvalues
        eigenvectors : np.ndarray
            Eigenvectors
        """
        # For small matrices, use NumPy
        if matrix.shape[0] < 1000:
            eigenvals, eigenvecs = np.linalg.eig(matrix)
            return eigenvals, eigenvecs
        
        # For large matrices, use distributed approach
        # This would require specialized distributed linear algebra
        # For now, fall back to NumPy
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        return eigenvals, eigenvecs


class SparkStatistics:
    """
    Distributed statistical operations using Spark
    
    Provides statistical calculations optimized for large datasets.
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
        self.math_ops = SparkMathOperations(spark_session)
    
    def correlation_matrix(self, 
                          data: np.ndarray,
                          method: str = 'pearson') -> np.ndarray:
        """
        Calculate correlation matrix using Spark MLlib
        
        Parameters:
        -----------
        data : np.ndarray
            2D array with variables as columns
        method : str
            Correlation method ('pearson' or 'spearman')
            
        Returns:
        --------
        corr_matrix : np.ndarray
            Correlation matrix
        """
        # Convert to Spark DataFrame
        n_vars = data.shape[1]
        columns = [f'var_{i}' for i in range(n_vars)]
        
        df_pandas = pd.DataFrame(data, columns=columns)
        df_spark = self.spark.createDataFrame(df_pandas)
        
        # Use VectorAssembler to create feature vector
        assembler = VectorAssembler(inputCols=columns, outputCol='features')
        df_assembled = assembler.transform(df_spark)
        
        # Calculate correlation matrix
        corr_matrix = Correlation.corr(df_assembled, 'features', method).collect()[0][0]
        
        # Convert to NumPy
        return corr_matrix.toArray()
    
    def autocorrelation_function(self, 
                                data: np.ndarray,
                                max_lags: int = 40) -> np.ndarray:
        """
        Calculate autocorrelation function using Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        max_lags : int
            Maximum number of lags
            
        Returns:
        --------
        acf : np.ndarray
            Autocorrelation function values
        """
        # Convert to Spark DataFrame
        df = self.converter.to_spark_dataframe(data)
        
        # Calculate mean
        mean_val = self.math_ops.vector_mean(df)
        
        # Calculate variance
        variance = self.math_ops.vector_variance(df, mean=mean_val)
        
        if variance == 0:
            return np.ones(max_lags + 1)
        
        # Calculate ACF for each lag
        acf_values = []
        for k in range(max_lags + 1):
            if k == 0:
                acf_values.append(1.0)
            else:
                # Create lagged data
                df_lagged = df.select(
                    col('index').alias('index_orig'),
                    col('value').alias('value_orig')
                ).join(
                    df.select(
                        (col('index') + lit(k)).alias('index_lag'),
                        col('value').alias('value_lag')
                    ),
                    col('index_orig') == col('index_lag')
                )
                
                # Calculate covariance
                cov_df = df_lagged.select(
                    ((col('value_orig') - lit(mean_val)) * 
                     (col('value_lag') - lit(mean_val))).alias('cov')
                )
                
                covariance = cov_df.agg({'cov': 'mean'}).collect()[0][0]
                acf_k = covariance / variance if covariance is not None else 0.0
                acf_values.append(acf_k)
        
        return np.array(acf_values)
    
    def partial_autocorrelation_function(self, 
                                       data: np.ndarray,
                                       max_lags: int = 40) -> np.ndarray:
        """
        Calculate partial autocorrelation function using Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        max_lags : int
            Maximum number of lags
            
        Returns:
        --------
        pacf : np.ndarray
            Partial autocorrelation function values
        """
        # First calculate ACF
        acf = self.autocorrelation_function(data, max_lags)
        
        # Use Durbin-Levinson algorithm
        pacf_values = np.zeros(max_lags + 1)
        pacf_values[0] = 1.0
        
        if max_lags == 0:
            return pacf_values
        
        pacf_values[1] = acf[1]
        
        # Durbin-Levinson algorithm
        for k in range(2, max_lags + 1):
            # Initialize coefficients
            phi = np.zeros(k)
            phi[k-1] = acf[k]
            
            # Iterative refinement
            for iteration in range(10):
                phi_old = phi.copy()
                
                # Calculate new coefficients
                for j in range(k-1):
                    phi[j] = phi_old[j] - phi_old[k-1] * phi_old[k-2-j]
                
                # Check convergence
                if np.max(np.abs(phi - phi_old)) < 1e-6:
                    break
            
            pacf_values[k] = phi[k-1]
        
        return pacf_values
    
    def rolling_statistics(self, 
                          data: np.ndarray,
                          window_size: int,
                          statistic: str = 'mean') -> np.ndarray:
        """
        Calculate rolling statistics using Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        window_size : int
            Window size for rolling calculation
        statistic : str
            Statistic to calculate ('mean', 'std', 'var')
            
        Returns:
        --------
        rolling_stats : np.ndarray
            Rolling statistics
        """
        # Convert to Spark DataFrame
        df = self.converter.to_spark_dataframe(data)
        
        # Use Spark SQL window functions
        from pyspark.sql.window import Window
        from pyspark.sql.functions import lag, lead
        
        window = Window.orderBy('index').rowsBetween(-window_size + 1, 0)
        
        if statistic == 'mean':
            result_df = df.withColumn('rolling_mean', spark_mean('value').over(window))
            result = self.converter.to_numpy(result_df, 'rolling_mean')
        elif statistic == 'std':
            result_df = df.withColumn('rolling_std', spark_stddev('value').over(window))
            result = self.converter.to_numpy(result_df, 'rolling_std')
        elif statistic == 'var':
            result_df = df.withColumn('rolling_var', spark_stddev('value').over(window) ** 2)
            result = self.converter.to_numpy(result_df, 'rolling_var')
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")
        
        return result


class SparkOptimization:
    """
    Distributed optimization operations using Spark
    
    Provides optimization algorithms that can leverage Spark's
    distributed computing for parallel function evaluations.
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
    
    def parallel_function_evaluation(self, 
                                   function: callable,
                                   parameter_sets: List[np.ndarray]) -> List[float]:
        """
        Evaluate function in parallel for multiple parameter sets
        
        Parameters:
        -----------
        function : callable
            Function to evaluate
        parameter_sets : list
            List of parameter arrays
            
        Returns:
        --------
        results : list
            Function evaluation results
        """
        # Convert parameter sets to RDD
        param_rdd = self.spark.sparkContext.parallelize(parameter_sets)
        
        # Evaluate function in parallel
        result_rdd = param_rdd.map(function)
        
        # Collect results
        return result_rdd.collect()
    
    def grid_search_optimization(self, 
                                objective_function: callable,
                                parameter_grid: Dict[str, List[float]],
                                maximize: bool = False) -> Tuple[Dict[str, float], float]:
        """
        Grid search optimization using Spark
        
        Parameters:
        -----------
        objective_function : callable
            Objective function to optimize
        parameter_grid : dict
            Dictionary with parameter names and value lists
        maximize : bool
            Whether to maximize (True) or minimize (False)
            
        Returns:
        --------
        best_params : dict
            Best parameter combination
        best_value : float
            Best objective function value
        """
        # Generate all parameter combinations
        import itertools
        
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Convert to list of parameter dictionaries
        param_dicts = []
        for combination in param_combinations:
            param_dict = dict(zip(param_names, combination))
            param_dicts.append(param_dict)
        
        # Evaluate objective function in parallel
        def evaluate_params(params):
            return objective_function(params)
        
        results = self.parallel_function_evaluation(evaluate_params, param_dicts)
        
        # Find best result
        if maximize:
            best_idx = np.argmax(results)
        else:
            best_idx = np.argmin(results)
        
        best_params = param_dicts[best_idx]
        best_value = results[best_idx]
        
        return best_params, best_value
    
    def stochastic_gradient_descent(self, 
                                   gradient_function: callable,
                                   initial_params: np.ndarray,
                                   learning_rate: float = 0.01,
                                   max_iterations: int = 1000,
                                   batch_size: int = 100) -> np.ndarray:
        """
        Stochastic gradient descent using Spark for batch processing
        
        Parameters:
        -----------
        gradient_function : callable
            Function that returns gradient for given parameters
        initial_params : np.ndarray
            Initial parameter values
        learning_rate : float
            Learning rate
        max_iterations : int
            Maximum number of iterations
        batch_size : int
            Batch size for gradient calculation
            
        Returns:
        --------
        optimized_params : np.ndarray
            Optimized parameters
        """
        params = initial_params.copy()
        
        for iteration in range(max_iterations):
            # Calculate gradient (this could be distributed)
            gradient = gradient_function(params)
            
            # Update parameters
            params = params - learning_rate * gradient
            
            # Check convergence (simplified)
            if np.linalg.norm(gradient) < 1e-6:
                break
        
        return params


class SparkTimeSeriesOperations:
    """
    Specialized time series operations using Spark
    
    Provides time series specific operations optimized for distributed computing.
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
        self.stats = SparkStatistics(spark_session)
    
    def differencing(self, 
                    data: np.ndarray,
                    order: int = 1) -> np.ndarray:
        """
        Calculate differencing using Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        order : int
            Order of differencing
            
        Returns:
        --------
        diff_data : np.ndarray
            Differenced data
        """
        # Convert to Spark DataFrame
        df = self.converter.to_spark_dataframe(data)
        
        # Apply differencing using window functions
        from pyspark.sql.window import Window
        from pyspark.sql.functions import lag
        
        window = Window.orderBy('index')
        
        current_df = df
        for _ in range(order):
            current_df = current_df.withColumn(
                'value',
                col('value') - lag('value', 1).over(window)
            )
        
        # Remove NaN values (first few observations)
        result_df = current_df.filter(col('value').isNotNull())
        
        return self.converter.to_numpy(result_df)
    
    def seasonal_decomposition(self, 
                              data: np.ndarray,
                              period: int,
                              model: str = 'additive') -> Dict[str, np.ndarray]:
        """
        Seasonal decomposition using Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        period : int
            Seasonal period
        model : str
            Decomposition model ('additive' or 'multiplicative')
            
        Returns:
        --------
        decomposition : dict
            Dictionary with trend, seasonal, and residual components
        """
        # Convert to Spark DataFrame
        df = self.converter.to_spark_dataframe(data)
        
        # Calculate trend using moving average
        window_size = period if period % 2 == 1 else period + 1
        window = Window.orderBy('index').rowsBetween(-window_size//2, window_size//2)
        
        trend_df = df.withColumn('trend', spark_mean('value').over(window))
        
        # Calculate seasonal component
        if model == 'additive':
            seasonal_df = trend_df.withColumn('detrended', col('value') - col('trend'))
        else:  # multiplicative
            seasonal_df = trend_df.withColumn('detrended', col('value') / col('trend'))
        
        # Calculate seasonal averages
        seasonal_df = seasonal_df.withColumn('seasonal_index', col('index') % period)
        seasonal_avg = seasonal_df.groupBy('seasonal_index').agg(
            spark_mean('detrended').alias('seasonal_avg')
        )
        
        # Join back to get seasonal component
        final_df = seasonal_df.join(seasonal_avg, 'seasonal_index')
        
        # Calculate residuals
        if model == 'additive':
            final_df = final_df.withColumn('residual', 
                col('value') - col('trend') - col('seasonal_avg'))
        else:
            final_df = final_df.withColumn('residual',
                col('value') / (col('trend') * col('seasonal_avg')))
        
        # Convert results to NumPy
        trend = self.converter.to_numpy(final_df, 'trend')
        seasonal = self.converter.to_numpy(final_df, 'seasonal_avg')
        residual = self.converter.to_numpy(final_df, 'residual')
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def cross_correlation(self, 
                         x: np.ndarray,
                         y: np.ndarray,
                         max_lags: int = 40) -> np.ndarray:
        """
        Calculate cross-correlation function using Spark
        
        Parameters:
        -----------
        x, y : np.ndarray
            Time series data
        max_lags : int
            Maximum number of lags
            
        Returns:
        --------
        ccf : np.ndarray
            Cross-correlation function
        """
        # Convert to Spark DataFrames
        df_x = self.converter.to_spark_dataframe(x)
        df_y = self.converter.to_spark_dataframe(y)
        
        # Calculate means
        mean_x = self.math_ops.vector_mean(df_x)
        mean_y = self.math_ops.vector_mean(df_y)
        
        # Calculate standard deviations
        std_x = np.sqrt(self.math_ops.vector_variance(df_x, mean_x))
        std_y = np.sqrt(self.math_ops.vector_variance(df_y, mean_y))
        
        if std_x == 0 or std_y == 0:
            return np.zeros(2 * max_lags + 1)
        
        # Calculate cross-correlation for each lag
        ccf_values = []
        for k in range(-max_lags, max_lags + 1):
            if k == 0:
                # Join on same index
                joined = df_x.alias('x').join(df_y.alias('y'), 'index')
            elif k > 0:
                # y is lagged by k
                joined = df_x.alias('x').join(
                    df_y.select(
                        (col('index') + lit(k)).alias('index'),
                        col('value').alias('y_value')
                    ).alias('y'),
                    col('x.index') == col('y.index')
                )
            else:
                # x is lagged by -k
                joined = df_x.select(
                    (col('index') + lit(-k)).alias('index'),
                    col('value').alias('x_value')
                ).alias('x').join(df_y.alias('y'), 'index')
            
            # Calculate cross-covariance
            if k >= 0:
                cov_df = joined.select(
                    ((col('x.value') - lit(mean_x)) * 
                     (col('y.value') - lit(mean_y))).alias('cov')
                )
            else:
                cov_df = joined.select(
                    ((col('x.x_value') - lit(mean_x)) * 
                     (col('y.value') - lit(mean_y))).alias('cov')
                )
            
            covariance = cov_df.agg({'cov': 'mean'}).collect()[0][0]
            ccf_k = covariance / (std_x * std_y) if covariance is not None else 0.0
            ccf_values.append(ccf_k)
        
        return np.array(ccf_values)




