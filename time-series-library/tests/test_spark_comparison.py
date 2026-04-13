"""
Tests comparing Spark vs normal ARIMA implementation

This module contains tests to ensure that the Spark implementation
produces consistent results with the normal implementation.
"""

import pytest
import numpy as np
import pandas as pd
from tslib.models import ARIMAModel
from tslib.utils.checks import check_spark_availability
from tests.spark_test_utils import get_spark_session_or_skip
from tests.spark_parallel_metrics import (
    per_series_forecast_agreement_pct,
    print_parallel_vs_sequential_accuracy,
    suite_arima_agreement_pct,
)


class TestSparkComparison:
    """Test Spark vs normal implementation consistency"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data for testing"""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        trend = 0.02 * t
        noise = np.random.normal(0, 1, n)
        return trend + noise
    
    @pytest.fixture
    def multiple_series_data(self):
        """Generate multiple time series for parallel processing"""
        np.random.seed(42)
        n_series = 3
        n_obs = 50
        
        data = []
        for i in range(n_series):
            # Generate different types of time series
            if i == 0:
                # Random walk
                series = np.cumsum(np.random.normal(0, 1, n_obs))
            elif i == 1:
                # Trend + noise
                t = np.arange(n_obs)
                series = 0.1 * t + np.random.normal(0, 1, n_obs)
            else:
                # AR(1) process
                phi = 0.7
                epsilon = np.random.normal(0, 1, n_obs)
                series = np.zeros(n_obs)
                series[0] = epsilon[0]
                for t in range(1, n_obs):
                    series[t] = phi * series[t-1] + epsilon[t]
            
            # Create DataFrame for this series
            series_df = pd.DataFrame({
                'series_id': f'series_{i}',
                'timestamp': range(n_obs),
                'value': series
            })
            data.append(series_df)
        
        return pd.concat(data, ignore_index=True)
    
    def test_spark_availability(self):
        """Test if Spark is available for testing"""
        spark_available = check_spark_availability()
        if not spark_available:
            pytest.skip("PySpark not available - skipping Spark comparison tests")
    
    def test_single_series_consistency(self, sample_data):
        """Test that single series results are consistent between implementations"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        from tslib.spark.parallel_arima import ParallelARIMAProcessor
        
        spark = None
        processor = None
        try:
            spark = get_spark_session_or_skip(
                app_name="TestComparison", master="local[1]"
            )
            # Normal implementation
            normal_model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
            normal_model.fit(sample_data)
            normal_predictions = normal_model.predict(steps=5)
            normal_aic = normal_model.get_model_selection_results()['best_aic'] if normal_model.get_model_selection_results() else None
            
            # Spark implementation
            # Convert single series to DataFrame format
            df_pandas = pd.DataFrame({
                'series_id': 'test_series',
                'timestamp': range(len(sample_data)),
                'value': sample_data
            })
            df_spark = spark.createDataFrame(df_pandas)
            
            processor = ParallelARIMAProcessor(spark_session=spark)
            
            # Fit model
            results_df = processor.fit_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=(1, 1, 1),
                auto_select=False
            )
            
            # Get predictions
            predictions_df = processor.predict_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=(1, 1, 1),
                steps=5,
                return_conf_int=False
            )
            
            # Convert results to pandas
            results_pandas = results_df.toPandas()
            predictions_pandas = predictions_df.toPandas()
            
            # Compare results
            if len(results_pandas) > 0 and len(predictions_pandas) > 0:
                spark_result = results_pandas.iloc[0]
                spark_predictions = predictions_pandas.iloc[0]['predictions']
                
                # Compare predictions (allow for small numerical differences)
                np.testing.assert_allclose(
                    normal_predictions, 
                    spark_predictions, 
                    rtol=1e-5, 
                    atol=1e-5,
                    err_msg="Predictions should be consistent between implementations"
                )
                
                # Compare AIC if available
                if normal_aic is not None:
                    spark_aic = spark_result['aic']
                    assert abs(normal_aic - spark_aic) < 1e-5, \
                        f"AIC values should be consistent: normal={normal_aic}, spark={spark_aic}"

                ag = per_series_forecast_agreement_pct(
                    np.asarray(normal_predictions, dtype=float),
                    np.asarray(spark_predictions, dtype=float),
                )
                print_parallel_vs_sequential_accuracy(
                    "spark_comparison.single_series",
                    ag,
                    1,
                    extra="ARIMA(1,1,1) pasos=5",
                )
            
        finally:
            if processor is not None:
                processor.close()
            if spark is not None:
                spark.stop()
    
    def test_multiple_series_processing(self, multiple_series_data):
        """Test that multiple series processing works correctly"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        from tslib.spark.parallel_arima import ParallelARIMAProcessor
        
        spark = None
        processor = None
        try:
            spark = get_spark_session_or_skip(
                app_name="TestMultipleSeries", master="local[2]"
            )
            # Convert to Spark DataFrame
            df_spark = spark.createDataFrame(multiple_series_data)
            
            processor = ParallelARIMAProcessor(spark_session=spark)
            
            # Fit models for all series
            results_df = processor.fit_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=(1, 1, 1),
                auto_select=False
            )
            
            # Generate predictions
            predictions_df = processor.predict_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=(1, 1, 1),
                steps=3,
                return_conf_int=True
            )
            
            # Convert to pandas for analysis
            results_pandas = results_df.toPandas()
            predictions_pandas = predictions_df.toPandas()
            
            # Verify results
            assert len(results_pandas) == len(multiple_series_data['series_id'].unique()), \
                "Should have results for all series"
            
            assert len(predictions_pandas) == len(multiple_series_data['series_id'].unique()), \
                "Should have predictions for all series"
            
            # Check that all series were processed successfully
            successful_fits = results_pandas[results_pandas['success'] == 'True']
            assert len(successful_fits) > 0, "At least some series should fit successfully"
            
            # Check prediction structure
            for _, row in predictions_pandas.iterrows():
                predictions = row['predictions']
                assert len(predictions) == 3, "Should have 3 step predictions"
                assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN"
                
                # Check confidence intervals if available
                if 'lower_bound' in row and 'upper_bound' in row:
                    lower = row['lower_bound']
                    upper = row['upper_bound']
                    assert len(lower) == 3, "Lower bound should have 3 values"
                    assert len(upper) == 3, "Upper bound should have 3 values"
                    assert np.all(lower <= upper), "Lower bound should be <= upper bound"

            normal_rows = []
            for series_id in multiple_series_data["series_id"].unique():
                series_data = multiple_series_data[
                    multiple_series_data["series_id"] == series_id
                ]["value"].values
                model = ARIMAModel(
                    order=(1, 1, 1), auto_select=False, validation=False
                )
                model.fit(series_data)
                normal_rows.append(
                    {
                        "series_id": series_id,
                        "predictions": model.predict(steps=3),
                        "success": True,
                    }
                )
            ag_m, n_m = suite_arima_agreement_pct(normal_rows, predictions_pandas)
            print_parallel_vs_sequential_accuracy(
                "spark_comparison.multi_series",
                ag_m,
                n_m,
                extra="pasos=3, intervalos",
            )
            
        finally:
            if processor is not None:
                processor.close()
            if spark is not None:
                spark.stop()
    
    def test_performance_comparison(self, multiple_series_data):
        """Test performance comparison between implementations"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        import time
        from tslib.spark.parallel_arima import ParallelARIMAProcessor
        
        spark = None
        processor = None
        try:
            spark = get_spark_session_or_skip(
                app_name="TestPerformance", master="local[2]"
            )
            # Normal implementation timing
            start_time = time.time()
            normal_results = []
            for series_id in multiple_series_data['series_id'].unique():
                series_data = multiple_series_data[
                    multiple_series_data['series_id'] == series_id
                ]['value'].values
                
                model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
                model.fit(series_data)
                predictions = model.predict(steps=3)
                normal_results.append({
                    'series_id': series_id,
                    'predictions': predictions,
                    'success': True,
                })
            normal_time = time.time() - start_time
            
            # Spark implementation timing
            start_time = time.time()
            df_spark = spark.createDataFrame(multiple_series_data)
            processor = ParallelARIMAProcessor(spark_session=spark)
            
            predictions_df = processor.predict_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=(1, 1, 1),
                steps=3,
                return_conf_int=False
            )
            
            # Force evaluation
            predictions_pandas = predictions_df.toPandas()
            spark_time = time.time() - start_time
            
            # Performance comparison
            print(f"\nPerformance Comparison:")
            print(f"Normal implementation: {normal_time:.3f} seconds")
            print(f"Spark implementation: {spark_time:.3f} seconds")
            print(f"Speedup: {normal_time / spark_time:.2f}x")
            
            # For small datasets, normal might be faster due to Spark overhead
            # This test mainly ensures both implementations work correctly
            assert len(predictions_pandas) == len(normal_results), \
                "Both implementations should process the same number of series"

            ag_p, n_p = suite_arima_agreement_pct(normal_results, predictions_pandas)
            print_parallel_vs_sequential_accuracy(
                "spark_comparison.performance",
                ag_p,
                n_p,
                extra="pasos=3",
            )
            
        finally:
            if processor is not None:
                processor.close()
            if spark is not None:
                spark.stop()
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent between implementations"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        from tslib.spark.parallel_arima import ParallelARIMAProcessor
        
        # Create invalid data (too short for ARIMA)
        invalid_data = pd.DataFrame({
            'series_id': 'invalid_series',
            'timestamp': [0, 1],
            'value': [1.0, 2.0]  # Only 2 points - too short for ARIMA(1,1,1)
        })
        
        spark = None
        processor = None
        try:
            spark = get_spark_session_or_skip(
                app_name="TestErrorHandling", master="local[1]"
            )
            # Normal implementation should raise an error
            with pytest.raises((ValueError, RuntimeError)):
                model = ARIMAModel(order=(1, 1, 1), auto_select=False, validation=False)
                model.fit(invalid_data['value'].values)
            
            # Spark implementation should handle errors gracefully
            df_spark = spark.createDataFrame(invalid_data)
            processor = ParallelARIMAProcessor(spark_session=spark)
            
            results_df = processor.fit_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=(1, 1, 1),
                auto_select=False
            )
            
            results_pandas = results_df.toPandas()
            
            # Should have a result but with success=False
            assert len(results_pandas) == 1, "Should have one result"
            assert results_pandas.iloc[0]['success'] == 'False', \
                "Should indicate failure for invalid data"
            
        finally:
            if processor is not None:
                processor.close()
            if spark is not None:
                spark.stop()


if __name__ == "__main__":
    pytest.main([__file__])
