"""
Comprehensive performance benchmark tests for Spark vs Normal ARIMA implementation

This module provides detailed performance comparisons between the standard
ARIMA implementation and the Spark-based parallel processing implementation.
It measures execution time, memory usage, scalability, and accuracy.
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Tuple, Any
from tslib.models import ARIMAModel
from tslib.utils.checks import check_spark_availability
from tests.spark_test_utils import get_spark_session_or_skip
from tests.spark_parallel_metrics import (
    benchmark_pair_agreement_pct,
    print_parallel_vs_sequential_accuracy,
)


def _print_spark_normal_agreement(label: str, normal_results: dict, spark_results: dict) -> None:
    if spark_results.get("execution_time") is None:
        return
    ag, n = benchmark_pair_agreement_pct(normal_results, spark_results)
    print_parallel_vs_sequential_accuracy(label, ag, n, extra="ARIMA Spark vs normal")


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark for ARIMA implementations
    
    Measures and compares:
    - Execution time for different dataset sizes
    - Scalability (1, 5, 10, 25, 50, 100 series)
    - Memory usage (Python vs Spark)
    - Result accuracy (numerical comparison)
    - Spark initialization overhead
    - Speedup factor and efficiency
    """
    
    def __init__(self):
        self.results = {}
        self.spark_available = check_spark_availability()
        
    def generate_time_series_data(self, 
                                 n_series: int, 
                                 n_obs: int, 
                                 series_type: str = 'mixed',
                                 seed: int = 42) -> pd.DataFrame:
        """
        Generate multiple time series for benchmarking
        
        Parameters:
        -----------
        n_series : int
            Number of time series to generate
        n_obs : int
            Number of observations per series
        series_type : str
            Type of series: 'ar', 'ma', 'arma', 'trend', 'mixed'
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with columns: series_id, timestamp, value
        """
        np.random.seed(seed)
        data = []
        
        for i in range(n_series):
            if series_type == 'ar':
                # AR(1) process
                phi = 0.7
                epsilon = np.random.normal(0, 1, n_obs)
                series = np.zeros(n_obs)
                series[0] = epsilon[0]
                for t in range(1, n_obs):
                    series[t] = phi * series[t-1] + epsilon[t]
                    
            elif series_type == 'ma':
                # MA(1) process
                theta = 0.5
                epsilon = np.random.normal(0, 1, n_obs)
                series = np.zeros(n_obs)
                for t in range(n_obs):
                    if t == 0:
                        series[t] = epsilon[t]
                    else:
                        series[t] = epsilon[t] + theta * epsilon[t-1]
                        
            elif series_type == 'arma':
                # ARMA(1,1) process
                phi, theta = 0.6, 0.4
                epsilon = np.random.normal(0, 1, n_obs)
                series = np.zeros(n_obs)
                series[0] = epsilon[0]
                for t in range(1, n_obs):
                    series[t] = phi * series[t-1] + epsilon[t] + theta * epsilon[t-1]
                    
            elif series_type == 'trend':
                # Trend + noise
                t = np.arange(n_obs)
                trend = 0.1 * t
                noise = np.random.normal(0, 1, n_obs)
                series = trend + noise
                
            else:  # mixed
                # Different types for different series
                if i % 4 == 0:
                    # AR(1)
                    phi = 0.7
                    epsilon = np.random.normal(0, 1, n_obs)
                    series = np.zeros(n_obs)
                    series[0] = epsilon[0]
                    for t in range(1, n_obs):
                        series[t] = phi * series[t-1] + epsilon[t]
                elif i % 4 == 1:
                    # MA(1)
                    theta = 0.5
                    epsilon = np.random.normal(0, 1, n_obs)
                    series = np.zeros(n_obs)
                    for t in range(n_obs):
                        if t == 0:
                            series[t] = epsilon[t]
                        else:
                            series[t] = epsilon[t] + theta * epsilon[t-1]
                elif i % 4 == 2:
                    # Trend
                    t = np.arange(n_obs)
                    trend = 0.05 * t
                    noise = np.random.normal(0, 1, n_obs)
                    series = trend + noise
                else:
                    # Random walk
                    series = np.cumsum(np.random.normal(0, 1, n_obs))
            
            # Create DataFrame for this series
            series_df = pd.DataFrame({
                'series_id': f'series_{i}',
                'timestamp': range(n_obs),
                'value': series
            })
            data.append(series_df)
        
        return pd.concat(data, ignore_index=True)
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_normal_implementation(self, 
                                      data: pd.DataFrame,
                                      order: Tuple[int, int, int] = (1, 1, 1),
                                      steps: int = 5) -> Dict[str, Any]:
        """
        Benchmark the normal (non-Spark) ARIMA implementation
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data
        order : tuple
            ARIMA order (p, d, q)
        steps : int
            Number of prediction steps
            
        Returns:
        --------
        results : dict
            Benchmark results including timing and memory usage
        """
        start_memory = self.measure_memory_usage()
        start_time = time.time()
        
        results = []
        successful_fits = 0
        
        for series_id in data['series_id'].unique():
            try:
                series_data = data[data['series_id'] == series_id]['value'].values
                
                # Fit model
                model = ARIMAModel(
                    order=order, 
                    auto_select=False, 
                    validation=False
                )
                model.fit(series_data)
                
                # Generate predictions
                predictions = model.predict(steps=steps)
                
                results.append({
                    'series_id': series_id,
                    'predictions': predictions,
                    'aic': model._fitted_params['aic'] if model._fitted_params else None,
                    'success': True
                })
                successful_fits += 1
                
            except Exception as e:
                results.append({
                    'series_id': series_id,
                    'predictions': None,
                    'aic': None,
                    'success': False,
                    'error': str(e)
                })
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        return {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'successful_fits': successful_fits,
            'total_series': len(data['series_id'].unique()),
            'success_rate': successful_fits / len(data['series_id'].unique()),
            'results': results,
            'implementation': 'normal'
        }
    
    def benchmark_spark_implementation(self, 
                                     data: pd.DataFrame,
                                     order: Tuple[int, int, int] = (1, 1, 1),
                                     steps: int = 5) -> Dict[str, Any]:
        """
        Benchmark the Spark-based ARIMA implementation
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data
        order : tuple
            ARIMA order (p, d, q)
        steps : int
            Number of prediction steps
            
        Returns:
        --------
        results : dict
            Benchmark results including timing and memory usage
        """
        if not self.spark_available:
            return {
                'execution_time': None,
                'memory_usage': None,
                'successful_fits': 0,
                'total_series': 0,
                'success_rate': 0,
                'results': [],
                'implementation': 'spark',
                'error': 'PySpark not available'
            }
        
        from tslib.spark.parallel_arima import ParallelARIMAProcessor
        
        start_memory = self.measure_memory_usage()
        start_time = time.time()
        
        spark = get_spark_session_or_skip(
            app_name="PerformanceBenchmark",
            master="local[2]",
            extra_config={"spark.sql.execution.arrow.pyspark.enabled": "true"},
        )
        processor = None
        try:
            # Convert to Spark DataFrame
            df_spark = spark.createDataFrame(data)
            
            processor = ParallelARIMAProcessor(spark_session=spark)
            
            # Fit models
            fit_results_df = processor.fit_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=order,
                auto_select=False
            )
            
            # Generate predictions
            predictions_df = processor.predict_multiple_arima(
                df=df_spark,
                group_column='series_id',
                value_column='value',
                time_column='timestamp',
                order=order,
                steps=steps,
                return_conf_int=False
            )
            
            # Convert to pandas for analysis
            fit_results_pandas = fit_results_df.toPandas()
            predictions_pandas = predictions_df.toPandas()
            
            successful_fits = len(fit_results_pandas[fit_results_pandas['success'] == 'True'])
            
            # Combine results
            results = []
            for _, row in fit_results_pandas.iterrows():
                series_id = row['group_id']
                success = row['success'] == 'True'
                
                # Find corresponding predictions
                pred_row = predictions_pandas[predictions_pandas['group_id'] == series_id]
                predictions = pred_row['predictions'].iloc[0] if len(pred_row) > 0 else None
                
                results.append({
                    'series_id': series_id,
                    'predictions': predictions,
                    'aic': row['aic'] if success else None,
                    'success': success,
                    'error': row['error'] if not success else None
                })
            
        finally:
            if processor is not None:
                processor.close()
            spark.stop()
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        return {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'successful_fits': successful_fits,
            'total_series': len(data['series_id'].unique()),
            'success_rate': successful_fits / len(data['series_id'].unique()),
            'results': results,
            'implementation': 'spark'
        }
    
    def compare_accuracy(self, 
                        normal_results: Dict[str, Any], 
                        spark_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare accuracy between normal and Spark implementations
        
        Parameters:
        -----------
        normal_results : dict
            Results from normal implementation
        spark_results : dict
            Results from Spark implementation
            
        Returns:
        --------
        accuracy_metrics : dict
            Accuracy comparison metrics
        """
        if not spark_results['results'] or not normal_results['results']:
            return {'mse': np.nan, 'mae': np.nan, 'max_diff': np.nan}
        
        # Create lookup dictionaries
        normal_lookup = {r['series_id']: r for r in normal_results['results']}
        spark_lookup = {r['series_id']: r for r in spark_results['results']}
        
        differences = []
        
        for series_id in normal_lookup:
            if (series_id in spark_lookup and 
                normal_lookup[series_id]['success'] and 
                spark_lookup[series_id]['success']):
                
                normal_pred = normal_lookup[series_id]['predictions']
                spark_pred = spark_lookup[series_id]['predictions']
                
                if normal_pred is not None and spark_pred is not None:
                    diff = np.abs(np.array(normal_pred) - np.array(spark_pred))
                    differences.extend(diff)
        
        if not differences:
            return {'mse': np.nan, 'mae': np.nan, 'max_diff': np.nan}
        
        differences = np.array(differences)
        
        return {
            'mse': np.mean(differences**2),
            'mae': np.mean(differences),
            'max_diff': np.max(differences),
            'rmse': np.sqrt(np.mean(differences**2))
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across different dataset sizes
        
        Returns:
        --------
        benchmark_results : dict
            Complete benchmark results
        """
        print("Starting comprehensive performance benchmark...")
        
        # Test configurations
        test_configs = [
            {'n_series': 1, 'n_obs': 100, 'name': 'single_series'},
            {'n_series': 5, 'n_obs': 100, 'name': 'small_batch'},
            {'n_series': 10, 'n_obs': 100, 'name': 'medium_batch'},
            {'n_series': 25, 'n_obs': 100, 'name': 'large_batch'},
            {'n_series': 50, 'n_obs': 100, 'name': 'xlarge_batch'},
            {'n_series': 100, 'n_obs': 100, 'name': 'massive_batch'},
        ]
        
        results = {}
        
        for config in test_configs:
            print(f"\nTesting {config['name']}: {config['n_series']} series, {config['n_obs']} obs each")
            
            # Generate test data
            data = self.generate_time_series_data(
                n_series=config['n_series'],
                n_obs=config['n_obs'],
                series_type='mixed'
            )
            
            # Benchmark normal implementation
            print("  Running normal implementation...")
            normal_results = self.benchmark_normal_implementation(data)
            
            # Benchmark Spark implementation
            print("  Running Spark implementation...")
            spark_results = self.benchmark_spark_implementation(data)
            
            # Compare accuracy
            accuracy = self.compare_accuracy(normal_results, spark_results)
            
            # Calculate speedup
            if (spark_results['execution_time'] is not None and 
                normal_results['execution_time'] is not None and
                spark_results['execution_time'] > 0):
                speedup = normal_results['execution_time'] / spark_results['execution_time']
            else:
                speedup = None
            
            results[config['name']] = {
                'config': config,
                'normal': normal_results,
                'spark': spark_results,
                'accuracy': accuracy,
                'speedup': speedup
            }

            ag_c, n_c = benchmark_pair_agreement_pct(normal_results, spark_results)
            print_parallel_vs_sequential_accuracy(
                f"perf_benchmark.comprehensive.{config['name']}",
                ag_c,
                n_c,
                extra="Spark vs normal",
            )
            
            # Print summary
            print(f"    Normal: {normal_results['execution_time']:.3f}s, "
                  f"{normal_results['success_rate']:.1%} success")
            if spark_results['execution_time'] is not None:
                print(f"    Spark:  {spark_results['execution_time']:.3f}s, "
                      f"{spark_results['success_rate']:.1%} success")
                if speedup is not None:
                    print(f"    Speedup: {speedup:.2f}x")
            else:
                print(f"    Spark:  Not available")
        
        self.results = results
        return results
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report
        
        Returns:
        --------
        report : str
            Formatted performance report
        """
        if not self.results:
            return "No benchmark results available. Run benchmark first."
        
        report = []
        report.append("=" * 80)
        report.append("TSLib Performance Benchmark Report")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        report.append("Performance Summary:")
        report.append("-" * 80)
        report.append(f"{'Test':<15} {'Series':<8} {'Normal(s)':<12} {'Spark(s)':<12} {'Speedup':<10} {'Accuracy':<12}")
        report.append("-" * 80)
        
        for test_name, result in self.results.items():
            config = result['config']
            normal = result['normal']
            spark = result['spark']
            speedup = result['speedup']
            accuracy = result['accuracy']
            
            normal_time = f"{normal['execution_time']:.3f}"
            spark_time = f"{spark['execution_time']:.3f}" if spark['execution_time'] else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
            accuracy_str = f"{accuracy['mae']:.6f}" if not np.isnan(accuracy['mae']) else "N/A"
            
            report.append(f"{test_name:<15} {config['n_series']:<8} {normal_time:<12} {spark_time:<12} {speedup_str:<10} {accuracy_str:<12}")
        
        report.append("")
        
        # Detailed analysis
        report.append("Detailed Analysis:")
        report.append("-" * 40)
        
        for test_name, result in self.results.items():
            config = result['config']
            normal = result['normal']
            spark = result['spark']
            accuracy = result['accuracy']
            
            report.append(f"\n{test_name.upper()} ({config['n_series']} series):")
            report.append(f"  Normal Implementation:")
            report.append(f"    Execution Time: {normal['execution_time']:.3f} seconds")
            report.append(f"    Memory Usage: {normal['memory_usage']:.2f} MB")
            report.append(f"    Success Rate: {normal['success_rate']:.1%}")
            
            if spark['execution_time'] is not None:
                report.append(f"  Spark Implementation:")
                report.append(f"    Execution Time: {spark['execution_time']:.3f} seconds")
                report.append(f"    Memory Usage: {spark['memory_usage']:.2f} MB")
                report.append(f"    Success Rate: {spark['success_rate']:.1%}")
                
                if result['speedup'] is not None:
                    report.append(f"  Performance:")
                    report.append(f"    Speedup: {result['speedup']:.2f}x")
                    report.append(f"    Efficiency: {result['speedup']/2:.2f}%")  # Assuming 2 cores
            else:
                report.append(f"  Spark Implementation: Not available")
            
            if not np.isnan(accuracy['mae']):
                report.append(f"  Accuracy Comparison:")
                report.append(f"    MAE: {accuracy['mae']:.6f}")
                report.append(f"    RMSE: {accuracy['rmse']:.6f}")
                report.append(f"    Max Difference: {accuracy['max_diff']:.6f}")
        
        # Conclusions
        report.append("\n" + "=" * 80)
        report.append("CONCLUSIONS:")
        report.append("=" * 80)
        
        # Find break-even point
        break_even = None
        for test_name, result in self.results.items():
            if result['speedup'] is not None and result['speedup'] > 1.0:
                break_even = result['config']['n_series']
                break
        
        if break_even:
            report.append(f"• Spark becomes more efficient than normal implementation at {break_even} series")
        else:
            report.append("• Normal implementation is more efficient for all tested dataset sizes")
        
        # Accuracy assessment
        all_accurate = all(
            np.isnan(result['accuracy']['mae']) or result['accuracy']['mae'] < 1e-10
            for result in self.results.values()
        )
        
        if all_accurate:
            report.append("• Both implementations produce numerically identical results")
        else:
            report.append("• Small numerical differences detected between implementations")
        
        # Memory usage
        avg_memory_normal = np.mean([r['normal']['memory_usage'] for r in self.results.values()])
        avg_memory_spark = np.mean([
            r['spark']['memory_usage'] for r in self.results.values() 
            if r['spark']['memory_usage'] is not None
        ])
        
        if not np.isnan(avg_memory_spark):
            report.append(f"• Average memory usage - Normal: {avg_memory_normal:.2f} MB, Spark: {avg_memory_spark:.2f} MB")
        
        return "\n".join(report)


class TestPerformanceBenchmark:
    """Test class for performance benchmarks"""
    
    def test_spark_availability(self):
        """Test if Spark is available for benchmarking"""
        spark_available = check_spark_availability()
        if not spark_available:
            pytest.skip("PySpark not available - skipping performance benchmark tests")
    
    def test_single_series_benchmark(self):
        """Test benchmark with single time series"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        benchmark = PerformanceBenchmark()
        
        # Generate single series
        data = benchmark.generate_time_series_data(n_series=1, n_obs=50, series_type='ar')
        
        # Run benchmarks
        normal_results = benchmark.benchmark_normal_implementation(data)
        spark_results = benchmark.benchmark_spark_implementation(data)
        
        # Verify results
        assert normal_results['total_series'] == 1
        assert normal_results['success_rate'] > 0
        assert normal_results['execution_time'] > 0
        
        if spark_results['execution_time'] is not None:
            assert spark_results['total_series'] == 1
            assert spark_results['success_rate'] > 0
            assert spark_results['execution_time'] > 0
        _print_spark_normal_agreement(
            "perf_benchmark.single_series", normal_results, spark_results
        )
    
    def test_multiple_series_benchmark(self):
        """Test benchmark with multiple time series"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        benchmark = PerformanceBenchmark()
        
        # Generate multiple series
        data = benchmark.generate_time_series_data(n_series=5, n_obs=50, series_type='mixed')
        
        # Run benchmarks
        normal_results = benchmark.benchmark_normal_implementation(data)
        spark_results = benchmark.benchmark_spark_implementation(data)
        
        # Verify results
        assert normal_results['total_series'] == 5
        assert normal_results['success_rate'] > 0
        
        if spark_results['execution_time'] is not None:
            assert spark_results['total_series'] == 5
            assert spark_results['success_rate'] > 0
        _print_spark_normal_agreement(
            "perf_benchmark.multiple_series", normal_results, spark_results
        )
    
    def test_accuracy_comparison(self):
        """Test accuracy comparison between implementations"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        benchmark = PerformanceBenchmark()
        
        # Generate test data
        data = benchmark.generate_time_series_data(n_series=3, n_obs=50, series_type='ar')
        
        # Run benchmarks
        normal_results = benchmark.benchmark_normal_implementation(data)
        spark_results = benchmark.benchmark_spark_implementation(data)
        
        # Compare accuracy
        accuracy = benchmark.compare_accuracy(normal_results, spark_results)
        ag_acc, n_acc = benchmark_pair_agreement_pct(normal_results, spark_results)
        _print_spark_normal_agreement(
            "perf_benchmark.accuracy", normal_results, spark_results
        )
        
        # Driver vs distributed fits use the same ARIMAModel but separate optimizations / float paths
        if not np.isnan(accuracy['mae']):
            assert n_acc > 0
            assert ag_acc >= 75.0, (
                f"Forecast agreement Spark vs normal too low: {ag_acc:.1f}% "
                f"(MAE={accuracy['mae']}, max_diff={accuracy['max_diff']})"
            )
            assert accuracy['max_diff'] < 2.0, f"Max difference too large: {accuracy['max_diff']}"
    
    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmark across multiple dataset sizes"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        benchmark = PerformanceBenchmark()
        
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that we have results for different dataset sizes
        expected_tests = ['single_series', 'small_batch', 'medium_batch']
        for test_name in expected_tests:
            assert test_name in results
            assert 'normal' in results[test_name]
            assert 'spark' in results[test_name]
            assert 'accuracy' in results[test_name]
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        if not check_spark_availability():
            pytest.skip("PySpark not available")
        
        benchmark = PerformanceBenchmark()
        
        # Run a small benchmark first
        data = benchmark.generate_time_series_data(n_series=2, n_obs=30, series_type='ar')
        nr = benchmark.benchmark_normal_implementation(data)
        sr = benchmark.benchmark_spark_implementation(data)
        _print_spark_normal_agreement("perf_benchmark.report_gen", nr, sr)

        # generate_performance_report() reads benchmark.results (filled by run_comprehensive_benchmark).
        accuracy = benchmark.compare_accuracy(nr, sr)
        sp = sr.get("execution_time")
        speedup = (nr["execution_time"] / sp) if sp and sp > 0 else None
        benchmark.results = {
            "report_smoke": {
                "config": {"n_series": 2, "n_obs": 30, "name": "report_smoke"},
                "normal": nr,
                "spark": sr,
                "accuracy": accuracy,
                "speedup": speedup,
            }
        }
        
        # Generate report
        report = benchmark.generate_performance_report()
        
        # Verify report content
        assert isinstance(report, str)
        assert "Performance Benchmark Report" in report
        assert "Performance Summary:" in report
        assert "CONCLUSIONS:" in report


if __name__ == "__main__":
    # Run comprehensive benchmark if executed directly
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    report = benchmark.generate_performance_report()
    print(report)
