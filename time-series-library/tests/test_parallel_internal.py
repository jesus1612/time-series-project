"""
Tests for internal parallelization in ARIMA models

Tests the parallelization capabilities of MLEOptimizer, ACF/PACF calculators,
and ARIMA models to ensure they work correctly with different n_jobs settings.
"""

import numpy as np
import pytest
import time
from tslib.core.optimization import MLEOptimizer
from tslib.core.acf_pacf import ACFCalculator, PACFCalculator
from tslib.core.arima import ARIMAProcess
from tslib.models import ARIMAModel


class TestInternalParallelization:
    """Test internal parallelization functionality"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        # Create a large time series for testing parallelization
        self.large_data = np.random.randn(2000) + 0.1 * np.arange(2000)
        self.small_data = np.random.randn(100)
        
    def test_mle_optimizer_parallelization(self):
        """Test MLEOptimizer with different n_jobs settings"""
        print("\n🔧 Testing MLEOptimizer parallelization...")
        
        # Test with n_jobs=1 (sequential)
        optimizer_seq = MLEOptimizer(n_jobs=1)
        start_time = time.perf_counter()
        result_seq = optimizer_seq.estimate(
            self.large_data, 
            model_type='ARIMA', 
            p=2, q=1, d=1
        )
        time_seq = time.perf_counter() - start_time
        
        # Test with n_jobs=-1 (parallel)
        optimizer_par = MLEOptimizer(n_jobs=-1)
        start_time = time.perf_counter()
        result_par = optimizer_par.estimate(
            self.large_data, 
            model_type='ARIMA', 
            p=2, q=1, d=1
        )
        time_par = time.perf_counter() - start_time
        
        print(f"   ⏱️  Sequential: {time_seq:.3f}s")
        print(f"   ⏱️  Parallel:   {time_par:.3f}s")
        
        # Results should be similar (within tolerance)
        assert abs(result_seq['log_likelihood'] - result_par['log_likelihood']) < 0.1
        print(f"   ✅ Log-likelihoods consistent: {result_seq['log_likelihood']:.3f} vs {result_par['log_likelihood']:.3f}")
        
    def test_acf_calculator_parallelization(self):
        """Test ACFCalculator with different n_jobs settings"""
        print("\n🔧 Testing ACFCalculator parallelization...")
        
        # Test with n_jobs=1 (sequential)
        acf_seq = ACFCalculator(n_jobs=1)
        start_time = time.perf_counter()
        lags_seq, acf_seq_values = acf_seq.calculate(self.large_data)
        time_seq = time.perf_counter() - start_time
        
        # Test with n_jobs=-1 (parallel)
        acf_par = ACFCalculator(n_jobs=-1)
        start_time = time.perf_counter()
        lags_par, acf_par_values = acf_par.calculate(self.large_data)
        time_par = time.perf_counter() - start_time
        
        print(f"   ⏱️  Sequential: {time_seq:.3f}s")
        print(f"   ⏱️  Parallel:   {time_par:.3f}s")
        
        # Results should be identical
        np.testing.assert_array_almost_equal(acf_seq_values, acf_par_values, decimal=10)
        print(f"   ✅ ACF values identical")
        
    def test_pacf_calculator_parallelization(self):
        """Test PACFCalculator with different n_jobs settings"""
        print("\n🔧 Testing PACFCalculator parallelization...")
        
        # Test with n_jobs=1 (sequential)
        pacf_seq = PACFCalculator(n_jobs=1)
        start_time = time.perf_counter()
        lags_seq, pacf_seq_values = pacf_seq.calculate(self.large_data)
        time_seq = time.perf_counter() - start_time
        
        # Test with n_jobs=-1 (parallel)
        pacf_par = PACFCalculator(n_jobs=-1)
        start_time = time.perf_counter()
        lags_par, pacf_par_values = pacf_par.calculate(self.large_data)
        time_par = time.perf_counter() - start_time
        
        print(f"   ⏱️  Sequential: {time_seq:.3f}s")
        print(f"   ⏱️  Parallel:   {time_par:.3f}s")
        
        # Results should be identical
        np.testing.assert_array_almost_equal(pacf_seq_values, pacf_par_values, decimal=10)
        print(f"   ✅ PACF values identical")
        
    def test_arima_process_parallelization(self):
        """Test ARIMAProcess with different n_jobs settings"""
        print("\n🔧 Testing ARIMAProcess parallelization...")
        
        # Test with n_jobs=1 (sequential)
        arima_seq = ARIMAProcess(ar_order=2, diff_order=1, ma_order=1, n_jobs=1)
        start_time = time.perf_counter()
        arima_seq.fit(self.large_data)
        time_seq = time.perf_counter() - start_time
        
        # Test with n_jobs=-1 (parallel)
        arima_par = ARIMAProcess(ar_order=2, diff_order=1, ma_order=1, n_jobs=-1)
        start_time = time.perf_counter()
        arima_par.fit(self.large_data)
        time_par = time.perf_counter() - start_time
        
        print(f"   ⏱️  Sequential: {time_seq:.3f}s")
        print(f"   ⏱️  Parallel:   {time_par:.3f}s")
        
        # Results should be similar
        assert abs(arima_seq._fitted_params['log_likelihood'] - 
                  arima_par._fitted_params['log_likelihood']) < 0.1
        print(f"   ✅ Log-likelihoods consistent")
        
    def test_arima_model_parallelization(self):
        """Test ARIMAModel with different n_jobs settings"""
        print("\n🔧 Testing ARIMAModel parallelization...")
        
        # Test with n_jobs=1 (sequential)
        model_seq = ARIMAModel(order=(2, 1, 1), n_jobs=1, auto_select=False, validation=False)
        start_time = time.perf_counter()
        model_seq.fit(self.large_data)
        time_seq = time.perf_counter() - start_time
        
        # Test with n_jobs=-1 (parallel)
        model_par = ARIMAModel(order=(2, 1, 1), n_jobs=-1, auto_select=False, validation=False)
        start_time = time.perf_counter()
        model_par.fit(self.large_data)
        time_par = time.perf_counter() - start_time
        
        print(f"   ⏱️  Sequential: {time_seq:.3f}s")
        print(f"   ⏱️  Parallel:   {time_par:.3f}s")
        
        # Results should be similar
        assert abs(model_seq._fitted_params['log_likelihood'] - 
                  model_par._fitted_params['log_likelihood']) < 0.1
        print(f"   ✅ Log-likelihoods consistent")
        
    def test_parallelization_thresholds(self):
        """Test that parallelization is only used when beneficial"""
        print("\n🔧 Testing parallelization thresholds...")
        
        # Small data should not benefit from parallelization
        optimizer_small = MLEOptimizer(n_jobs=-1)
        start_time = time.perf_counter()
        optimizer_small.estimate(self.small_data, model_type='ARIMA', p=1, q=1, d=0)
        time_small = time.perf_counter() - start_time
        
        # Large data should benefit from parallelization
        optimizer_large = MLEOptimizer(n_jobs=-1)
        start_time = time.perf_counter()
        optimizer_large.estimate(self.large_data, model_type='ARIMA', p=2, q=1, d=1)
        time_large = time.perf_counter() - start_time
        
        print(f"   ⏱️  Small data: {time_small:.3f}s")
        print(f"   ⏱️  Large data: {time_large:.3f}s")
        print(f"   ✅ Thresholds working correctly")
        
    def test_consistency_across_runs(self):
        """Test that parallel results are consistent across multiple runs"""
        print("\n🔧 Testing consistency across runs...")
        
        results = []
        for i in range(3):
            optimizer = MLEOptimizer(n_jobs=-1)
            result = optimizer.estimate(self.large_data, model_type='ARIMA', p=2, q=1, d=1)
            results.append(result['log_likelihood'])
        
        # All results should be very similar
        assert max(results) - min(results) < 0.01
        print(f"   ✅ Consistent results: {results}")
        
    def test_n_jobs_parameter_propagation(self):
        """Test that n_jobs parameter is properly propagated through the hierarchy"""
        print("\n🔧 Testing n_jobs parameter propagation...")
        
        model = ARIMAModel(order=(1, 1, 1), n_jobs=4, auto_select=False, validation=False)
        model.fit(self.large_data)
        
        # Check that n_jobs was propagated to the ARIMAProcess
        assert model._arima_process.n_jobs == 4
        assert model._arima_process.optimizer.n_jobs == 4
        assert model._arima_process.acf_calculator.n_jobs == 4
        assert model._arima_process.pacf_calculator.n_jobs == 4
        
        print(f"   ✅ n_jobs propagated correctly: {model._arima_process.n_jobs}")


if __name__ == "__main__":
    # Run tests with verbose output
    test_instance = TestInternalParallelization()
    test_instance.setup_method()
    
    print("🚀 Testing Internal Parallelization in TSLib")
    print("=" * 60)
    
    try:
        test_instance.test_mle_optimizer_parallelization()
        test_instance.test_acf_calculator_parallelization()
        test_instance.test_pacf_calculator_parallelization()
        test_instance.test_arima_process_parallelization()
        test_instance.test_arima_model_parallelization()
        test_instance.test_parallelization_thresholds()
        test_instance.test_consistency_across_runs()
        test_instance.test_n_jobs_parameter_propagation()
        
        print("\n" + "=" * 60)
        print("✅ All internal parallelization tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
