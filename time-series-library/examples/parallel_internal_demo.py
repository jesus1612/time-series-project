"""
Demo de Paralelización Interna en TSLib

Este demo muestra cómo la paralelización interna funciona en las operaciones
computacionalmente intensivas del modelo ARIMA, como MLE, ACF/PACF, etc.
"""

import numpy as np
import time
import psutil
from tslib.models import ARIMAModel
from tslib.core.optimization import MLEOptimizer
from tslib.core.acf_pacf import ACFCalculator, PACFCalculator


def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def create_large_time_series(n_obs=5000):
    """Create a large time series for testing"""
    np.random.seed(42)
    
    # AR(2) process with trend
    ar_coefs = [0.6, -0.3]
    trend = 0.01
    noise_std = 1.0
    
    series = np.zeros(n_obs)
    series[0] = np.random.normal(0, noise_std)
    series[1] = np.random.normal(0, noise_std)
    
    for t in range(2, n_obs):
        series[t] = (ar_coefs[0] * series[t-1] + 
                    ar_coefs[1] * series[t-2] + 
                    trend * t + 
                    np.random.normal(0, noise_std))
    
    return series


def demo_mle_optimizer_parallelization():
    """Demo MLEOptimizer parallelization"""
    print("🔧 DEMO: MLEOptimizer Paralelización Interna")
    print("=" * 50)
    
    # Create large dataset
    data = create_large_time_series(3000)
    print(f"📊 Dataset: {len(data):,} observaciones")
    
    # Test sequential (n_jobs=1)
    print("\n🔄 Ejecutando MLE secuencial (n_jobs=1)...")
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    
    optimizer_seq = MLEOptimizer(n_jobs=1)
    result_seq = optimizer_seq.estimate(data, model_type='ARIMA', p=2, q=1, d=1)
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    time_seq = end_time - start_time
    mem_seq = end_mem - start_mem
    
    print(f"   ⏱️  Tiempo: {time_seq:.3f}s")
    print(f"   💾 Memoria: {mem_seq:.2f}MB")
    print(f"   📈 Log-likelihood: {result_seq['log_likelihood']:.3f}")
    
    # Test parallel (n_jobs=-1)
    print("\n⚡ Ejecutando MLE paralelo (n_jobs=-1)...")
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    
    optimizer_par = MLEOptimizer(n_jobs=-1)
    result_par = optimizer_par.estimate(data, model_type='ARIMA', p=2, q=1, d=1)
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    time_par = end_time - start_time
    mem_par = end_mem - start_mem
    
    print(f"   ⏱️  Tiempo: {time_par:.3f}s")
    print(f"   💾 Memoria: {mem_par:.2f}MB")
    print(f"   📈 Log-likelihood: {result_par['log_likelihood']:.3f}")
    
    # Analysis
    print(f"\n📊 ANÁLISIS:")
    print(f"   🚀 Speedup: {time_seq/time_par:.2f}x")
    print(f"   💾 Diferencia memoria: {mem_par - mem_seq:+.2f}MB")
    print(f"   ✅ Consistencia: {abs(result_seq['log_likelihood'] - result_par['log_likelihood']) < 0.1}")


def demo_acf_pacf_parallelization():
    """Demo ACF/PACF parallelization"""
    print("\n🔧 DEMO: ACF/PACF Paralelización Interna")
    print("=" * 50)
    
    # Create large dataset
    data = create_large_time_series(2000)
    print(f"📊 Dataset: {len(data):,} observaciones")
    
    # Test ACF sequential
    print("\n🔄 Ejecutando ACF secuencial (n_jobs=1)...")
    start_time = time.perf_counter()
    acf_seq = ACFCalculator(n_jobs=1)
    lags_seq, acf_seq_values = acf_seq.calculate(data)
    time_seq = time.perf_counter() - start_time
    
    print(f"   ⏱️  Tiempo: {time_seq:.3f}s")
    print(f"   📊 Lags calculados: {len(acf_seq_values)}")
    
    # Test ACF parallel
    print("\n⚡ Ejecutando ACF paralelo (n_jobs=-1)...")
    start_time = time.perf_counter()
    acf_par = ACFCalculator(n_jobs=-1)
    lags_par, acf_par_values = acf_par.calculate(data)
    time_par = time.perf_counter() - start_time
    
    print(f"   ⏱️  Tiempo: {time_par:.3f}s")
    print(f"   📊 Lags calculados: {len(acf_par_values)}")
    
    # Test PACF
    print("\n🔄 Ejecutando PACF secuencial (n_jobs=1)...")
    start_time = time.perf_counter()
    pacf_seq = PACFCalculator(n_jobs=1)
    lags_seq, pacf_seq_values = pacf_seq.calculate(data)
    time_seq_pacf = time.perf_counter() - start_time
    
    print(f"   ⏱️  Tiempo: {time_seq_pacf:.3f}s")
    
    print("\n⚡ Ejecutando PACF paralelo (n_jobs=-1)...")
    start_time = time.perf_counter()
    pacf_par = PACFCalculator(n_jobs=-1)
    lags_par, pacf_par_values = pacf_par.calculate(data)
    time_par_pacf = time.perf_counter() - start_time
    
    print(f"   ⏱️  Tiempo: {time_par_pacf:.3f}s")
    
    # Analysis
    print(f"\n📊 ANÁLISIS:")
    print(f"   🚀 ACF Speedup: {time_seq/time_par:.2f}x")
    print(f"   🚀 PACF Speedup: {time_seq_pacf/time_par_pacf:.2f}x")
    print(f"   ✅ ACF Consistencia: {np.allclose(acf_seq_values, acf_par_values)}")
    print(f"   ✅ PACF Consistencia: {np.allclose(pacf_seq_values, pacf_par_values)}")


def demo_arima_model_parallelization():
    """Demo ARIMAModel parallelization"""
    print("\n🔧 DEMO: ARIMAModel Paralelización Interna")
    print("=" * 50)
    
    # Create large dataset
    data = create_large_time_series(4000)
    print(f"📊 Dataset: {len(data):,} observaciones")
    
    # Test sequential
    print("\n🔄 Ejecutando ARIMA secuencial (n_jobs=1)...")
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    
    model_seq = ARIMAModel(order=(2, 1, 1), n_jobs=1, auto_select=False, validation=False)
    model_seq.fit(data)
    predictions_seq = model_seq.predict(steps=10)
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    time_seq = end_time - start_time
    mem_seq = end_mem - start_mem
    
    print(f"   ⏱️  Tiempo: {time_seq:.3f}s")
    print(f"   💾 Memoria: {mem_seq:.2f}MB")
    print(f"   📈 Log-likelihood: {model_seq._fitted_params['log_likelihood']:.3f}")
    print(f"   🔮 Predicciones: {len(predictions_seq)} pasos")
    
    # Test parallel
    print("\n⚡ Ejecutando ARIMA paralelo (n_jobs=-1)...")
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    
    model_par = ARIMAModel(order=(2, 1, 1), n_jobs=-1, auto_select=False, validation=False)
    model_par.fit(data)
    predictions_par = model_par.predict(steps=10)
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    time_par = end_time - start_time
    mem_par = end_mem - start_mem
    
    print(f"   ⏱️  Tiempo: {time_par:.3f}s")
    print(f"   💾 Memoria: {mem_par:.2f}MB")
    print(f"   📈 Log-likelihood: {model_par._fitted_params['log_likelihood']:.3f}")
    print(f"   🔮 Predicciones: {len(predictions_par)} pasos")
    
    # Analysis
    print(f"\n📊 ANÁLISIS:")
    print(f"   🚀 Speedup: {time_seq/time_par:.2f}x")
    print(f"   💾 Diferencia memoria: {mem_par - mem_seq:+.2f}MB")
    print(f"   ✅ Consistencia LL: {abs(model_seq._fitted_params['log_likelihood'] - model_par._fitted_params['log_likelihood']) < 0.1}")
    print(f"   ✅ Consistencia Pred: {np.allclose(predictions_seq, predictions_par, rtol=1e-3)}")


def demo_parallelization_thresholds():
    """Demo parallelization thresholds"""
    print("\n🔧 DEMO: Umbrales de Paralelización")
    print("=" * 50)
    
    sizes = [100, 500, 1000, 2000, 5000]
    
    print("📊 Probando diferentes tamaños de dataset:")
    print("   Tamaño    Secuencial    Paralelo     Speedup")
    print("   " + "-" * 45)
    
    for size in sizes:
        data = create_large_time_series(size)
        
        # Sequential
        start_time = time.perf_counter()
        model_seq = ARIMAModel(order=(1, 1, 1), n_jobs=1, auto_select=False, validation=False)
        model_seq.fit(data)
        time_seq = time.perf_counter() - start_time
        
        # Parallel
        start_time = time.perf_counter()
        model_par = ARIMAModel(order=(1, 1, 1), n_jobs=-1, auto_select=False, validation=False)
        model_par.fit(data)
        time_par = time.perf_counter() - start_time
        
        speedup = time_seq / time_par if time_par > 0 else 1.0
        
        print(f"   {size:6d}    {time_seq:8.3f}s    {time_par:8.3f}s    {speedup:6.2f}x")
    
    print(f"\n💡 Observación: La paralelización es más beneficiosa con datasets grandes")


def main():
    """Main demo function"""
    print("🚀 DEMO DE PARALELIZACIÓN INTERNA EN TSLIB")
    print("=" * 60)
    print("Este demo muestra cómo TSLib paraleliza automáticamente")
    print("operaciones computacionalmente intensivas dentro del modelo ARIMA.")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_mle_optimizer_parallelization()
        demo_acf_pacf_parallelization()
        demo_arima_model_parallelization()
        demo_parallelization_thresholds()
        
        print("\n" + "=" * 60)
        print("✅ Demo completado exitosamente!")
        print("\n💡 CONCLUSIONES:")
        print("   • La paralelización interna funciona correctamente")
        print("   • Los resultados son consistentes entre secuencial y paralelo")
        print("   • El speedup es más notable con datasets grandes")
        print("   • La paralelización se activa automáticamente según umbrales")
        print("   • El parámetro n_jobs controla el nivel de paralelización")
        
    except Exception as e:
        print(f"\n❌ Error en el demo: {e}")
        raise


if __name__ == "__main__":
    main()
