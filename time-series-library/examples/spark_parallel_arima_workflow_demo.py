"""
Demo: Parallel ARIMA Workflow with Spark

This demo shows how to use ParallelARIMAWorkflow to fit ARIMA models
using the complete 11-step parallel process with Spark.

The workflow automatically:
1. Determines differencing order
2-3. Generates parameter combinations
4-5. Fits models in parallel across windows
6. Selects best global model
7-8. Validates with backtesting
9. Performs residual diagnostics
10. Adjusts if needed
11. Generates forecasts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslib.spark import ParallelARIMAWorkflow
from tslib.utils.checks import check_spark_availability

def generate_sample_data(n=500, seed=42):
    """Generate sample ARIMA time series"""
    np.random.seed(seed)
    
    # Generate ARIMA(2,1,1) data
    # AR: y_t = 0.5*y_{t-1} - 0.3*y_{t-2} + e_t + 0.4*e_{t-1}
    n_total = n + 100  # Extra for warmup
    y = np.zeros(n_total)
    e = np.random.normal(0, 1, n_total)
    
    for t in range(2, n_total):
        y[t] = 0.5*y[t-1] - 0.3*y[t-2] + e[t] + 0.4*e[t-1]
    
    # Add trend through cumsum (differencing order 1)
    y_trend = np.cumsum(y[100:])  # Remove warmup
    
    return y_trend

def main():
    """Main demo function"""
    
    print("="*70)
    print("PARALLEL ARIMA WORKFLOW DEMO")
    print("="*70)
    print()
    
    # Check if Spark is available
    if not check_spark_availability():
        print("❌ PySpark not available!")
        print("Install with: pip install -r requirements-spark.txt")
        return
    
    print("✓ PySpark available")
    print()
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(n=500, seed=42)
    print(f"  Data: {len(data)} observations")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Std: {np.std(data):.2f}")
    print()
    
    # Initialize workflow
    print("Initializing ParallelARIMAWorkflow...")
    workflow = ParallelARIMAWorkflow(verbose=True)
    print()

    # Fit the workflow (executes all 10 steps)
    print("Fitting workflow (this will take a moment)...")
    print()
    workflow.fit(data)
    
    # Print summary
    print("\n")
    print(workflow.summary())
    
    # Generate forecast
    print("\n")
    print("="*70)
    print("GENERATING FORECAST")
    print("="*70)
    steps = 10
    forecast, conf_int = workflow.predict(steps=steps, return_conf_int=True)
    
    print(f"\n{steps}-step forecast:")
    for i, (pred, lower, upper) in enumerate(zip(forecast, conf_int[0], conf_int[1]), 1):
        print(f"  Step {i}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")
    
    # Plot results
    print("\n")
    print("Generating plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Data and forecast
    ax = axes[0]
    n_plot = min(100, len(data))
    ax.plot(range(n_plot), data[-n_plot:], label='Historical Data', color='blue')
    forecast_index = range(n_plot, n_plot + steps)
    ax.plot(forecast_index, forecast, label='Forecast', color='red', linewidth=2)
    ax.fill_between(forecast_index, conf_int[0], conf_int[1], 
                     alpha=0.3, color='red', label='95% CI')
    ax.set_title('Parallel ARIMA Workflow - Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Full time series
    ax = axes[1]
    ax.plot(data, label='Time Series', color='blue', alpha=0.7)
    ax.set_title('Complete Time Series')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parallel_arima_workflow_demo.png', dpi=150)
    print("  ✓ Plot saved as 'parallel_arima_workflow_demo.png'")
    
    # Get detailed results
    results = workflow.get_results()
    print("\n")
    print("="*70)
    print("DETAILED RESULTS SUMMARY")
    print("="*70)
    print(f"\nFinal Order: ARIMA{results['order']}")
    print(f"\nConfiguration:")
    for key, value in results['config'].items():
        print(f"  {key}: {value}")
    
    print(f"\nValidation Metrics:")
    val_metrics = results['step_results']['step7_8_validation']['metrics']
    print(f"  MAE: {val_metrics['avg_mae']:.4f}")
    print(f"  RMSE: {val_metrics['avg_rmse']:.4f}")
    print(f"  MAPE: {val_metrics['avg_mape']:.2f}%")
    
    print(f"\nDiagnostic Pass Rates:")
    diag_rates = results['step_results']['step9_diagnostics']['pass_rates']
    print(f"  Overall: {diag_rates['overall']*100:.1f}%")
    print(f"  Ljung-Box: {diag_rates['ljung_box']*100:.1f}%")
    print(f"  ACF: {diag_rates['acf']*100:.1f}%")
    
    print("\n")
    print("="*70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*70)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()

