"""
Basic ARIMA Example

Demonstrates how to use the TSLib ARIMA model for time series analysis.
This example shows the complete workflow from data loading to forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslib.models import ARIMAModel


def generate_sample_data(n=100, trend=0.02, noise_std=1.0, seed=42):
    """
    Generate sample time series data with trend and noise
    
    Parameters:
    -----------
    n : int
        Number of observations
    trend : float
        Linear trend coefficient
    noise_std : float
        Standard deviation of noise
    seed : int
        Random seed
        
    Returns:
    --------
    data : np.ndarray
        Generated time series data
    """
    np.random.seed(seed)
    t = np.arange(n)
    trend_component = trend * t
    noise = np.random.normal(0, noise_std, n)
    return trend_component + noise


def main():
    """Main example function"""
    print("TSLib - Basic ARIMA Example")
    print("=" * 50)
    
    # Generate sample data
    print("1. Generating sample time series data...")
    data = generate_sample_data(n=100, trend=0.02, noise_std=1.0)
    print(f"   Generated {len(data)} observations")
    print(f"   Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    
    # Create and fit ARIMA model with automatic order selection
    print("\n2. Fitting ARIMA model with automatic order selection...")
    model = ARIMAModel(
        auto_select=True,
        max_p=3,
        max_d=2,
        max_q=3,
        validation=True
    )
    
    model.fit(data)
    print(f"   Selected order: ARIMA{model.order}")
    
    # Display model summary
    print("\n3. Model Summary:")
    print(model.summary())
    
    # Generate predictions
    print("\n4. Generating forecasts...")
    steps = 20
    predictions, conf_int = model.predict(steps=steps, return_conf_int=True)
    
    print(f"   Generated {steps} step-ahead forecasts")
    print(f"   Forecast mean: {np.mean(predictions):.4f}")
    print(f"   Forecast std: {np.std(predictions):.4f}")
    
    # Plot results
    print("\n5. Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot historical data
    plt.subplot(2, 1, 1)
    plt.plot(data, label='Historical Data', color='blue', linewidth=1.5)
    
    # Plot forecast
    forecast_index = np.arange(len(data), len(data) + steps)
    plt.plot(forecast_index, predictions, label='Forecast', color='red', linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(forecast_index, conf_int[0], conf_int[1], 
                    alpha=0.3, color='red', label='95% Confidence Interval')
    
    plt.title('ARIMA Model Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    residuals = model.get_residuals()
    plt.plot(residuals, label='Residuals', color='green', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.title('Model Residuals')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Model diagnostics
    print("\n6. Model Diagnostics:")
    print(f"   Residuals mean: {np.mean(residuals):.6f}")
    print(f"   Residuals std: {np.std(residuals):.6f}")
    print(f"   Residuals skewness: {np.mean((residuals - np.mean(residuals))**3) / np.std(residuals)**3:.6f}")
    
    # Get exploratory analysis results
    print("\n7. Exploratory Analysis Results:")
    analysis = model.get_exploratory_analysis()
    
    if analysis['acf_pacf']:
        suggested = analysis['acf_pacf']['suggested_orders']
        print(f"   ACF/PACF suggested p: {suggested['suggested_p']}")
        print(f"   ACF/PACF suggested q: {suggested['suggested_q']}")
    
    if analysis['stationarity']:
        print(f"   Stationarity test: {'Stationary' if analysis['stationarity']['is_stationary'] else 'Non-stationary'}")
        print(f"   Suggested differencing order: {analysis['stationarity']['suggested_differencing_order']}")
    
    # Model selection results
    print("\n8. Model Selection Results:")
    selection_results = model.get_model_selection_results()
    if selection_results:
        print(f"   Best AIC: {selection_results['best_aic']:.4f}")
        print(f"   Total models tested: {len(selection_results['all_results'])}")
        
        # Show top 3 models
        all_results = selection_results['all_results']
        all_results.sort(key=lambda x: x[3])  # Sort by AIC
        print("   Top 3 models by AIC:")
        for i, (p, d, q, aic) in enumerate(all_results[:3]):
            print(f"     {i+1}. ARIMA({p},{d},{q}): AIC = {aic:.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()




