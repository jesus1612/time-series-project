"""
AR Model Example

Demonstrates how to use the TSLib AR model for time series analysis.
This example shows the complete workflow from data generation to forecasting
using an Autoregressive model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslib.models import ARModel


def generate_ar_data(n=200, order=2, phi=None, sigma=1.0, seed=42):
    """
    Generate synthetic AR(p) data
    
    Parameters:
    -----------
    n : int
        Number of observations
    order : int
        AR order
    phi : array-like, optional
        AR parameters. If None, will use default values
    sigma : float
        Standard deviation of noise
    seed : int
        Random seed
        
    Returns:
    --------
    data : np.ndarray
        Generated AR time series
    """
    np.random.seed(seed)
    
    if phi is None:
        # Default AR parameters
        if order == 1:
            phi = [0.7]
        elif order == 2:
            phi = [0.6, -0.3]
        elif order == 3:
            phi = [0.5, -0.3, 0.2]
        else:
            phi = [0.6] + [-0.2] * (order - 1)
    
    phi = np.array(phi)
    epsilon = np.random.normal(0, sigma, n)
    
    y = np.zeros(n)
    for t in range(order, n):
        y[t] = np.sum(phi * y[t-order:t][::-1]) + epsilon[t]
    
    return y


def main():
    """Main example function"""
    print("=" * 70)
    print("TSLib - AR Model Example")
    print("=" * 70)
    
    # ============================================================================
    # Example 1: AR(2) with Automatic Order Selection
    # ============================================================================
    print("\n" + "="*70)
    print("Example 1: AR(2) Model with Automatic Order Selection")
    print("="*70)
    
    # Generate AR(2) data: y_t = 0.6*y_{t-1} - 0.3*y_{t-2} + ε_t
    print("\n1. Generating AR(2) synthetic data...")
    data = generate_ar_data(n=200, order=2, phi=[0.6, -0.3], sigma=1.0)
    print(f"   Generated {len(data)} observations")
    print(f"   True parameters: φ₁ = 0.6, φ₂ = -0.3")
    print(f"   Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    
    # Fit AR model with automatic order selection
    print("\n2. Fitting AR model with automatic order selection...")
    model = ARModel(
        auto_select=True,
        max_order=5,
        selection_method='pacf',  # Use PACF cutoff method
        validation=True
    )
    
    model.fit(data)
    print(f"   Selected order: AR({model.order})")
    print(f"   Expected: AR(2)")
    
    # Display model summary
    print("\n3. Model Summary:")
    print(model.summary())
    
    # Generate predictions
    print("\n4. Generating forecasts...")
    steps = 20
    predictions, conf_int = model.predict(steps=steps, return_conf_int=True)
    
    print(f"   Generated {steps} step-ahead forecasts")
    print(f"   First 5 predictions: {predictions[:5]}")
    
    # Model diagnostics
    print("\n5. Model Diagnostics:")
    residuals = model.get_residuals()
    print(f"   Residuals mean: {np.mean(residuals):.6f} (should be ≈ 0)")
    print(f"   Residuals std: {np.std(residuals):.6f}")
    
    residual_diag = model.get_residual_diagnostics()
    print(f"   Ljung-Box p-value: {residual_diag['ljung_box_test']['p_value']:.4f}")
    print(f"   (p > 0.05 indicates residuals are white noise)")
    
    # Visualization
    print("\n6. Creating visualizations...")
    
    # Plot diagnostics
    model.plot_diagnostics(figsize=(14, 10))
    
    # Plot forecast
    model.plot_forecast(steps=steps, figsize=(14, 6))
    
    # ============================================================================
    # Example 2: AR(1) - Simple Random Walk with Mean Reversion
    # ============================================================================
    print("\n" + "="*70)
    print("Example 2: AR(1) Model - Mean Reverting Process")
    print("="*70)
    
    # Generate AR(1) data with strong autocorrelation
    print("\n1. Generating AR(1) data with φ₁ = 0.8...")
    data_ar1 = generate_ar_data(n=150, order=1, phi=[0.8], sigma=0.5)
    print(f"   This represents a mean-reverting process")
    print(f"   Strong persistence (high autocorrelation)")
    
    # Fit AR(1) model
    print("\n2. Fitting AR(1) model...")
    model_ar1 = ARModel(order=1, validation=False)  # Specify order directly
    model_ar1.fit(data_ar1)
    
    print(f"\n   Estimated φ₁: {model_ar1._ar_process.ar_params[0]:.4f}")
    print(f"   True φ₁: 0.8000")
    print(f"   AIC: {model_ar1._fitted_params['aic']:.2f}")
    
    # Short-term forecast
    forecast_ar1 = model_ar1.predict(steps=10)
    print(f"\n3. 10-step forecast converges toward mean: {np.mean(data_ar1):.4f}")
    print(f"   Forecast[1]: {forecast_ar1[0]:.4f}")
    print(f"   Forecast[5]: {forecast_ar1[4]:.4f}")
    print(f"   Forecast[10]: {forecast_ar1[9]:.4f}")
    
    # ============================================================================
    # Example 3: Comparing Different AR Orders
    # ============================================================================
    print("\n" + "="*70)
    print("Example 3: Comparing Different AR Orders")
    print("="*70)
    
    # Generate AR(3) data
    data_ar3 = generate_ar_data(n=200, order=3, phi=[0.5, -0.3, 0.2])
    
    print("\n1. Fitting models with different orders...")
    orders_to_try = [1, 2, 3, 4]
    results = []
    
    for order in orders_to_try:
        try:
            m = ARModel(order=order, validation=False)
            m.fit(data_ar3)
            aic = m._fitted_params['aic']
            bic = m._fitted_params['bic']
            results.append((order, aic, bic))
            print(f"   AR({order}): AIC={aic:.2f}, BIC={bic:.2f}")
        except:
            print(f"   AR({order}): Failed to fit")
    
    # Find best model by AIC
    best_by_aic = min(results, key=lambda x: x[1])
    best_by_bic = min(results, key=lambda x: x[2])
    
    print(f"\n2. Best model by AIC: AR({best_by_aic[0]}) (AIC={best_by_aic[1]:.2f})")
    print(f"   Best model by BIC: AR({best_by_bic[0]}) (BIC={best_by_bic[2]:.2f})")
    print(f"   True order: AR(3)")
    
    # ============================================================================
    # Example 4: Real-World Style Analysis
    # ============================================================================
    print("\n" + "="*70)
    print("Example 4: Complete Workflow with Exploratory Analysis")
    print("="*70)
    
    # Generate data
    data_real = generate_ar_data(n=250, order=2, phi=[0.7, -0.2], sigma=1.5)
    
    print("\n1. Loading and exploring data...")
    print(f"   Observations: {len(data_real)}")
    print(f"   Mean: {np.mean(data_real):.4f}")
    print(f"   Std: {np.std(data_real):.4f}")
    print(f"   Min: {np.min(data_real):.4f}")
    print(f"   Max: {np.max(data_real):.4f}")
    
    # Fit model
    print("\n2. Fitting AR model...")
    model_real = ARModel(auto_select=True, max_order=5)
    model_real.fit(data_real)
    
    # Get exploratory analysis
    print("\n3. Exploratory Analysis Results:")
    analysis = model_real.get_exploratory_analysis()
    
    if analysis['stationarity']:
        print(f"   Stationarity: {analysis['stationarity']['is_stationary']}")
        print(f"   ADF p-value: {analysis['stationarity']['adf_test']['p_value']:.4f}")
    
    if analysis['order_selection']:
        print(f"   Selection method: {analysis['order_selection']['method']}")
        print(f"   Selected order: {model_real.order}")
    
    # Evaluate forecast accuracy (using train-test split)
    print("\n4. Forecast Evaluation (Train-Test Split):")
    train_size = int(0.8 * len(data_real))
    train_data = data_real[:train_size]
    test_data = data_real[train_size:]
    
    # Fit on train data
    model_eval = ARModel(auto_select=True, max_order=5, validation=False)
    model_eval.fit(train_data)
    
    # Predict test period
    test_predictions = model_eval.predict(steps=len(test_data))
    
    # Calculate metrics
    metrics = model_eval.evaluate_forecast(test_data, test_predictions)
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_data)), train_data, label='Training Data', color='blue')
    plt.plot(range(len(train_data), len(data_real)), test_data, 
             label='Actual Test Data', color='green')
    plt.plot(range(len(train_data), len(data_real)), test_predictions, 
             label='Predicted', color='red', linestyle='--')
    plt.title('AR Model: Train-Test Split Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    
    # Key takeaways
    print("\nKey Takeaways:")
    print("• AR models capture autocorrelation in stationary time series")
    print("• PACF shows clear cutoff at lag p for AR(p) models")
    print("• Automatic order selection uses PACF analysis")
    print("• Forecasts converge to the mean for long horizons")
    print("• Always check residuals for white noise properties")


if __name__ == "__main__":
    main()

