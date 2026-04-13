"""
MA Model Example

Demonstrates how to use the TSLib MA model for time series analysis.
This example shows the complete workflow from data generation to forecasting
using a Moving Average model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslib.models import MAModel


def generate_ma_data(n=200, order=2, theta=None, mu=10.0, sigma=1.0, seed=42):
    """
    Generate synthetic MA(q) data
    
    Parameters:
    -----------
    n : int
        Number of observations
    order : int
        MA order
    theta : array-like, optional
        MA parameters. If None, will use default values
    mu : float
        Mean of the process
    sigma : float
        Standard deviation of innovations
    seed : int
        Random seed
        
    Returns:
    --------
    data : np.ndarray
        Generated MA time series
    """
    np.random.seed(seed)
    
    if theta is None:
        # Default MA parameters
        if order == 1:
            theta = [0.6]
        elif order == 2:
            theta = [0.7, -0.4]
        elif order == 3:
            theta = [0.6, -0.3, 0.2]
        else:
            theta = [0.5] + [-0.2] * (order - 1)
    
    theta = np.array(theta)
    epsilon = np.random.normal(0, sigma, n)
    
    y = np.zeros(n)
    for t in range(n):
        y[t] = mu + epsilon[t]
        for i in range(min(order, t)):
            y[t] += theta[i] * epsilon[t-i-1]
    
    return y


def main():
    """Main example function"""
    print("=" * 70)
    print("TSLib - MA Model Example")
    print("=" * 70)
    
    # ============================================================================
    # Example 1: MA(2) with Automatic Order Selection
    # ============================================================================
    print("\n" + "="*70)
    print("Example 1: MA(2) Model with Automatic Order Selection")
    print("="*70)
    
    # Generate MA(2) data: y_t = μ + ε_t + 0.7*ε_{t-1} - 0.4*ε_{t-2}
    print("\n1. Generating MA(2) synthetic data...")
    data = generate_ma_data(n=200, order=2, theta=[0.7, -0.4], mu=10.0, sigma=1.0)
    print(f"   Generated {len(data)} observations")
    print(f"   True parameters: μ = 10.0, θ₁ = 0.7, θ₂ = -0.4")
    print(f"   Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    
    # Fit MA model with automatic order selection
    print("\n2. Fitting MA model with automatic order selection...")
    model = MAModel(
        auto_select=True,
        max_order=5,
        selection_method='acf',  # Use ACF cutoff method
        validation=True
    )
    
    model.fit(data)
    print(f"   Selected order: MA({model.order})")
    print(f"   Expected: MA(2)")
    
    # Display model summary
    print("\n3. Model Summary:")
    print(model.summary())
    
    # Generate predictions
    print("\n4. Generating forecasts...")
    steps = 20
    predictions, conf_int = model.predict(steps=steps, return_conf_int=True)
    
    print(f"   Generated {steps} step-ahead forecasts")
    print(f"   First 5 predictions: {predictions[:5]}")
    print(f"   Note: MA forecasts beyond order q converge to the mean")
    print(f"   Prediction[1]: {predictions[0]:.4f}")
    print(f"   Prediction[5]: {predictions[4]:.4f}")
    print(f"   Prediction[20]: {predictions[19]:.4f}")
    print(f"   Mean: {np.mean(data):.4f}")
    
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
    
    # Plot forecast (notice convergence to mean)
    model.plot_forecast(steps=steps, figsize=(14, 6))
    
    # ============================================================================
    # Example 2: MA(1) - Response to Shocks
    # ============================================================================
    print("\n" + "="*70)
    print("Example 2: MA(1) Model - Modeling Shock Effects")
    print("="*70)
    
    # Generate MA(1) data with moderate shock effect
    print("\n1. Generating MA(1) data with θ₁ = 0.8...")
    data_ma1 = generate_ma_data(n=150, order=1, theta=[0.8], mu=5.0, sigma=1.0)
    print(f"   This represents a process where shocks affect the next period")
    print(f"   Suitable for modeling transitory effects")
    
    # Fit MA(1) model
    print("\n2. Fitting MA(1) model...")
    model_ma1 = MAModel(order=1, validation=False)  # Specify order directly
    model_ma1.fit(data_ma1)
    
    print(f"\n   Estimated μ: {model_ma1._ma_process.mean:.4f}")
    print(f"   True μ: 5.0000")
    print(f"   Estimated θ₁: {model_ma1._ma_process.ma_params[0]:.4f}")
    print(f"   True θ₁: 0.8000")
    print(f"   AIC: {model_ma1._fitted_params['aic']:.2f}")
    
    # Demonstrate forecast convergence
    forecast_ma1 = model_ma1.predict(steps=10)
    print(f"\n3. Forecast Behavior (converges quickly to mean):")
    print(f"   1-step ahead: {forecast_ma1[0]:.4f}")
    print(f"   2-step ahead: {forecast_ma1[1]:.4f} (= mean, as q=1)")
    print(f"   3-step ahead: {forecast_ma1[2]:.4f} (= mean)")
    print(f"   Mean: {model_ma1._ma_process.mean:.4f}")
    
    # ============================================================================
    # Example 3: Comparing Different MA Orders
    # ============================================================================
    print("\n" + "="*70)
    print("Example 3: Comparing Different MA Orders")
    print("="*70)
    
    # Generate MA(3) data
    data_ma3 = generate_ma_data(n=200, order=3, theta=[0.6, -0.3, 0.2], mu=8.0)
    
    print("\n1. Fitting models with different orders...")
    orders_to_try = [1, 2, 3, 4]
    results = []
    
    for order in orders_to_try:
        try:
            m = MAModel(order=order, validation=False)
            m.fit(data_ma3)
            aic = m._fitted_params['aic']
            bic = m._fitted_params['bic']
            results.append((order, aic, bic))
            print(f"   MA({order}): AIC={aic:.2f}, BIC={bic:.2f}")
        except:
            print(f"   MA({order}): Failed to fit")
    
    # Find best model by AIC
    best_by_aic = min(results, key=lambda x: x[1])
    best_by_bic = min(results, key=lambda x: x[2])
    
    print(f"\n2. Best model by AIC: MA({best_by_aic[0]}) (AIC={best_by_aic[1]:.2f})")
    print(f"   Best model by BIC: MA({best_by_bic[0]}) (BIC={best_by_bic[2]:.2f})")
    print(f"   True order: MA(3)")
    
    # ============================================================================
    # Example 4: MA vs White Noise
    # ============================================================================
    print("\n" + "="*70)
    print("Example 4: Distinguishing MA from White Noise")
    print("="*70)
    
    print("\n1. Generating two series...")
    # White noise
    np.random.seed(100)
    white_noise = np.random.normal(10, 1, 200)
    
    # MA(1)
    ma_series = generate_ma_data(n=200, order=1, theta=[0.7], mu=10.0, sigma=1.0, seed=100)
    
    print("   a) White Noise (no structure)")
    print("   b) MA(1) with θ₁ = 0.7")
    
    print("\n2. Fitting MA(1) to both series...")
    
    # Fit white noise
    model_wn = MAModel(order=1, validation=False)
    model_wn.fit(white_noise)
    print(f"\n   White Noise fitted as MA(1):")
    print(f"     θ₁: {model_wn._ma_process.ma_params[0]:.4f} (should be ≈ 0)")
    print(f"     AIC: {model_wn._fitted_params['aic']:.2f}")
    
    # Fit MA(1)
    model_ma = MAModel(order=1, validation=False)
    model_ma.fit(ma_series)
    print(f"\n   True MA(1) fitted:")
    print(f"     θ₁: {model_ma._ma_process.ma_params[0]:.4f} (true: 0.7)")
    print(f"     AIC: {model_ma._fitted_params['aic']:.2f}")
    
    print(f"\n3. Interpretation:")
    print(f"   - White noise: θ ≈ 0 indicates no MA structure")
    print(f"   - MA(1): θ ≈ 0.7 captures shock persistence")
    
    # ============================================================================
    # Example 5: Real-World Style Analysis - Forecast Error Modeling
    # ============================================================================
    print("\n" + "="*70)
    print("Example 5: Using MA for Forecast Error Correction")
    print("="*70)
    
    print("\n1. Simulating forecast errors...")
    # Imagine these are forecast errors from another model
    # Errors often have MA structure
    forecast_errors = generate_ma_data(n=250, order=2, theta=[0.5, -0.3], mu=0.0, sigma=2.0, seed=123)
    
    print(f"   Forecast errors generated: n = {len(forecast_errors)}")
    print(f"   Mean error: {np.mean(forecast_errors):.4f} (should be ≈ 0)")
    print(f"   Std error: {np.std(forecast_errors):.4f}")
    
    # Fit MA model to errors
    print("\n2. Fitting MA model to errors...")
    error_model = MAModel(auto_select=True, max_order=5, validation=False)
    error_model.fit(forecast_errors)
    
    print(f"   Identified structure: MA({error_model.order})")
    print(f"   This suggests forecast errors have short-term correlation")
    
    # Predict next error
    next_error_pred = error_model.predict(steps=1)
    print(f"\n3. Predicting next forecast error: {next_error_pred[0]:.4f}")
    print(f"   This can be used to adjust future forecasts")
    
    # Get exploratory analysis
    print("\n4. Error Structure Analysis:")
    analysis = error_model.get_exploratory_analysis()
    
    if analysis['acf_pacf']:
        acf_results = analysis['acf_pacf']
        print(f"   ACF shows significant lags up to: {error_model.order}")
        print(f"   This indicates {error_model.order}-period shock effect")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    
    # Key takeaways
    print("\nKey Takeaways:")
    print("• MA models capture the effect of past shocks/innovations")
    print("• ACF shows clear cutoff at lag q for MA(q) models")
    print("• Forecasts beyond q steps converge to the mean")
    print("• MA models are always stationary (unlike AR)")
    print("• Useful for modeling transitory effects and forecast errors")
    print("• Estimation requires iterative methods (MLE)")


if __name__ == "__main__":
    main()

