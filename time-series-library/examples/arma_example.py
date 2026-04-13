"""
ARMA Model Example

Demonstrates how to use the TSLib ARMA model for time series analysis.
This example shows the complete workflow from data generation to forecasting
using an Autoregressive Moving Average model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslib.models import ARMAModel


def generate_arma_data(n=200, ar_order=1, ma_order=1, phi=None, theta=None, 
                      mu=10.0, sigma=1.0, seed=42):
    """
    Generate synthetic ARMA(p,q) data
    
    Parameters:
    -----------
    n : int
        Number of observations
    ar_order : int
        AR order (p)
    ma_order : int
        MA order (q)
    phi : array-like, optional
        AR parameters
    theta : array-like, optional
        MA parameters
    mu : float
        Mean of the process
    sigma : float
        Standard deviation of innovations
    seed : int
        Random seed
        
    Returns:
    --------
    data : np.ndarray
        Generated ARMA time series
    """
    np.random.seed(seed)
    
    if phi is None:
        phi = [0.6] * ar_order if ar_order > 0 else []
    if theta is None:
        theta = [0.5] * ma_order if ma_order > 0 else []
    
    phi = np.array(phi)
    theta = np.array(theta)
    epsilon = np.random.normal(0, sigma, n)
    
    y = np.zeros(n)
    max_order = max(ar_order, ma_order)
    
    for t in range(max_order, n):
        # AR component
        if ar_order > 0:
            y[t] = np.sum(phi * y[t-ar_order:t][::-1])
        
        # MA component
        ma_term = epsilon[t]
        for i in range(ma_order):
            if t - i - 1 >= 0:
                ma_term += theta[i] * epsilon[t-i-1]
        
        y[t] += ma_term
    
    return y + mu


def main():
    """Main example function"""
    print("=" * 70)
    print("TSLib - ARMA Model Example")
    print("=" * 70)
    
    # ============================================================================
    # Example 1: ARMA(2,1) with Automatic Order Selection
    # ============================================================================
    print("\n" + "="*70)
    print("Example 1: ARMA(2,1) Model with Automatic Order Selection")
    print("="*70)
    
    # Generate ARMA(2,1) data
    print("\n1. Generating ARMA(2,1) synthetic data...")
    data = generate_arma_data(n=200, ar_order=2, ma_order=1, 
                             phi=[0.6, -0.3], theta=[0.5], mu=10.0, sigma=1.0)
    print(f"   Generated {len(data)} observations")
    print(f"   True parameters: φ₁ = 0.6, φ₂ = -0.3, θ₁ = 0.5")
    print(f"   Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    
    # Fit ARMA model with automatic order selection
    print("\n2. Fitting ARMA model with automatic order selection...")
    model = ARMAModel(
        auto_select=True,
        max_ar=5,
        max_ma=5,
        criterion='aic',
        validation=True
    )
    
    model.fit(data)
    print(f"   Selected order: ARMA({model.order[0]},{model.order[1]})")
    print(f"   Expected: ARMA(2,1)")
    
    # Display model summary
    print("\n3. Model Summary:")
    print(model.summary())
    
    # Generate predictions
    print("\n4. Generating forecasts...")
    steps = 20
    predictions, conf_int = model.predict(steps=steps, return_conf_int=True)
    
    print(f"   Generated {steps} step-ahead forecasts")
    print(f"   First 5 predictions: {predictions[:5]}")
    print(f"   Note: ARMA combines AR persistence with MA shock effects")
    
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
    model.plot_diagnostics(figsize=(14, 12))
    
    # Plot forecast
    model.plot_forecast(steps=steps, figsize=(14, 6))
    
    # ============================================================================
    # Example 2: ARMA(1,1) - The Workhorse Model
    # ============================================================================
    print("\n" + "="*70)
    print("Example 2: ARMA(1,1) - Most Common ARMA Specification")
    print("="*70)
    
    # Generate ARMA(1,1) data
    print("\n1. Generating ARMA(1,1) data...")
    data_11 = generate_arma_data(n=150, ar_order=1, ma_order=1, 
                                phi=[0.7], theta=[0.4], mu=5.0, sigma=1.0)
    print(f"   ARMA(1,1) is the most parsimonious mixed model")
    print(f"   Combines persistence (AR) with shock effects (MA)")
    
    # Fit ARMA(1,1) model
    print("\n2. Fitting ARMA(1,1) model...")
    model_11 = ARMAModel(order=(1, 1), validation=False)
    model_11.fit(data_11)
    
    print(f"\n   Estimated φ₁: {model_11._arma_process.ar_params[0]:.4f} (true: 0.7)")
    print(f"   Estimated θ₁: {model_11._arma_process.ma_params[0]:.4f} (true: 0.4)")
    print(f"   AIC: {model_11._fitted_params['aic']:.2f}")
    print(f"   BIC: {model_11._fitted_params['bic']:.2f}")
    
    # Compare with pure AR and MA
    print("\n3. Comparing with pure AR(1) and MA(1)...")
    from tslib.models import ARModel, MAModel
    
    ar1 = ARModel(order=1, validation=False)
    ar1.fit(data_11)
    
    ma1 = MAModel(order=1, validation=False)
    ma1.fit(data_11)
    
    print(f"   AR(1) AIC: {ar1._fitted_params['aic']:.2f}")
    print(f"   MA(1) AIC: {ma1._fitted_params['aic']:.2f}")
    print(f"   ARMA(1,1) AIC: {model_11._fitted_params['aic']:.2f}")
    print(f"   → ARMA(1,1) provides best fit (lowest AIC)")
    
    # ============================================================================
    # Example 3: Grid Search for Optimal Order
    # ============================================================================
    print("\n" + "="*70)
    print("Example 3: Grid Search for Optimal ARMA Order")
    print("="*70)
    
    # Generate ARMA(2,2) data
    data_22 = generate_arma_data(n=200, ar_order=2, ma_order=2,
                                phi=[0.5, -0.3], theta=[0.6, -0.2], mu=8.0)
    
    print("\n1. Performing grid search over (p,q) orders...")
    results = []
    
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                m = ARMAModel(order=(p, q), validation=False)
                m.fit(data_22)
                aic = m._fitted_params['aic']
                bic = m._fitted_params['bic']
                results.append((p, q, aic, bic))
            except:
                pass
    
    # Display results sorted by AIC
    results_sorted = sorted(results, key=lambda x: x[2])
    print("\n2. Top 5 models by AIC:")
    for i, (p, q, aic, bic) in enumerate(results_sorted[:5]):
        print(f"   {i+1}. ARMA({p},{q}): AIC={aic:.2f}, BIC={bic:.2f}")
    
    print(f"\n   True order: ARMA(2,2)")
    print(f"   Best by AIC: ARMA({results_sorted[0][0]},{results_sorted[0][1]})")
    
    # ============================================================================
    # Example 4: ACF/PACF Patterns for ARMA
    # ============================================================================
    print("\n" + "="*70)
    print("Example 4: Understanding ACF/PACF for ARMA")
    print("="*70)
    
    print("\n1. Generating different model types...")
    
    # Pure AR(2)
    ar2_data = generate_arma_data(n=200, ar_order=2, ma_order=0, 
                                 phi=[0.6, -0.3], theta=[], mu=0.0, seed=10)
    
    # Pure MA(2)
    ma2_data = generate_arma_data(n=200, ar_order=0, ma_order=2,
                                 phi=[], theta=[0.6, -0.3], mu=0.0, seed=20)
    
    # Mixed ARMA(2,2)
    arma22_data = generate_arma_data(n=200, ar_order=2, ma_order=2,
                                    phi=[0.6, -0.3], theta=[0.6, -0.3], mu=0.0, seed=30)
    
    # Fit and analyze patterns
    print("\n2. Analyzing ACF/PACF patterns...")
    
    models_list = [
        ("AR(2)", ARMAModel(order=(2, 0), validation=False), ar2_data),
        ("MA(2)", ARMAModel(order=(0, 2), validation=False), ma2_data),
        ("ARMA(2,2)", ARMAModel(order=(2, 2), validation=False), arma22_data)
    ]
    
    for name, model, data in models_list:
        model.fit(data)
        analysis = model.get_exploratory_analysis()
        print(f"\n   {name}:")
        if analysis['acf_pacf']:
            acf_vals = np.array(analysis['acf_pacf']['acf_values'])
            pacf_vals = np.array(analysis['acf_pacf']['pacf_values'])
            print(f"     ACF pattern: {'cutoff' if np.sum(np.abs(acf_vals[3:]) > 0.1) < 2 else 'gradual decay'}")
            print(f"     PACF pattern: {'cutoff' if np.sum(np.abs(pacf_vals[3:]) > 0.1) < 2 else 'gradual decay'}")
    
    print("\n   Key insight: ARMA shows gradual decay in both ACF and PACF")
    
    # ============================================================================
    # Example 5: ARMA for Economic/Financial Data
    # ============================================================================
    print("\n" + "="*70)
    print("Example 5: ARMA for Modeling Returns (Financial Application)")
    print("="*70)
    
    print("\n1. Simulating financial returns with ARMA structure...")
    # Returns often show ARMA patterns
    returns = generate_arma_data(n=500, ar_order=1, ma_order=1,
                                phi=[0.2], theta=[-0.3], mu=0.0005, sigma=0.02, seed=99)
    
    print(f"   Simulated {len(returns)} daily returns")
    print(f"   Mean return: {np.mean(returns):.6f}")
    print(f"   Volatility: {np.std(returns):.6f}")
    print(f"   Min: {np.min(returns):.6f}, Max: {np.max(returns):.6f}")
    
    # Fit ARMA model
    print("\n2. Fitting ARMA model to returns...")
    returns_model = ARMAModel(auto_select=True, max_ar=3, max_ma=3, validation=False)
    returns_model.fit(returns)
    
    print(f"   Identified model: ARMA({returns_model.order[0]},{returns_model.order[1]})")
    
    # Forecast next period return
    next_return = returns_model.predict(steps=1)
    print(f"\n3. Next period forecast: {next_return[0]:.6f}")
    print(f"   This can inform trading decisions")
    
    # Rolling forecast evaluation
    print("\n4. Rolling forecast evaluation...")
    train_size = 400
    test_size = 100
    
    predictions = []
    actuals = []
    
    for t in range(train_size, train_size + test_size):
        # Fit on expanding window
        train_data = returns[:t]
        test_point = returns[t]
        
        # Fit and predict
        temp_model = ARMAModel(order=returns_model.order, validation=False)
        temp_model.fit(train_data)
        pred = temp_model.predict(steps=1)[0]
        
        predictions.append(pred)
        actuals.append(test_point)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    mae = np.mean(np.abs(actuals - predictions))
    
    print(f"   Rolling forecast RMSE: {rmse:.6f}")
    print(f"   Rolling forecast MAE: {mae:.6f}")
    print(f"   Directional accuracy: {np.mean(np.sign(predictions) == np.sign(actuals)):.2%}")
    
    # Plot rolling forecasts
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(actuals)), actuals, label='Actual Returns', alpha=0.7)
    plt.plot(range(len(predictions)), predictions, label='Predicted Returns', alpha=0.7)
    plt.title('ARMA Model: Rolling Forecast of Returns')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    
    # Key takeaways
    print("\nKey Takeaways:")
    print("• ARMA combines AR (persistence) and MA (shocks) components")
    print("• Both ACF and PACF decay gradually for ARMA models")
    print("• ARMA(1,1) is often sufficient for many real-world applications")
    print("• More parsimonious than pure AR or MA for complex patterns")
    print("• Grid search with AIC/BIC is used for order selection")
    print("• Widely used in economics and finance")
    print("• Requires stationary data (like AR and MA)")


if __name__ == "__main__":
    main()

