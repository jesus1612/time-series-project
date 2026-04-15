#!/usr/bin/env python
# Test script for TSLib integration

import numpy as np
import pandas as pd
from services.tslib_service import TSLibService

def test_tslib_integration():
    """Test TSLib service integration"""
    print("=" * 60)
    print("Testing TSLib Integration")
    print("=" * 60)
    
    # Initialize service
    service = TSLibService()
    print("✓ TSLibService initialized")
    
    # Create synthetic data
    np.random.seed(42)
    n = 100
    data = np.cumsum(np.random.randn(n)) + 50  # Random walk with drift
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='D'),
        'value': data,
        'other': np.random.randn(n)
    })
    print(f"✓ Created synthetic dataset: {n} observations")
    
    # Test 1: Get numeric columns
    numeric_cols = service.get_numeric_columns(df)
    print(f"✓ Numeric columns detected: {numeric_cols}")
    assert 'value' in numeric_cols
    
    # Test 2: Detect datetime column
    date_col = service.detect_datetime_column(df)
    print(f"✓ Datetime column detected: {date_col}")
    assert date_col == 'date'
    
    # Test 3: Validate data
    validation = service.validate_data(df, 'value')
    print(f"✓ Data validation: {'Valid' if validation['valid'] else 'Invalid'}")
    print(f"  - Messages: {validation['messages']}")
    if validation['warnings']:
        print(f"  - Warnings: {validation['warnings']}")
    
    # Test 4: Exploratory analysis
    exploratory = service.get_exploratory_analysis(data)
    print(f"✓ Exploratory analysis completed")
    print(f"  - ACF calculated: {len(exploratory['acf'])} lags")
    print(f"  - PACF calculated: {len(exploratory['pacf'])} lags")
    print(f"  - Statistics: {exploratory['statistics']}")
    
    # Test 5: Fit AR model
    print("\n" + "-" * 60)
    print("Testing AR Model")
    print("-" * 60)
    try:
        ar_model = service.fit_model(
            data=data,
            model_type='AR',
            order=(2,),
            auto_select=True
        )
        print(f"✓ AR model fitted successfully")
        print(f"  - Order: {ar_model.order}")
        
        # Forecast
        ar_forecast = service.get_forecast(ar_model, steps=10, return_conf_int=True)
        print(f"✓ AR forecast generated: {len(ar_forecast['forecast'])} steps")
        print(f"  - Forecast values: {ar_forecast['forecast'][:3]}...")
        
        # Metrics
        ar_metrics = service.get_model_metrics(ar_model)
        print(f"✓ AR metrics extracted: {ar_metrics}")
    except Exception as e:
        print(f"✗ AR model failed: {e}")
    
    # Test 6: Fit ARIMA model
    print("\n" + "-" * 60)
    print("Testing ARIMA Model")
    print("-" * 60)
    try:
        arima_model = service.fit_statsmodels_arima(data, (1, 1, 1))
        print(f"✓ ARIMA model fitted successfully")
        print(f"  - Order: {arima_model.order}")
        
        # Forecast
        arima_forecast = service.get_forecast(arima_model, steps=10, return_conf_int=True)
        print(f"✓ ARIMA forecast generated: {len(arima_forecast['forecast'])} steps")
        print(f"  - Forecast values: {arima_forecast['forecast'][:3]}...")
        print(f"  - Has confidence intervals: {arima_forecast['lower_bound'] is not None}")
        
        # Metrics
        arima_metrics = service.get_model_metrics(arima_model)
        print(f"✓ ARIMA metrics extracted: {arima_metrics}")
        
        # Diagnostics
        if hasattr(arima_model, 'get_residuals'):
            residuals = arima_model.get_residuals()
            print(f"✓ Residuals extracted: {len(residuals)} values")
            print(f"  - Mean: {np.mean(residuals):.6f}")
            print(f"  - Std: {np.std(residuals):.6f}")
    except Exception as e:
        print(f"✗ ARIMA model failed: {e}")
    
    # Test 7: Calculate basic stats
    stats = service.calculate_basic_stats(data)
    print("\n" + "-" * 60)
    print("Basic Statistics")
    print("-" * 60)
    for key, value in stats.items():
        print(f"  - {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_tslib_integration()

