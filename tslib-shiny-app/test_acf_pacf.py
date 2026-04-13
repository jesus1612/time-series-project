#!/usr/bin/env python3
"""
Test script to debug ACF/PACF calculation issues
"""
import numpy as np
import pandas as pd
import sys

# Test with dummy data
print("Testing ACF/PACF calculation...")
print(f"NumPy version: {np.__version__}")

# Create test data
test_data = np.array([100.5, 102.3, 104.05, 105.8, 107.2, 108.65, 110.1, 112.5, 115.3, 117.0,
                      118.7, 120.2, 122.8, 124.1, 125.4, 127.9, 130.1, 131.8, 133.5, 135.8])

print(f"\nTest data length: {len(test_data)}")
print(f"Test data sample: {test_data[:5]}")

try:
    from tslib.core.acf_pacf import ACFCalculator, PACFCalculator
    
    print("\n=== Testing ACFCalculator ===")
    acf_calc = ACFCalculator()
    print(f"ACFCalculator type: {type(acf_calc)}")
    print(f"ACFCalculator methods: {[m for m in dir(acf_calc) if not m.startswith('_')]}")
    
    # Try different ways to call calculate
    print("\n--- Method 1: calculate(data) ---")
    try:
        acf1 = acf_calc.calculate(test_data)
        print(f"Result type: {type(acf1)}")
        print(f"Result value: {acf1}")
        if hasattr(acf1, '__len__'):
            print(f"Result length: {len(acf1)}")
            if len(acf1) > 0:
                print(f"First 5 values: {acf1[:5] if hasattr(acf1, '__getitem__') else 'N/A'}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Method 2: calculate(data, nlags=10) ---")
    try:
        acf2 = acf_calc.calculate(test_data, nlags=10)
        print(f"Result type: {type(acf2)}")
        print(f"Result value: {acf2}")
        if hasattr(acf2, '__len__'):
            print(f"Result length: {len(acf2)}")
            if len(acf2) > 0:
                print(f"First 5 values: {acf2[:5] if hasattr(acf2, '__getitem__') else 'N/A'}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing PACFCalculator ===")
    pacf_calc = PACFCalculator()
    print(f"PACFCalculator type: {type(pacf_calc)}")
    print(f"PACFCalculator methods: {[m for m in dir(pacf_calc) if not m.startswith('_')]}")
    
    print("\n--- Method 1: calculate(data) ---")
    try:
        pacf1 = pacf_calc.calculate(test_data)
        print(f"Result type: {type(pacf1)}")
        print(f"Result value: {pacf1}")
        if hasattr(pacf1, '__len__'):
            print(f"Result length: {len(pacf1)}")
            if len(pacf1) > 0:
                print(f"First 5 values: {pacf1[:5] if hasattr(pacf1, '__getitem__') else 'N/A'}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Method 2: calculate(data, nlags=10) ---")
    try:
        pacf2 = pacf_calc.calculate(test_data, nlags=10)
        print(f"Result type: {type(pacf2)}")
        print(f"Result value: {pacf2}")
        if hasattr(pacf2, '__len__'):
            print(f"Result length: {len(pacf2)}")
            if len(pacf2) > 0:
                print(f"First 5 values: {pacf2[:5] if hasattr(pacf2, '__getitem__') else 'N/A'}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test with larger dataset (like the CSV)
    print("\n=== Testing with larger dataset (180 points) ===")
    large_data = np.linspace(100, 426, 180)
    print(f"Large data length: {len(large_data)}")
    
    try:
        acf_large = acf_calc.calculate(large_data)
        print(f"ACF result type: {type(acf_large)}")
        if hasattr(acf_large, '__len__'):
            print(f"ACF result length: {len(acf_large)}")
        else:
            print(f"ACF result: {acf_large}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    try:
        pacf_large = pacf_calc.calculate(large_data)
        print(f"PACF result type: {type(pacf_large)}")
        if hasattr(pacf_large, '__len__'):
            print(f"PACF result length: {len(pacf_large)}")
        else:
            print(f"PACF result: {pacf_large}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"ERROR: Could not import ACFCalculator/PACFCalculator: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Test completed ===")

