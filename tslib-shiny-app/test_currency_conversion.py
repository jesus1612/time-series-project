#!/usr/bin/env python
# Test currency conversion

import pandas as pd
from services.tslib_service import TSLibService

def test_currency_conversion():
    """Test that currency format columns are detected and converted"""
    print("=" * 60)
    print("Testing Currency Format Detection and Conversion")
    print("=" * 60)
    
    # Create sample data with currency format (like your data)
    data = {
        'Mes': ['01/04/2020', '01/05/2020', '01/06/2020', '01/07/2020', '01/08/2020'],
        'Cash In': ['$1,228,301,354.06', '$981,108,675.92', '$1,587,365,909.22', '$1,593,938,361.08', '$1,531,477,837.65'],
        'Cash Out': ['$3,928,297,963.57', '$1,917,797,050.54', '$1,962,988,215.81', '$3,569,340,520.08', '$4,605,225,475.67']
    }
    
    df = pd.DataFrame(data)
    
    print("\n📊 Sample Data:")
    print(df)
    print(f"\nData types:")
    print(df.dtypes)
    
    # Initialize service
    service = TSLibService()
    
    # Test 1: Detect numeric columns
    print("\n" + "-" * 60)
    print("Test 1: Detecting Numeric Columns")
    print("-" * 60)
    
    numeric_cols = service.get_numeric_columns(df)
    print(f"✓ Numeric columns detected: {numeric_cols}")
    
    assert 'Cash In' in numeric_cols, "Cash In should be detected as numeric"
    assert 'Cash Out' in numeric_cols, "Cash Out should be detected as numeric"
    assert 'Mes' not in numeric_cols, "Mes should NOT be detected as numeric"
    print("✓ All assertions passed!")
    
    # Test 2: Convert to numeric
    print("\n" + "-" * 60)
    print("Test 2: Converting Currency to Numeric")
    print("-" * 60)
    
    cash_in_numeric = service.convert_to_numeric(df, 'Cash In')
    print(f"✓ Converted Cash In:")
    print(cash_in_numeric)
    print(f"\n  Original type: {df['Cash In'].dtype}")
    print(f"  Converted type: {cash_in_numeric.dtype}")
    print(f"  First value: {cash_in_numeric.iloc[0]:,.2f}")
    
    assert pd.api.types.is_numeric_dtype(cash_in_numeric), "Should be numeric type"
    assert cash_in_numeric.iloc[0] == 1228301354.06, "First value should match"
    print("✓ Conversion successful!")
    
    # Test 3: Calculate statistics
    print("\n" + "-" * 60)
    print("Test 3: Calculating Statistics")
    print("-" * 60)
    
    stats = service.calculate_basic_stats(cash_in_numeric.values)
    print(f"✓ Statistics calculated:")
    print(f"  Mean: ${stats['mean']:,.2f}")
    print(f"  Std: ${stats['std']:,.2f}")
    print(f"  Min: ${stats['min']:,.2f}")
    print(f"  Max: ${stats['max']:,.2f}")
    
    # Test 4: Validate data
    print("\n" + "-" * 60)
    print("Test 4: Validating Currency Data")
    print("-" * 60)
    
    validation = service.validate_data(df, 'Cash In')
    print(f"✓ Validation result: {'Valid' if validation['valid'] else 'Invalid'}")
    print(f"  Messages: {validation['messages']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    print("\n" + "=" * 60)
    print("✅ All currency conversion tests passed!")
    print("=" * 60)
    print("\n💡 Your data with currency format ($) will now be detected correctly!")

if __name__ == "__main__":
    test_currency_conversion()

