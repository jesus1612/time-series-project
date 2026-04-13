#!/usr/bin/env python3
"""
Script to generate dummy time series data with missing values
for testing the TSLib Shiny app
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_data(
    start_date='2023-01-01',
    n_days=180,
    missing_rate=0.1,
    trend=2.0,
    noise_std=1.5,
    output_file='dummy_with_missing.csv'
):
    """
    Generate dummy time series data with missing values
    
    Args:
        start_date: Start date for the time series
        n_days: Number of days to generate
        missing_rate: Proportion of values to set as missing (0.0 to 1.0)
        trend: Daily trend value
        noise_std: Standard deviation of noise
        output_file: Output CSV filename
    """
    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate values with trend and noise
    base_value = 100.0
    values = []
    
    for i in range(n_days):
        # Linear trend + some noise
        value = base_value + (trend * i) + np.random.normal(0, noise_std)
        values.append(value)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    # Introduce missing values randomly
    n_missing = int(n_days * missing_rate)
    missing_indices = np.random.choice(n_days, size=n_missing, replace=False)
    df.loc[missing_indices, 'value'] = np.nan
    
    # Save to CSV (empty strings for NaN values)
    df.to_csv(output_file, index=False, na_rep='')
    
    print(f"Generated {n_days} days of data with {n_missing} missing values")
    print(f"Saved to: {output_file}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nMissing values at indices: {sorted(missing_indices)}")
    
    return df

if __name__ == '__main__':
    # Generate dummy data
    df = generate_dummy_data(
        start_date='2023-01-01',
        n_days=180,
        missing_rate=0.1,  # 10% missing values
        trend=2.0,  # Daily increase
        noise_std=1.5,
        output_file='dummy_with_missing.csv'
    )

