"""
PySpark Parallel ARIMA Example

Demonstrates how to use TSLib with PySpark for distributed time series analysis.
This example shows how to process multiple time series in parallel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslib.spark.parallel_arima import ParallelARIMAProcessor
from tslib.spark.utils import create_spark_session, prepare_time_series_dataframe
from tslib.utils.checks import check_spark_availability


def generate_multiple_time_series(n_series=5, n_obs=100, seed=42):
    """
    Generate multiple time series for parallel processing
    
    Parameters:
    -----------
    n_series : int
        Number of time series to generate
    n_obs : int
        Number of observations per series
    seed : int
        Random seed
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with multiple time series
    """
    np.random.seed(seed)
    
    data = []
    for i in range(n_series):
        # Generate different types of time series
        if i == 0:
            # Random walk
            series = np.cumsum(np.random.normal(0, 1, n_obs))
        elif i == 1:
            # Trend + noise
            t = np.arange(n_obs)
            series = 0.1 * t + np.random.normal(0, 1, n_obs)
        elif i == 2:
            # AR(1) process
            phi = 0.7
            epsilon = np.random.normal(0, 1, n_obs)
            series = np.zeros(n_obs)
            series[0] = epsilon[0]
            for t in range(1, n_obs):
                series[t] = phi * series[t-1] + epsilon[t]
        elif i == 3:
            # MA(1) process
            theta = 0.5
            epsilon = np.random.normal(0, 1, n_obs)
            series = np.zeros(n_obs)
            for t in range(n_obs):
                if t == 0:
                    series[t] = epsilon[t]
                else:
                    series[t] = epsilon[t] + theta * epsilon[t-1]
        else:
            # Seasonal pattern
            t = np.arange(n_obs)
            seasonal = 2 * np.sin(2 * np.pi * t / 12)
            trend = 0.05 * t
            noise = np.random.normal(0, 1, n_obs)
            series = trend + seasonal + noise
        
        # Create DataFrame for this series
        series_df = pd.DataFrame({
            'series_id': f'series_{i}',
            'timestamp': range(n_obs),
            'value': series
        })
        data.append(series_df)
    
    return pd.concat(data, ignore_index=True)


def main():
    """Main example function"""
    print("TSLib - PySpark Parallel ARIMA Example")
    print("=" * 60)
    
    # Check if PySpark is available
    if not check_spark_availability():
        print("PySpark not available. Please install with: pip install pyspark")
        print("This example requires PySpark for distributed processing.")
        return
    
    try:
        # Create Spark session
        print("1. Creating Spark session...")
        spark = create_spark_session(
            app_name="ParallelARIMAExample",
            master="local[*]"
        )
        print("   Spark session created successfully")

        # Generate sample data
        print("\n2. Generating multiple time series data...")
        df_pandas = generate_multiple_time_series(n_series=5, n_obs=100)
        print(f"   Generated {len(df_pandas)} observations across {df_pandas['series_id'].nunique()} series")
        
        # Convert to Spark DataFrame
        df_spark = spark.createDataFrame(df_pandas)
        print("   Converted to Spark DataFrame")
        
        # Show data schema
        print("\n3. Data Schema:")
        df_spark.printSchema()
        
        # Show sample data
        print("\n4. Sample Data:")
        df_spark.show(10)
        
        # Initialize parallel ARIMA processor
        print("\n5. Initializing parallel ARIMA processor...")
        processor = ParallelARIMAProcessor(spark_session=spark)
        
        # Fit ARIMA models for all series
        print("\n6. Fitting ARIMA models in parallel...")
        results_df = processor.fit_multiple_arima(
            df=df_spark,
            group_column='series_id',
            value_column='value',
            time_column='timestamp',
            order=(1, 1, 1),
            auto_select=False
        )
        
        # Show fitting results
        print("\n7. Model Fitting Results:")
        results_df.show()
        
        # Convert results to pandas for analysis
        results_pandas = results_df.toPandas()
        
        # Analyze results
        print("\n8. Results Analysis:")
        successful_fits = results_pandas[results_pandas['success'] == 'True']
        failed_fits = results_pandas[results_pandas['success'] == 'False']
        
        print(f"   Successful fits: {len(successful_fits)}")
        print(f"   Failed fits: {len(failed_fits)}")
        
        if len(successful_fits) > 0:
            print(f"   Average AIC: {successful_fits['aic'].mean():.4f}")
            print(f"   Average BIC: {successful_fits['bic'].mean():.4f}")
            
            # Show best model by AIC
            best_model = successful_fits.loc[successful_fits['aic'].idxmin()]
            print(f"   Best model: {best_model['group_id']} with AIC = {best_model['aic']:.4f}")
        
        # Generate predictions for all series
        print("\n9. Generating predictions in parallel...")
        predictions_df = processor.predict_multiple_arima(
            df=df_spark,
            group_column='series_id',
            value_column='value',
            time_column='timestamp',
            order=(1, 1, 1),
            steps=10,
            return_conf_int=True
        )
        
        # Show prediction results
        print("\n10. Prediction Results:")
        predictions_df.show()
        
        # Convert predictions to pandas for visualization
        predictions_pandas = predictions_df.toPandas()
        
        # Create visualization
        print("\n11. Creating visualization...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Parallel ARIMA Results', fontsize=16)
        
        # Plot each series
        for i, series_id in enumerate(df_pandas['series_id'].unique()):
            if i >= 6:  # Limit to 6 plots
                break
            
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Get historical data
            series_data = df_pandas[df_pandas['series_id'] == series_id]
            ax.plot(series_data['timestamp'], series_data['value'], 
                   label='Historical', color='blue', linewidth=1.5)
            
            # Get predictions
            pred_data = predictions_pandas[predictions_pandas['group_id'] == series_id]
            if len(pred_data) > 0:
                pred_row = pred_data.iloc[0]
                predictions = pred_row['predictions']
                lower_bound = pred_row['lower_bound']
                upper_bound = pred_row['upper_bound']
                
                # Plot forecast
                forecast_index = range(len(series_data), len(series_data) + len(predictions))
                ax.plot(forecast_index, predictions, label='Forecast', color='red', linewidth=2)
                
                # Plot confidence intervals
                ax.fill_between(forecast_index, lower_bound, upper_bound, 
                               alpha=0.3, color='red', label='95% CI')
            
            ax.set_title(f'{series_id}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(df_pandas['series_id'].unique()), 6):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Performance metrics
        print("\n12. Performance Summary:")
        print(f"   Total series processed: {len(df_pandas['series_id'].unique())}")
        print(f"   Successful model fits: {len(successful_fits)}")
        print(f"   Failed model fits: {len(failed_fits)}")
        print(f"   Success rate: {len(successful_fits) / len(df_pandas['series_id'].unique()) * 100:.1f}%")
        
        if len(successful_fits) > 0:
            print(f"   Average AIC: {successful_fits['aic'].mean():.4f}")
            print(f"   AIC range: {successful_fits['aic'].min():.4f} - {successful_fits['aic'].max():.4f}")
        
        # Clean up
        print("\n13. Cleaning up...")
        processor.close()
        spark.stop()
        print("   Spark session closed")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure PySpark is properly installed and configured.")
    
    print("\n" + "=" * 60)
    print("Example completed!")


if __name__ == "__main__":
    main()




