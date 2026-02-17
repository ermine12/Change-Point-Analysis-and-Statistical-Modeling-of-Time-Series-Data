"""Time series property analysis script.

Analyzes stationarity, trends, and volatility of the observation data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import paths

def load_data() -> pd.DataFrame:
    """Load and preprocess the Brent oil price data.
    
    Returns:
        DataFrame with Date index, Price, Log_Price, and Log_Returns columns.
    """
    if not paths.brent_prices.exists():
        raise FileNotFoundError(f"Data file not found at {paths.brent_prices}")
        
    df = pd.read_csv(paths.brent_prices)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)
    
    # Calculate derived features
    df['Log_Price'] = np.log(df['Price'])
    df['Log_Returns'] = df['Log_Price'].diff()
    df['SMA_250'] = df['Price'].rolling(window=250).mean()
    df['Volatility_21'] = df['Log_Returns'].rolling(window=21).std() * np.sqrt(252) # Annualized
    
    return df

def run_adf(series: pd.Series, name: str) -> tuple:
    """Run Augmented Dickey-Fuller test.
    
    Args:
        series: Time series data
        name: Name of the series for display
        
    Returns:
        ADF test result tuple
    """
    print(f"\n--- ADF Test for {name} ---")
    clean_series = series.dropna()
    result = adfuller(clean_series)
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value:.4f}')
        
    return result

def plot_properties(df: pd.DataFrame, output_dir: str) -> None:
    """Create and save time series property plots.
    
    Args:
        df: DataFrame with price and return data
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    # Price and Trend
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Price'], label='Brent Price', alpha=0.7)
    plt.plot(df.index, df['SMA_250'], label='250-day SMA', color='red')
    plt.title('Brent Oil Price and 250-day SMA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log Returns
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['Log_Returns'], label='Log Returns', color='green', alpha=0.5)
    plt.title('Daily Log Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Volatility
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['Volatility_21'], label='21-day Rolling Volatility (Annualized)', color='orange')
    plt.title('Volatility Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'time_series_properties.png')
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")

def save_report(df: pd.DataFrame, adf_price: tuple, adf_returns: tuple, output_dir: str) -> None:
    """Save analysis report to text file.
    
    Args:
        df: Data DataFrame
        adf_price: ADF result for price
        adf_returns: ADF result for returns
        output_dir: Directory to save report
    """
    report_path = os.path.join(output_dir, 'analysis_results.txt')
    
    with open(report_path, 'w') as f:
        f.write("Time Series Analysis Results\n")
        f.write("============================\n\n")
        f.write(f"Data Range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Total Observations: {len(df)}\n\n")
        
        f.write("ADF Test Results:\n")
        f.write(f"Price Level p-value: {adf_price[1]:.4f}\n")
        f.write(f"Log Returns p-value: {adf_returns[1]:.4f}\n\n")
        
        f.write("Observations:\n")
        if adf_price[1] > 0.05:
            f.write("- Price level is non-stationary (fail to reject null hypothesis).\n")
        else:
            f.write("- Price level appears stationary.\n")
        
        if adf_returns[1] < 0.05:
            f.write("- Log returns are stationary (reject null hypothesis).\n")
        else:
            f.write("- Log returns are non-stationary.\n")

    print(f"Report saved to {report_path}")

def main():
    """Main execution function."""
    print("Starting Time Series Property Analysis...")
    paths.ensure_directories()
    
    # Load Data
    df = load_data()
    
    # Run Tests
    adf_price = run_adf(df['Price'], "Price Level")
    adf_returns = run_adf(df['Log_Returns'], "Log Returns")
    
    # Generate Outputs
    plot_properties(df, str(paths.reports))
    save_report(df, adf_price, adf_returns, str(paths.reports))
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
