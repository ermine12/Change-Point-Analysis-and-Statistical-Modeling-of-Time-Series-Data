import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import os

# Set paths
data_path = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data\data\BrentOilPrices.csv'
output_dir = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data\reports'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Load Data
df = pd.read_csv(data_path)
# Inconsistent date formats found (DD-Mon-YY and "Mon DD, YYYY")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# 2. Daily Log Returns
df['Log_Price'] = np.log(df['Price'])
df['Log_Returns'] = df['Log_Price'].diff().dropna()

# 3. Stationarity Testing (ADF)
def run_adf(series, name):
    print(f"\n--- ADF Test for {name} ---")
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
    return result

adf_price = run_adf(df['Price'], "Price Level")
adf_returns = run_adf(df['Log_Returns'], "Log Returns")

# 4. Trend Analysis
# Simple moving average to visualize trend
df['SMA_250'] = df['Price'].rolling(window=250).mean()

# 5. Volatility Patterns
# Rolling volatility (21-day)
df['Volatility_21'] = df['Log_Returns'].rolling(window=21).std() * np.sqrt(252) # Annualized

# 6. Plotting
plt.figure(figsize=(15, 10))

# Price and Trend
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Price'], label='Brent Price', alpha=0.7)
plt.plot(df.index, df['SMA_250'], label='250-day SMA', color='red')
plt.title('Brent Oil Price and 250-day SMA')
plt.legend()

# Log Returns
plt.subplot(3, 1, 2)
plt.plot(df.index, df['Log_Returns'], label='Log Returns', color='green', alpha=0.5)
plt.title('Daily Log Returns')
plt.legend()

# Volatility
plt.subplot(3, 1, 3)
plt.plot(df.index, df['Volatility_21'], label='21-day Rolling Volatility (Annualized)', color='orange')
plt.title('Volatility Patterns')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_series_properties.png'))
# plt.show() # Can't show in terminal, so saving.

# Save results to a text file for Task 1 report
with open(os.path.join(output_dir, 'analysis_results.txt'), 'w') as f:
    f.write("Time Series Analysis Results\n")
    f.write("============================\n\n")
    f.write(f"Data Range: {df.index.min()} to {df.index.max()}\n\n")
    f.write("ADF Test Results:\n")
    f.write(f"Price Level p-value: {adf_price[1]}\n")
    f.write(f"Log Returns p-value: {adf_returns[1]}\n\n")
    f.write("Observations:\n")
    if adf_price[1] > 0.05:
        f.write("- Price level is non-stationary.\n")
    else:
        f.write("- Price level appears stationary.\n")
    
    if adf_returns[1] < 0.05:
        f.write("- Log returns are stationary.\n")
    else:
        f.write("- Log returns are non-stationary.\n")

print("\nAnalysis complete. Results saved to reports directory.")
