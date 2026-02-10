import pandas as pd
import numpy as np
import os
os.environ['PYTENSOR_FLAGS'] = 'cxx='
import pymc as pm
import arviz as az
import json

def run_analysis():
    # Paths
    base_path = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data'
    data_path = os.path.join(base_path, 'data', 'BrentOilPrices.csv')
    output_path = os.path.join(base_path, 'data', 'model_results.json')
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Focus on a smaller period for faster demo: 2020 (COVID-19 impact)
    df_zoom = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2020-12-31')].copy()
    
    prices = df_zoom['Price'].values
    dates = df_zoom['Date'].dt.strftime('%Y-%m-%d').values
    n_days = len(prices)
    days = np.arange(n_days)
    
    print(f"Running model on {n_days} days of data...")
    
    # 2. PyMC Model
    with pm.Model() as model:
        tau = pm.DiscreteUniform('tau', lower=0, upper=n_days - 1)
        mu_1 = pm.Normal('mu_1', mu=prices.mean(), sigma=prices.std())
        mu_2 = pm.Normal('mu_2', mu=prices.mean(), sigma=prices.std())
        sigma = pm.HalfNormal('sigma', sigma=prices.std())
        
        mu = pm.math.switch(tau > days, mu_1, mu_2)
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=prices)
        
        # Using nuts_sampler='nutpie' or 'blackjax' often avoids C++ issues if they're installed,
        # but here we'll just try to ensure it runs with minimal cores if needed.
        trace = pm.sample(1000, tune=500, chains=2, cores=1, return_inferencedata=True, random_seed=42)
    
    # 3. Extract Results
    summary = az.summary(trace, var_names=['mu_1', 'mu_2', 'tau']).to_dict()
    
    tau_samples = trace.posterior['tau'].values.flatten()
    tau_mode = int(pd.Series(tau_samples).mode()[0])
    change_date = dates[tau_mode]
    
    mu1_mean = float(trace.posterior['mu_1'].mean())
    mu2_mean = float(trace.posterior['mu_2'].mean())
    
    # Quantify impact
    impact_pct = (mu2_mean - mu1_mean) / mu1_mean * 100
    
    results = {
        "change_point_index": float(tau_mode),
        "change_point_date": change_date,
        "mu_1_mean": mu1_mean,
        "mu_2_mean": mu2_mean,
        "impact_percentage": impact_pct,
        "summary": summary,
        "dates": dates.tolist(),
        "prices": prices.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Model run complete. Change point detected at {change_date}.")
    print(f"Mean Price shifted from {mu1_mean:.2f} to {mu2_mean:.2f} ({impact_pct:.2f}%)")

if __name__ == "__main__":
    run_analysis()
