"""Demonstration script for Bayesian change point detection.

This script demonstrates both single and multi-change point detection
on Brent oil price data using the refactored BayesianChangePointModel.
"""
import pandas as pd
import numpy as np
import os
import json
import sys

# Set PyTensor configuration before importing PyMC
os.environ['PYTENSOR_FLAGS'] = 'cxx='

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from change_point.bayesian_model import BayesianChangePointModel


def run_single_change_point_analysis():
    """Run single change point analysis on 2020 COVID-19 period."""
    print("\n" + "="*80)
    print("SINGLE CHANGE POINT ANALYSIS - 2020 COVID-19 Impact")
    print("="*80 + "\n")
    
    # Paths
    base_path = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data'
    data_path = os.path.join(base_path, 'data', 'BrentOilPrices.csv')
    output_path = os.path.join(base_path, 'data', 'model_results_single.json')
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Focus on 2020 for fast demonstration
    df_2020 = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2020-12-31')].copy()
    df_2020.set_index('Date', inplace=True)
    
    print(f"Loaded {len(df_2020)} days of data from 2020")
    print(f"Price range: ${df_2020['Price'].min():.2f} - ${df_2020['Price'].max():.2f}\n")
    
    # Initialize and fit model
    model = BayesianChangePointModel(df_2020['Price'], use_log_returns=True)
    print("Fitting single change point model with log-returns...")
    model.fit_single_change_point(draws=1000, tune=500, chains=2, random_seed=42, heteroskedastic=True)
    
    # Get summary
    summary = model.summary()
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    print(f"\nDetected Change Point: {summary['change_points'][0]}")
    print(f"\nConvergence Status:")
    print(f"  - Converged (R-hat < 1.05): {summary['convergence']['converged']}")
    print(f"  - Sufficient ESS (> 400): {summary['convergence']['sufficient_ess']}")
    print(f"\nR-hat values:")
    for param, val in summary['convergence']['r_hat'].items():
        print(f"  - {param}: {val:.4f}")
    print(f"\nRegime Parameters (Log-Returns):")
    print(f"  - Pre-change mean: {summary['regime_parameters']['mu_1']['mean']:.6f}")
    print(f"  - Post-change mean: {summary['regime_parameters']['mu_2']['mean']:.6f}")
    print(f"  - Impact: {summary['impact']['mean_shift_percentage']:.2f}% ({summary['impact']['direction']})")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*80 + "\n")
    
    return model, summary


def run_multi_change_point_analysis():
    """Run multi-change point analysis on a longer period (2007-2022)."""
    print("\n" + "="*80)
    print("MULTI-CHANGE POINT ANALYSIS - 2007-2022 (Major Events)")
    print("="*80 + "\n")
    
    # Paths
    base_path = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data'
    data_path = os.path.join(base_path, 'data', 'BrentOilPrices.csv')
    output_path = os.path.join(base_path, 'data', 'model_results_multi.json')
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Focus on 2007-2022 to capture: Financial Crisis (2008), Shale Boom (2014), COVID-19 (2020)
    df_period = df[(df['Date'] >= '2007-01-01') & (df['Date'] <= '2022-12-31')].copy()
    df_period.set_index('Date', inplace=True)
    
    print(f"Loaded {len(df_period)} days of data from 2007-2022")
    print(f"Price range: ${df_period['Price'].min():.2f} - ${df_period['Price'].max():.2f}\n")
    
    # Initialize and fit model
    model = BayesianChangePointModel(df_period['Price'], use_log_returns=True)
    print("Fitting multi-change point model with recursive binary segmentation...")
    print("This may take several minutes...\n")
    
    model.fit_multi_change_point(
        max_change_points=3,
        min_segment_size=30,
        draws=1000,
        tune=500,
        chains=2,
        random_seed=42
    )
    
    # Get summary
    summary = model.summary()
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    print(f"\nDetected {len(summary['change_points'])} Change Points:")
    for i, cp in enumerate(summary['change_points'], 1):
        print(f"  {i}. {cp}")
    
    # Save results
    results_extended = {
        'method': 'recursive_binary_segmentation',
        'max_depth': 3,
        'change_points': summary['change_points'],
        'use_log_returns': True
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_extended, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*80 + "\n")
    
    return model, summary


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("BRENT OIL PRICE - BAYESIAN CHANGE POINT DETECTION")
    print("Refactored Model with Log-Return Transformation")
    print("="*80)
    
    # Run single change point analysis (fast)
    single_model, single_summary = run_single_change_point_analysis()
    
    # Ask user if they want to run multi-change point (slow)
    print("\nWould you like to run multi-change point analysis?")
    print("WARNING: This will take 5-10 minutes to complete.")
    
    # For automated execution, we'll run it
    try:
        multi_model, multi_summary = run_multi_change_point_analysis()
    except KeyboardInterrupt:
        print("\nMulti-change point analysis skipped.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review results in data/model_results_single.json and data/model_results_multi.json")
    print("2. Start the Flask API: python src/app.py")
    print("3. Launch the dashboard: cd dashboard && npm run dev")
    

if __name__ == "__main__":
    main()
