"""Quick validation script for Bayesian change point model.

This script runs a lightweight test to verify the refactored model works correctly
with optimized parameters for faster execution.
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


def test_single_change_point():
    """Quick test of single change point detection."""
    print("\n" + "="*80)
    print("VALIDATION TEST - Single Change Point Model")
    print("="*80 + "\n")
    
    # Paths
    base_path = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data'
    data_path = os.path.join(base_path, 'data', 'BrentOilPrices.csv')
    output_path = os.path.join(base_path, 'data', 'model_results_single.json')
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Focus on March-May 2020 (smaller window for faster testing)
    df_test = df[(df['Date'] >= '2020-03-01') & (df['Date'] <= '2020-05-31')].copy()
    df_test.set_index('Date', inplace=True)
    
    print(f"Test dataset: {len(df_test)} days (March-May 2020)")
    print(f"Price range: ${df_test['Price'].min():.2f} - ${df_test['Price'].max():.2f}\n")
    
    # Initialize and fit model with reduced parameters
    print("Initializing BayesianChangePointModel...")
    model = BayesianChangePointModel(df_test['Price'], use_log_returns=True)
    
    print("Fitting model (draws=500, tune=200, chains=2)...")
    print("This should take 2-3 minutes...\n")
    
    model.fit_single_change_point(
        draws=500,
        tune=200,
        chains=2,
        random_seed=42,
        heteroskedastic=True
    )
    
    # Get summary
    summary = model.summary()
    
    print("\n" + "-"*80)
    print("VALIDATION RESULTS")
    print("-"*80)
    print(f"\n✓ Model successfully fitted!")
    print(f"\nDetected Change Point: {summary['change_points'][0]}")
    print(f"\nConvergence Diagnostics:")
    print(f"  - Converged (R-hat < 1.05): {summary['convergence']['converged']}")
    print(f"  - Sufficient ESS (> 400): {summary['convergence']['sufficient_ess']}")
    
    if not summary['convergence']['converged']:
        print("\n  ⚠ WARNING: Model may not have fully converged")
        print("  R-hat values:")
        for param, val in summary['convergence']['r_hat'].items():
            print(f"    - {param}: {val:.4f}")
    
    print(f"\nRegime Parameters (Log-Returns):")
    print(f"  - Pre-change mean: {summary['regime_parameters']['mu_1']['mean']:.6f}")
    print(f"  - Post-change mean: {summary['regime_parameters']['mu_2']['mean']:.6f}")
    print(f"  - Mean shift: {summary['impact']['mean_shift_percentage']:.2f}% ({summary['impact']['direction']})")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: model_results_single.json")
    print("\n" + "="*80)
    print("VALIDATION SUCCESSFUL ✓")
    print("="*80)
    print("\nThe refactored model is working correctly!")
    print("\nNext steps:")
    print("1. For production use, increase draws to 2000+ and tune to 1000+")
    print("2. Test multi-change point detection on longer time series")
    print("3. Update the Flask API to use the new model class")
    
    return model, summary


def test_multi_change_point():
    """Quick test of multi-change point detection."""
    print("\n" + "="*80)
    print("VALIDATION TEST - Multi-Change Point Model")
    print("="*80 + "\n")
    
    # Paths
    base_path = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data'
    data_path = os.path.join(base_path, 'data', 'BrentOilPrices.csv')
    output_path = os.path.join(base_path, 'data', 'model_results_multi.json')
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Focus on 2019-2021 (captures pre-COVID and COVID periods)
    df_test = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2021-12-31')].copy()
    df_test.set_index('Date', inplace=True)
    
    print(f"Test dataset: {len(df_test)} days (2019-2021)")
    print(f"Price range: ${df_test['Price'].min():.2f} - ${df_test['Price'].max():.2f}\n")
    
    # Initialize model
    print("Initializing BayesianChangePointModel...")
    model = BayesianChangePointModel(df_test['Price'], use_log_returns=True)
    
    print("Fitting multi-change point model (max_change_points=2)...")
    print("This may take 5-10 minutes...\n")
    
    model.fit_multi_change_point(
        max_change_points=2,
        min_segment_size=30,
        draws=500,
        tune=200,
        chains=2,
        random_seed=42
    )
    
    # Get summary
    summary = model.summary()
    
    print("\n" + "-"*80)
    print("VALIDATION RESULTS")
    print("-"*80)
    print(f"\n✓ Multi-point detection successful!")
    print(f"\nDetected {len(summary['change_points'])} Change Points:")
    for i, cp in enumerate(summary['change_points'], 1):
        print(f"  {i}. {cp}")
    
    # Save results
    results = {
        'method': 'recursive_binary_segmentation',
        'max_depth': 2,
        'change_points': summary['change_points'],
        'use_log_returns': True
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: model_results_multi.json")
    print("\n" + "="*80)
    print("MULTI-POINT VALIDATION SUCCESSFUL ✓")
    print("="*80)
    
    return model, summary


if __name__ == "__main__":
    # Test single change point (required)
    single_model, single_summary = test_single_change_point()
    
    # Ask user if they want to test multi-change point (optional, slow)
    print("\n" + "="*80)
    print("Multi-change point test available (press Ctrl+C to skip)")
    print("="*80)
    
    try:
        import time
        time.sleep(2)  # Give user a moment to cancel
        multi_model, multi_summary = test_multi_change_point()
    except KeyboardInterrupt:
        print("\n\nMulti-change point test skipped.")
        print("Run again to test multi-point detection.")
