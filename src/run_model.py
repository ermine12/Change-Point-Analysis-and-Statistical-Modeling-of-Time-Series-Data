"""Demonstration script for Bayesian change point detection.

This script demonstrates both single and multi-change point detection
on Brent oil price data using the refactored BayesianChangePointModel.
"""
import pandas as pd
import numpy as np
import os
import json
import sys
from typing import Tuple, Dict, Any, Optional

# Set PyTensor configuration before importing PyMC
os.environ['PYTENSOR_FLAGS'] = 'cxx='

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.change_point.bayesian_model import BayesianChangePointModel
from src.config import paths, params


def load_and_preprocess_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Load and preprocess the Brent oil price data.
    
    Args:
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with Date index and Price column
    """
    if not paths.brent_prices.exists():
        raise FileNotFoundError(f"Data file not found at {paths.brent_prices}")
        
    df = pd.read_csv(paths.brent_prices)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
        
    df.set_index('Date', inplace=True)
    return df.copy()


def save_results(summary: Dict[str, Any], output_path: str) -> None:
    """Save analysis results to JSON file.
    
    Args:
        summary: Dictionary of model results
        output_path: Path to save the JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {output_path}")


def run_single_change_point_analysis() -> Tuple[BayesianChangePointModel, Dict[str, Any]]:
    """Run single change point analysis on 2020 COVID-19 period.
    
    Returns:
        Tuple of (fitted model, summary dictionary)
    """
    print("\n" + "="*80)
    print("SINGLE CHANGE POINT ANALYSIS - 2020 COVID-19 Impact")
    print("="*80 + "\n")
    
    # Load 2020 data
    df_2020 = load_and_preprocess_data(start_date='2020-01-01', end_date='2020-12-31')
    
    print(f"Loaded {len(df_2020)} days of data from 2020")
    print(f"Price range: ${df_2020['Price'].min():.2f} - ${df_2020['Price'].max():.2f}\n")
    
    # Initialize and fit model
    model = BayesianChangePointModel(df_2020['Price'], use_log_returns=True)
    print("Fitting single change point model with log-returns...")
    model.fit_single_change_point(
        draws=params.draws, 
        tune=params.tune, 
        chains=params.chains, 
        random_seed=params.random_seed, 
        heteroskedastic=True
    )
    
    # Get summary
    summary = model.summary()
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    
    cp_info = summary['change_points'][0]
    print(f"\nDetected Change Point: {cp_info['date']}")
    if cp_info['credible_interval']:
        print(f"95% Credible Interval: {cp_info['credible_interval'][0]} to {cp_info['credible_interval'][1]}")
        
    print(f"\nConvergence Status:")
    print(f"  - Converged (R-hat < 1.05): {summary['convergence']['converged']}")
    print(f"  - Sufficient ESS (> 400): {summary['convergence']['sufficient_ess']}")
    
    # Save results using config paths
    save_results(summary, str(paths.model_results_single))
    save_results(summary, str(paths.model_results_canonical))
    
    print("="*80 + "\n")
    return model, summary


def run_multi_change_point_analysis() -> Tuple[BayesianChangePointModel, Dict[str, Any]]:
    """Run multi-change point analysis on a longer period (2007-2022).
    
    Returns:
        Tuple of (fitted model, summary dictionary)
    """
    print("\n" + "="*80)
    print("MULTI-CHANGE POINT ANALYSIS - 2007-2022 (Major Events)")
    print("="*80 + "\n")
    
    # Load long-term data
    df_period = load_and_preprocess_data(start_date='2007-01-01', end_date='2022-12-31')
    
    print(f"Loaded {len(df_period)} days of data from 2007-2022")
    print(f"Price range: ${df_period['Price'].min():.2f} - ${df_period['Price'].max():.2f}\n")
    
    # Initialize and fit model
    model = BayesianChangePointModel(df_period['Price'], use_log_returns=True)
    print("Fitting multi-change point model with recursive binary segmentation...")
    print("This may take several minutes...\n")
    
    model.fit_multi_change_point(
        max_change_points=3,
        min_segment_size=30,
        draws=1000, # Use slightly fewer draws for speed in multi-change point
        tune=500,
        chains=2,
        random_seed=params.random_seed
    )
    
    # Get summary
    summary = model.summary()
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    print(f"\nDetected {len(summary['change_points'])} Change Points:")
    for i, cp in enumerate(summary['change_points'], 1):
        print(f"  {i}. {cp['date']}")
    
    # Extend summary with metadata
    results_extended = {
        'method': 'recursive_binary_segmentation',
        'max_depth': 3,
        'change_points': summary['change_points'],
        'use_log_returns': True
    }
    
    save_results(results_extended, str(paths.model_results_multi))
    
    print("="*80 + "\n")
    
    return model, summary


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("BRENT OIL PRICE - BAYESIAN CHANGE POINT DETECTION")
    print("Refactored Model with Log-Return Transformation")
    print("="*80)
    
    # Ensure directories exist
    paths.ensure_directories()
    
    # Run single change point analysis (fast)
    try:
        single_model, single_summary = run_single_change_point_analysis()
    except Exception as e:
        print(f"Error in single change point analysis: {e}")
        return

    # Ask user if they want to run multi-change point (slow)
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        run_multi = True
    else:
        print("\nTo run multi-change point analysis, pass --all argument.")
        print("Skipping slow multi-change point analysis by default.")
        run_multi = False
    
    if run_multi:
        try:
            multi_model, multi_summary = run_multi_change_point_analysis()
        except KeyboardInterrupt:
            print("\nMulti-change point analysis skipped.")
        except Exception as e:
            print(f"Error in multi change point analysis: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review results in {paths.model_results_single}")
    print(f"2. Start the Flask API: python src/app.py")
    print(f"3. Launch the dashboard: streamlit run streamlit_app.py")
    

if __name__ == "__main__":
    main()
