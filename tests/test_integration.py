"""Integration tests for Change Point Analysis scripts.

These tests verify that the scripts run correctly end-to-end using
mocked data and configuration.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_model import main as run_model_main
from src.analyze_properties import main as analyze_properties_main
from src.config import ProjectPaths

@pytest.fixture
def mock_paths(tmp_path):
    """Create a mock ProjectPaths object with temporary directories."""
    base = tmp_path / "project"
    base.mkdir()
    
    data = base / "data"
    data.mkdir()
    
    reports = base / "reports"
    reports.mkdir()
    
    # Create sample price data
    dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
    prices = np.random.normal(loc=50, scale=2, size=50)
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    csv_path = data / "BrentOilPrices.csv"
    df.to_csv(csv_path, index=False)
    
    # Create mock paths object
    paths = ProjectPaths(
        base=base,
        data=data,
        reports=reports,
        src=base / "src",
        tests=base / "tests",
        brent_prices=csv_path,
        events=base / "events.csv",
        model_results_single=data / "model_results_single.json",
        model_results_multi=data / "model_results_multi.json",
        model_results_canonical=data / "model_results.json"
    )
    
    return paths

def test_run_model_integration(mock_paths):
    """Test run_model.py end-to-end."""
    with patch('src.run_model.paths', mock_paths):
        # Mock params to reduce sampling time
        with patch('src.run_model.params') as mock_params:
            mock_params.draws = 10
            mock_params.tune = 5
            mock_params.chains = 1
            mock_params.random_seed = 42
            
            # Run main function
            run_model_main()
            
            # Verify outputs
            assert mock_paths.model_results_single.exists()
            assert mock_paths.model_results_canonical.exists()

def test_analyze_properties_integration(mock_paths):
    """Test analyze_properties.py end-to-end."""
    with patch('src.analyze_properties.paths', mock_paths):
        # Run main function
        analyze_properties_main()
        
        # Verify outputs
        assert (mock_paths.reports / "time_series_properties.png").exists()
        assert (mock_paths.reports / "analysis_results.txt").exists()
