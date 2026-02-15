"""Pytest fixtures for Change Point Analysis tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_price_series():
    """Create a sample price series for testing.
    
    Returns:
        pd.Series with DatetimeIndex and price values
    """
    # 100 days of prices with a clear change point at day 50
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Regime 1: mean=50, std=2
    prices_1 = np.random.normal(loc=50, scale=2, size=50)
    
    # Regime 2: mean=60, std=3
    prices_2 = np.random.normal(loc=60, scale=3, size=50)
    
    prices = np.concatenate([prices_1, prices_2])
    
    return pd.Series(prices, index=dates, name='Price')


@pytest.fixture
def sample_price_series_no_change():
    """Create a price series without change points.
    
    Returns:
        pd.Series with DatetimeIndex and stable prices
    """
    np.random.seed(123)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.random.normal(loc=50, scale=2, size=100)
    
    return pd.Series(prices, index=dates, name='Price')


@pytest.fixture
def sample_log_returns():
    """Create sample log returns for testing.
    
    Returns:
        pd.Series with DatetimeIndex and log return values
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    returns = np.random.normal(loc=0.001, scale=0.02, size=100)
    
    return pd.Series(returns, index=dates, name='Log_Returns')


@pytest.fixture
def sample_events_data():
    """Create sample events data for testing.
    
    Returns:
        pd.DataFrame with event information
    """
    events = pd.DataFrame({
        'Date': ['2020-03-01', '2020-06-01'],
        'Event': ['COVID-19 Pandemic', 'OPEC+ Supply Cut'],
        'Type': ['Health Crisis', 'Supply Shock'],
        'Impact': ['Negative', 'Positive']
    })
    events['Date'] = pd.to_datetime(events['Date'])
    return events


@pytest.fixture
def mock_model_results():
    """Create mock model results for testing.
    
    Returns:
        dict with model result structure
    """
    return {
        'change_points': [
            {
                'date': '2020-02-15',
                'index': 45,
                'credible_interval': ['2020-02-10', '2020-02-20']
            }
        ],
        'regime_parameters': [
            {'regime': 1, 'mean': -0.001, 'std': 0.02},
            {'regime': 2, 'mean': 0.005, 'std': 0.03}
        ],
        'convergence': {
            'r_hat_ok': True,
            'ess_ok': True,
            'max_r_hat': 1.01
        }
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data.
    
    Args:
        tmp_path: pytest's tmp_path fixture
        
    Returns:
        Path to temporary directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_price_series):
    """Create a temporary CSV file with sample price data.
    
    Args:
        temp_data_dir: Temporary directory fixture
        sample_price_series: Sample price series fixture
        
    Returns:
        Path to CSV file
    """
    csv_path = temp_data_dir / "test_prices.csv"
    df = sample_price_series.reset_index()
    df.columns = ['Date', 'Price']
    df.to_csv(csv_path, index=False)
    return csv_path
