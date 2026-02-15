"""Unit tests for utility functions."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestDataValidation:
    """Tests for data validation utilities."""
    
    @pytest.mark.unit
    def test_valid_price_series(self, sample_price_series):
        """Test validation of valid price series."""
        # Basic checks that a proper series should pass
        assert isinstance(sample_price_series, pd.Series)
        assert len(sample_price_series) > 0
        assert sample_price_series.dtype in [np.float64, np.int64, float, int]
    
    @pytest.mark.unit
    def test_datetime_index_validation(self, sample_price_series):
        """Test that price series has DatetimeIndex."""
        assert isinstance(sample_price_series.index, pd.DatetimeIndex)
    
    @pytest.mark.unit
    def test_no_missing_values(self, sample_price_series):
        """Test that sample series has no missing values."""
        assert not sample_price_series.isna().any()
        assert not sample_price_series.isnull().any()


class TestLogReturnCalculations:
    """Tests for log return utility functions."""
    
    @pytest.mark.unit
    def test_log_return_formula(self):
        """Test manual log return calculation."""
        prices = pd.Series([100, 105, 103, 108])
        
        # Manual calculation: log(P_t / P_{t-1})
        expected = np.log(prices / prices.shift(1)).dropna()
        
        # Using numpy directly
        calculated = np.log(prices.iloc[1:].values / prices.iloc[:-1].values)
        
        np.testing.assert_array_almost_equal(expected.values, calculated, decimal=10)
    
    @pytest.mark.unit
    def test_log_return_properties(self, sample_price_series):
        """Test that log returns have expected statistical properties."""
        log_returns = np.log(sample_price_series / sample_price_series.shift(1)).dropna()
        
        # Log returns should be roughly centered around 0 for stationary data
        # (though not necessarily exactly 0)
        assert abs(log_returns.mean()) < 1.0  # Reasonable bound
        
        # Standard deviation should be positive
        assert log_returns.std() > 0


class TestDateHandling:
    """Tests for date/time handling utilities."""
    
    @pytest.mark.unit
    def test_date_sorting(self):
        """Test that dates can be sorted correctly."""
        dates = pd.to_datetime(['2020-03-01', '2020-01-01', '2020-02-01'])
        sorted_dates = sorted(dates)
        
        expected = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        
        assert all(sorted_dates == expected)
    
    @pytest.mark.unit
    def test_date_range_calculation(self, sample_price_series):
        """Test calculating date range of series."""
        min_date = sample_price_series.index.min()
        max_date = sample_price_series.index.max()
        
        # Should be valid dates
        assert isinstance(min_date, pd.Timestamp)
        assert isinstance(max_date, pd.Timestamp)
        
        # Max should be after min
        assert max_date > min_date


class TestStatisticalFunctions:
    """Tests for statistical utility functions."""
    
    @pytest.mark.unit
    def test_mean_calculation(self, sample_log_returns):
        """Test mean calculation."""
        mean = sample_log_returns.mean()
        
        assert isinstance(mean, (float, np.floating))
        assert not np.isnan(mean)
    
    @pytest.mark.unit
    def test_std_calculation(self, sample_log_returns):
        """Test standard deviation calculation."""
        std = sample_log_returns.std()
        
        assert isinstance(std, (float, np.floating))
        assert std > 0
        assert not np.isnan(std)
    
    @pytest.mark.unit
    def test_volatility_annualization(self):
        """Test volatility annualization (daily to annual)."""
        daily_std = 0.02  # 2% daily volatility
        annual_std = daily_std * np.sqrt(252)  # 252 trading days
        
        assert annual_std > daily_std
        assert abs(annual_std - 0.3176) < 0.001  # ~31.76% annual


class TestFileOperations:
    """Tests for file I/O operations."""
    
    @pytest.mark.unit
    def test_csv_read_write(self, temp_data_dir):
        """Test CSV read and write operations."""
        # Create test data
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Price': np.random.randn(10) + 100
        })
        
        # Write to CSV
        csv_path = temp_data_dir / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Read back
        df_read = pd.read_csv(csv_path)
        df_read['Date'] = pd.to_datetime(df_read['Date'])
        
        # Verify
        assert len(df_read) == len(df)
        assert list(df_read.columns) == ['Date', 'Price']
    
    @pytest.mark.unit
    def test_json_read_write(self, temp_data_dir, mock_model_results):
        """Test JSON read and write operations."""
        import json
        
        # Write to JSON
        json_path = temp_data_dir / "test.json"
        with open(json_path, 'w') as f:
            json.dump(mock_model_results, f)
        
        # Read back
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Verify
        assert data == mock_model_results
        assert 'change_points' in data
