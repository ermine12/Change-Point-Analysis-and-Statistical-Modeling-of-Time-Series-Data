"""Unit tests for Bayesian Change Point Model."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from change_point.bayesian_model import BayesianChangePointModel


class TestBayesianModelInitialization:
    """Tests for model initialization."""
    
    @pytest.mark.unit
    def test_initialization_with_valid_series(self, sample_price_series):
        """Test model initializes correctly with valid price series."""
        model = BayesianChangePointModel(sample_price_series, use_log_returns=True)
        
        assert model.data is not None
        assert model.use_log_returns is True
        assert len(model.returns) == len(sample_price_series) - 1  # log returns are one shorter
        
    @pytest.mark.unit
    def test_initialization_with_insufficient_data(self):
        """Test model raises error with too little data."""
        # Only 5 data points - too small
        short_series = pd.Series(
            [10, 11, 12, 13, 14],
            index=pd.date_range('2020-01-01', periods=5, freq='D')
        )
        
        with pytest.raises(ValueError, match="at least 20 observations"):
            BayesianChangePointModel(short_series)
    
    @pytest.mark.unit
    def test_initialization_without_datetime_index_raises_error(self):
        """Test model raises error without DatetimeIndex."""
        # Series with default integer index
        series = pd.Series(np.random.randn(50))
        
        with pytest.raises(TypeError, match="must have a DatetimeIndex"):
            BayesianChangePointModel(series, use_log_returns=False)
        

class TestLogReturnTransformation:
    """Tests for log return transformation."""
    
    @pytest.mark.unit
    def test_log_return_calculation(self, sample_price_series):
        """Test log returns are calculated correctly."""
        model = BayesianChangePointModel(sample_price_series, use_log_returns=True)
        
        # Manually calculate expected log returns
        expected_returns = np.log(sample_price_series / sample_price_series.shift(1)).dropna()
        
        # Compare (allowing for small floating point differences)
        np.testing.assert_array_almost_equal(
            model.returns.values,
            expected_returns.values,
            decimal=10
        )
    
    @pytest.mark.unit
    def test_prices_mode_no_transformation(self, sample_price_series):
        """Test that use_log_returns=False doesn't transform data."""
        model = BayesianChangePointModel(sample_price_series, use_log_returns=False)
        
        # Should be same length (no transformation)
        assert len(model.returns) == len(sample_price_series)
        
        # Values should be identical
        pd.testing.assert_series_equal(model.returns, sample_price_series)


class TestModelFitting:
    """Tests for model fitting functionality."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_single_change_point_fit(self, sample_price_series):
        """Test single change point detection completes without error."""
        model = BayesianChangePointModel(sample_price_series, use_log_returns=True)
        
        # Use minimal draws for fast testing
        model.fit_single_change_point(draws=100, tune=50, chains=1, random_seed=42)
        
        assert model.trace is not None
        assert model.model is not None
    
    @pytest.mark.unit
    def test_summary_before_fit_raises_error(self, sample_price_series):
        """Test that calling summary before fit raises appropriate error."""
        model = BayesianChangePointModel(sample_price_series)
        
        with pytest.raises(ValueError, match="not been fitted"):
            model.summary()
    
    @pytest.mark.unit
    def test_predict_before_fit_raises_error(self, sample_price_series):
        """Test that calling predict before fit raises appropriate error."""
        model = BayesianChangePointModel(sample_price_series)
        
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict()


class TestConvergenceChecking:
    """Tests for convergence diagnostics."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_convergence_check_runs(self, sample_price_series_no_change):
        """Test convergence checking completes."""
        model = BayesianChangePointModel(sample_price_series_no_change, use_log_returns=False)
        
        # Fit with minimal parameters
        model.fit_single_change_point(draws=100, tune=50, chains=2, random_seed=42)
        
        # Convergence check should run without error
        convergence_ok = model._check_convergence()
        
        # Should return boolean
        assert isinstance(convergence_ok, bool)


class TestSummaryOutput:
    """Tests for summary output format."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_summary_returns_dict(self, sample_price_series):
        """Test that summary returns properly formatted dictionary."""
        model = BayesianChangePointModel(sample_price_series, use_log_returns=True)
        model.fit_single_change_point(draws=100, tune=50, chains=1, random_seed=42)
        
        summary = model.summary()
        
        # Check required keys
        assert 'change_points' in summary
        assert 'convergence' in summary
        assert isinstance(summary['change_points'], list)
        assert isinstance(summary['convergence'], dict)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.unit
    def test_empty_series_raises_error(self):
        """Test that empty series raises ValueError."""
        empty_series = pd.Series([], dtype=float)
        
        with pytest.raises((ValueError, Exception)):
            BayesianChangePointModel(empty_series)
    
    @pytest.mark.unit  
    def test_series_with_nan_values(self):
        """Test handling of NaN values in series."""
        series_with_nan = pd.Series([10, 20, np.nan, 30, 40, 50, 60, 70, 80, 90] * 5)
        
        # Should either handle gracefully or raise informative error
        try:
            model = BayesianChangePointModel(series_with_nan, use_log_returns=False)
            # If it accepts, series should have NaN removed or handled
            assert not model.returns.isna().any()
        except ValueError as e:
            # If it rejects, error message should mention NaN
            assert "NaN" in str(e) or "missing" in str(e).lower()
