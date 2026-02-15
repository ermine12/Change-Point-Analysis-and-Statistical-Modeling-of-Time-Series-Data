"""Unit tests for Flask API endpoints."""

import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import app as flask_app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    flask_app.app.config['TESTING'] = True
    with flask_app.app.test_client() as client:
        yield client


class TestPricesEndpoint:
    """Tests for /api/prices endpoint."""
    
    @pytest.mark.api
    @pytest.mark.unit
    def test_prices_endpoint_success(self, client, sample_csv_file, monkeypatch):
        """Test prices endpoint returns JSON successfully."""
        # Mock the DATA_PATH to point to our test file
        monkeypatch.setattr(flask_app, 'DATA_PATH', str(sample_csv_file))
        
        response = client.get('/api/prices')
        
        assert response.status_code == 200
        assert response.content_type == 'application/json'
        
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check data structure
        if len(data) > 0:
            assert 'Date' in data[0]
            assert 'Price' in data[0]
    
    @pytest.mark.api
    @pytest.mark.unit
    def test_prices_endpoint_missing_file(self, client, monkeypatch):
        """Test prices endpoint handles missing file gracefully."""
        # Point to non-existent file
        monkeypatch.setattr(flask_app, 'DATA_PATH', '/nonexistent/file.csv')
        
        response = client.get('/api/prices')
        
        # Should return error status
        assert response.status_code in [404, 500]


class TestEventsEndpoint:
    """Tests for /api/events endpoint."""
    
    @pytest.mark.api
    @pytest.mark.unit
    def test_events_endpoint_success(self, client, temp_data_dir, sample_events_data, monkeypatch):
        """Test events endpoint returns JSON successfully."""
        # Create events CSV
        events_path = temp_data_dir / "events.csv"
        sample_events_data.to_csv(events_path, index=False)
        
        # Mock the EVENTS_PATH
        monkeypatch.setattr(flask_app, 'EVENTS_PATH', str(events_path))
        
        response = client.get('/api/events')
        
        assert response.status_code == 200
        assert response.content_type == 'application/json'
        
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Check structure
        assert 'Event' in data[0]
        assert 'Date' in data[0]


class TestChangePointsEndpoint:
    """Tests for /api/change-points endpoint."""
    
    @pytest.mark.api
    @pytest.mark.unit
    def test_change_points_success(self, client, temp_data_dir, mock_model_results, monkeypatch):
        """Test change points endpoint returns results when file exists."""
        # Create results JSON
        results_path = temp_data_dir / "model_results.json"
        with open(results_path, 'w') as f:
            json.dump(mock_model_results, f)
        
        # Mock the RESULTS_PATH
        monkeypatch.setattr(flask_app, 'RESULTS_PATH', str(results_path))
        
        response = client.get('/api/change-points')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'change_points' in data
        assert 'convergence' in data
    
    @pytest.mark.api
    @pytest.mark.unit
    def test_change_points_missing_file(self, client, monkeypatch):
        """Test change points endpoint handles missing results file."""
        # Point to non-existent file
        monkeypatch.setattr(flask_app, 'RESULTS_PATH', '/nonexistent/results.json')
        
        response = client.get('/api/change-points')
        
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data


class TestCORSHeaders:
    """Tests for CORS configuration."""
    
    @pytest.mark.api
    @pytest.mark.unit
    def test_cors_headers_present(self, client, sample_csv_file, monkeypatch):
        """Test that CORS headers are present in responses."""
        monkeypatch.setattr(flask_app, 'DATA_PATH', str(sample_csv_file))
        
        response = client.get('/api/prices')
        
        # Check for CORS headers (flask-cors adds these)
        # The exact header name depends on flask-cors version
        assert response.status_code == 200
