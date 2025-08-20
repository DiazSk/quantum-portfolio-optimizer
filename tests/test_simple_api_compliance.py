"""
Simple API test for compliance integration
Tests basic API functionality without heavy database setup
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Set testing mode before importing
os.environ['TESTING'] = '1'

from src.api.api_server import app


class TestSimpleAPICompliance:
    """Simple API compliance tests"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_request(self):
        """Simple portfolio request"""
        return {
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "optimization_method": "max_sharpe",
            "use_ml_predictions": True,
            "use_alternative_data": True,
            "skip_compliance_check": True,  # Skip compliance for simplicity
            "initial_capital": 100000,
            "risk_free_rate": 0.04
        }
    
    def test_api_basic_functionality(self, client, sample_request):
        """Test basic API functionality without compliance"""
        response = client.post("/api/optimize", json=sample_request)
        
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check basic structure
        assert "weights" in data
        assert "expected_return" in data
        assert "volatility" in data
        assert "sharpe_ratio" in data
        
        # Check compliance fields exist (even if not used)
        assert "compliance_status" in data
        assert "compliance_violations" in data
        assert "compliance_score" in data
        
        # When compliance is skipped, status should be "not_checked"
        assert data["compliance_status"] == "not_checked"
        assert data["compliance_violations"] == []
        assert data["compliance_score"] is None
    
    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_api_with_mocked_compliance(self, mock_engine_class, client):
        """Test API with mocked compliance engine"""
        # Setup async mock
        from unittest.mock import AsyncMock
        
        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.is_compliant = True
        mock_result.violations = []
        mock_result.compliance_score = 0.95
        mock_result.validation_timestamp = "2024-01-01T00:00:00"
        
        mock_engine.validate_portfolio.return_value = mock_result
        mock_engine_class.return_value = mock_engine
        
        request_data = {
            "tickers": ["AAPL", "GOOGL"],
            "optimization_method": "max_sharpe",
            "skip_compliance_check": False,  # Enable compliance
            "compliance_rule_sets": [1, 2]
        }
        
        response = client.post("/api/optimize", json=request_data)
        
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have compliance data
        assert data["compliance_status"] == "compliant"
        assert data["compliance_score"] == 0.95
    
    def test_api_health_check(self, client):
        """Test basic API health"""
        # Test root endpoint exists
        response = client.get("/")
        
        # Should not error (might be 200, 404, or 405 depending on implementation)
        assert response.status_code in [200, 404, 405]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
