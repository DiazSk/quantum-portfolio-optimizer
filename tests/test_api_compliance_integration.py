"""
Test suite for API compliance integration
Tests the integration of compliance validation with the portfolio optimization API
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json
from dataclasses import dataclass
from typing import List, Optional

from src.api.api_server import app
from src.models.compliance import ViolationDetails, ComplianceValidationResult


@dataclass
class ValidationResult:
    """Simple validation result for testing"""
    is_compliant: bool
    violations: List[ViolationDetails]
    compliance_score: float
    validation_timestamp: datetime


class TestAPIComplianceIntegration:
    """Test compliance integration with API endpoints"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_portfolio_request(self):
        """Sample portfolio optimization request"""
        return {
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "optimization_method": "max_sharpe",
            "use_ml_predictions": True,
            "use_alternative_data": True,
            "skip_compliance_check": False,
            "compliance_rule_sets": [1, 2],  # Use integer IDs
            "initial_capital": 100000,
            "risk_free_rate": 0.04
        }
    
    @pytest.fixture
    def mock_compliance_result(self):
        """Mock compliance validation result"""
        return ValidationResult(
            is_compliant=True,
            violations=[],
            compliance_score=0.95,
            validation_timestamp=datetime.now()
        )
    
    @pytest.fixture
    def mock_violation_result(self):
        """Mock compliance result with violations"""
        violations = [
            ViolationDetails(
                rule_name="Single Position Limit",
                rule_type="position_limit",
                violation_description="Position exceeds single asset limit for AAPL",
                current_value=0.35,
                threshold_value=0.30,
                affected_assets=["AAPL"],
                recommended_action="Reduce AAPL position to below 30% of portfolio"
            )
        ]
        return ValidationResult(
            is_compliant=False,
            violations=violations,
            compliance_score=0.72,
            validation_timestamp=datetime.now()
        )

    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_optimize_with_compliance_success(self, mock_engine_class, client, 
                                            sample_portfolio_request, mock_compliance_result):
        """Test successful portfolio optimization with compliance validation"""
        # Setup mock compliance engine
        mock_engine = AsyncMock()
        mock_engine.validate_portfolio.return_value = mock_compliance_result
        mock_engine_class.return_value = mock_engine
        
        # Make API request
        response = client.post("/api/optimize", json=sample_portfolio_request)
        
        # Debug the response
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        # Check portfolio response structure
        assert "weights" in data
        assert "expected_return" in data
        assert "volatility" in data  # Changed from "risk" to "volatility"
        assert "compliance_status" in data
        assert "compliance_violations" in data
        assert "compliance_score" in data
        
        # Check compliance fields
        assert data["compliance_status"] == "compliant"
        assert data["compliance_violations"] == []
        assert data["compliance_score"] == 0.95
        
        # Verify compliance engine was called
        mock_engine.validate_portfolio.assert_called_once()

    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_optimize_with_compliance_violations(self, mock_engine_class, client, 
                                               sample_portfolio_request, mock_violation_result):
        """Test portfolio optimization with compliance violations"""
        # Setup mock compliance engine
        mock_engine = AsyncMock()
        mock_engine.validate_portfolio.return_value = mock_violation_result
        mock_engine_class.return_value = mock_engine
        
        # Make API request
        response = client.post("/api/optimize", json=sample_portfolio_request)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        # Check compliance fields
        assert data["compliance_status"] == "violations_detected"
        assert len(data["compliance_violations"]) == 1
        assert data["compliance_score"] == 0.72
        
        # Check violation details
        violation = data["compliance_violations"][0]
        assert violation["rule_name"] == "Single Position Limit"
        assert violation["rule_type"] == "position_limit"
        assert violation["violation_description"] == "Position exceeds single asset limit for AAPL"

    def test_optimize_skip_compliance(self, client):
        """Test portfolio optimization with compliance checking disabled"""
        request_data = {
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "optimization_method": "max_sharpe",
            "use_ml_predictions": True,
            "use_alternative_data": True,
            "skip_compliance_check": True,
            "initial_capital": 100000,
            "risk_free_rate": 0.04
        }
        
        response = client.post("/api/optimize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have default compliance values when skipped
        assert data["compliance_status"] == "not_checked"
        assert data["compliance_violations"] == []
        assert data["compliance_score"] is None

    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_compliance_validate_endpoint(self, mock_engine_class, client, mock_compliance_result):
        """Test dedicated compliance validation endpoint"""
        mock_engine = AsyncMock()
        mock_engine.validate_portfolio.return_value = mock_compliance_result
        mock_engine_class.return_value = mock_engine
        
        request_data = {
            "allocations": {
                "AAPL": 0.30,
                "GOOGL": 0.40,
                "MSFT": 0.30
            },
            "rule_sets": ["default", "esg"]
        }
        
        response = client.post("/api/compliance/validate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["is_compliant"] is True
        assert data["violations"] == []
        assert data["compliance_score"] == 0.95
        assert "validation_timestamp" in data

    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_get_compliance_rules(self, mock_engine_class, client):
        """Test fetching compliance rules"""
        mock_engine = AsyncMock()
        mock_rules = [
            Mock(
                rule_id="SINGLE_POSITION_LIMIT",
                rule_set="default",
                rule_type="position_limit",
                description="Limit single asset position",
                parameters={"max_weight": 0.30},
                severity="high",
                is_active=True
            )
        ]
        mock_engine.get_rules.return_value = mock_rules
        mock_engine_class.return_value = mock_engine
        
        response = client.get("/api/compliance/rules")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 1
        rule = data[0]
        assert rule["rule_id"] == "SINGLE_POSITION_LIMIT"
        assert rule["rule_type"] == "position_limit"
        assert rule["severity"] == "high"

    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_get_compliance_violations(self, mock_engine_class, client):
        """Test fetching compliance violations history"""
        mock_engine = AsyncMock()
        mock_violations = [
            Mock(
                violation_id="viol_001",
                portfolio_id="port_123",
                rule_id="SINGLE_POSITION_LIMIT",
                asset_symbol="AAPL",
                violation_type="position_limit",
                severity="high",
                message="Position exceeds limit",
                current_value=0.35,
                threshold=0.30,
                detected_at=datetime(2024, 1, 15, 10, 30),
                resolved_at=None
            )
        ]
        mock_engine.get_violations_history.return_value = mock_violations
        mock_engine_class.return_value = mock_engine
        
        response = client.get("/api/compliance/violations?portfolio_id=port_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 1
        violation = data[0]
        assert violation["violation_id"] == "viol_001"
        assert violation["rule_id"] == "SINGLE_POSITION_LIMIT"
        assert violation["severity"] == "high"

    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_create_compliance_rule(self, mock_engine_class, client):
        """Test creating a new compliance rule"""
        mock_engine = AsyncMock()
        mock_created_rule = Mock(
            rule_id="NEW_RULE_001",
            rule_set="custom",
            rule_type="position_limit",
            description="Custom position limit",
            parameters={"max_weight": 0.25},
            severity="medium",
            is_active=True
        )
        mock_engine.create_rule.return_value = mock_created_rule
        mock_engine_class.return_value = mock_engine
        
        rule_data = {
            "rule_set": "custom",
            "rule_type": "position_limit",
            "description": "Custom position limit",
            "parameters": {"max_weight": 0.25},
            "severity": "medium"
        }
        
        response = client.post("/api/compliance/rules", json=rule_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["rule_id"] == "NEW_RULE_001"
        assert data["message"] == "Compliance rule created successfully"
        assert data["rule"]["rule_type"] == "position_limit"

    @patch('src.api.api_server.ProductionComplianceEngine')
    def test_compliance_engine_error_handling(self, mock_engine_class, client):
        """Test error handling in compliance endpoints"""
        mock_engine = AsyncMock()
        mock_engine.validate_portfolio.side_effect = Exception("Database connection failed")
        mock_engine_class.return_value = mock_engine
        
        request_data = {
            "allocations": {"AAPL": 0.50, "GOOGL": 0.50},
            "rule_sets": ["default"]
        }
        
        response = client.post("/api/compliance/validate", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "Compliance validation failed" in data["detail"]

    def test_backward_compatibility(self, client):
        """Test that existing API clients without compliance fields still work"""
        # Old format request without compliance fields
        old_request = {
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "optimization_method": "max_sharpe",
            "use_ml_predictions": True,
            "use_alternative_data": True,
            "initial_capital": 100000,
            "risk_free_rate": 0.04
        }
        
        response = client.post("/api/optimize", json=old_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have basic structure
        assert "weights" in data
        assert "expected_return" in data
        assert "volatility" in data
        
        # Should have default compliance values
        assert data["compliance_status"] == "not_checked"
        assert data["compliance_violations"] == []
        assert data["compliance_score"] is None


class TestComplianceAPIRobustness:
    """Test robustness and edge cases for compliance API"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_compliance_with_empty_allocations(self, client):
        """Test compliance validation with empty allocations"""
        request_data = {
            "allocations": {},
            "rule_sets": ["default"]
        }
        
        response = client.post("/api/compliance/validate", json=request_data)
        
        # Should handle gracefully (depends on implementation)
        assert response.status_code in [200, 400]
    
    def test_compliance_with_invalid_rule_set(self, client):
        """Test compliance validation with non-existent rule set"""
        request_data = {
            "allocations": {"AAPL": 1.0},
            "rule_sets": ["non_existent_rule_set"]
        }
        
        response = client.post("/api/compliance/validate", json=request_data)
        
        # Should handle gracefully or return appropriate error
        assert response.status_code in [200, 400, 404]
    
    def test_compliance_rules_filtering(self, client):
        """Test compliance rules endpoint with various filters"""
        # Test with rule_set filter
        response = client.get("/api/compliance/rules?rule_set=esg")
        assert response.status_code == 200
        
        # Test with active_only filter
        response = client.get("/api/compliance/rules?active_only=false")
        assert response.status_code == 200
        
        # Test with both filters
        response = client.get("/api/compliance/rules?rule_set=default&active_only=true")
        assert response.status_code == 200
    
    def test_violations_date_filtering(self, client):
        """Test violations endpoint with date range filtering"""
        # Test with date range
        response = client.get(
            "/api/compliance/violations?start_date=2024-01-01&end_date=2024-12-31"
        )
        assert response.status_code == 200
        
        # Test with severity filter
        response = client.get("/api/compliance/violations?severity=high")
        assert response.status_code == 200
        
        # Test with limit
        response = client.get("/api/compliance/violations?limit=50")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
