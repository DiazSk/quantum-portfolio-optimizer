"""
Test the new compliance reporting API endpoints
==============================================

Tests for dashboard, summary, and analytics endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
from datetime import datetime

# Set testing mode
os.environ['TESTING'] = '1'

from src.api.api_server import app


class TestComplianceReportingAPI:
    """Test compliance reporting API endpoints"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_violation_reporter(self):
        """Mock violation reporter for testing"""
        with patch('src.api.api_server.violation_reporter') as mock:
            # Mock dashboard widgets
            mock_widget = MagicMock()
            mock_widget.widget_id = "test_widget"
            mock_widget.widget_type = "metric"
            mock_widget.title = "Test Widget"
            mock_widget.description = "Test Description"
            mock_widget.data = {"value": 42}
            mock_widget.last_updated = datetime.now()
            mock_widget.refresh_interval = 300
            
            mock.get_dashboard_widgets.return_value = [mock_widget]
            
            # Mock summary
            mock_summary = MagicMock()
            mock_summary.total_violations = 25
            mock_summary.violations_by_severity = {"high": 5, "medium": 10, "low": 10}
            mock_summary.violations_by_category = {"position_limits": 15, "esg_requirements": 10}
            mock_summary.resolution_rate = 0.85
            mock_summary.average_resolution_time = 2.5
            mock_summary.affected_portfolios = ["portfolio_1", "portfolio_2"]
            mock_summary.most_violated_rules = [("RULE_001", 10), ("RULE_002", 8)]
            
            mock.get_violation_summary.return_value = mock_summary
            mock._get_critical_issues.return_value = ["No critical issues"]
            mock._generate_recommendation.return_value = "Continue monitoring"
            
            # Mock trends
            mock_trends = MagicMock()
            mock_trends.violation_counts = [(datetime.now().date(), 5)]
            mock_trends.severity_trends = {"high": [(datetime.now().date(), 2)]}
            mock_trends.category_trends = {"position_limits": [(datetime.now().date(), 3)]}
            
            mock.get_violation_trends.return_value = mock_trends
            
            # Mock report
            mock.generate_violation_report.return_value = {
                "report_metadata": {"generated_at": datetime.now().isoformat()},
                "executive_summary": {"total_violations": 25},
                "detailed_analysis": {"summary": "test"},
                "dashboard_data": {"widgets": []}
            }
            
            yield mock
    
    def test_get_compliance_dashboard(self, client, mock_violation_reporter):
        """Test compliance dashboard endpoint"""
        response = client.get("/api/compliance/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "widgets" in data
        assert "dashboard_metadata" in data
        assert len(data["widgets"]) == 1
        
        widget = data["widgets"][0]
        assert widget["widget_id"] == "test_widget"
        assert widget["widget_type"] == "metric"
        assert widget["title"] == "Test Widget"
        assert widget["data"]["value"] == 42
    
    def test_get_compliance_dashboard_with_filters(self, client, mock_violation_reporter):
        """Test dashboard endpoint with portfolio filter"""
        response = client.get("/api/compliance/dashboard?portfolio_id=test_portfolio&timeframe=last_7d")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["dashboard_metadata"]["portfolio_id"] == "test_portfolio"
        assert data["dashboard_metadata"]["timeframe"] == "last_7d"
    
    def test_get_violation_summary(self, client, mock_violation_reporter):
        """Test violation summary endpoint"""
        response = client.get("/api/compliance/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "summary_statistics" in data
        assert "trend_analysis" in data
        assert "key_insights" in data
        assert "metadata" in data
        
        # Check summary statistics
        stats = data["summary_statistics"]
        assert stats["total_violations"] == 25
        assert stats["resolution_rate"] == 0.85
        assert stats["affected_portfolios_count"] == 2
        
        # Check key insights
        insights = data["key_insights"]
        assert "most_violated_rules" in insights
        assert "critical_issues" in insights
        assert "recommendations" in insights
    
    def test_get_violation_summary_with_timeframe(self, client, mock_violation_reporter):
        """Test summary endpoint with different timeframes"""
        response = client.get("/api/compliance/summary?timeframe=last_24h&portfolio_id=portfolio_1")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metadata"]["timeframe"] == "last_24h"
        assert data["metadata"]["portfolio_id"] == "portfolio_1"
    
    def test_generate_compliance_report(self, client, mock_violation_reporter):
        """Test comprehensive report generation endpoint"""
        request_data = {
            "timeframe": "last_30d",
            "portfolio_id": "test_portfolio",
            "include_resolved": True,
            "format": "json"
        }
        
        response = client.post("/api/compliance/report", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "report_metadata" in data
        assert "executive_summary" in data
        assert "detailed_analysis" in data
        assert "dashboard_data" in data
    
    def test_get_compliance_analytics(self, client, mock_violation_reporter):
        """Test advanced compliance analytics endpoint"""
        response = client.get("/api/compliance/analytics?include_predictions=true")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "analytics" in data
        assert "metadata" in data
        
        analytics = data["analytics"]
        assert "risk_assessment" in analytics
        assert "performance_metrics" in analytics
        assert "correlation_analysis" in analytics
        assert "benchmarks" in analytics
        
        # Should include predictions when requested
        assert data["metadata"]["includes_predictions"] is True
    
    def test_get_compliance_analytics_without_predictions(self, client, mock_violation_reporter):
        """Test analytics endpoint without predictions"""
        response = client.get("/api/compliance/analytics?include_predictions=false")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metadata"]["includes_predictions"] is False
    
    def test_compliance_dashboard_error_handling(self, client):
        """Test error handling in compliance endpoints"""
        with patch('src.api.api_server.violation_reporter') as mock:
            mock.get_dashboard_widgets.side_effect = Exception("Database error")
            
            response = client.get("/api/compliance/dashboard")
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to fetch compliance dashboard" in data["detail"]
    
    def test_invalid_timeframe_handling(self, client, mock_violation_reporter):
        """Test handling of invalid timeframe parameters"""
        # The FastAPI validation should handle invalid enum values
        response = client.get("/api/compliance/summary?timeframe=invalid_timeframe")
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_multiple_endpoints_integration(self, client, mock_violation_reporter):
        """Test that multiple endpoints work together consistently"""
        # Get dashboard
        dashboard_response = client.get("/api/compliance/dashboard")
        assert dashboard_response.status_code == 200
        
        # Get summary
        summary_response = client.get("/api/compliance/summary")
        assert summary_response.status_code == 200
        
        # Generate report
        report_response = client.post("/api/compliance/report", json={
            "timeframe": "last_30d",
            "format": "json"
        })
        assert report_response.status_code == 200
        
        # Get analytics
        analytics_response = client.get("/api/compliance/analytics")
        assert analytics_response.status_code == 200
        
        # All should succeed
        assert all([
            dashboard_response.status_code == 200,
            summary_response.status_code == 200,
            report_response.status_code == 200,
            analytics_response.status_code == 200
        ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
