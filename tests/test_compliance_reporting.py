"""
Tests for Compliance Violation Reporting System
==============================================

Comprehensive test suite for violation reporting, dashboard widgets,
and analytics functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import json

from src.portfolio.compliance_reporting import (
    ComplianceViolationReporter,
    ViolationCategory,
    TimeFrame,
    ViolationSummary,
    ViolationTrend,
    DashboardWidget
)
from src.models.compliance_reporting import (
    ViolationReportRequest,
    ViolationStatistics,
    DashboardWidgetData,
    WidgetType,
    ChartType,
    ReportTimeFrame,
    ViolationCategoryType
)
from src.models.compliance import ViolationSeverity


class TestComplianceViolationReporter:
    """Test the main violation reporter class"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing"""
        with patch('src.portfolio.compliance_reporting.get_db_manager') as mock:
            db_manager = Mock()
            session = Mock()
            db_manager.SessionLocal.return_value = session
            mock.return_value = db_manager
            yield db_manager, session
    
    @pytest.fixture
    def sample_violations(self):
        """Sample violation data for testing"""
        violations = []
        base_time = datetime.now() - timedelta(days=5)
        
        for i in range(10):
            violation = Mock()
            violation.violation_id = f"viol_{i:03d}"
            violation.portfolio_id = f"portfolio_{i % 3}"
            violation.rule_id = f"RULE_{i % 4}"
            violation.severity = ["low", "medium", "high", "critical"][i % 4]
            violation.violation_type = ["position_limit", "esg_score", "credit_rating"][i % 3]
            violation.detected_at = base_time + timedelta(hours=i)
            violation.resolved_at = (base_time + timedelta(hours=i+2)) if i % 2 == 0 else None
            violations.append(violation)
        
        return violations
    
    @pytest.fixture
    def reporter(self, mock_db_manager):
        """Violation reporter instance with mocked database"""
        return ComplianceViolationReporter()
    
    def test_get_violation_summary(self, reporter, mock_db_manager, sample_violations):
        """Test violation summary generation"""
        db_manager, session = mock_db_manager
        session.query.return_value.filter.return_value.all.return_value = sample_violations
        
        summary = reporter.get_violation_summary(TimeFrame.LAST_7D)
        
        assert isinstance(summary, ViolationSummary)
        assert summary.total_violations == 10
        assert len(summary.violations_by_severity) > 0
        assert len(summary.violations_by_category) > 0
        assert summary.resolution_rate >= 0.0
        assert summary.resolution_rate <= 1.0
    
    def test_get_violation_summary_with_portfolio_filter(self, reporter, mock_db_manager, sample_violations):
        """Test violation summary with portfolio filtering"""
        db_manager, session = mock_db_manager
        
        # Filter violations for specific portfolio
        filtered_violations = [v for v in sample_violations if v.portfolio_id == "portfolio_0"]
        session.query.return_value.filter.return_value.filter.return_value.all.return_value = filtered_violations
        
        summary = reporter.get_violation_summary(TimeFrame.LAST_7D, portfolio_id="portfolio_0")
        
        assert summary.total_violations == len(filtered_violations)
        assert "portfolio_0" in summary.affected_portfolios
    
    def test_get_violation_trends(self, reporter, mock_db_manager, sample_violations):
        """Test violation trend analysis"""
        db_manager, session = mock_db_manager
        session.query.return_value.filter.return_value.all.return_value = sample_violations
        
        trends = reporter.get_violation_trends(TimeFrame.LAST_7D)
        
        assert isinstance(trends, ViolationTrend)
        assert trends.timeframe == TimeFrame.LAST_7D
        assert len(trends.violation_counts) > 0
        assert len(trends.severity_trends) > 0
        assert len(trends.category_trends) > 0
    
    def test_get_dashboard_widgets(self, reporter, mock_db_manager, sample_violations):
        """Test dashboard widget generation"""
        db_manager, session = mock_db_manager
        session.query.return_value.filter.return_value.all.return_value = sample_violations
        
        widgets = reporter.get_dashboard_widgets()
        
        assert len(widgets) == 6  # Expected number of widgets
        
        # Check widget types and IDs
        widget_ids = {w.widget_id for w in widgets}
        expected_ids = {
            "current_violations", "severity_distribution", "top_violated_rules",
            "violation_trends", "resolution_rate", "category_breakdown"
        }
        assert widget_ids == expected_ids
        
        # Check widget data structure
        for widget in widgets:
            assert isinstance(widget, DashboardWidget)
            assert widget.widget_id
            assert widget.title
            assert widget.data
            assert isinstance(widget.last_updated, datetime)
    
    def test_generate_violation_report(self, reporter, mock_db_manager, sample_violations):
        """Test comprehensive report generation"""
        db_manager, session = mock_db_manager
        session.query.return_value.filter.return_value.all.return_value = sample_violations
        
        report = reporter.generate_violation_report(TimeFrame.LAST_30D)
        
        assert "report_metadata" in report
        assert "executive_summary" in report
        assert "detailed_analysis" in report
        assert "dashboard_data" in report
        
        # Check report metadata
        metadata = report["report_metadata"]
        assert metadata["timeframe"] == "last_30d"
        assert metadata["report_type"] == "compliance_violation_analysis"
        
        # Check executive summary
        summary = report["executive_summary"]
        assert "total_violations" in summary
        assert "resolution_rate" in summary
        assert "most_critical_issues" in summary
        assert "recommendation" in summary
    
    def test_categorize_violation(self, reporter):
        """Test violation categorization logic"""
        # Test position limit categorization
        category = reporter._categorize_violation("position_limit")
        assert category == ViolationCategory.POSITION_LIMITS
        
        # Test ESG categorization
        category = reporter._categorize_violation("esg_score")
        assert category == ViolationCategory.ESG_REQUIREMENTS
        
        # Test unknown type defaults to operational risks
        category = reporter._categorize_violation("unknown_type")
        assert category == ViolationCategory.OPERATIONAL_RISKS
    
    def test_get_start_date(self, reporter):
        """Test start date calculation for different timeframes"""
        now = datetime.now()
        
        # Test 24 hours
        start_date = reporter._get_start_date(TimeFrame.LAST_24H)
        expected = now - timedelta(hours=24)
        assert abs((start_date - expected).total_seconds()) < 60  # Within 1 minute
        
        # Test 7 days
        start_date = reporter._get_start_date(TimeFrame.LAST_7D)
        expected = now - timedelta(days=7)
        assert abs((start_date - expected).total_seconds()) < 3600  # Within 1 hour
        
        # Test 30 days
        start_date = reporter._get_start_date(TimeFrame.LAST_30D)
        expected = now - timedelta(days=30)
        assert abs((start_date - expected).total_seconds()) < 3600  # Within 1 hour


class TestViolationReportingModels:
    """Test the reporting data models"""
    
    def test_violation_report_request_validation(self):
        """Test validation of report request model"""
        # Valid request
        request = ViolationReportRequest(
            timeframe=ReportTimeFrame.LAST_7D,
            portfolio_id="test_portfolio",
            format="json"
        )
        assert request.timeframe == ReportTimeFrame.LAST_7D
        assert request.portfolio_id == "test_portfolio"
        assert request.format == "json"
        
        # Test invalid format
        with pytest.raises(ValueError):
            ViolationReportRequest(format="invalid_format")
    
    def test_dashboard_widget_data_validation(self):
        """Test dashboard widget data validation"""
        # Valid metric widget
        widget = DashboardWidgetData(
            widget_id="test_metric",
            widget_type=WidgetType.METRIC,
            title="Test Metric",
            description="Test description",
            data={"value": 42},
            last_updated=datetime.now()
        )
        assert widget.widget_type == WidgetType.METRIC
        assert widget.data["value"] == 42
        
        # Invalid metric widget (missing required value)
        with pytest.raises(ValueError):
            DashboardWidgetData(
                widget_id="invalid_metric",
                widget_type=WidgetType.METRIC,
                title="Invalid Metric",
                description="Missing value field",
                data={},  # Missing 'value' field
                last_updated=datetime.now()
            )
        
        # Valid chart widget
        chart_widget = DashboardWidgetData(
            widget_id="test_chart",
            widget_type=WidgetType.CHART,
            title="Test Chart",
            description="Test chart description",
            data={"chart_type": "line", "data": [1, 2, 3]},
            last_updated=datetime.now()
        )
        assert chart_widget.data["chart_type"] == "line"
    
    def test_violation_statistics_model(self):
        """Test violation statistics data model"""
        stats = ViolationStatistics(
            total_violations=100,
            violations_by_severity={ViolationSeverity.HIGH: 10, ViolationSeverity.MEDIUM: 20},
            violations_by_category={ViolationCategoryType.POSITION_LIMITS: 30},
            violations_by_day={"2024-01-01": 5, "2024-01-02": 8},
            resolution_rate=0.85,
            most_violated_rules=[{"rule_id": "RULE_001", "count": 15}],
            affected_portfolios_count=5
        )
        
        assert stats.total_violations == 100
        assert stats.resolution_rate == 0.85
        assert stats.affected_portfolios_count == 5
        
        # Test resolution rate bounds
        with pytest.raises(ValueError):
            ViolationStatistics(
                total_violations=10,
                violations_by_severity={},
                violations_by_category={},
                violations_by_day={},
                resolution_rate=1.5,  # Invalid: > 1.0
                most_violated_rules=[],
                affected_portfolios_count=1
            )


class TestDashboardWidgets:
    """Test dashboard widget functionality"""
    
    def test_metric_widget_creation(self):
        """Test creation of metric widgets"""
        widget = DashboardWidget(
            widget_id="violations_count",
            widget_type="metric",
            title="Total Violations",
            description="Current violation count",
            data={
                "value": 25,
                "change": 5.2,
                "trend": "up"
            },
            last_updated=datetime.now()
        )
        
        assert widget.widget_id == "violations_count"
        assert widget.widget_type == "metric"
        assert widget.data["value"] == 25
        assert widget.data["change"] == 5.2
    
    def test_chart_widget_creation(self):
        """Test creation of chart widgets"""
        widget = DashboardWidget(
            widget_id="severity_chart",
            widget_type="chart",
            title="Violation Severity Distribution",
            description="Pie chart of violation severities",
            data={
                "chart_type": "pie",
                "data": {"high": 10, "medium": 20, "low": 5},
                "colors": {"high": "#dc2626", "medium": "#d97706", "low": "#65a30d"}
            },
            last_updated=datetime.now()
        )
        
        assert widget.widget_type == "chart"
        assert widget.data["chart_type"] == "pie"
        assert "colors" in widget.data
    
    def test_table_widget_creation(self):
        """Test creation of table widgets"""
        widget = DashboardWidget(
            widget_id="top_rules",
            widget_type="table",
            title="Most Violated Rules",
            description="Rules with highest violation frequency",
            data={
                "headers": ["Rule ID", "Count", "Last Violation"],
                "rows": [
                    ["RULE_001", "15", "2024-01-15"],
                    ["RULE_002", "12", "2024-01-14"]
                ]
            },
            last_updated=datetime.now()
        )
        
        assert widget.widget_type == "table"
        assert len(widget.data["headers"]) == 3
        assert len(widget.data["rows"]) == 2


class TestViolationReportingIntegration:
    """Integration tests for violation reporting system"""
    
    @patch('src.portfolio.compliance_reporting.get_db_manager')
    def test_end_to_end_report_generation(self, mock_db_manager):
        """Test complete report generation workflow"""
        # Setup mock database
        db_manager = Mock()
        session = Mock()
        db_manager.SessionLocal.return_value = session
        mock_db_manager.return_value = db_manager
        
        # Mock violation data
        violations = []
        for i in range(5):
            violation = Mock()
            violation.violation_id = f"viol_{i}"
            violation.portfolio_id = "test_portfolio"
            violation.rule_id = f"RULE_{i}"
            violation.severity = "high"
            violation.violation_type = "position_limit"
            violation.detected_at = datetime.now() - timedelta(hours=i)
            violation.resolved_at = None
            violations.append(violation)
        
        session.query.return_value.filter.return_value.all.return_value = violations
        
        # Generate report
        reporter = ComplianceViolationReporter()
        report = reporter.generate_violation_report()
        
        # Validate report structure
        assert "report_metadata" in report
        assert "executive_summary" in report
        assert "detailed_analysis" in report
        assert "dashboard_data" in report
        
        # Validate dashboard widgets
        widgets = report["dashboard_data"]["widgets"]
        assert len(widgets) == 6
        
        # Check that all widgets have required fields
        for widget_data in widgets:
            assert "widget_id" in widget_data
            assert "widget_type" in widget_data
            assert "title" in widget_data
            assert "data" in widget_data
    
    def test_real_time_dashboard_updates(self):
        """Test real-time dashboard update functionality"""
        # This would test WebSocket integration for real-time updates
        # For now, verify that widgets have proper timestamps for refresh
        
        widget = DashboardWidget(
            widget_id="live_violations",
            widget_type="metric",
            title="Live Violation Count",
            description="Real-time violation monitoring",
            data={"value": 10},
            last_updated=datetime.now(),
            refresh_interval=60  # 1 minute refresh
        )
        
        assert widget.refresh_interval == 60
        assert isinstance(widget.last_updated, datetime)
        
        # Simulate data refresh
        old_timestamp = widget.last_updated
        widget.last_updated = datetime.now()
        widget.data["value"] = 12
        
        assert widget.last_updated > old_timestamp
        assert widget.data["value"] == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
