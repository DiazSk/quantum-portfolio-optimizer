"""
Compliance Violation Reporting System
====================================

This module provides comprehensive reporting and dashboard capabilities
for compliance violations, including categorization, severity analysis,
and monitoring widgets.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from collections import defaultdict

from ..models.compliance import (
    ComplianceViolation, 
    ViolationDetails, 
    ViolationSeverity,
    ComplianceRule
)
from ..database.production_db import get_db_manager


class ViolationCategory(Enum):
    """Categories for grouping compliance violations"""
    POSITION_LIMITS = "position_limits"
    INVESTMENT_MANDATES = "investment_mandates"
    RISK_MANAGEMENT = "risk_management"
    ESG_REQUIREMENTS = "esg_requirements"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    OPERATIONAL_RISKS = "operational_risks"


class TimeFrame(Enum):
    """Time frames for violation analysis"""
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"
    LAST_90D = "last_90d"
    LAST_YEAR = "last_year"


@dataclass
class ViolationSummary:
    """Summary statistics for compliance violations"""
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_category: Dict[str, int]
    violations_by_timeframe: Dict[str, int]
    most_violated_rules: List[Tuple[str, int]]
    affected_portfolios: List[str]
    resolution_rate: float
    average_resolution_time: Optional[float]  # in hours


@dataclass
class ViolationTrend:
    """Trend analysis for violations over time"""
    timeframe: TimeFrame
    violation_counts: List[Tuple[datetime, int]]
    severity_trends: Dict[str, List[Tuple[datetime, int]]]
    category_trends: Dict[str, List[Tuple[datetime, int]]]


@dataclass
class DashboardWidget:
    """Individual dashboard widget for violation monitoring"""
    widget_id: str
    widget_type: str  # chart, metric, alert, table
    title: str
    description: str
    data: Dict[str, Any]
    last_updated: datetime
    refresh_interval: int = 300  # seconds


class ComplianceViolationReporter:
    """Main class for compliance violation reporting and analytics"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        
    def get_violation_summary(
        self, 
        timeframe: TimeFrame = TimeFrame.LAST_30D,
        portfolio_id: Optional[str] = None
    ) -> ViolationSummary:
        """
        Get comprehensive violation summary for specified timeframe
        """
        session = self.db_manager.SessionLocal()
        try:
            # Calculate date range
            start_date = self._get_start_date(timeframe)
            
            # Base query
            query = session.query(ComplianceViolation).filter(
                ComplianceViolation.detected_at >= start_date
            )
            
            if portfolio_id:
                query = query.filter(ComplianceViolation.portfolio_id == portfolio_id)
            
            violations = query.all()
            
            # Calculate summary statistics
            total_violations = len(violations)
            
            # Violations by severity
            violations_by_severity = defaultdict(int)
            for v in violations:
                violations_by_severity[v.severity] += 1
            
            # Violations by category (derived from rule type)
            violations_by_category = defaultdict(int)
            for v in violations:
                category = self._categorize_violation(v.violation_type)
                violations_by_category[category.value] += 1
            
            # Violations by timeframe (daily breakdown)
            violations_by_timeframe = self._get_daily_breakdown(violations, timeframe)
            
            # Most violated rules
            rule_counts = defaultdict(int)
            for v in violations:
                rule_counts[v.rule_id] += 1
            most_violated_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Affected portfolios
            affected_portfolios = list(set(v.portfolio_id for v in violations if v.portfolio_id))
            
            # Resolution metrics
            resolved_violations = [v for v in violations if v.resolved_at is not None]
            resolution_rate = len(resolved_violations) / total_violations if total_violations > 0 else 0.0
            
            # Average resolution time
            if resolved_violations:
                resolution_times = [
                    (v.resolved_at - v.detected_at).total_seconds() / 3600  # hours
                    for v in resolved_violations
                ]
                average_resolution_time = sum(resolution_times) / len(resolution_times)
            else:
                average_resolution_time = None
            
            return ViolationSummary(
                total_violations=total_violations,
                violations_by_severity=dict(violations_by_severity),
                violations_by_category=dict(violations_by_category),
                violations_by_timeframe=violations_by_timeframe,
                most_violated_rules=most_violated_rules,
                affected_portfolios=affected_portfolios,
                resolution_rate=resolution_rate,
                average_resolution_time=average_resolution_time
            )
            
        finally:
            session.close()
    
    def get_violation_trends(
        self, 
        timeframe: TimeFrame = TimeFrame.LAST_30D,
        portfolio_id: Optional[str] = None
    ) -> ViolationTrend:
        """
        Get trend analysis for violations over time
        """
        session = self.db_manager.SessionLocal()
        try:
            start_date = self._get_start_date(timeframe)
            
            # Base query
            query = session.query(ComplianceViolation).filter(
                ComplianceViolation.detected_at >= start_date
            )
            
            if portfolio_id:
                query = query.filter(ComplianceViolation.portfolio_id == portfolio_id)
            
            violations = query.all()
            
            # Create daily buckets
            daily_counts = defaultdict(int)
            severity_trends = defaultdict(lambda: defaultdict(int))
            category_trends = defaultdict(lambda: defaultdict(int))
            
            for v in violations:
                day = v.detected_at.date()
                daily_counts[day] += 1
                severity_trends[v.severity][day] += 1
                
                category = self._categorize_violation(v.violation_type)
                category_trends[category.value][day] += 1
            
            # Convert to lists of tuples for easier plotting
            violation_counts = sorted(daily_counts.items())
            
            # Convert trends to proper format
            severity_trend_data = {}
            for severity, daily_data in severity_trends.items():
                severity_trend_data[severity] = sorted(daily_data.items())
            
            category_trend_data = {}
            for category, daily_data in category_trends.items():
                category_trend_data[category] = sorted(daily_data.items())
            
            return ViolationTrend(
                timeframe=timeframe,
                violation_counts=violation_counts,
                severity_trends=severity_trend_data,
                category_trends=category_trend_data
            )
            
        finally:
            session.close()
    
    def get_dashboard_widgets(
        self, 
        portfolio_id: Optional[str] = None
    ) -> List[DashboardWidget]:
        """
        Generate dashboard widgets for violation monitoring
        """
        widgets = []
        
        # Widget 1: Current Violation Count
        summary = self.get_violation_summary(TimeFrame.LAST_24H, portfolio_id)
        widgets.append(DashboardWidget(
            widget_id="current_violations",
            widget_type="metric",
            title="Current Violations (24h)",
            description="Total compliance violations in the last 24 hours",
            data={
                "value": summary.total_violations,
                "change": self._calculate_change(TimeFrame.LAST_24H, portfolio_id),
                "severity_breakdown": summary.violations_by_severity
            },
            last_updated=datetime.now()
        ))
        
        # Widget 2: Violation Severity Distribution
        widgets.append(DashboardWidget(
            widget_id="severity_distribution",
            widget_type="chart",
            title="Violation Severity Distribution",
            description="Distribution of violations by severity level",
            data={
                "chart_type": "pie",
                "data": summary.violations_by_severity,
                "colors": {
                    "critical": "#dc2626",
                    "high": "#ea580c", 
                    "medium": "#d97706",
                    "low": "#65a30d"
                }
            },
            last_updated=datetime.now()
        ))
        
        # Widget 3: Top Violated Rules
        widgets.append(DashboardWidget(
            widget_id="top_violated_rules",
            widget_type="table",
            title="Most Violated Rules",
            description="Rules with the highest violation frequency",
            data={
                "headers": ["Rule ID", "Violation Count", "Last Violation"],
                "rows": self._format_rule_table(summary.most_violated_rules[:5])
            },
            last_updated=datetime.now()
        ))
        
        # Widget 4: Violation Trends
        trends = self.get_violation_trends(TimeFrame.LAST_7D, portfolio_id)
        widgets.append(DashboardWidget(
            widget_id="violation_trends",
            widget_type="chart",
            title="Violation Trends (7 days)",
            description="Daily violation count over the last week",
            data={
                "chart_type": "line",
                "x_axis": [d.strftime("%m/%d") for d, _ in trends.violation_counts],
                "y_axis": [count for _, count in trends.violation_counts],
                "trend": "up" if len(trends.violation_counts) > 1 and 
                         trends.violation_counts[-1][1] > trends.violation_counts[0][1] else "down"
            },
            last_updated=datetime.now()
        ))
        
        # Widget 5: Resolution Rate
        widgets.append(DashboardWidget(
            widget_id="resolution_rate",
            widget_type="metric",
            title="Resolution Rate",
            description="Percentage of violations that have been resolved",
            data={
                "value": f"{summary.resolution_rate:.1%}",
                "average_resolution_time": summary.average_resolution_time,
                "target": "95%"
            },
            last_updated=datetime.now()
        ))
        
        # Widget 6: Category Breakdown
        widgets.append(DashboardWidget(
            widget_id="category_breakdown",
            widget_type="chart",
            title="Violations by Category",
            description="Breakdown of violations by compliance category",
            data={
                "chart_type": "bar",
                "data": summary.violations_by_category
            },
            last_updated=datetime.now()
        ))
        
        return widgets
    
    def generate_violation_report(
        self, 
        timeframe: TimeFrame = TimeFrame.LAST_30D,
        portfolio_id: Optional[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive violation report
        """
        summary = self.get_violation_summary(timeframe, portfolio_id)
        trends = self.get_violation_trends(timeframe, portfolio_id)
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "timeframe": timeframe.value,
                "portfolio_id": portfolio_id,
                "report_type": "compliance_violation_analysis"
            },
            "executive_summary": {
                "total_violations": summary.total_violations,
                "resolution_rate": summary.resolution_rate,
                "most_critical_issues": self._get_critical_issues(summary),
                "recommendation": self._generate_recommendation(summary)
            },
            "detailed_analysis": {
                "violation_summary": summary.__dict__,
                "trend_analysis": trends.__dict__,
                "risk_assessment": self._assess_compliance_risk(summary, trends)
            },
            "dashboard_data": {
                "widgets": [w.__dict__ for w in self.get_dashboard_widgets(portfolio_id)]
            }
        }
        
        return report
    
    # Helper methods
    
    def _get_start_date(self, timeframe: TimeFrame) -> datetime:
        """Calculate start date based on timeframe"""
        now = datetime.now()
        if timeframe == TimeFrame.LAST_24H:
            return now - timedelta(hours=24)
        elif timeframe == TimeFrame.LAST_7D:
            return now - timedelta(days=7)
        elif timeframe == TimeFrame.LAST_30D:
            return now - timedelta(days=30)
        elif timeframe == TimeFrame.LAST_90D:
            return now - timedelta(days=90)
        elif timeframe == TimeFrame.LAST_YEAR:
            return now - timedelta(days=365)
        else:
            return now - timedelta(days=30)  # default
    
    def _categorize_violation(self, violation_type: str) -> ViolationCategory:
        """Categorize violation based on type"""
        category_mapping = {
            "position_limit": ViolationCategory.POSITION_LIMITS,
            "sector_limit": ViolationCategory.POSITION_LIMITS,
            "geographic_limit": ViolationCategory.POSITION_LIMITS,
            "excluded_asset": ViolationCategory.POSITION_LIMITS,
            "esg_score": ViolationCategory.ESG_REQUIREMENTS,
            "esg_exclusion": ViolationCategory.ESG_REQUIREMENTS,
            "credit_rating": ViolationCategory.INVESTMENT_MANDATES,
            "liquidity": ViolationCategory.RISK_MANAGEMENT,
            "var_limit": ViolationCategory.RISK_MANAGEMENT,
            "concentration": ViolationCategory.RISK_MANAGEMENT
        }
        
        return category_mapping.get(violation_type, ViolationCategory.OPERATIONAL_RISKS)
    
    def _get_daily_breakdown(self, violations: List, timeframe: TimeFrame) -> Dict[str, int]:
        """Get daily breakdown of violations"""
        daily_counts = defaultdict(int)
        for v in violations:
            day_key = v.detected_at.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1
        
        return dict(daily_counts)
    
    def _calculate_change(self, timeframe: TimeFrame, portfolio_id: Optional[str]) -> float:
        """Calculate percentage change compared to previous period"""
        # Implementation would compare current period to previous period
        # For now, return mock change
        return 5.2  # +5.2% change
    
    def _format_rule_table(self, rule_data: List[Tuple[str, int]]) -> List[List[str]]:
        """Format rule violation data for table display"""
        rows = []
        for rule_id, count in rule_data:
            rows.append([
                rule_id,
                str(count),
                "2024-08-20 14:30"  # Mock last violation time
            ])
        return rows
    
    def _get_critical_issues(self, summary: ViolationSummary) -> List[str]:
        """Identify critical compliance issues"""
        issues = []
        
        if summary.violations_by_severity.get("critical", 0) > 0:
            issues.append(f"{summary.violations_by_severity['critical']} critical violations require immediate attention")
        
        if summary.resolution_rate < 0.8:
            issues.append(f"Low resolution rate ({summary.resolution_rate:.1%}) indicates process inefficiency")
        
        if summary.total_violations > 50:
            issues.append("High violation volume may indicate systemic compliance issues")
        
        return issues or ["No critical issues identified"]
    
    def _generate_recommendation(self, summary: ViolationSummary) -> str:
        """Generate actionable recommendations based on summary"""
        if summary.total_violations == 0:
            return "Excellent compliance performance. Continue current practices."
        
        if summary.violations_by_severity.get("critical", 0) > 0:
            return "Immediate action required on critical violations. Review and update compliance controls."
        
        if summary.resolution_rate < 0.9:
            return "Focus on improving violation resolution processes and response times."
        
        return "Monitor ongoing compliance trends and consider preventive measures for recurring violations."
    
    def _assess_compliance_risk(self, summary: ViolationSummary, trends: ViolationTrend) -> Dict[str, Any]:
        """Assess overall compliance risk level"""
        risk_score = 0
        
        # Factor in violation volume
        if summary.total_violations > 100:
            risk_score += 30
        elif summary.total_violations > 50:
            risk_score += 20
        elif summary.total_violations > 10:
            risk_score += 10
        
        # Factor in severity
        critical_count = summary.violations_by_severity.get("critical", 0)
        high_count = summary.violations_by_severity.get("high", 0)
        risk_score += critical_count * 10 + high_count * 5
        
        # Factor in resolution rate
        if summary.resolution_rate < 0.7:
            risk_score += 25
        elif summary.resolution_rate < 0.9:
            risk_score += 10
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "key_risk_factors": self._identify_risk_factors(summary, trends)
        }
    
    def _identify_risk_factors(self, summary: ViolationSummary, trends: ViolationTrend) -> List[str]:
        """Identify key risk factors"""
        factors = []
        
        if summary.violations_by_severity.get("critical", 0) > 0:
            factors.append("Critical severity violations present")
        
        if summary.resolution_rate < 0.8:
            factors.append("Poor violation resolution rate")
        
        if len(trends.violation_counts) > 0 and trends.violation_counts[-1][1] > 10:
            factors.append("High recent violation volume")
        
        return factors or ["No significant risk factors identified"]


# Global reporter instance
violation_reporter = ComplianceViolationReporter()
