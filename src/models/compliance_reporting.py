"""
Enhanced Compliance Models for Violation Reporting
================================================

Extended models for comprehensive violation reporting,
categorization, and dashboard functionality.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

from .compliance import ViolationDetails, ViolationSeverity


class ViolationCategoryType(str, Enum):
    """Enhanced violation categories for reporting"""
    POSITION_LIMITS = "position_limits"
    INVESTMENT_MANDATES = "investment_mandates"
    RISK_MANAGEMENT = "risk_management"
    ESG_REQUIREMENTS = "esg_requirements"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    OPERATIONAL_RISKS = "operational_risks"


class ReportTimeFrame(str, Enum):
    """Time frames for violation reporting"""
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"
    LAST_90D = "last_90d"
    LAST_YEAR = "last_year"


class WidgetType(str, Enum):
    """Dashboard widget types"""
    METRIC = "metric"
    CHART = "chart"
    TABLE = "table"
    ALERT = "alert"
    GAUGE = "gauge"


class ChartType(str, Enum):
    """Chart types for dashboard widgets"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"


class ViolationReportRequest(BaseModel):
    """Request model for violation reports"""
    timeframe: ReportTimeFrame = ReportTimeFrame.LAST_30D
    portfolio_id: Optional[str] = None
    include_resolved: bool = True
    severity_filter: Optional[List[ViolationSeverity]] = None
    category_filter: Optional[List[ViolationCategoryType]] = None
    format: str = Field(default="json", pattern="^(json|pdf|excel)$")


class ViolationStatistics(BaseModel):
    """Statistical summary of violations"""
    total_violations: int
    violations_by_severity: Dict[ViolationSeverity, int]
    violations_by_category: Dict[ViolationCategoryType, int]
    violations_by_day: Dict[str, int]  # ISO date string -> count
    resolution_rate: float = Field(ge=0, le=1)
    average_resolution_time_hours: Optional[float] = None
    most_violated_rules: List[Dict[str, Union[str, int]]]
    affected_portfolios_count: int


class ViolationTrendData(BaseModel):
    """Trend analysis data for violations"""
    timeframe: ReportTimeFrame
    daily_counts: List[Dict[str, Union[str, int]]]  # [{"date": "2024-01-01", "count": 5}]
    severity_trends: Dict[ViolationSeverity, List[Dict[str, Union[str, int]]]]
    category_trends: Dict[ViolationCategoryType, List[Dict[str, Union[str, int]]]]
    trend_direction: str = Field(pattern="^(up|down|stable)$")
    percentage_change: float


class DashboardWidgetData(BaseModel):
    """Data structure for dashboard widgets"""
    widget_id: str
    widget_type: WidgetType
    title: str
    description: str
    data: Dict[str, Any]
    last_updated: datetime
    refresh_interval: int = Field(default=300, ge=60)  # minimum 1 minute
    
    @validator('data')
    def validate_widget_data(cls, v, values):
        """Validate widget data based on widget type"""
        widget_type = values.get('widget_type')
        
        if widget_type == WidgetType.METRIC:
            required_fields = ['value']
            if not all(field in v for field in required_fields):
                raise ValueError(f"Metric widget requires fields: {required_fields}")
        
        elif widget_type == WidgetType.CHART:
            required_fields = ['chart_type']
            if not all(field in v for field in required_fields):
                raise ValueError(f"Chart widget requires fields: {required_fields}")
        
        elif widget_type == WidgetType.TABLE:
            required_fields = ['headers', 'rows']
            if not all(field in v for field in required_fields):
                raise ValueError(f"Table widget requires fields: {required_fields}")
        
        return v


class ComplianceAlertConfig(BaseModel):
    """Configuration for compliance alerts"""
    alert_id: str
    alert_type: str = Field(pattern="^(threshold|trend|anomaly)$")
    description: str
    severity_threshold: ViolationSeverity
    violation_count_threshold: int = Field(gt=0)
    time_window_minutes: int = Field(default=60, gt=0)
    notification_channels: List[str] = ["email", "dashboard"]
    is_active: bool = True


class ComplianceAlert(BaseModel):
    """Active compliance alert"""
    alert_id: str
    alert_config_id: str
    triggered_at: datetime
    message: str
    severity: ViolationSeverity
    affected_portfolios: List[str]
    violation_count: int
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class ViolationReportResponse(BaseModel):
    """Comprehensive violation report response"""
    report_metadata: Dict[str, Any]
    executive_summary: Dict[str, Any]
    violation_statistics: ViolationStatistics
    trend_analysis: ViolationTrendData
    dashboard_widgets: List[DashboardWidgetData]
    active_alerts: List[ComplianceAlert]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]


class DashboardConfig(BaseModel):
    """Configuration for compliance dashboard"""
    dashboard_id: str
    dashboard_name: str
    widget_layout: List[Dict[str, Any]]  # Grid layout configuration
    refresh_interval: int = Field(default=300, ge=60)
    auto_refresh: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    user_permissions: List[str] = ["view"]


class ViolationExportRequest(BaseModel):
    """Request for exporting violation data"""
    timeframe: ReportTimeFrame
    portfolio_id: Optional[str] = None
    export_format: str = Field(pattern="^(csv|excel|pdf)$")
    include_charts: bool = True
    include_summary: bool = True
    include_raw_data: bool = False
    email_recipients: Optional[List[str]] = None


class ComplianceMetrics(BaseModel):
    """Key compliance performance metrics"""
    compliance_score: float = Field(ge=0, le=1)
    violation_trend: str = Field(pattern="^(improving|deteriorating|stable)$")
    resolution_efficiency: float = Field(ge=0, le=1)
    rule_effectiveness: Dict[str, float]
    portfolio_risk_scores: Dict[str, float]
    benchmark_comparison: Optional[Dict[str, float]] = None


class ViolationEnrichmentData(BaseModel):
    """Additional context data for violations"""
    violation_id: str
    business_impact: str = Field(pattern="^(low|medium|high|critical)$")
    estimated_cost: Optional[float] = None
    remediation_effort: str = Field(pattern="^(low|medium|high)$")
    related_violations: List[str] = Field(default_factory=list)
    regulatory_citations: List[str] = Field(default_factory=list)
    stakeholder_notifications: List[str] = Field(default_factory=list)


class ViolationWorkflowStatus(BaseModel):
    """Workflow status for violation resolution"""
    violation_id: str
    workflow_stage: str = Field(pattern="^(detected|triaged|investigating|resolving|resolved|closed)$")
    assigned_to: Optional[str] = None
    priority: str = Field(pattern="^(low|medium|high|urgent)$")
    due_date: Optional[datetime] = None
    progress_notes: List[Dict[str, str]] = Field(default_factory=list)
    approvals_required: List[str] = Field(default_factory=list)
    escalation_rules: Dict[str, Any] = Field(default_factory=dict)


# Enhanced violation details with reporting context
class EnhancedViolationDetails(ViolationDetails):
    """Extended violation details with reporting and workflow context"""
    category: ViolationCategoryType
    business_impact: str = "medium"
    estimated_cost: Optional[float] = None
    workflow_status: Optional[ViolationWorkflowStatus] = None
    enrichment_data: Optional[ViolationEnrichmentData] = None
    related_portfolio_metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


# API Response Models
class ViolationDashboardResponse(BaseModel):
    """Response for dashboard data"""
    widgets: List[DashboardWidgetData]
    alerts: List[ComplianceAlert]
    last_updated: datetime
    dashboard_config: DashboardConfig


class ViolationSummaryResponse(BaseModel):
    """Response for violation summary endpoint"""
    summary: ViolationStatistics
    trends: ViolationTrendData
    top_violations: List[EnhancedViolationDetails]
    compliance_score: float
    generated_at: datetime


class ViolationAnalyticsResponse(BaseModel):
    """Response for advanced violation analytics"""
    time_series_data: Dict[str, List[Dict[str, Any]]]
    correlation_analysis: Dict[str, float]
    predictive_metrics: Dict[str, Any]
    benchmark_comparisons: Dict[str, float]
    risk_factors: List[str]
    recommended_actions: List[str]


# WebSocket message types for real-time updates
class ViolationUpdateMessage(BaseModel):
    """WebSocket message for real-time violation updates"""
    message_type: str = "violation_update"
    violation: EnhancedViolationDetails
    timestamp: datetime
    affected_dashboards: List[str]


class AlertMessage(BaseModel):
    """WebSocket message for compliance alerts"""
    message_type: str = "compliance_alert"
    alert: ComplianceAlert
    timestamp: datetime
    priority: str = Field(pattern="^(low|medium|high|urgent)$")
