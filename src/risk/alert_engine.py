"""
Risk Alert Engine
Implements configurable alert thresholds and dynamic threshold checking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json

from .realtime_monitor import RiskMetricsSnapshot
from ..utils.professional_logging import log_risk_event

# Configure logging
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Risk metric types for alerts"""
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"
    VAR_99 = "var_99"
    CVAR_99 = "cvar_99"
    MAX_DRAWDOWN = "max_drawdown"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    CORRELATION = "correlation"


class AlertStatus(Enum):
    """Alert status tracking"""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    id: str
    user_id: str
    portfolio_id: Optional[str]  # None for global thresholds
    metric_type: MetricType
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<='
    severity: AlertSeverity
    is_active: bool
    description: str
    created_at: datetime
    updated_at: datetime


@dataclass
class RiskAlert:
    """Risk alert event"""
    id: str
    threshold_id: str
    portfolio_id: str
    metric_type: MetricType
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    status: AlertStatus
    description: str
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    acknowledgment_user: Optional[str] = None
    resolution_note: Optional[str] = None


@dataclass
class AlertRule:
    """Complex alert rule with conditions"""
    id: str
    name: str
    conditions: List[Dict[str, Any]]  # Multiple conditions that must be met
    severity: AlertSeverity
    cooldown_minutes: int  # Minimum time between alerts
    is_active: bool


class AlertEngine:
    """
    Risk alert engine with configurable thresholds and dynamic checking
    """
    
    def __init__(self):
        """Initialize alert engine"""
        self.thresholds: Dict[str, AlertThreshold] = {}
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        self.alert_history: List[RiskAlert] = []
        
        # Configuration
        self.max_alerts_per_minute = 10
        self.alert_suppression_window = 300  # 5 minutes
        self.auto_resolve_timeout = 3600  # 1 hour
        
        logger.info("AlertEngine initialized")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Remove callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def create_threshold(self, 
                        user_id: str,
                        metric_type: MetricType,
                        threshold_value: float,
                        comparison_operator: str = '>',
                        severity: AlertSeverity = AlertSeverity.MEDIUM,
                        portfolio_id: Optional[str] = None,
                        description: str = "") -> AlertThreshold:
        """
        Create a new alert threshold
        
        Args:
            user_id: User creating the threshold
            metric_type: Risk metric to monitor
            threshold_value: Threshold value to trigger alert
            comparison_operator: Comparison operator ('>', '<', '>=', '<=')
            severity: Alert severity level
            portfolio_id: Specific portfolio (None for global)
            description: Human-readable description
            
        Returns:
            Created AlertThreshold
        """
        threshold_id = f"threshold_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.thresholds)}"
        
        threshold = AlertThreshold(
            id=threshold_id,
            user_id=user_id,
            portfolio_id=portfolio_id,
            metric_type=metric_type,
            threshold_value=threshold_value,
            comparison_operator=comparison_operator,
            severity=severity,
            is_active=True,
            description=description or f"{metric_type.value} {comparison_operator} {threshold_value}",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.thresholds[threshold_id] = threshold
        
        log_risk_event(
            portfolio_id=portfolio_id or "global",
            event_type="alert_threshold_created",
            event_data={
                "threshold_id": threshold_id,
                "metric_type": metric_type.value,
                "threshold_value": threshold_value,
                "severity": severity.value
            }
        )
        
        logger.info(f"Created alert threshold: {threshold_id}")
        return threshold
    
    def update_threshold(self, threshold_id: str, **kwargs) -> Optional[AlertThreshold]:
        """Update an existing threshold"""
        if threshold_id not in self.thresholds:
            return None
        
        threshold = self.thresholds[threshold_id]
        
        # Update allowed fields
        for field, value in kwargs.items():
            if hasattr(threshold, field):
                setattr(threshold, field, value)
        
        threshold.updated_at = datetime.now()
        
        logger.info(f"Updated alert threshold: {threshold_id}")
        return threshold
    
    def delete_threshold(self, threshold_id: str) -> bool:
        """Delete an alert threshold"""
        if threshold_id in self.thresholds:
            del self.thresholds[threshold_id]
            logger.info(f"Deleted alert threshold: {threshold_id}")
            return True
        return False
    
    def get_thresholds(self, 
                      user_id: Optional[str] = None,
                      portfolio_id: Optional[str] = None,
                      metric_type: Optional[MetricType] = None,
                      active_only: bool = True) -> List[AlertThreshold]:
        """Get thresholds with optional filtering"""
        thresholds = list(self.thresholds.values())
        
        if user_id:
            thresholds = [t for t in thresholds if t.user_id == user_id]
        
        if portfolio_id:
            thresholds = [t for t in thresholds if t.portfolio_id == portfolio_id]
        
        if metric_type:
            thresholds = [t for t in thresholds if t.metric_type == metric_type]
        
        if active_only:
            thresholds = [t for t in thresholds if t.is_active]
        
        return sorted(thresholds, key=lambda x: x.created_at, reverse=True)
    
    async def check_thresholds(self, snapshot: RiskMetricsSnapshot) -> List[RiskAlert]:
        """
        Check risk metrics against configured thresholds
        
        Args:
            snapshot: Risk metrics snapshot to check
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        # Get relevant thresholds for this portfolio
        relevant_thresholds = []
        
        # Global thresholds
        relevant_thresholds.extend([
            t for t in self.thresholds.values()
            if t.is_active and t.portfolio_id is None
        ])
        
        # Portfolio-specific thresholds
        relevant_thresholds.extend([
            t for t in self.thresholds.values()
            if t.is_active and t.portfolio_id == snapshot.portfolio_id
        ])
        
        # Check each threshold
        for threshold in relevant_thresholds:
            try:
                current_value = self._extract_metric_value(snapshot, threshold.metric_type)
                
                if current_value is None:
                    continue
                
                # Check if threshold is breached
                if self._evaluate_threshold(current_value, threshold.threshold_value, threshold.comparison_operator):
                    
                    # Check if this alert is already active and not resolved
                    existing_alert = self._find_active_alert(threshold.id, snapshot.portfolio_id)
                    
                    if existing_alert is None:
                        # Create new alert
                        alert = await self._create_alert(threshold, snapshot, current_value)
                        triggered_alerts.append(alert)
                    else:
                        # Update existing alert
                        existing_alert.current_value = current_value
                        existing_alert.triggered_at = datetime.now()
                
            except Exception as e:
                logger.error(f"Error checking threshold {threshold.id}: {e}")
        
        return triggered_alerts
    
    def _extract_metric_value(self, snapshot: RiskMetricsSnapshot, metric_type: MetricType) -> Optional[float]:
        """Extract metric value from snapshot"""
        metric_map = {
            MetricType.VAR_95: snapshot.var_95,
            MetricType.CVAR_95: snapshot.cvar_95,
            MetricType.VAR_99: snapshot.var_99,
            MetricType.CVAR_99: snapshot.cvar_99,
            MetricType.MAX_DRAWDOWN: snapshot.max_drawdown,
            MetricType.LEVERAGE: snapshot.leverage_ratio,
            MetricType.VOLATILITY: snapshot.annual_volatility,
            MetricType.SHARPE_RATIO: snapshot.sharpe_ratio,
            MetricType.CONCENTRATION: snapshot.concentration_risk.get('max_position', 0.0)
        }
        
        return metric_map.get(metric_type)
    
    def _evaluate_threshold(self, current_value: float, threshold_value: float, operator: str) -> bool:
        """Evaluate if threshold is breached"""
        if operator == '>':
            return current_value > threshold_value
        elif operator == '<':
            return current_value < threshold_value
        elif operator == '>=':
            return current_value >= threshold_value
        elif operator == '<=':
            return current_value <= threshold_value
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    
    def _find_active_alert(self, threshold_id: str, portfolio_id: str) -> Optional[RiskAlert]:
        """Find active alert for threshold and portfolio"""
        for alert in self.active_alerts.values():
            if (alert.threshold_id == threshold_id and 
                alert.portfolio_id == portfolio_id and 
                alert.status == AlertStatus.TRIGGERED):
                return alert
        return None
    
    async def _create_alert(self, threshold: AlertThreshold, snapshot: RiskMetricsSnapshot, current_value: float) -> RiskAlert:
        """Create a new alert"""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_alerts)}"
        
        alert = RiskAlert(
            id=alert_id,
            threshold_id=threshold.id,
            portfolio_id=snapshot.portfolio_id,
            metric_type=threshold.metric_type,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            severity=threshold.severity,
            status=AlertStatus.TRIGGERED,
            description=f"{threshold.description} - Current: {current_value:.4f}, Threshold: {threshold.threshold_value:.4f}",
            triggered_at=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert event
        log_risk_event(
            portfolio_id=snapshot.portfolio_id,
            event_type="risk_alert_triggered",
            event_data={
                "alert_id": alert_id,
                "metric_type": threshold.metric_type.value,
                "current_value": current_value,
                "threshold_value": threshold.threshold_value,
                "severity": threshold.severity.value
            }
        )
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Risk alert triggered: {alert_id} - {alert.description}")
        return alert
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledgment_user = user_id
            
            log_risk_event(
                portfolio_id=alert.portfolio_id,
                event_type="risk_alert_acknowledged",
                event_data={
                    "alert_id": alert_id,
                    "acknowledged_by": user_id
                }
            )
            
            logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_note: str = "") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.resolution_note = resolution_note
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            log_risk_event(
                portfolio_id=alert.portfolio_id,
                event_type="risk_alert_resolved",
                event_data={
                    "alert_id": alert_id,
                    "resolved_by": user_id,
                    "resolution_note": resolution_note
                }
            )
            
            logger.info(f"Alert resolved: {alert_id} by {user_id}")
            return True
        return False
    
    def get_active_alerts(self, 
                         portfolio_id: Optional[str] = None,
                         severity: Optional[AlertSeverity] = None) -> List[RiskAlert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if portfolio_id:
            alerts = [a for a in alerts if a.portfolio_id == portfolio_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_history(self, 
                         portfolio_id: Optional[str] = None,
                         hours: int = 24,
                         limit: int = 100) -> List[RiskAlert]:
        """Get alert history with filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            a for a in self.alert_history
            if a.triggered_at >= cutoff_time
        ]
        
        if portfolio_id:
            alerts = [a for a in alerts if a.portfolio_id == portfolio_id]
        
        alerts = sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
        
        return alerts[:limit]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            a for a in self.alert_history
            if a.triggered_at >= cutoff_time
        ]
        
        stats = {
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': {},
            'alerts_by_metric': {},
            'alerts_by_portfolio': {},
            'average_resolution_time': None
        }
        
        # Count by severity
        for severity in AlertSeverity:
            stats['alerts_by_severity'][severity.value] = len([
                a for a in recent_alerts if a.severity == severity
            ])
        
        # Count by metric type
        for metric in MetricType:
            stats['alerts_by_metric'][metric.value] = len([
                a for a in recent_alerts if a.metric_type == metric
            ])
        
        # Count by portfolio
        for alert in recent_alerts:
            portfolio_id = alert.portfolio_id
            if portfolio_id not in stats['alerts_by_portfolio']:
                stats['alerts_by_portfolio'][portfolio_id] = 0
            stats['alerts_by_portfolio'][portfolio_id] += 1
        
        # Calculate average resolution time
        resolved_alerts = [
            a for a in recent_alerts 
            if a.status == AlertStatus.RESOLVED and a.resolved_at
        ]
        
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.triggered_at).total_seconds()
                for a in resolved_alerts
            ]
            stats['average_resolution_time'] = sum(resolution_times) / len(resolution_times)
        
        return stats


# Create common alert thresholds
def create_default_thresholds(alert_engine: AlertEngine, user_id: str) -> List[AlertThreshold]:
    """Create default alert thresholds for risk monitoring"""
    
    default_configs = [
        # High severity thresholds
        {
            'metric_type': MetricType.VAR_95,
            'threshold_value': -0.05,  # -5% daily VaR
            'comparison_operator': '<',
            'severity': AlertSeverity.HIGH,
            'description': 'Daily VaR (95%) exceeds -5%'
        },
        {
            'metric_type': MetricType.MAX_DRAWDOWN,
            'threshold_value': -0.20,  # -20% max drawdown
            'comparison_operator': '<',
            'severity': AlertSeverity.CRITICAL,
            'description': 'Maximum drawdown exceeds -20%'
        },
        {
            'metric_type': MetricType.CONCENTRATION,
            'threshold_value': 0.30,  # 30% single position
            'comparison_operator': '>',
            'severity': AlertSeverity.MEDIUM,
            'description': 'Single position concentration exceeds 30%'
        },
        {
            'metric_type': MetricType.LEVERAGE,
            'threshold_value': 1.5,  # 150% leverage
            'comparison_operator': '>',
            'severity': AlertSeverity.HIGH,
            'description': 'Portfolio leverage exceeds 150%'
        },
        {
            'metric_type': MetricType.VOLATILITY,
            'threshold_value': 0.25,  # 25% annual volatility
            'comparison_operator': '>',
            'severity': AlertSeverity.MEDIUM,
            'description': 'Annual volatility exceeds 25%'
        }
    ]
    
    thresholds = []
    for config in default_configs:
        threshold = alert_engine.create_threshold(user_id, **config)
        thresholds.append(threshold)
    
    return thresholds


# Example usage and testing
async def test_alert_engine():
    """Test the alert engine"""
    from .realtime_monitor import RiskMetricsSnapshot
    
    # Initialize alert engine
    alert_engine = AlertEngine()
    
    # Create default thresholds
    thresholds = create_default_thresholds(alert_engine, "test_user")
    print(f"Created {len(thresholds)} default thresholds")
    
    # Create test snapshot that triggers alerts
    snapshot = RiskMetricsSnapshot(
        timestamp=datetime.now(),
        portfolio_id="test_portfolio",
        var_95=-0.06,  # Exceeds threshold
        cvar_95=-0.08,
        var_99=-0.10,
        cvar_99=-0.12,
        max_drawdown=-0.25,  # Exceeds threshold
        sharpe_ratio=0.8,
        sortino_ratio=1.2,
        calmar_ratio=0.5,
        annual_volatility=0.30,  # Exceeds threshold
        daily_volatility=0.019,
        correlation_matrix={},
        concentration_risk={'max_position': 0.35},  # Exceeds threshold
        leverage_ratio=1.6  # Exceeds threshold
    )
    
    # Check thresholds
    alerts = await alert_engine.check_thresholds(snapshot)
    print(f"Triggered {len(alerts)} alerts:")
    
    for alert in alerts:
        print(f"  - {alert.id}: {alert.description} (Severity: {alert.severity.value})")
    
    # Get statistics
    stats = alert_engine.get_alert_statistics()
    print(f"\nAlert Statistics:")
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Active alerts: {stats['active_alerts']}")
    print(f"  By severity: {stats['alerts_by_severity']}")


if __name__ == "__main__":
    asyncio.run(test_alert_engine())
