"""
Alert Escalation Manager
Implements automated escalation workflows for different severity levels
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json

from ..risk.alert_engine import RiskAlert, AlertSeverity, AlertStatus
from ..utils.professional_logging import log_risk_event

# Configure logging
logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Escalation levels"""
    LEVEL_1 = "level_1"  # Initial alert
    LEVEL_2 = "level_2"  # Supervisor notification
    LEVEL_3 = "level_3"  # Management notification
    LEVEL_4 = "level_4"  # Executive notification


class UserRole(Enum):
    """User roles for escalation paths"""
    ANALYST = "analyst"
    SENIOR_ANALYST = "senior_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"
    HEAD_OF_RISK = "head_of_risk"
    CIO = "cio"
    CEO = "ceo"


@dataclass
class EscalationRule:
    """Escalation rule configuration"""
    id: str
    name: str
    severity: AlertSeverity
    escalation_path: List[Dict[str, Any]]  # [{"level": EscalationLevel, "roles": [UserRole], "delay_minutes": int}]
    auto_escalate: bool
    max_escalation_level: EscalationLevel
    require_acknowledgment: bool
    escalation_conditions: List[str]  # Conditions that trigger escalation
    is_active: bool
    created_at: datetime


@dataclass
class EscalationEvent:
    """Escalation event tracking"""
    id: str
    alert_id: str
    escalation_rule_id: str
    from_level: EscalationLevel
    to_level: EscalationLevel
    escalated_at: datetime
    escalated_by: Optional[str]  # User ID or "system" for auto-escalation
    reason: str
    notification_sent: bool
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class UserEscalationContact:
    """User contact information for escalations"""
    user_id: str
    role: UserRole
    email: Optional[str] = None
    phone: Optional[str] = None
    backup_users: List[str] = None  # Backup user IDs
    is_available: bool = True
    timezone: str = "UTC"
    working_hours_start: Optional[str] = None  # HH:MM format
    working_hours_end: Optional[str] = None


class EscalationManager:
    """
    Alert escalation manager with configurable workflows
    """
    
    def __init__(self):
        """Initialize escalation manager"""
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.user_contacts: Dict[str, UserEscalationContact] = {}
        self.escalation_events: List[EscalationEvent] = []
        self.active_escalations: Dict[str, List[EscalationEvent]] = {}  # alert_id -> escalation events
        
        # Callbacks for escalation actions
        self.escalation_callbacks: List[Callable[[EscalationEvent], None]] = []
        
        # Configuration
        self.check_interval = 60  # Check escalations every minute
        self.max_auto_escalations_per_hour = 20
        
        # Create default escalation rules
        self._create_default_rules()
        
        logger.info("EscalationManager initialized")
    
    def add_escalation_callback(self, callback: Callable[[EscalationEvent], None]):
        """Add callback function for escalation events"""
        self.escalation_callbacks.append(callback)
    
    def remove_escalation_callback(self, callback: Callable[[EscalationEvent], None]):
        """Remove callback function"""
        if callback in self.escalation_callbacks:
            self.escalation_callbacks.remove(callback)
    
    def create_escalation_rule(self, 
                              name: str,
                              severity: AlertSeverity,
                              escalation_path: List[Dict[str, Any]],
                              auto_escalate: bool = True,
                              require_acknowledgment: bool = True) -> EscalationRule:
        """
        Create a new escalation rule
        
        Args:
            name: Rule name
            severity: Alert severity this rule applies to
            escalation_path: List of escalation steps
            auto_escalate: Whether to auto-escalate
            require_acknowledgment: Whether acknowledgment is required
            
        Returns:
            Created EscalationRule
        """
        rule_id = f"escalation_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.escalation_rules)}"
        
        rule = EscalationRule(
            id=rule_id,
            name=name,
            severity=severity,
            escalation_path=escalation_path,
            auto_escalate=auto_escalate,
            max_escalation_level=EscalationLevel.LEVEL_4,
            require_acknowledgment=require_acknowledgment,
            escalation_conditions=["unacknowledged", "time_threshold"],
            is_active=True,
            created_at=datetime.now()
        )
        
        self.escalation_rules[rule_id] = rule
        
        log_risk_event(
            portfolio_id="system",
            event_type="escalation_rule_created",
            event_data={
                "rule_id": rule_id,
                "severity": severity.value,
                "auto_escalate": auto_escalate
            }
        )
        
        logger.info(f"Created escalation rule: {rule_id}")
        return rule
    
    def add_user_contact(self, user_contact: UserEscalationContact):
        """Add user contact information"""
        self.user_contacts[user_contact.user_id] = user_contact
        logger.info(f"Added user contact: {user_contact.user_id} ({user_contact.role.value})")
    
    def get_escalation_rule(self, severity: AlertSeverity) -> Optional[EscalationRule]:
        """Get escalation rule for alert severity"""
        for rule in self.escalation_rules.values():
            if rule.severity == severity and rule.is_active:
                return rule
        return None
    
    async def process_alert_for_escalation(self, alert: RiskAlert):
        """
        Process an alert for potential escalation
        
        Args:
            alert: Risk alert to process
        """
        # Get escalation rule for this severity
        rule = self.get_escalation_rule(alert.severity)
        if not rule:
            logger.debug(f"No escalation rule found for severity {alert.severity.value}")
            return
        
        # Initialize escalation tracking if not exists
        if alert.id not in self.active_escalations:
            self.active_escalations[alert.id] = []
            
            # Create initial escalation event (Level 1)
            if rule.escalation_path:
                await self._create_escalation_event(
                    alert, rule, EscalationLevel.LEVEL_1, EscalationLevel.LEVEL_1,
                    "Initial alert triggered", "system"
                )
    
    async def check_escalations(self):
        """Check all active alerts for escalation conditions"""
        current_time = datetime.now()
        
        for alert_id, escalation_events in self.active_escalations.items():
            if not escalation_events:
                continue
            
            latest_escalation = max(escalation_events, key=lambda x: x.escalated_at)
            
            # Skip if already at max level or acknowledged
            if (latest_escalation.to_level == EscalationLevel.LEVEL_4 or
                latest_escalation.acknowledged_at):
                continue
            
            # Get the rule for this escalation
            rule = None
            for r in self.escalation_rules.values():
                if r.id == latest_escalation.escalation_rule_id:
                    rule = r
                    break
            
            if not rule or not rule.auto_escalate:
                continue
            
            # Check if escalation delay has passed
            next_level = self._get_next_escalation_level(latest_escalation.to_level)
            if next_level:
                delay_minutes = self._get_escalation_delay(rule, next_level)
                time_since_escalation = current_time - latest_escalation.escalated_at
                
                if time_since_escalation >= timedelta(minutes=delay_minutes):
                    await self._escalate_alert(alert_id, rule, next_level, "Automatic escalation - time threshold")
    
    async def manual_escalate(self, alert_id: str, user_id: str, reason: str = "Manual escalation") -> bool:
        """
        Manually escalate an alert
        
        Args:
            alert_id: Alert ID to escalate
            user_id: User initiating escalation
            reason: Reason for escalation
            
        Returns:
            True if escalation was successful
        """
        if alert_id not in self.active_escalations:
            logger.error(f"No active escalation found for alert {alert_id}")
            return False
        
        escalation_events = self.active_escalations[alert_id]
        if not escalation_events:
            logger.error(f"No escalation events found for alert {alert_id}")
            return False
        
        latest_escalation = max(escalation_events, key=lambda x: x.escalated_at)
        
        # Get escalation rule
        rule = None
        for r in self.escalation_rules.values():
            if r.id == latest_escalation.escalation_rule_id:
                rule = r
                break
        
        if not rule:
            logger.error(f"Escalation rule not found for alert {alert_id}")
            return False
        
        # Get next level
        next_level = self._get_next_escalation_level(latest_escalation.to_level)
        if not next_level:
            logger.warning(f"Alert {alert_id} is already at maximum escalation level")
            return False
        
        await self._escalate_alert(alert_id, rule, next_level, reason, user_id)
        return True
    
    async def acknowledge_escalation(self, alert_id: str, user_id: str) -> bool:
        """
        Acknowledge an escalation
        
        Args:
            alert_id: Alert ID to acknowledge
            user_id: User acknowledging the alert
            
        Returns:
            True if acknowledgment was successful
        """
        if alert_id not in self.active_escalations:
            return False
        
        escalation_events = self.active_escalations[alert_id]
        if not escalation_events:
            return False
        
        # Acknowledge the latest escalation
        latest_escalation = max(escalation_events, key=lambda x: x.escalated_at)
        latest_escalation.acknowledged_by = user_id
        latest_escalation.acknowledged_at = datetime.now()
        
        log_risk_event(
            portfolio_id="system",
            event_type="escalation_acknowledged",
            event_data={
                "alert_id": alert_id,
                "escalation_id": latest_escalation.id,
                "acknowledged_by": user_id,
                "escalation_level": latest_escalation.to_level.value
            }
        )
        
        logger.info(f"Escalation acknowledged for alert {alert_id} by {user_id}")
        return True
    
    async def _escalate_alert(self, alert_id: str, rule: EscalationRule, to_level: EscalationLevel, reason: str, user_id: str = "system"):
        """Internal method to escalate an alert"""
        escalation_events = self.active_escalations[alert_id]
        latest_escalation = max(escalation_events, key=lambda x: x.escalated_at) if escalation_events else None
        
        from_level = latest_escalation.to_level if latest_escalation else EscalationLevel.LEVEL_1
        
        await self._create_escalation_event(
            None,  # We don't have the alert object here
            rule, from_level, to_level, reason, user_id, alert_id
        )
    
    async def _create_escalation_event(self, 
                                     alert: Optional[RiskAlert],
                                     rule: EscalationRule,
                                     from_level: EscalationLevel,
                                     to_level: EscalationLevel,
                                     reason: str,
                                     escalated_by: str,
                                     alert_id: Optional[str] = None) -> EscalationEvent:
        """Create an escalation event"""
        event_id = f"escalation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.escalation_events)}"
        
        event = EscalationEvent(
            id=event_id,
            alert_id=alert_id or alert.id,
            escalation_rule_id=rule.id,
            from_level=from_level,
            to_level=to_level,
            escalated_at=datetime.now(),
            escalated_by=escalated_by,
            reason=reason,
            notification_sent=False
        )
        
        # Add to tracking
        self.escalation_events.append(event)
        target_alert_id = alert_id or alert.id
        if target_alert_id not in self.active_escalations:
            self.active_escalations[target_alert_id] = []
        self.active_escalations[target_alert_id].append(event)
        
        # Log event
        log_risk_event(
            portfolio_id=alert.portfolio_id if alert else "unknown",
            event_type="alert_escalated",
            event_data={
                "escalation_id": event_id,
                "alert_id": target_alert_id,
                "from_level": from_level.value,
                "to_level": to_level.value,
                "escalated_by": escalated_by,
                "reason": reason
            }
        )
        
        # Trigger callbacks
        for callback in self.escalation_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in escalation callback: {e}")
        
        logger.warning(f"Alert escalated: {target_alert_id} to level {to_level.value} - {reason}")
        return event
    
    def _get_next_escalation_level(self, current_level: EscalationLevel) -> Optional[EscalationLevel]:
        """Get the next escalation level"""
        level_order = [EscalationLevel.LEVEL_1, EscalationLevel.LEVEL_2, EscalationLevel.LEVEL_3, EscalationLevel.LEVEL_4]
        
        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:
                return level_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _get_escalation_delay(self, rule: EscalationRule, level: EscalationLevel) -> int:
        """Get escalation delay in minutes for a level"""
        for step in rule.escalation_path:
            if step.get("level") == level:
                return step.get("delay_minutes", 15)
        
        # Default delays by level
        default_delays = {
            EscalationLevel.LEVEL_1: 0,
            EscalationLevel.LEVEL_2: 15,
            EscalationLevel.LEVEL_3: 30,
            EscalationLevel.LEVEL_4: 60
        }
        
        return default_delays.get(level, 15)
    
    def _get_escalation_recipients(self, rule: EscalationRule, level: EscalationLevel) -> List[str]:
        """Get recipients for an escalation level"""
        for step in rule.escalation_path:
            if step.get("level") == level:
                roles = step.get("roles", [])
                recipients = []
                
                for user_id, contact in self.user_contacts.items():
                    if contact.role in roles and contact.is_available:
                        recipients.append(user_id)
                
                return recipients
        
        return []
    
    def _create_default_rules(self):
        """Create default escalation rules"""
        # Critical alert escalation
        critical_path = [
            {"level": EscalationLevel.LEVEL_1, "roles": [UserRole.ANALYST, UserRole.PORTFOLIO_MANAGER], "delay_minutes": 0},
            {"level": EscalationLevel.LEVEL_2, "roles": [UserRole.RISK_MANAGER], "delay_minutes": 5},
            {"level": EscalationLevel.LEVEL_3, "roles": [UserRole.HEAD_OF_RISK, UserRole.CIO], "delay_minutes": 15},
            {"level": EscalationLevel.LEVEL_4, "roles": [UserRole.CEO], "delay_minutes": 30}
        ]
        
        self.create_escalation_rule(
            "Critical Alert Escalation",
            AlertSeverity.CRITICAL,
            critical_path,
            auto_escalate=True,
            require_acknowledgment=True
        )
        
        # High alert escalation
        high_path = [
            {"level": EscalationLevel.LEVEL_1, "roles": [UserRole.ANALYST, UserRole.PORTFOLIO_MANAGER], "delay_minutes": 0},
            {"level": EscalationLevel.LEVEL_2, "roles": [UserRole.RISK_MANAGER], "delay_minutes": 15},
            {"level": EscalationLevel.LEVEL_3, "roles": [UserRole.HEAD_OF_RISK], "delay_minutes": 60}
        ]
        
        self.create_escalation_rule(
            "High Alert Escalation",
            AlertSeverity.HIGH,
            high_path,
            auto_escalate=True,
            require_acknowledgment=True
        )
        
        # Medium alert escalation (manual only)
        medium_path = [
            {"level": EscalationLevel.LEVEL_1, "roles": [UserRole.ANALYST], "delay_minutes": 0},
            {"level": EscalationLevel.LEVEL_2, "roles": [UserRole.SENIOR_ANALYST], "delay_minutes": 30}
        ]
        
        self.create_escalation_rule(
            "Medium Alert Escalation",
            AlertSeverity.MEDIUM,
            medium_path,
            auto_escalate=False,
            require_acknowledgment=False
        )
    
    async def start_escalation_monitoring(self):
        """Start the escalation monitoring loop"""
        logger.info("Starting escalation monitoring")
        
        while True:
            try:
                await self.check_escalations()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in escalation monitoring: {e}")
                await asyncio.sleep(30)  # Brief pause before retry
    
    def get_escalation_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get escalation statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_escalations = [
            e for e in self.escalation_events
            if e.escalated_at >= cutoff_time
        ]
        
        stats = {
            'total_escalations': len(recent_escalations),
            'auto_escalations': len([e for e in recent_escalations if e.escalated_by == "system"]),
            'manual_escalations': len([e for e in recent_escalations if e.escalated_by != "system"]),
            'by_level': {},
            'acknowledgment_rate': 0.0,
            'average_escalation_time': None
        }
        
        # Count by level
        for level in EscalationLevel:
            stats['by_level'][level.value] = len([
                e for e in recent_escalations if e.to_level == level
            ])
        
        # Calculate acknowledgment rate
        acknowledged_escalations = len([e for e in recent_escalations if e.acknowledged_at])
        if recent_escalations:
            stats['acknowledgment_rate'] = acknowledged_escalations / len(recent_escalations)
        
        # Calculate average escalation time (time to acknowledgment)
        acknowledged = [e for e in recent_escalations if e.acknowledged_at]
        if acknowledged:
            escalation_times = [
                (e.acknowledged_at - e.escalated_at).total_seconds()
                for e in acknowledged
            ]
            stats['average_escalation_time'] = sum(escalation_times) / len(escalation_times)
        
        return stats


# Helper functions for creating user contacts
def create_user_contact(user_id: str, 
                       role: UserRole,
                       email: Optional[str] = None,
                       phone: Optional[str] = None) -> UserEscalationContact:
    """Create a user escalation contact"""
    return UserEscalationContact(
        user_id=user_id,
        role=role,
        email=email,
        phone=phone,
        is_available=True,
        working_hours_start="09:00",
        working_hours_end="17:00"
    )


# Example usage and testing
async def test_escalation_manager():
    """Test the escalation manager"""
    from ..risk.alert_engine import RiskAlert, AlertSeverity, MetricType
    
    # Initialize escalation manager
    escalation_manager = EscalationManager()
    
    # Add user contacts
    contacts = [
        create_user_contact("analyst_1", UserRole.ANALYST, "analyst1@company.com"),
        create_user_contact("pm_1", UserRole.PORTFOLIO_MANAGER, "pm1@company.com"),
        create_user_contact("risk_mgr", UserRole.RISK_MANAGER, "risk@company.com"),
        create_user_contact("head_risk", UserRole.HEAD_OF_RISK, "head.risk@company.com"),
        create_user_contact("cio", UserRole.CIO, "cio@company.com")
    ]
    
    for contact in contacts:
        escalation_manager.add_user_contact(contact)
    
    # Create test alert
    alert = RiskAlert(
        id="test_alert_esc_001",
        threshold_id="threshold_001",
        portfolio_id="test_portfolio",
        metric_type=MetricType.VAR_95,
        current_value=-0.08,
        threshold_value=-0.05,
        severity=AlertSeverity.CRITICAL,
        status=AlertStatus.TRIGGERED,
        description="Critical VaR breach requiring immediate attention",
        triggered_at=datetime.now()
    )
    
    # Process alert for escalation
    await escalation_manager.process_alert_for_escalation(alert)
    
    print(f"Processed alert {alert.id} for escalation")
    
    # Check active escalations
    active = escalation_manager.active_escalations.get(alert.id, [])
    print(f"Active escalations for alert: {len(active)}")
    
    for escalation in active:
        print(f"  - Level {escalation.to_level.value}: {escalation.reason}")
    
    # Manually escalate
    success = await escalation_manager.manual_escalate(alert.id, "pm_1", "Urgent manual escalation")
    print(f"Manual escalation result: {success}")
    
    # Get statistics
    stats = escalation_manager.get_escalation_statistics()
    print(f"\nEscalation Statistics:")
    print(f"  Total escalations: {stats['total_escalations']}")
    print(f"  Auto escalations: {stats['auto_escalations']}")
    print(f"  Manual escalations: {stats['manual_escalations']}")
    print(f"  By level: {stats['by_level']}")


if __name__ == "__main__":
    asyncio.run(test_escalation_manager())
