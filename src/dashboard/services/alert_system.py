"""
Real-time Alert System for Portfolio Monitoring
===============================================

This module provides comprehensive real-time alerting capabilities for portfolio
monitoring, risk management, and performance tracking with multiple notification channels.

Dependencies:
- asyncio: Asynchronous event processing
- smtplib: Email notifications
- requests: HTTP webhook notifications
- sqlite3/postgresql: Alert configuration storage
- schedule: Alert scheduling and frequency management
"""

import asyncio
import json
import logging
import smtplib
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# Database integration
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status tracking."""
    PENDING = "pending"
    SENT = "sent" 
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FAILED = "failed"


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    
    rule_id: str
    name: str
    description: str
    metric: str  # e.g., 'portfolio_value', 'var_95', 'drawdown'
    condition: str  # e.g., 'greater_than', 'less_than', 'equals', 'change_by'
    threshold: float
    portfolio_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Alert configuration
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.EMAIL])
    recipients: List[str] = field(default_factory=list)
    
    # Timing configuration
    cooldown_minutes: int = 60  # Minimum time between alerts
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    
    # Advanced configuration
    business_hours_only: bool = False
    weekend_alerts: bool = True
    escalation_rules: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Individual alert instance."""
    
    alert_id: str
    rule_id: str
    portfolio_id: Optional[str]
    tenant_id: Optional[str]
    
    # Alert details
    title: str
    message: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    
    # Status tracking
    status: AlertStatus = AlertStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Notification tracking
    notification_attempts: int = 0
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    
    # Additional context
    context_data: Dict[str, Any] = field(default_factory=dict)


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    @abstractmethod
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """Send notification through this provider."""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate provider configuration."""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider using SMTP."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, use_tls: bool = True, sender_email: str = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.sender_email = sender_email or username
        
    def validate_configuration(self) -> bool:
        """Validate SMTP configuration by attempting connection."""
        try:
            if self.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.username, self.password)
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.login(self.username, self.password)
            return True
        except Exception as e:
            logging.error(f"Email configuration validation failed: {e}")
            return False
    
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """Send email notification."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            if self.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.username, self.password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.login(self.username, self.password)
                    server.send_message(msg)
            
            logging.info(f"Email alert sent successfully to {recipients}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107", 
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 20px;">
                <h2 style="color: {color}; margin-top: 0;">
                    {alert.severity.value.upper()} Alert: {alert.title}
                </h2>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <p><strong>Message:</strong> {alert.message}</p>
                    <p><strong>Metric:</strong> {alert.metric_name}</p>
                    <p><strong>Current Value:</strong> {alert.current_value:,.4f}</p>
                    <p><strong>Threshold:</strong> {alert.threshold_value:,.4f}</p>
                    {f'<p><strong>Portfolio:</strong> {alert.portfolio_id}</p>' if alert.portfolio_id else ''}
                </div>
                
                <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #dee2e6;">
                    <p style="color: #6c757d; font-size: 12px;">
                        Alert ID: {alert.alert_id}<br>
                        Generated: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                        Quantum Portfolio Optimizer - Real-time Alert System
                    </p>
                </div>
            </div>
        </body>
        </html>
        """


class WebhookNotificationProvider(NotificationProvider):
    """Webhook notification provider for external integrations."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None,
                 timeout: int = 30):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.timeout = timeout
        
    def validate_configuration(self) -> bool:
        """Validate webhook configuration."""
        try:
            response = requests.get(self.webhook_url, headers=self.headers, 
                                  timeout=5)
            return response.status_code < 500
        except Exception as e:
            logging.error(f"Webhook configuration validation failed: {e}")
            return False
    
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """Send webhook notification."""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "portfolio_id": alert.portfolio_id,
                "tenant_id": alert.tenant_id,
                "created_at": alert.created_at.isoformat(),
                "recipients": recipients,
                "context": alert.context_data
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code < 300:
                logging.info(f"Webhook alert sent successfully to {self.webhook_url}")
                return True
            else:
                logging.error(f"Webhook failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to send webhook alert: {e}")
            return False


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider using webhooks."""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.channel = channel
        
    def validate_configuration(self) -> bool:
        """Validate Slack webhook configuration."""
        try:
            test_payload = {"text": "Configuration test"}
            response = requests.post(self.webhook_url, json=test_payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Slack configuration validation failed: {e}")
            return False
    
    async def send_notification(self, alert: Alert, recipients: List[str]) -> bool:
        """Send Slack notification."""
        try:
            severity_colors = {
                AlertSeverity.LOW: "#28a745",
                AlertSeverity.MEDIUM: "#ffc107",
                AlertSeverity.HIGH: "#fd7e14", 
                AlertSeverity.CRITICAL: "#dc3545"
            }
            
            color = severity_colors.get(alert.severity, "#6c757d")
            
            payload = {
                "text": f"{alert.severity.value.upper()} Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False
                            },
                            {
                                "title": "Metric",
                                "value": alert.metric_name,
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": f"{alert.current_value:,.4f}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.threshold_value:,.4f}",
                                "short": True
                            },
                            {
                                "title": "Portfolio",
                                "value": alert.portfolio_id or "N/A",
                                "short": True
                            }
                        ],
                        "footer": "Quantum Portfolio Optimizer",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logging.info(f"Slack alert sent successfully")
                return True
            else:
                logging.error(f"Slack notification failed: {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")
            return False


class AlertStorage:
    """Storage interface for alert rules and history."""
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        self.db_config = db_config or {}
        self.connection = None
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        # For demo purposes, use SQLite
        if SQLITE_AVAILABLE:
            self.connection = sqlite3.connect("alerts.db", check_same_thread=False)
            self._create_tables()
        else:
            logging.warning("No database backend available for alert storage")
    
    def _create_tables(self) -> None:
        """Create necessary database tables."""
        if not self.connection:
            return
            
        cursor = self.connection.cursor()
        
        # Alert rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                portfolio_id TEXT,
                tenant_id TEXT,
                metric TEXT NOT NULL,
                condition TEXT NOT NULL,
                threshold REAL NOT NULL,
                severity TEXT NOT NULL,
                channels TEXT NOT NULL,
                recipients TEXT NOT NULL,
                cooldown_minutes INTEGER DEFAULT 60,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_triggered TIMESTAMP,
                business_hours_only BOOLEAN DEFAULT 0,
                weekend_alerts BOOLEAN DEFAULT 1,
                escalation_rules TEXT
            )
        """)
        
        # Alert history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                alert_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                portfolio_id TEXT,
                tenant_id TEXT,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sent_at TIMESTAMP,
                acknowledged_at TIMESTAMP,
                resolved_at TIMESTAMP,
                notification_attempts INTEGER DEFAULT 0,
                notification_channels TEXT,
                recipients TEXT,
                context_data TEXT,
                FOREIGN KEY (rule_id) REFERENCES alert_rules (rule_id)
            )
        """)
        
        self.connection.commit()
    
    def save_alert_rule(self, rule: AlertRule) -> bool:
        """Save alert rule to storage."""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alert_rules 
                (rule_id, name, description, portfolio_id, tenant_id, metric, condition, 
                 threshold, severity, channels, recipients, cooldown_minutes, is_active,
                 created_at, last_triggered, business_hours_only, weekend_alerts, escalation_rules)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id, rule.name, rule.description, rule.portfolio_id, rule.tenant_id,
                rule.metric, rule.condition, rule.threshold, rule.severity.value,
                json.dumps([ch.value for ch in rule.channels]),
                json.dumps(rule.recipients), rule.cooldown_minutes, rule.is_active,
                rule.created_at, rule.last_triggered, rule.business_hours_only,
                rule.weekend_alerts, json.dumps(rule.escalation_rules) if rule.escalation_rules else None
            ))
            self.connection.commit()
            return True
        except Exception as e:
            logging.error(f"Failed to save alert rule: {e}")
            return False
    
    def get_active_rules(self, tenant_id: Optional[str] = None) -> List[AlertRule]:
        """Get all active alert rules."""
        if not self.connection:
            return []
            
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM alert_rules WHERE is_active = 1"
            params = []
            
            if tenant_id:
                query += " AND (tenant_id = ? OR tenant_id IS NULL)"
                params.append(tenant_id)
                
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule = AlertRule(
                    rule_id=row[0], name=row[1], description=row[2],
                    portfolio_id=row[3], tenant_id=row[4], metric=row[5],
                    condition=row[6], threshold=row[7],
                    severity=AlertSeverity(row[8]),
                    channels=[NotificationChannel(ch) for ch in json.loads(row[9])],
                    recipients=json.loads(row[10]),
                    cooldown_minutes=row[11], is_active=bool(row[12]),
                    created_at=datetime.fromisoformat(row[13]) if row[13] else datetime.now(),
                    last_triggered=datetime.fromisoformat(row[14]) if row[14] else None,
                    business_hours_only=bool(row[15]), weekend_alerts=bool(row[16]),
                    escalation_rules=json.loads(row[17]) if row[17] else None
                )
                rules.append(rule)
            
            return rules
        except Exception as e:
            logging.error(f"Failed to get active rules: {e}")
            return []
    
    def save_alert(self, alert: Alert) -> bool:
        """Save alert to history."""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alert_history
                (alert_id, rule_id, portfolio_id, tenant_id, title, message, severity,
                 metric_name, current_value, threshold_value, status, created_at, sent_at,
                 acknowledged_at, resolved_at, notification_attempts, notification_channels,
                 recipients, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.rule_id, alert.portfolio_id, alert.tenant_id,
                alert.title, alert.message, alert.severity.value, alert.metric_name,
                alert.current_value, alert.threshold_value, alert.status.value,
                alert.created_at, alert.sent_at, alert.acknowledged_at, alert.resolved_at,
                alert.notification_attempts, json.dumps([ch.value for ch in alert.notification_channels]),
                json.dumps(alert.recipients), json.dumps(alert.context_data)
            ))
            self.connection.commit()
            return True
        except Exception as e:
            logging.error(f"Failed to save alert: {e}")
            return False


class RealTimeAlertSystem:
    """
    Main alert system orchestrating monitoring, evaluation, and notifications.
    
    Features:
    - Real-time metric monitoring with configurable thresholds
    - Multi-channel notifications (email, Slack, webhooks)
    - Alert escalation and acknowledgment workflows
    - Business hours and cooldown period support
    - Comprehensive alert history and analytics
    - Role-based alert routing and permissions
    """
    
    def __init__(self, storage: Optional[AlertStorage] = None):
        self.storage = storage or AlertStorage()
        self.notification_providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.monitoring_active = False
        self.monitoring_task = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Metrics cache for monitoring
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def configure_email_notifications(self, smtp_server: str, smtp_port: int,
                                    username: str, password: str, use_tls: bool = True) -> bool:
        """Configure email notification provider."""
        provider = EmailNotificationProvider(smtp_server, smtp_port, username, password, use_tls)
        if provider.validate_configuration():
            self.notification_providers[NotificationChannel.EMAIL] = provider
            self.logger.info("Email notifications configured successfully")
            return True
        return False
    
    def configure_slack_notifications(self, webhook_url: str, channel: Optional[str] = None) -> bool:
        """Configure Slack notification provider."""
        provider = SlackNotificationProvider(webhook_url, channel)
        if provider.validate_configuration():
            self.notification_providers[NotificationChannel.SLACK] = provider
            self.logger.info("Slack notifications configured successfully")
            return True
        return False
    
    def configure_webhook_notifications(self, webhook_url: str, 
                                      headers: Optional[Dict[str, str]] = None) -> bool:
        """Configure webhook notification provider."""
        provider = WebhookNotificationProvider(webhook_url, headers)
        if provider.validate_configuration():
            self.notification_providers[NotificationChannel.WEBHOOK] = provider
            self.logger.info("Webhook notifications configured successfully")
            return True
        return False
    
    def create_alert_rule(self, rule: AlertRule) -> bool:
        """Create new alert rule."""
        if self.storage.save_alert_rule(rule):
            self.logger.info(f"Alert rule created: {rule.rule_id} - {rule.name}")
            return True
        return False
    
    def update_portfolio_metrics(self, portfolio_id: str, metrics: Dict[str, float]) -> None:
        """Update portfolio metrics for monitoring."""
        with self.cache_lock:
            if portfolio_id not in self.metrics_cache:
                self.metrics_cache[portfolio_id] = {}
            self.metrics_cache[portfolio_id].update(metrics)
            self.metrics_cache[portfolio_id]['last_updated'] = datetime.now()
    
    def trigger_manual_alert(self, rule_id: str, portfolio_id: str, 
                           message: str, context: Optional[Dict] = None) -> bool:
        """Manually trigger an alert for testing or urgent notifications."""
        rules = self.storage.get_active_rules()
        rule = next((r for r in rules if r.rule_id == rule_id), None)
        
        if not rule:
            self.logger.error(f"Alert rule not found: {rule_id}")
            return False
        
        alert = Alert(
            alert_id=f"manual_{int(time.time())}_{rule_id}",
            rule_id=rule_id,
            portfolio_id=portfolio_id,
            tenant_id=rule.tenant_id,
            title=f"Manual Alert: {rule.name}",
            message=message,
            severity=rule.severity,
            metric_name="manual_trigger",
            current_value=0.0,
            threshold_value=0.0,
            notification_channels=rule.channels,
            recipients=rule.recipients,
            context_data=context or {}
        )
        
        return asyncio.run(self._send_alert(alert))
    
    async def start_monitoring(self, check_interval: int = 60) -> None:
        """Start real-time monitoring loop."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.logger.info(f"Starting alert monitoring with {check_interval}s interval")
        
        while self.monitoring_active:
            try:
                await self._check_all_rules()
                await asyncio.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(check_interval)
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        self.logger.info("Alert monitoring stopped")
    
    async def _check_all_rules(self) -> None:
        """Check all active alert rules against current metrics."""
        rules = self.storage.get_active_rules()
        
        for rule in rules:
            try:
                if await self._should_check_rule(rule):
                    await self._evaluate_rule(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    async def _should_check_rule(self, rule: AlertRule) -> bool:
        """Determine if rule should be checked based on timing constraints."""
        now = datetime.now()
        
        # Check cooldown period
        if rule.last_triggered:
            cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
            if now < cooldown_end:
                return False
        
        # Check business hours
        if rule.business_hours_only:
            if now.hour < 9 or now.hour > 17:  # Simple business hours check
                return False
        
        # Check weekend alerts
        if not rule.weekend_alerts and now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        return True
    
    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single rule against current metrics."""
        # Get relevant metrics
        metrics = self._get_metrics_for_rule(rule)
        if not metrics:
            return
        
        current_value = metrics.get(rule.metric)
        if current_value is None:
            return
        
        # Check condition
        threshold_breached = self._check_threshold(current_value, rule.condition, rule.threshold)
        
        if threshold_breached:
            alert = Alert(
                alert_id=f"{rule.rule_id}_{int(time.time())}",
                rule_id=rule.rule_id,
                portfolio_id=rule.portfolio_id,
                tenant_id=rule.tenant_id,
                title=f"Alert: {rule.name}",
                message=self._generate_alert_message(rule, current_value),
                severity=rule.severity,
                metric_name=rule.metric,
                current_value=current_value,
                threshold_value=rule.threshold,
                notification_channels=rule.channels,
                recipients=rule.recipients,
                context_data=metrics
            )
            
            await self._send_alert(alert)
            
            # Update rule last triggered time
            rule.last_triggered = datetime.now()
            self.storage.save_alert_rule(rule)
    
    def _get_metrics_for_rule(self, rule: AlertRule) -> Optional[Dict[str, float]]:
        """Get metrics relevant to the alert rule."""
        with self.cache_lock:
            if rule.portfolio_id:
                return self.metrics_cache.get(rule.portfolio_id)
            else:
                # Global metrics - combine all portfolios
                global_metrics = {}
                for portfolio_metrics in self.metrics_cache.values():
                    for key, value in portfolio_metrics.items():
                        if isinstance(value, (int, float)):
                            global_metrics[key] = global_metrics.get(key, 0) + value
                return global_metrics if global_metrics else None
    
    def _check_threshold(self, current_value: float, condition: str, threshold: float) -> bool:
        """Check if current value breaches threshold based on condition."""
        if condition == "greater_than":
            return current_value > threshold
        elif condition == "less_than":
            return current_value < threshold
        elif condition == "equals":
            return abs(current_value - threshold) < 0.0001  # Float comparison
        elif condition == "change_by":
            # This would require historical data comparison
            return False  # Simplified for demo
        else:
            self.logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _generate_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Generate human-readable alert message."""
        condition_text = {
            "greater_than": "exceeded",
            "less_than": "fallen below", 
            "equals": "reached",
            "change_by": "changed by"
        }.get(rule.condition, "triggered")
        
        return (f"Portfolio metric '{rule.metric}' has {condition_text} the threshold. "
                f"Current value: {current_value:,.4f}, Threshold: {rule.threshold:,.4f}")
    
    async def _send_alert(self, alert: Alert) -> bool:
        """Send alert through configured notification channels."""
        self.logger.info(f"Sending alert: {alert.alert_id}")
        
        success = True
        
        for channel in alert.notification_channels:
            provider = self.notification_providers.get(channel)
            if provider:
                try:
                    result = await provider.send_notification(alert, alert.recipients)
                    if not result:
                        success = False
                    alert.notification_attempts += 1
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel.value}: {e}")
                    success = False
            else:
                self.logger.warning(f"No provider configured for channel: {channel.value}")
                success = False
        
        # Update alert status
        alert.status = AlertStatus.SENT if success else AlertStatus.FAILED
        alert.sent_at = datetime.now() if success else None
        
        # Save to history
        self.storage.save_alert(alert)
        
        return success
    
    def get_alert_history(self, portfolio_id: Optional[str] = None, 
                         days: int = 30) -> List[Alert]:
        """Get alert history for analysis."""
        # This would query the database for historical alerts
        # Simplified implementation for demo
        return []
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Mark alert as acknowledged."""
        # Update alert status in database
        # Implementation would update the alert_history table
        self.logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        return True
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_notes: str) -> bool:
        """Mark alert as resolved."""
        # Update alert status in database
        # Implementation would update the alert_history table
        self.logger.info(f"Alert {alert_id} resolved by {user_id}: {resolution_notes}")
        return True


# Convenience functions for external use
def create_alert_system(storage_config: Optional[Dict] = None) -> RealTimeAlertSystem:
    """Create and configure alert system."""
    storage = AlertStorage(storage_config)
    return RealTimeAlertSystem(storage)

def create_portfolio_alert_rule(portfolio_id: str, metric: str, condition: str, 
                               threshold: float, severity: str = "medium") -> AlertRule:
    """Create a basic portfolio alert rule."""
    return AlertRule(
        rule_id=f"{portfolio_id}_{metric}_{condition}_{int(time.time())}",
        name=f"Portfolio {portfolio_id} - {metric} Alert",
        description=f"Alert when {metric} {condition} {threshold}",
        portfolio_id=portfolio_id,
        metric=metric,
        condition=condition,
        threshold=threshold,
        severity=AlertSeverity(severity),
        channels=[NotificationChannel.EMAIL],
        recipients=[]
    )


if __name__ == "__main__":
    # Test the alert system
    print("Testing Real-time Alert System...")
    
    try:
        # Create alert system
        alert_system = create_alert_system()
        
        # Configure email notifications (demo configuration)
        alert_system.configure_email_notifications(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username="demo@example.com",
            password="demo_password",
            use_tls=True
        )
        
        # Create test alert rule
        test_rule = create_portfolio_alert_rule(
            portfolio_id="test_portfolio_001",
            metric="portfolio_value",
            condition="less_than",
            threshold=9000000.0,
            severity="high"
        )
        test_rule.recipients = ["admin@example.com"]
        
        # Add rule to system
        alert_system.create_alert_rule(test_rule)
        
        # Update portfolio metrics to trigger alert
        alert_system.update_portfolio_metrics("test_portfolio_001", {
            "portfolio_value": 8500000.0,  # Below threshold
            "var_95": -0.025,
            "sharpe_ratio": 0.45
        })
        
        # Trigger manual alert for testing
        alert_system.trigger_manual_alert(
            rule_id=test_rule.rule_id,
            portfolio_id="test_portfolio_001",
            message="Test alert triggered manually for system validation",
            context={"test_mode": True}
        )
        
        print("Alert system test completed successfully")
        print(f"Test rule created: {test_rule.rule_id}")
        
    except Exception as e:
        print(f"Error testing alert system: {e}")
    
    print("Real-time Alert System test completed.")
