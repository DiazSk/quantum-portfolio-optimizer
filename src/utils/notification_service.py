"""
Notification Service for Risk Alerts
Implements email and SMS notifications with template system and delivery tracking
"""

import smtplib
import ssl
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from jinja2 import Template
import json

# Optional SMS providers (install with pip install twilio boto3)
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

try:
    import boto3
    AWS_SNS_AVAILABLE = True
except ImportError:
    AWS_SNS_AVAILABLE = False

from ..risk.alert_engine import RiskAlert, AlertSeverity
from ..utils.professional_logging import log_risk_event

# Configure logging
logger = logging.getLogger(__name__)


class NotificationType:
    """Notification delivery types"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


class DeliveryStatus:
    """Notification delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


# Alias for backward compatibility with tests
NotificationStatus = DeliveryStatus


@dataclass
class NotificationTemplate:
    """Notification template definition"""
    template_id: str
    subject_template: str
    body_template: str
    notification_type: str
    variables: Optional[List[str]] = None
    created_at: Optional[datetime] = None


@dataclass
class NotificationPreference:
    """User notification preferences"""
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    webhook_url: Optional[str] = None
    enabled_types: List[str] = None
    severity_filter: List[AlertSeverity] = None
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None
    timezone: str = "UTC"


@dataclass
class NotificationDelivery:
    """Notification delivery tracking"""
    id: str
    alert_id: str
    notification_type: str
    recipient: str
    status: str
    attempts: int
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    template_used: Optional[str] = None


class NotificationTemplateEngine:
    """Template engine for notification formatting"""
    
    def __init__(self):
        """Initialize template engine"""
        self.templates = {
            'email': {
                'critical': self._get_email_critical_template(),
                'high': self._get_email_high_template(),
                'medium': self._get_email_medium_template(),
                'low': self._get_email_low_template()
            },
            'sms': {
                'critical': self._get_sms_critical_template(),
                'high': self._get_sms_high_template(),
                'medium': self._get_sms_medium_template(),
                'low': self._get_sms_low_template()
            }
        }
    
    def render_template(self, notification_type: str, severity: AlertSeverity, context: Dict[str, Any]) -> str:
        """Render notification template with context"""
        template_str = self.templates.get(notification_type, {}).get(severity.value)
        
        if not template_str:
            # Fallback template
            template_str = self._get_fallback_template(notification_type)
        
        template = Template(template_str)
        return template.render(**context)
    
    def _get_email_critical_template(self) -> str:
        return """
🚨 CRITICAL RISK ALERT - Immediate Action Required

Portfolio: {{ alert.portfolio_id }}
Alert: {{ alert.description }}

Risk Details:
- Metric: {{ alert.metric_type.value }}
- Current Value: {{ "%.4f"|format(alert.current_value) }}
- Threshold: {{ "%.4f"|format(alert.threshold_value) }}
- Triggered: {{ alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}

This is a CRITICAL alert requiring immediate attention. Please review your portfolio positions and consider rebalancing or hedging strategies.

Alert ID: {{ alert.id }}
Portfolio Dashboard: {{ dashboard_url }}

This is an automated alert from the Quantum Portfolio Risk Monitoring System.
        """
    
    def _get_email_high_template(self) -> str:
        return """
⚠️ HIGH RISK ALERT

Portfolio: {{ alert.portfolio_id }}
Alert: {{ alert.description }}

Risk Details:
- Metric: {{ alert.metric_type.value }}
- Current Value: {{ "%.4f"|format(alert.current_value) }}
- Threshold: {{ "%.4f"|format(alert.threshold_value) }}
- Triggered: {{ alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}

Please review your portfolio risk exposure at your earliest convenience.

Alert ID: {{ alert.id }}
Portfolio Dashboard: {{ dashboard_url }}
        """
    
    def _get_email_medium_template(self) -> str:
        return """
📊 Risk Alert Notification

Portfolio: {{ alert.portfolio_id }}
Alert: {{ alert.description }}

Risk Details:
- Metric: {{ alert.metric_type.value }}
- Current Value: {{ "%.4f"|format(alert.current_value) }}
- Threshold: {{ "%.4f"|format(alert.threshold_value) }}
- Triggered: {{ alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}

Alert ID: {{ alert.id }}
        """
    
    def _get_email_low_template(self) -> str:
        return """
ℹ️ Risk Monitoring Update

Portfolio {{ alert.portfolio_id }}: {{ alert.description }}
Triggered: {{ alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}

Alert ID: {{ alert.id }}
        """
    
    def _get_sms_critical_template(self) -> str:
        return "🚨 CRITICAL RISK ALERT: {{ alert.portfolio_id }} - {{ alert.description }}. Current: {{ '%.3f'|format(alert.current_value) }}. IMMEDIATE ACTION REQUIRED."
    
    def _get_sms_high_template(self) -> str:
        return "⚠️ HIGH RISK: {{ alert.portfolio_id }} - {{ alert.description }}. Current: {{ '%.3f'|format(alert.current_value) }}."
    
    def _get_sms_medium_template(self) -> str:
        return "📊 Risk Alert: {{ alert.portfolio_id }} - {{ alert.metric_type.value }} breach. Current: {{ '%.3f'|format(alert.current_value) }}."
    
    def _get_sms_low_template(self) -> str:
        return "ℹ️ {{ alert.portfolio_id }}: {{ alert.metric_type.value }} alert."
    
    def _get_fallback_template(self, notification_type: str) -> str:
        if notification_type == 'sms':
            return "Risk Alert: {{ alert.portfolio_id }} - {{ alert.description }}"
        else:
            return "Risk Alert: {{ alert.description }} for portfolio {{ alert.portfolio_id }}"


class EmailNotificationService:
    """Email notification service using SMTP"""
    
    def __init__(self, 
                 smtp_server: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 from_email: str,
                 use_tls: bool = True):
        """
        Initialize email service
        
        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            use_tls: Use TLS encryption
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_tls = use_tls
        
        logger.info(f"EmailNotificationService initialized for {smtp_server}")
    
    async def send_email(self, 
                        to_email: str,
                        subject: str,
                        body: str,
                        html_body: Optional[str] = None) -> bool:
        """
        Send email notification
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (plain text)
            html_body: Optional HTML body
            
        Returns:
            True if sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            # Add plain text part
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
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
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False


class SMSNotificationService:
    """SMS notification service supporting Twilio and AWS SNS"""
    
    def __init__(self, provider: str = 'twilio', **config):
        """
        Initialize SMS service
        
        Args:
            provider: SMS provider ('twilio' or 'aws_sns')
            **config: Provider-specific configuration
        """
        self.provider = provider
        self.config = config
        self.client = None
        
        if provider == 'twilio' and TWILIO_AVAILABLE:
            self.client = TwilioClient(
                config.get('account_sid'),
                config.get('auth_token')
            )
        elif provider == 'aws_sns' and AWS_SNS_AVAILABLE:
            self.client = boto3.client(
                'sns',
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key'),
                region_name=config.get('region', 'us-east-1')
            )
        
        logger.info(f"SMSNotificationService initialized with {provider}")
    
    async def send_sms(self, to_phone: str, message: str) -> bool:
        """
        Send SMS notification
        
        Args:
            to_phone: Recipient phone number (E.164 format)
            message: SMS message text
            
        Returns:
            True if sent successfully
        """
        try:
            if self.provider == 'twilio' and self.client:
                message = self.client.messages.create(
                    body=message,
                    from_=self.config.get('from_phone'),
                    to=to_phone
                )
                logger.info(f"SMS sent via Twilio to {to_phone}: {message.sid}")
                return True
                
            elif self.provider == 'aws_sns' and self.client:
                response = self.client.publish(
                    PhoneNumber=to_phone,
                    Message=message
                )
                logger.info(f"SMS sent via AWS SNS to {to_phone}: {response['MessageId']}")
                return True
                
            else:
                logger.error(f"SMS provider {self.provider} not available or configured")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send SMS to {to_phone}: {e}")
            return False


class NotificationService:
    """
    Main notification service orchestrating email, SMS, and other channels
    """
    
    def __init__(self):
        """Initialize notification service"""
        self.template_engine = NotificationTemplateEngine()
        self.email_service: Optional[EmailNotificationService] = None
        self.sms_service: Optional[SMSNotificationService] = None
        self.user_preferences: Dict[str, NotificationPreference] = {}
        self.delivery_history: List[NotificationDelivery] = []
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 300  # 5 minutes
        
        # Initialize services from environment
        self._initialize_services()
        
        logger.info("NotificationService initialized")
    
    def _initialize_services(self):
        """Initialize notification services from environment variables"""
        # Initialize email service
        smtp_server = os.getenv('SMTP_SERVER')
        if smtp_server:
            self.email_service = EmailNotificationService(
                smtp_server=smtp_server,
                smtp_port=int(os.getenv('SMTP_PORT', 587)),
                username=os.getenv('SMTP_USERNAME'),
                password=os.getenv('SMTP_PASSWORD'),
                from_email=os.getenv('SMTP_FROM_EMAIL'),
                use_tls=os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
            )
        
        # Initialize SMS service (Twilio)
        twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
        if twilio_sid and TWILIO_AVAILABLE:
            self.sms_service = SMSNotificationService(
                provider='twilio',
                account_sid=twilio_sid,
                auth_token=os.getenv('TWILIO_AUTH_TOKEN'),
                from_phone=os.getenv('TWILIO_FROM_PHONE')
            )
        
        # Initialize SMS service (AWS SNS) as fallback
        elif os.getenv('AWS_ACCESS_KEY_ID') and AWS_SNS_AVAILABLE:
            self.sms_service = SMSNotificationService(
                provider='aws_sns',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region=os.getenv('AWS_REGION', 'us-east-1')
            )
    
    def set_user_preferences(self, user_id: str, preferences: NotificationPreference):
        """Set notification preferences for a user"""
        preferences.user_id = user_id
        self.user_preferences[user_id] = preferences
        logger.info(f"Updated notification preferences for user {user_id}")
    
    def get_user_preferences(self, user_id: str) -> Optional[NotificationPreference]:
        """Get notification preferences for a user"""
        return self.user_preferences.get(user_id)
    
    async def send_alert_notification(self, alert: RiskAlert, user_id: str) -> List[NotificationDelivery]:
        """
        Send notification for a risk alert
        
        Args:
            alert: Risk alert to notify about
            user_id: User to notify
            
        Returns:
            List of delivery records
        """
        preferences = self.get_user_preferences(user_id)
        if not preferences:
            logger.warning(f"No notification preferences found for user {user_id}")
            return []
        
        # Check if user wants notifications for this severity
        if preferences.severity_filter and alert.severity not in preferences.severity_filter:
            return []
        
        deliveries = []
        
        # Send email notification
        if (NotificationType.EMAIL in (preferences.enabled_types or []) and 
            preferences.email and self.email_service):
            
            delivery = await self._send_email_notification(alert, preferences.email, user_id)
            if delivery:
                deliveries.append(delivery)
        
        # Send SMS notification
        if (NotificationType.SMS in (preferences.enabled_types or []) and 
            preferences.phone and self.sms_service):
            
            delivery = await self._send_sms_notification(alert, preferences.phone, user_id)
            if delivery:
                deliveries.append(delivery)
        
        return deliveries
    
    async def _send_email_notification(self, alert: RiskAlert, email: str, user_id: str) -> Optional[NotificationDelivery]:
        """Send email notification for alert"""
        delivery_id = f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.delivery_history)}"
        
        delivery = NotificationDelivery(
            id=delivery_id,
            alert_id=alert.id,
            notification_type=NotificationType.EMAIL,
            recipient=email,
            status=DeliveryStatus.PENDING,
            attempts=0,
            created_at=datetime.now(),
            template_used=f"email_{alert.severity.value}"
        )
        
        try:
            # Render email content
            context = {
                'alert': alert,
                'user_id': user_id,
                'dashboard_url': os.getenv('DASHBOARD_URL', 'http://localhost:8501')
            }
            
            subject = f"Risk Alert: {alert.severity.value.upper()} - {alert.portfolio_id}"
            body = self.template_engine.render_template('email', alert.severity, context)
            
            # Send email
            success = await self.email_service.send_email(email, subject, body)
            
            delivery.attempts += 1
            if success:
                delivery.status = DeliveryStatus.SENT
                delivery.sent_at = datetime.now()
                logger.info(f"Email notification sent for alert {alert.id}")
            else:
                delivery.status = DeliveryStatus.FAILED
                delivery.failed_at = datetime.now()
                delivery.error_message = "SMTP send failed"
            
        except Exception as e:
            delivery.status = DeliveryStatus.FAILED
            delivery.failed_at = datetime.now()
            delivery.error_message = str(e)
            logger.error(f"Failed to send email notification: {e}")
        
        self.delivery_history.append(delivery)
        return delivery
    
    async def _send_sms_notification(self, alert: RiskAlert, phone: str, user_id: str) -> Optional[NotificationDelivery]:
        """Send SMS notification for alert"""
        delivery_id = f"sms_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.delivery_history)}"
        
        delivery = NotificationDelivery(
            id=delivery_id,
            alert_id=alert.id,
            notification_type=NotificationType.SMS,
            recipient=phone,
            status=DeliveryStatus.PENDING,
            attempts=0,
            created_at=datetime.now(),
            template_used=f"sms_{alert.severity.value}"
        )
        
        try:
            # Render SMS content
            context = {
                'alert': alert,
                'user_id': user_id
            }
            
            message = self.template_engine.render_template('sms', alert.severity, context)
            
            # Truncate SMS to 160 characters
            if len(message) > 160:
                message = message[:157] + "..."
            
            # Send SMS
            success = await self.sms_service.send_sms(phone, message)
            
            delivery.attempts += 1
            if success:
                delivery.status = DeliveryStatus.SENT
                delivery.sent_at = datetime.now()
                logger.info(f"SMS notification sent for alert {alert.id}")
            else:
                delivery.status = DeliveryStatus.FAILED
                delivery.failed_at = datetime.now()
                delivery.error_message = "SMS send failed"
            
        except Exception as e:
            delivery.status = DeliveryStatus.FAILED
            delivery.failed_at = datetime.now()
            delivery.error_message = str(e)
            logger.error(f"Failed to send SMS notification: {e}")
        
        self.delivery_history.append(delivery)
        return delivery
    
    def get_delivery_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get notification delivery statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_deliveries = [
            d for d in self.delivery_history
            if d.created_at >= cutoff_time
        ]
        
        stats = {
            'total_notifications': len(recent_deliveries),
            'successful_deliveries': len([d for d in recent_deliveries if d.status == DeliveryStatus.SENT]),
            'failed_deliveries': len([d for d in recent_deliveries if d.status == DeliveryStatus.FAILED]),
            'by_type': {},
            'by_status': {},
            'delivery_rate': 0.0
        }
        
        # Count by type
        for delivery in recent_deliveries:
            notification_type = delivery.notification_type
            if notification_type not in stats['by_type']:
                stats['by_type'][notification_type] = 0
            stats['by_type'][notification_type] += 1
        
        # Count by status
        for delivery in recent_deliveries:
            status = delivery.status
            if status not in stats['by_status']:
                stats['by_status'][status] = 0
            stats['by_status'][status] += 1
        
        # Calculate delivery rate
        if stats['total_notifications'] > 0:
            stats['delivery_rate'] = stats['successful_deliveries'] / stats['total_notifications']
        
        return stats


# Helper function to create default notification preferences
def create_default_preferences(user_id: str, 
                             email: Optional[str] = None,
                             phone: Optional[str] = None) -> NotificationPreference:
    """Create default notification preferences for a user"""
    return NotificationPreference(
        user_id=user_id,
        email=email,
        phone=phone,
        enabled_types=[NotificationType.EMAIL, NotificationType.DASHBOARD],
        severity_filter=[AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL],
        quiet_hours_start="22:00",
        quiet_hours_end="08:00",
        timezone="UTC"
    )


# Example usage and testing
async def test_notification_service():
    """Test the notification service"""
    from ..risk.alert_engine import RiskAlert, AlertSeverity, MetricType
    
    # Initialize notification service
    notification_service = NotificationService()
    
    # Set user preferences
    preferences = create_default_preferences(
        "test_user",
        email="test@example.com",
        phone="+1234567890"
    )
    notification_service.set_user_preferences("test_user", preferences)
    
    # Create test alert
    alert = RiskAlert(
        id="test_alert_001",
        threshold_id="threshold_001",
        portfolio_id="test_portfolio",
        metric_type=MetricType.VAR_95,
        current_value=-0.06,
        threshold_value=-0.05,
        severity=AlertSeverity.HIGH,
        status="triggered",
        description="Daily VaR (95%) exceeds -5%",
        triggered_at=datetime.now()
    )
    
    # Send notifications (will fail without actual SMTP/SMS config)
    deliveries = await notification_service.send_alert_notification(alert, "test_user")
    print(f"Created {len(deliveries)} notification deliveries")
    
    for delivery in deliveries:
        print(f"  - {delivery.notification_type}: {delivery.status}")
    
    # Get statistics
    stats = notification_service.get_delivery_statistics()
    print(f"\nNotification Statistics:")
    print(f"  Total notifications: {stats['total_notifications']}")
    print(f"  Successful deliveries: {stats['successful_deliveries']}")
    print(f"  Failed deliveries: {stats['failed_deliveries']}")
    print(f"  Delivery rate: {stats['delivery_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(test_notification_service())
