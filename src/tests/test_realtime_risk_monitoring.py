"""
Comprehensive Testing Suite for Real-time Risk Monitoring System
Tests all components: monitor, alerts, notifications, escalation, WebSocket
"""

import pytest
import asyncio
import json
import time
import redis
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Import all components to test
from src.risk.realtime_monitor import RealTimeRiskMonitor, RiskMetricsSnapshot
from src.risk.alert_engine import AlertEngine, AlertThreshold, RiskAlert, AlertSeverity, AlertStatus
from src.utils.notification_service import NotificationService, NotificationTemplate, NotificationStatus
from src.risk.escalation_manager import EscalationManager, EscalationRule, EscalationEvent
from src.api.websocket_risk import WebSocketRiskAPI, ConnectionManager


class TestRealTimeRiskMonitor:
    """Test suite for RealTimeRiskMonitor"""
    
    @pytest.fixture
    def mock_risk_manager(self):
        """Mock risk manager for testing"""
        risk_manager = Mock()
        risk_manager.portfolio_data = {
            'portfolio_001': {
                'holdings': {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'TSLA': 0.15, 'AMZN': 0.1},
                'values': {'AAPL': 30000, 'GOOGL': 25000, 'MSFT': 20000, 'TSLA': 15000, 'AMZN': 10000}
            }
        }
        risk_manager.calculate_var = Mock(return_value=(-0.0342, -0.0598))  # 95%, 99%
        risk_manager.calculate_max_drawdown = Mock(return_value=-0.1245)
        risk_manager.calculate_performance_ratios = Mock(return_value={
            'sharpe_ratio': 1.23,
            'sortino_ratio': 1.45,
            'calmar_ratio': 0.98
        })
        return risk_manager
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing"""
        redis_client = Mock()
        redis_client.get = Mock(return_value=None)
        redis_client.set = Mock(return_value=True)
        redis_client.expire = Mock(return_value=True)
        redis_client.ping = Mock(return_value=True)
        return redis_client
    
    @pytest.fixture
    def monitor(self, mock_redis):
        """Create RealTimeRiskMonitor instance for testing"""
        # Mock Redis connection
        with patch('redis.Redis', return_value=mock_redis):
            return RealTimeRiskMonitor(redis_host='localhost', redis_port=6379, redis_db=0)
    
    def test_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.redis_client is not None
        assert monitor.monitoring_active is False
        assert monitor.refresh_interval == 30
        assert monitor.cache_ttl == 60
        assert monitor.cache_prefix == "risk_metrics"
        assert isinstance(monitor.websocket_clients, dict)
        assert isinstance(monitor.risk_callbacks, list)
    
    def test_calculate_realtime_metrics(self, monitor):
        """Test real-time metrics calculation"""
        portfolio_id = "portfolio_001"
        metrics = monitor.calculate_realtime_metrics(portfolio_id)
        
        assert isinstance(metrics, RiskMetricsSnapshot)
        assert metrics.portfolio_id == portfolio_id
        assert metrics.var_95 == -0.0342
        assert metrics.var_99 == -0.0598
        assert metrics.max_drawdown == -0.1245
        assert metrics.sharpe_ratio == 1.23
        
        # Test calculated fields
        assert 0 <= metrics.annual_volatility <= 1
        assert 0 <= metrics.daily_volatility <= 1
        assert metrics.leverage_ratio >= 1.0
        assert isinstance(metrics.concentration_risk, dict)
    
    def test_concentration_risk_calculation(self, monitor):
        """Test concentration risk calculation"""
        portfolio_id = "portfolio_001"
        metrics = monitor.calculate_realtime_metrics(portfolio_id)
        
        # Check concentration metrics
        concentration = metrics.concentration_risk
        assert 'max_position' in concentration
        assert 'top_5_concentration' in concentration
        assert 'effective_positions' in concentration
        
        # Max position should be 30% (AAPL)
        assert concentration['max_position'] == 0.30
        
        # Top 5 should be 100% (all positions)
        assert concentration['top_5_concentration'] == 1.0
        
        # Effective positions should be reasonable
        assert 3 <= concentration['effective_positions'] <= 6
    
    def test_cache_operations(self, monitor, mock_redis):
        """Test Redis caching operations"""
        portfolio_id = "portfolio_001"
        
        # Test cache miss
        mock_redis.get.return_value = None
        metrics = monitor.get_cached_metrics(portfolio_id)
        assert metrics is None
        
        # Test cache set
        new_metrics = monitor.calculate_realtime_metrics(portfolio_id)
        monitor.cache_metrics(portfolio_id, new_metrics)
        
        # Verify cache operations
        mock_redis.set.assert_called()
        mock_redis.expire.assert_called_with(f"risk_metrics:{portfolio_id}", 60)
    
    def test_cache_hit(self, monitor, mock_redis):
        """Test cache hit scenario"""
        portfolio_id = "portfolio_001"
        
        # Mock cached data
        cached_data = {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now().isoformat(),
            'var_95': -0.035,
            'var_99': -0.060,
            'max_drawdown': -0.125,
            'sharpe_ratio': 1.25,
            'concentration_risk': {'max_position': 0.30}
        }
        
        mock_redis.get.return_value = json.dumps(cached_data)
        
        metrics = monitor.get_cached_metrics(portfolio_id)
        assert metrics is not None
        assert metrics.portfolio_id == portfolio_id
        assert metrics.var_95 == -0.035
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitor):
        """Test monitoring lifecycle"""
        portfolio_ids = ["portfolio_001", "portfolio_002"]
        
        # Start monitoring
        monitor.start_monitoring(portfolio_ids, update_interval=1)
        assert monitor.monitoring_active is True
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring_active is False
    
    def test_leverage_calculation(self, monitor):
        """Test leverage ratio calculation"""
        # Test with leveraged portfolio
        monitor.risk_manager.portfolio_data['portfolio_001']['leverage'] = {
            'cash': -10000,  # Negative cash indicates borrowing
            'total_assets': 110000
        }
        
        metrics = monitor.calculate_realtime_metrics("portfolio_001")
        
        # Should detect leverage
        assert metrics.leverage_ratio > 1.0
    
    def test_error_handling(self, monitor):
        """Test error handling in metrics calculation"""
        # Test with invalid portfolio
        with patch.object(monitor.risk_manager, 'calculate_var', side_effect=Exception("Test error")):
            metrics = monitor.calculate_realtime_metrics("invalid_portfolio")
            
            # Should return metrics with default values
            assert metrics is not None
            assert metrics.portfolio_id == "invalid_portfolio"
            # Error values should be set to safe defaults
            assert metrics.var_95 == 0.0
            assert metrics.var_99 == 0.0


class TestAlertEngine:
    """Test suite for AlertEngine"""
    
    @pytest.fixture
    def mock_monitor(self):
        """Mock real-time monitor"""
        monitor = Mock()
        monitor.get_cached_metrics = Mock()
        return monitor
    
    @pytest.fixture
    def alert_engine(self, mock_monitor):
        """Create AlertEngine instance for testing"""
        return AlertEngine()
    
    @pytest.fixture
    def sample_thresholds(self):
        """Sample alert thresholds for testing"""
        return [
            AlertThreshold(
                portfolio_id="portfolio_001",
                metric_type="var_95",
                threshold_value=-0.05,
                comparison_type="less_than",
                severity=AlertSeverity.HIGH,
                description="VaR exceeds 5%"
            ),
            AlertThreshold(
                portfolio_id="portfolio_001",
                metric_type="max_drawdown",
                threshold_value=-0.20,
                comparison_type="less_than",
                severity=AlertSeverity.CRITICAL,
                description="Drawdown exceeds 20%"
            ),
            AlertThreshold(
                portfolio_id="portfolio_001",
                metric_type="concentration_risk.max_position",
                threshold_value=0.30,
                comparison_type="greater_than",
                severity=AlertSeverity.MEDIUM,
                description="Position concentration too high"
            )
        ]
    
    def test_initialization(self, alert_engine):
        """Test alert engine initialization"""
        assert isinstance(alert_engine.thresholds, dict)
        assert isinstance(alert_engine.active_alerts, dict)
        assert isinstance(alert_engine.alert_rules, dict)
        assert isinstance(alert_engine.alert_callbacks, list)
        assert isinstance(alert_engine.alert_history, list)
        assert alert_engine.max_alerts_per_minute == 10
        assert alert_engine.alert_suppression_window == 300
        assert alert_engine.auto_resolve_timeout == 3600
    
    def test_add_threshold(self, alert_engine, sample_thresholds):
        """Test adding alert thresholds"""
        threshold = sample_thresholds[0]
        alert_engine.add_threshold(threshold)
        
        assert len(alert_engine.thresholds) == 1
        assert alert_engine.thresholds[0] == threshold
    
    def test_check_thresholds_no_breach(self, alert_engine, sample_thresholds, mock_monitor):
        """Test threshold checking with no breaches"""
        # Setup thresholds
        for threshold in sample_thresholds:
            alert_engine.add_threshold(threshold)
        
        # Mock metrics within safe ranges
        safe_metrics = RiskMetricsSnapshot(
            portfolio_id="portfolio_001",
            timestamp=datetime.now(),
            var_95=-0.03,  # Better than -0.05 threshold
            max_drawdown=-0.15,  # Better than -0.20 threshold
            concentration_risk={'max_position': 0.25}  # Better than 0.30 threshold
        )
        
        mock_monitor.get_cached_metrics.return_value = safe_metrics
        
        # Check thresholds
        alerts = alert_engine.check_thresholds("portfolio_001")
        
        # Should not trigger any alerts
        assert len(alerts) == 0
        assert len(alert_engine.active_alerts) == 0
    
    def test_check_thresholds_with_breaches(self, alert_engine, sample_thresholds, mock_monitor):
        """Test threshold checking with breaches"""
        # Setup thresholds
        for threshold in sample_thresholds:
            alert_engine.add_threshold(threshold)
        
        # Mock metrics with breaches
        breach_metrics = RiskMetricsSnapshot(
            portfolio_id="portfolio_001",
            timestamp=datetime.now(),
            var_95=-0.07,  # Worse than -0.05 threshold
            max_drawdown=-0.25,  # Worse than -0.20 threshold
            concentration_risk={'max_position': 0.35}  # Worse than 0.30 threshold
        )
        
        mock_monitor.get_cached_metrics.return_value = breach_metrics
        
        # Check thresholds
        alerts = alert_engine.check_thresholds("portfolio_001")
        
        # Should trigger all 3 alerts
        assert len(alerts) == 3
        assert len(alert_engine.active_alerts) == 3
        
        # Check alert details
        var_alert = next(a for a in alerts if a.metric_type == "var_95")
        assert var_alert.severity == AlertSeverity.HIGH
        assert var_alert.current_value == -0.07
        assert var_alert.threshold_value == -0.05
        assert var_alert.status == AlertStatus.TRIGGERED
    
    def test_nested_metric_access(self, alert_engine, mock_monitor):
        """Test accessing nested metrics (concentration_risk.max_position)"""
        threshold = AlertThreshold(
            portfolio_id="portfolio_001",
            metric_type="concentration_risk.max_position",
            threshold_value=0.30,
            comparison_type="greater_than",
            severity=AlertSeverity.MEDIUM,
            description="Position concentration too high"
        )
        
        alert_engine.add_threshold(threshold)
        
        # Mock metrics with nested breach
        metrics = RiskMetricsSnapshot(
            portfolio_id="portfolio_001",
            timestamp=datetime.now(),
            concentration_risk={'max_position': 0.35, 'other_metric': 0.1}
        )
        
        mock_monitor.get_cached_metrics.return_value = metrics
        
        alerts = alert_engine.check_thresholds("portfolio_001")
        assert len(alerts) == 1
        assert alerts[0].current_value == 0.35
    
    def test_alert_suppression(self, alert_engine, sample_thresholds, mock_monitor):
        """Test alert suppression for duplicate alerts"""
        threshold = sample_thresholds[0]
        alert_engine.add_threshold(threshold)
        
        # Mock breaching metrics
        breach_metrics = RiskMetricsSnapshot(
            portfolio_id="portfolio_001",
            timestamp=datetime.now(),
            var_95=-0.07
        )
        
        mock_monitor.get_cached_metrics.return_value = breach_metrics
        
        # First check should trigger alert
        alerts1 = alert_engine.check_thresholds("portfolio_001")
        assert len(alerts1) == 1
        
        # Second check should suppress duplicate
        alerts2 = alert_engine.check_thresholds("portfolio_001")
        assert len(alerts2) == 0  # Suppressed
    
    def test_alert_resolution(self, alert_engine, sample_thresholds, mock_monitor):
        """Test alert auto-resolution when metrics improve"""
        threshold = sample_thresholds[0]
        alert_engine.add_threshold(threshold)
        
        # First: trigger alert with bad metrics
        bad_metrics = RiskMetricsSnapshot(
            portfolio_id="portfolio_001",
            timestamp=datetime.now(),
            var_95=-0.07
        )
        mock_monitor.get_cached_metrics.return_value = bad_metrics
        alerts = alert_engine.check_thresholds("portfolio_001")
        assert len(alerts) == 1
        
        # Second: resolve alert with good metrics
        good_metrics = RiskMetricsSnapshot(
            portfolio_id="portfolio_001",
            timestamp=datetime.now(),
            var_95=-0.03
        )
        mock_monitor.get_cached_metrics.return_value = good_metrics
        alerts = alert_engine.check_thresholds("portfolio_001")
        
        # Should auto-resolve the alert
        assert len(alert_engine.active_alerts) == 0


class TestNotificationService:
    """Test suite for NotificationService"""
    
    @pytest.fixture
    def mock_email_service(self):
        """Mock email service"""
        email_service = Mock()
        email_service.send_email = Mock(return_value=True)
        return email_service
    
    @pytest.fixture
    def mock_sms_service(self):
        """Mock SMS service"""
        sms_service = Mock()
        sms_service.send_sms = Mock(return_value=True)
        return sms_service
    
    @pytest.fixture
    def notification_service(self, mock_email_service, mock_sms_service):
        """Create NotificationService instance for testing"""
        return NotificationService(
            email_service=mock_email_service,
            sms_service=mock_sms_service
        )
    
    @pytest.fixture
    def sample_template(self):
        """Sample notification template"""
        return NotificationTemplate(
            template_id="risk_alert",
            subject_template="Risk Alert: {metric_type} for {portfolio_id}",
            body_template="""
            RISK ALERT: {severity}
            
            Portfolio: {portfolio_id}
            Metric: {metric_type}
            Current Value: {current_value}
            Threshold: {threshold_value}
            Time: {timestamp}
            
            Description: {description}
            """,
            notification_type="email"
        )
    
    def test_template_registration(self, notification_service, sample_template):
        """Test template registration"""
        notification_service.register_template(sample_template)
        
        assert "risk_alert" in notification_service.templates
        assert notification_service.templates["risk_alert"] == sample_template
    
    def test_email_notification(self, notification_service, sample_template, mock_email_service):
        """Test email notification sending"""
        notification_service.register_template(sample_template)
        
        # Send notification
        result = notification_service.send_notification(
            template_id="risk_alert",
            recipient="risk-manager@company.com",
            context={
                "portfolio_id": "portfolio_001",
                "metric_type": "var_95",
                "severity": "HIGH",
                "current_value": "-0.07",
                "threshold_value": "-0.05",
                "timestamp": "2024-01-15 10:30:00",
                "description": "VaR exceeds threshold"
            }
        )
        
        assert result.status == NotificationStatus.SENT
        mock_email_service.send_email.assert_called_once()
        
        # Check email content
        call_args = mock_email_service.send_email.call_args
        assert "Risk Alert: var_95 for portfolio_001" in call_args[1]['subject']
        assert "portfolio_001" in call_args[1]['body']
        assert "HIGH" in call_args[1]['body']
    
    def test_sms_notification(self, notification_service, mock_sms_service):
        """Test SMS notification sending"""
        # Register SMS template
        sms_template = NotificationTemplate(
            template_id="risk_alert_sms",
            subject_template="Risk Alert",
            body_template="RISK ALERT: {severity} - {metric_type} for {portfolio_id}: {current_value}",
            notification_type="sms"
        )
        
        notification_service.register_template(sms_template)
        
        # Send SMS
        result = notification_service.send_notification(
            template_id="risk_alert_sms",
            recipient="+1234567890",
            context={
                "portfolio_id": "portfolio_001",
                "metric_type": "var_95",
                "severity": "CRITICAL",
                "current_value": "-0.08"
            }
        )
        
        assert result.status == NotificationStatus.SENT
        mock_sms_service.send_sms.assert_called_once()
    
    def test_notification_failure_handling(self, notification_service, sample_template, mock_email_service):
        """Test notification failure handling"""
        notification_service.register_template(sample_template)
        
        # Mock email service failure
        mock_email_service.send_email.return_value = False
        
        result = notification_service.send_notification(
            template_id="risk_alert",
            recipient="risk-manager@company.com",
            context={"portfolio_id": "portfolio_001"}
        )
        
        assert result.status == NotificationStatus.FAILED
        assert "Failed to send" in result.error_message
    
    def test_template_rendering(self, notification_service, sample_template):
        """Test template rendering with context"""
        notification_service.register_template(sample_template)
        
        context = {
            "portfolio_id": "portfolio_001",
            "metric_type": "var_95",
            "severity": "HIGH",
            "current_value": "-0.07",
            "threshold_value": "-0.05",
            "timestamp": "2024-01-15 10:30:00",
            "description": "VaR exceeds threshold"
        }
        
        subject, body = notification_service._render_template(sample_template, context)
        
        assert "Risk Alert: var_95 for portfolio_001" == subject
        assert "Portfolio: portfolio_001" in body
        assert "RISK ALERT: HIGH" in body
        assert "Current Value: -0.07" in body


class TestEscalationManager:
    """Test suite for EscalationManager"""
    
    @pytest.fixture
    def mock_notification_service(self):
        """Mock notification service"""
        service = Mock()
        service.send_notification = Mock(return_value=Mock(status=NotificationStatus.SENT))
        return service
    
    @pytest.fixture
    def escalation_manager(self, mock_notification_service):
        """Create EscalationManager instance for testing"""
        return EscalationManager(notification_service=mock_notification_service)
    
    @pytest.fixture
    def sample_rules(self):
        """Sample escalation rules"""
        return [
            {
                'severity': AlertSeverity.CRITICAL,
                'initial_delay_minutes': 0,
                'escalation_steps': [
                    {
                        'level': 'immediate',
                        'delay_minutes': 0,
                        'recipients': ['on-call@company.com', '+1234567890'],
                        'notification_type': 'both'
                    },
                    {
                        'level': 'management',
                        'delay_minutes': 15,
                        'recipients': ['risk-manager@company.com', 'cro@company.com'],
                        'notification_type': 'email'
                    }
                ]
            },
            {
                'severity': AlertSeverity.HIGH,
                'initial_delay_minutes': 5,
                'escalation_steps': [
                    {
                        'level': 'team',
                        'delay_minutes': 0,
                        'recipients': ['risk-team@company.com'],
                        'notification_type': 'email'
                    },
                    {
                        'level': 'management',
                        'delay_minutes': 30,
                        'recipients': ['risk-manager@company.com'],
                        'notification_type': 'email'
                    }
                ]
            }
        ]
    
    def test_initialization(self, escalation_manager):
        """Test escalation manager initialization"""
        assert escalation_manager.notification_service is not None
        assert escalation_manager.escalation_rules == []
        assert escalation_manager.active_escalations == {}
    
    def test_add_escalation_rules(self, escalation_manager, sample_rules):
        """Test adding escalation rules"""
        escalation_manager.load_escalation_rules(sample_rules)
        
        assert len(escalation_manager.escalation_rules) == 2
        
        # Check critical rule
        critical_rule = next(r for r in escalation_manager.escalation_rules if r.severity == AlertSeverity.CRITICAL)
        assert critical_rule.initial_delay_minutes == 0
        assert len(critical_rule.escalation_steps) == 2
    
    def test_immediate_escalation(self, escalation_manager, sample_rules, mock_notification_service):
        """Test immediate escalation for critical alerts"""
        escalation_manager.load_escalation_rules(sample_rules)
        
        # Create critical alert
        alert = RiskAlert(
            id="alert_001",
            portfolio_id="portfolio_001",
            metric_type="var_95",
            threshold_value=-0.05,
            current_value=-0.10,
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.TRIGGERED,
            description="Critical VaR breach",
            triggered_at=datetime.now()
        )
        
        # Start escalation
        escalation_manager.start_escalation(alert)
        
        # Should immediately notify for critical alert
        assert len(escalation_manager.active_escalations) == 1
        
        # Should have sent immediate notifications
        assert mock_notification_service.send_notification.called
    
    def test_delayed_escalation(self, escalation_manager, sample_rules):
        """Test delayed escalation for high alerts"""
        escalation_manager.load_escalation_rules(sample_rules)
        
        # Create high severity alert
        alert = RiskAlert(
            id="alert_002",
            portfolio_id="portfolio_001",
            metric_type="max_drawdown",
            threshold_value=-0.20,
            current_value=-0.25,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.TRIGGERED,
            description="High drawdown",
            triggered_at=datetime.now()
        )
        
        # Start escalation
        escalation_manager.start_escalation(alert)
        
        # Should be in active escalations
        assert len(escalation_manager.active_escalations) == 1
        
        # Should schedule future escalation steps
        escalation = escalation_manager.active_escalations[alert.id]
        assert len(escalation.scheduled_steps) > 0
    
    def test_escalation_cancellation(self, escalation_manager, sample_rules):
        """Test escalation cancellation when alert resolves"""
        escalation_manager.load_escalation_rules(sample_rules)
        
        # Start escalation
        alert = RiskAlert(
            id="alert_003",
            portfolio_id="portfolio_001",
            metric_type="var_95",
            threshold_value=-0.05,
            current_value=-0.07,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.TRIGGERED,
            description="VaR breach",
            triggered_at=datetime.now()
        )
        
        escalation_manager.start_escalation(alert)
        assert len(escalation_manager.active_escalations) == 1
        
        # Cancel escalation
        escalation_manager.cancel_escalation(alert.id, "Alert resolved")
        assert len(escalation_manager.active_escalations) == 0
    
    @pytest.mark.asyncio
    async def test_escalation_processing(self, escalation_manager, sample_rules, mock_notification_service):
        """Test escalation step processing"""
        escalation_manager.load_escalation_rules(sample_rules)
        
        # Start escalation for high alert
        alert = RiskAlert(
            id="alert_004",
            portfolio_id="portfolio_001",
            metric_type="var_95",
            threshold_value=-0.05,
            current_value=-0.07,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.TRIGGERED,
            description="VaR breach",
            triggered_at=datetime.now()
        )
        
        escalation_manager.start_escalation(alert)
        
        # Process immediate steps (should happen right away)
        await escalation_manager.process_escalations()
        
        # Should have sent notifications
        assert mock_notification_service.send_notification.called


class TestWebSocketRiskAPI:
    """Test suite for WebSocket Risk API"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies"""
        return {
            'monitor': Mock(),
            'alert_engine': Mock(),
            'escalation_manager': Mock()
        }
    
    @pytest.fixture
    def websocket_api(self, mock_dependencies):
        """Create WebSocket API instance"""
        return WebSocketRiskAPI(
            realtime_monitor=mock_dependencies['monitor'],
            alert_engine=mock_dependencies['alert_engine'],
            escalation_manager=mock_dependencies['escalation_manager']
        )
    
    def test_connection_manager_initialization(self, websocket_api):
        """Test connection manager initialization"""
        manager = websocket_api.connection_manager
        assert len(manager.active_connections) == 0
        assert len(manager.portfolio_subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, websocket_api):
        """Test WebSocket connection handling"""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.receive_text = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        # Mock authentication
        with patch('src.api.websocket_risk.verify_websocket_token', return_value={'user_id': 'test_user'}):
            # Simulate connection
            await websocket_api.connection_manager.connect(mock_websocket, 'test_user')
            
            assert len(websocket_api.connection_manager.active_connections) == 1
            assert 'test_user' in websocket_api.connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_risk_update(self, websocket_api):
        """Test broadcasting risk updates"""
        # Setup mock connections
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        websocket_api.connection_manager.active_connections = {
            'user1': mock_websocket1,
            'user2': mock_websocket2
        }
        
        websocket_api.connection_manager.portfolio_subscriptions = {
            'portfolio_001': ['user1', 'user2']
        }
        
        # Broadcast update
        await websocket_api.broadcast_risk_update(
            portfolio_id='portfolio_001',
            metrics={'var_95': -0.05, 'max_drawdown': -0.15}
        )
        
        # Both users should receive the update
        mock_websocket1.send_text.assert_called_once()
        mock_websocket2.send_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_alert_notification(self, websocket_api):
        """Test alert notification via WebSocket"""
        # Setup mock connection
        mock_websocket = AsyncMock()
        websocket_api.connection_manager.active_connections = {'user1': mock_websocket}
        websocket_api.connection_manager.portfolio_subscriptions = {'portfolio_001': ['user1']}
        
        # Create mock alert
        alert = RiskAlert(
            id="alert_001",
            portfolio_id="portfolio_001",
            metric_type="var_95",
            threshold_value=-0.05,
            current_value=-0.07,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.TRIGGERED,
            description="VaR breach",
            triggered_at=datetime.now()
        )
        
        # Send alert notification
        await websocket_api.broadcast_alert(alert)
        
        # Should send alert to subscribed user
        mock_websocket.send_text.assert_called_once()
        
        # Check message content
        call_args = mock_websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message['type'] == 'new_alert'
        assert message['alert']['id'] == "alert_001"
        assert message['alert']['severity'] == 'HIGH'


class TestIntegrationWorkflows:
    """Integration tests for complete workflows"""
    
    @pytest.fixture
    def full_system(self):
        """Setup complete integrated system"""
        # Mock external dependencies
        mock_risk_manager = Mock()
        mock_redis = Mock()
        mock_email_service = Mock()
        mock_sms_service = Mock()
        
        # Create components
        monitor = RealTimeRiskMonitor(mock_risk_manager, mock_redis)
        notification_service = NotificationService(mock_email_service, mock_sms_service)
        alert_engine = AlertEngine(monitor)
        escalation_manager = EscalationManager(notification_service)
        websocket_api = WebSocketRiskAPI(monitor, alert_engine, escalation_manager)
        
        return {
            'monitor': monitor,
            'alert_engine': alert_engine,
            'notification_service': notification_service,
            'escalation_manager': escalation_manager,
            'websocket_api': websocket_api,
            'mocks': {
                'risk_manager': mock_risk_manager,
                'redis': mock_redis,
                'email': mock_email_service,
                'sms': mock_sms_service
            }
        }
    
    def test_complete_alert_workflow(self, full_system):
        """Test complete workflow from risk calculation to notification"""
        system = full_system
        
        # Setup risk data
        system['mocks']['risk_manager'].portfolio_data = {
            'portfolio_001': {
                'holdings': {'AAPL': 0.6, 'GOOGL': 0.4},  # High concentration
                'values': {'AAPL': 60000, 'GOOGL': 40000}
            }
        }
        system['mocks']['risk_manager'].calculate_var = Mock(return_value=(-0.08, -0.12))  # High VaR
        
        # Setup alert threshold
        threshold = AlertThreshold(
            portfolio_id="portfolio_001",
            metric_type="var_95",
            threshold_value=-0.05,
            comparison_type="less_than",
            severity=AlertSeverity.CRITICAL,
            description="Critical VaR breach"
        )
        system['alert_engine'].add_threshold(threshold)
        
        # Setup escalation rules
        escalation_rules = [{
            'severity': AlertSeverity.CRITICAL,
            'initial_delay_minutes': 0,
            'escalation_steps': [{
                'level': 'immediate',
                'delay_minutes': 0,
                'recipients': ['risk-manager@company.com'],
                'notification_type': 'email'
            }]
        }]
        system['escalation_manager'].load_escalation_rules(escalation_rules)
        
        # Setup notification template
        template = NotificationTemplate(
            template_id="risk_alert",
            subject_template="Critical Risk Alert: {portfolio_id}",
            body_template="VaR breach: {current_value} exceeds {threshold_value}",
            notification_type="email"
        )
        system['notification_service'].register_template(template)
        
        # Execute workflow
        # 1. Calculate metrics
        metrics = system['monitor'].calculate_realtime_metrics("portfolio_001")
        assert metrics.var_95 == -0.08
        
        # 2. Check alerts
        alerts = system['alert_engine'].check_thresholds("portfolio_001")
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        
        # 3. Start escalation
        system['escalation_manager'].start_escalation(alerts[0])
        
        # 4. Verify notification sent
        system['mocks']['email'].send_email.assert_called()
    
    def test_performance_under_load(self, full_system):
        """Test system performance under load"""
        system = full_system
        
        # Setup multiple portfolios
        portfolio_ids = [f"portfolio_{i:03d}" for i in range(100)]
        
        # Setup mock data for all portfolios
        for pid in portfolio_ids:
            system['mocks']['risk_manager'].portfolio_data[pid] = {
                'holdings': {'AAPL': 0.5, 'GOOGL': 0.5},
                'values': {'AAPL': 50000, 'GOOGL': 50000}
            }
        
        system['mocks']['risk_manager'].calculate_var = Mock(return_value=(-0.03, -0.05))
        
        # Time the metrics calculation
        start_time = time.time()
        
        for pid in portfolio_ids:
            metrics = system['monitor'].calculate_realtime_metrics(pid)
            assert metrics is not None
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should process 100 portfolios in reasonable time (< 10 seconds)
        assert duration < 10.0
        
        # Performance metrics
        portfolios_per_second = len(portfolio_ids) / duration
        print(f"Performance: {portfolios_per_second:.1f} portfolios/second")
        
        # Should handle at least 10 portfolios per second
        assert portfolios_per_second >= 10
    
    @pytest.mark.asyncio
    async def test_websocket_stress_test(self, full_system):
        """Test WebSocket handling under stress"""
        system = full_system
        
        # Create multiple mock connections
        connections = {}
        for i in range(50):
            mock_ws = AsyncMock()
            user_id = f"user_{i}"
            connections[user_id] = mock_ws
            system['websocket_api'].connection_manager.active_connections[user_id] = mock_ws
            system['websocket_api'].connection_manager.portfolio_subscriptions[f"portfolio_{i % 10}"] = [user_id]
        
        # Broadcast multiple updates rapidly
        for i in range(100):
            await system['websocket_api'].broadcast_risk_update(
                portfolio_id=f"portfolio_{i % 10}",
                metrics={'var_95': -0.03 - (i * 0.001), 'timestamp': datetime.now().isoformat()}
            )
        
        # All connections should still be working
        for mock_ws in connections.values():
            assert mock_ws.send_text.called


# Performance and stress test configurations
pytest_plugins = ["pytest_asyncio"]

# Test configuration
TEST_CONFIG = {
    'redis_url': 'redis://localhost:6379/1',  # Use test database
    'test_portfolios': ['portfolio_test_001', 'portfolio_test_002', 'portfolio_test_003'],
    'performance_thresholds': {
        'metrics_calculation_ms': 100,  # Max 100ms per portfolio
        'alert_check_ms': 50,  # Max 50ms per alert check
        'notification_send_ms': 1000,  # Max 1s per notification
        'websocket_broadcast_ms': 100  # Max 100ms per broadcast
    }
}


if __name__ == "__main__":
    # Run specific test suites
    import sys
    
    if len(sys.argv) > 1:
        test_suite = sys.argv[1]
        if test_suite == "unit":
            pytest.main(["-v", "test_realtime_risk_monitoring.py::TestRealTimeRiskMonitor"])
        elif test_suite == "integration":
            pytest.main(["-v", "test_realtime_risk_monitoring.py::TestIntegrationWorkflows"])
        elif test_suite == "performance":
            pytest.main(["-v", "test_realtime_risk_monitoring.py::TestIntegrationWorkflows::test_performance_under_load"])
        else:
            pytest.main(["-v", "test_realtime_risk_monitoring.py"])
    else:
        # Run all tests
        pytest.main(["-v", "test_realtime_risk_monitoring.py"])
