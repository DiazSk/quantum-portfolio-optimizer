"""
Professional Logging Configuration for Quantum Portfolio Optimizer
Production-grade logging system for FAANG-level data analytics project
"""

import logging
import logging.config
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


class PortfolioAnalyticsLogger:
    """
    Professional logging system for portfolio analytics
    Implements structured logging with JSON format for production deployment
    """
    
    def __init__(self, log_level="INFO", log_dir="logs"):
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log files
        self.log_files = {
            'main': self.log_dir / 'portfolio_optimizer.log',
            'performance': self.log_dir / 'performance_metrics.log',
            'risk': self.log_dir / 'risk_events.log',
            'ml_model': self.log_dir / 'ml_model_performance.log',
            'api': self.log_dir / 'api_requests.log',
            'audit': self.log_dir / 'audit_trail.log'
        }
        
        self._setup_logging()
        if STRUCTLOG_AVAILABLE:
            self._setup_structlog()
        
    def _setup_logging(self):
        """Configure Python logging with JSON formatter"""
        
        # Custom JSON formatter or fallback
        if JSON_LOGGER_AVAILABLE:
            class CustomJsonFormatter(jsonlogger.JsonFormatter):
                def add_fields(self, log_record, record, message_dict):
                    super().add_fields(log_record, record, message_dict)
                    log_record['timestamp'] = datetime.utcnow().isoformat()
                    log_record['level'] = record.levelname
                    log_record['module'] = record.module
                    log_record['function'] = record.funcName
                    log_record['line'] = record.lineno
                    log_record['process_id'] = os.getpid()
            
            json_formatter = {
                '()': CustomJsonFormatter,
                'format': '%(timestamp)s %(level)s %(name)s %(message)s'
            }
        else:
            # Fallback to standard formatter
            json_formatter = {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            }
                
        # Logging configuration
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': json_formatter,
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'detailed',
                    'stream': sys.stdout
                },
                'main_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.log_level,
                    'formatter': 'json' if JSON_LOGGER_AVAILABLE else 'detailed',
                    'filename': str(self.log_files['main']),
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5
                },
                'performance_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json' if JSON_LOGGER_AVAILABLE else 'detailed',
                    'filename': str(self.log_files['performance']),
                    'maxBytes': 10485760,
                    'backupCount': 5
                },
                'risk_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'WARNING',
                    'formatter': 'json',
                    'filename': str(self.log_files['risk']),
                    'maxBytes': 10485760,
                    'backupCount': 5
                },
                'ml_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': str(self.log_files['ml_model']),
                    'maxBytes': 10485760,
                    'backupCount': 5
                },
                'api_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': str(self.log_files['api']),
                    'maxBytes': 10485760,
                    'backupCount': 5
                },
                'audit_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': str(self.log_files['audit']),
                    'maxBytes': 10485760,
                    'backupCount': 10  # Keep more audit logs
                }
            },
            'loggers': {
                'portfolio_optimizer': {
                    'handlers': ['console', 'main_file'],
                    'level': self.log_level,
                    'propagate': False
                },
                'performance_metrics': {
                    'handlers': ['performance_file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'risk_events': {
                    'handlers': ['console', 'risk_file'],
                    'level': 'WARNING',
                    'propagate': False
                },
                'ml_model': {
                    'handlers': ['ml_file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'api_requests': {
                    'handlers': ['api_file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'audit_trail': {
                    'handlers': ['audit_file'],
                    'level': 'INFO',
                    'propagate': False
                }
            }
        }
        
        logging.config.dictConfig(config)
        
    def _setup_structlog(self):
        """Configure structured logging with structlog"""
        if not STRUCTLOG_AVAILABLE:
            return  # Skip if structlog not available
            
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
    def get_logger(self, name="portfolio_optimizer"):
        """Get configured logger instance"""
        return logging.getLogger(name)
    
    def get_structured_logger(self, name="portfolio_optimizer"):
        """Get structured logger instance"""
        if STRUCTLOG_AVAILABLE:
            return structlog.get_logger(name)
        else:
            # Fallback to regular logger if structlog not available
            return logging.getLogger(name)


class PerformanceMonitor:
    """
    Performance monitoring for portfolio analytics
    Tracks system performance, business metrics, and alerts
    """
    
    def __init__(self, logger_config):
        self.logger = logger_config.get_logger('performance_metrics')
        self.structured_logger = logger_config.get_structured_logger('performance_metrics')
        
    def log_portfolio_performance(self, portfolio_id, metrics, timestamp=None):
        """Log portfolio performance metrics"""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.structured_logger.info(
            "portfolio_performance_update",
            portfolio_id=portfolio_id,
            timestamp=timestamp.isoformat(),
            annual_return=metrics.get('annual_return'),
            volatility=metrics.get('volatility'),
            sharpe_ratio=metrics.get('sharpe_ratio'),
            max_drawdown=metrics.get('max_drawdown'),
            var_95=metrics.get('var_95'),
            beta=metrics.get('beta'),
            alpha=metrics.get('alpha')
        )
        
    def log_model_performance(self, model_name, metrics, validation_data):
        """Log ML model performance metrics"""
        self.structured_logger.info(
            "ml_model_performance",
            model_name=model_name,
            timestamp=datetime.utcnow().isoformat(),
            accuracy=metrics.get('accuracy'),
            precision=metrics.get('precision'),
            recall=metrics.get('recall'),
            f1_score=metrics.get('f1_score'),
            auc_roc=metrics.get('auc_roc'),
            validation_samples=len(validation_data),
            feature_importance=metrics.get('feature_importance', {}),
            prediction_confidence=metrics.get('prediction_confidence')
        )
        
    def log_risk_event(self, event_type, severity, details):
        """Log risk management events"""
        risk_logger = logging.getLogger('risk_events')
        
        risk_logger.warning(
            json.dumps({
                "event_type": event_type,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details,
                "requires_attention": severity in ['HIGH', 'CRITICAL']
            })
        )
        
    def log_api_request(self, endpoint, method, status_code, response_time, user_id=None):
        """Log API request metrics"""
        api_logger = logging.getLogger('api_requests')
        
        api_logger.info(
            json.dumps({
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time_ms": response_time,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "success": 200 <= status_code < 300
            })
        )
        
    def log_audit_event(self, action, user_id, resource, details=None):
        """Log audit trail events for compliance"""
        audit_logger = logging.getLogger('audit_trail')
        
        audit_logger.info(
            json.dumps({
                "action": action,
                "user_id": user_id,
                "resource": resource,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details or {},
                "session_id": getattr(self, 'session_id', None)
            })
        )


class AlertManager:
    """
    Alert management system for critical events
    Implements thresholds and notification logic
    """
    
    def __init__(self, logger_config):
        self.logger = logger_config.get_logger('risk_events')
        self.performance_monitor = PerformanceMonitor(logger_config)
        
        # Alert thresholds
        self.thresholds = {
            'max_drawdown': -0.15,  # -15% max drawdown
            'var_95': -0.05,        # -5% daily VaR
            'sharpe_ratio': 0.5,    # Minimum Sharpe ratio
            'model_accuracy': 0.6,  # Minimum model accuracy
            'api_response_time': 1000,  # 1 second max response time
            'portfolio_concentration': 0.3  # Max 30% in single asset
        }
        
    def check_portfolio_risk(self, portfolio_id, metrics):
        """Check portfolio metrics against risk thresholds"""
        alerts = []
        
        # Drawdown check
        if metrics.get('max_drawdown', 0) < self.thresholds['max_drawdown']:
            alerts.append({
                'type': 'MAX_DRAWDOWN_BREACH',
                'severity': 'HIGH',
                'value': metrics['max_drawdown'],
                'threshold': self.thresholds['max_drawdown']
            })
            
        # VaR check
        if metrics.get('var_95', 0) < self.thresholds['var_95']:
            alerts.append({
                'type': 'VAR_BREACH',
                'severity': 'MEDIUM',
                'value': metrics['var_95'],
                'threshold': self.thresholds['var_95']
            })
            
        # Sharpe ratio check
        if metrics.get('sharpe_ratio', 0) < self.thresholds['sharpe_ratio']:
            alerts.append({
                'type': 'LOW_SHARPE_RATIO',
                'severity': 'MEDIUM',
                'value': metrics['sharpe_ratio'],
                'threshold': self.thresholds['sharpe_ratio']
            })
            
        # Log all alerts
        for alert in alerts:
            self.performance_monitor.log_risk_event(
                event_type=alert['type'],
                severity=alert['severity'],
                details={
                    'portfolio_id': portfolio_id,
                    'metric_value': alert['value'],
                    'threshold': alert['threshold'],
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        return alerts
        
    def check_model_performance(self, model_name, metrics):
        """Check ML model performance thresholds"""
        alerts = []
        
        if metrics.get('accuracy', 0) < self.thresholds['model_accuracy']:
            alerts.append({
                'type': 'MODEL_ACCURACY_DEGRADATION',
                'severity': 'HIGH',
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'threshold': self.thresholds['model_accuracy']
            })
            
        for alert in alerts:
            self.performance_monitor.log_risk_event(
                event_type=alert['type'],
                severity=alert['severity'],
                details=alert
            )
            
        return alerts


# Example usage and configuration
def setup_production_logging():
    """
    Set up production-grade logging configuration
    Call this at the start of your application
    """
    # Initialize logging system
    logger_config = PortfolioAnalyticsLogger(
        log_level="INFO",
        log_dir="logs"
    )
    
    # Initialize monitoring
    performance_monitor = PerformanceMonitor(logger_config)
    alert_manager = AlertManager(logger_config)
    
    # Test logging
    logger = logger_config.get_logger()
    logger.info("Portfolio analytics logging system initialized")
    
    return logger_config, performance_monitor, alert_manager


if __name__ == "__main__":
    # Example usage
    logger_config, perf_monitor, alert_manager = setup_production_logging()
    
    # Example portfolio metrics
    sample_metrics = {
        'annual_return': 0.12,
        'volatility': 0.15,
        'sharpe_ratio': 0.8,
        'max_drawdown': -0.08,
        'var_95': -0.025,
        'beta': 1.1,
        'alpha': 0.02
    }
    
    # Log performance
    perf_monitor.log_portfolio_performance('portfolio_001', sample_metrics)
    
    # Log risk event
    perf_monitor.log_risk_event('VaR_BREACH', 'HIGH', {
        'portfolio_id': 'portfolio_001',
        'var_95': -0.035,
        'threshold': -0.03
    })


# Global instance for convenience functions
_global_logger_config = None
_global_performance_monitor = None

def get_global_logger():
    """Get global logger instance"""
    global _global_logger_config
    if _global_logger_config is None:
        _global_logger_config = PortfolioAnalyticsLogger()
    return _global_logger_config.get_logger()

def get_logger(name="portfolio_optimizer"):
    """Get logger instance - convenience function for imports"""
    global _global_logger_config
    if _global_logger_config is None:
        _global_logger_config = PortfolioAnalyticsLogger()
    return _global_logger_config.get_logger(name)

def log_risk_event(event_type: str, severity: str, details: dict):
    """Convenience function for logging risk events"""
    global _global_performance_monitor, _global_logger_config
    if _global_performance_monitor is None:
        if _global_logger_config is None:
            _global_logger_config = PortfolioAnalyticsLogger()
        _global_performance_monitor = PerformanceMonitor(_global_logger_config)
    
    _global_performance_monitor.log_risk_event(event_type, severity, details)
    # Check for alerts
    alerts = alert_manager.check_portfolio_risk('portfolio_001', sample_metrics)
    
    print(f"✅ Logging system setup complete")
    print(f"📊 Sample metrics logged for portfolio_001")
    print(f"🚨 Alerts generated: {len(alerts)}")
