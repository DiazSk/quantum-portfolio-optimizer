"""
Production Configuration for Quantum Portfolio Optimizer
FAANG-ready configuration management for data analytics project
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import json


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "portfolio_optimizer"
    user: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisConfig:
    """Redis configuration for caching"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    
    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class APIConfig:
    """API service configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: str = "100/minute"
    
    # API Keys (from environment)
    alpha_vantage_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    news_api_key: Optional[str] = None
    fmp_api_key: Optional[str] = None


@dataclass
class AnalyticsConfig:
    """Analytics and ML configuration"""
    default_lookback_years: int = 2
    rebalance_frequency: str = "Q"  # Q=Quarterly, M=Monthly
    risk_free_rate: float = 0.04
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.02
    
    # ML Model parameters
    model_retrain_days: int = 30
    validation_split: float = 0.2
    max_features: int = 50
    n_estimators: int = 100
    max_depth: int = 6
    
    # Risk thresholds
    max_position_size: float = 0.30
    max_drawdown_threshold: float = -0.15
    var_threshold: float = -0.05
    minimum_sharpe: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    log_dir: str = "logs"
    max_bytes: int = 10_485_760  # 10MB
    backup_count: int = 5
    
    # Log file names
    main_log: str = "portfolio_optimizer.log"
    performance_log: str = "performance_metrics.log"
    risk_log: str = "risk_events.log"
    api_log: str = "api_requests.log"
    audit_log: str = "audit_trail.log"


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    health_check_interval: int = 60
    
    # Alert settings
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    alert_threshold_high: float = 0.8
    alert_threshold_critical: float = 0.95
    
    # Performance monitoring
    track_response_times: bool = True
    track_memory_usage: bool = True
    track_model_performance: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # HTTPS settings
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Authentication
    enable_auth: bool = False
    jwt_secret: Optional[str] = None


class ProductionConfig:
    """
    Complete production configuration for Portfolio Optimizer
    Manages all configuration aspects with environment variable support
    """
    
    def __init__(self, config_file: Optional[str] = None, env_prefix: str = "PORTFOLIO_"):
        self.env_prefix = env_prefix
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        else:
            self._load_defaults()
        
        # Override with environment variables
        self._load_from_env()
        
    def _load_defaults(self):
        """Load default configuration"""
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.analytics = AnalyticsConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        self.database = DatabaseConfig(**config_data.get('database', {}))
        self.redis = RedisConfig(**config_data.get('redis', {}))
        self.api = APIConfig(**config_data.get('api', {}))
        self.analytics = AnalyticsConfig(**config_data.get('analytics', {}))
        self.logging = LoggingConfig(**config_data.get('logging', {}))
        self.monitoring = MonitoringConfig(**config_data.get('monitoring', {}))
        self.security = SecurityConfig(**config_data.get('security', {}))
        
    def _load_from_env(self):
        """Override configuration with environment variables"""
        # Database
        self.database.host = os.getenv(f'{self.env_prefix}DB_HOST', self.database.host)
        self.database.port = int(os.getenv(f'{self.env_prefix}DB_PORT', self.database.port))
        self.database.name = os.getenv(f'{self.env_prefix}DB_NAME', self.database.name)
        self.database.user = os.getenv(f'{self.env_prefix}DB_USER', self.database.user)
        self.database.password = os.getenv(f'{self.env_prefix}DB_PASSWORD', self.database.password)
        
        # Redis
        self.redis.host = os.getenv(f'{self.env_prefix}REDIS_HOST', self.redis.host)
        self.redis.port = int(os.getenv(f'{self.env_prefix}REDIS_PORT', self.redis.port))
        self.redis.password = os.getenv(f'{self.env_prefix}REDIS_PASSWORD', self.redis.password)
        
        # API Keys
        self.api.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.api.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.api.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.api.news_api_key = os.getenv('NEWS_API_KEY')
        self.api.fmp_api_key = os.getenv('FMP_API_KEY')
        
        # Security
        self.security.secret_key = os.getenv(f'{self.env_prefix}SECRET_KEY', self.security.secret_key)
        self.security.jwt_secret = os.getenv(f'{self.env_prefix}JWT_SECRET')
        
        # Environment-specific overrides
        env = os.getenv('ENVIRONMENT', 'development')
        if env == 'production':
            self._apply_production_overrides()
        elif env == 'testing':
            self._apply_testing_overrides()
            
    def _apply_production_overrides(self):
        """Apply production-specific configuration"""
        self.api.reload = False
        self.database.echo = False
        self.logging.level = "INFO"
        self.security.ssl_enabled = True
        self.security.enable_auth = True
        self.monitoring.enable_prometheus = True
        
    def _apply_testing_overrides(self):
        """Apply testing-specific configuration"""
        self.database.name = "portfolio_optimizer_test"
        self.logging.level = "DEBUG"
        self.analytics.model_retrain_days = 1  # Faster retraining for tests
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'api': self.api.__dict__,
            'analytics': self.analytics.__dict__,
            'logging': self.logging.__dict__,
            'monitoring': self.monitoring.__dict__,
            'security': {k: v for k, v in self.security.__dict__.items() if 'secret' not in k.lower()}
        }
        
    def save_to_file(self, config_file: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        with open(config_file, 'w') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
                
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required API keys
        if not self.api.alpha_vantage_key:
            issues.append("Alpha Vantage API key not configured")
        if not self.api.reddit_client_id:
            issues.append("Reddit API credentials not configured")
        if not self.api.news_api_key:
            issues.append("News API key not configured")
            
        # Check database connectivity
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.database.url)
            engine.connect()
        except Exception as e:
            issues.append(f"Database connection failed: {e}")
            
        # Validate analytics parameters
        if self.analytics.confidence_level <= 0 or self.analytics.confidence_level >= 1:
            issues.append("Confidence level must be between 0 and 1")
        if self.analytics.max_position_size <= 0 or self.analytics.max_position_size > 1:
            issues.append("Max position size must be between 0 and 1")
            
        return issues
        
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv('ENVIRONMENT', 'development') == 'production'
        
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return os.getenv('ENVIRONMENT', 'development') == 'testing'


# Global configuration instance
config: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get global configuration instance"""
    global config
    if config is None:
        config_file = os.getenv('CONFIG_FILE')
        config = ProductionConfig(config_file)
    return config


def setup_config(config_file: Optional[str] = None) -> ProductionConfig:
    """Setup configuration with optional file"""
    global config
    config = ProductionConfig(config_file)
    return config


# Configuration templates for different environments
DEVELOPMENT_CONFIG = {
    'database': {
        'host': 'localhost',
        'port': 5432,
        'name': 'portfolio_optimizer_dev',
        'echo': True
    },
    'logging': {
        'level': 'DEBUG'
    },
    'api': {
        'reload': True,
        'workers': 1
    },
    'monitoring': {
        'enable_prometheus': False
    }
}

PRODUCTION_CONFIG = {
    'database': {
        'host': 'prod-db-host',
        'pool_size': 20,
        'echo': False
    },
    'logging': {
        'level': 'INFO'
    },
    'api': {
        'reload': False,
        'workers': 8
    },
    'security': {
        'ssl_enabled': True,
        'enable_auth': True
    },
    'monitoring': {
        'enable_prometheus': True,
        'enable_email_alerts': True
    }
}

TESTING_CONFIG = {
    'database': {
        'name': 'portfolio_optimizer_test'
    },
    'logging': {
        'level': 'WARNING'
    },
    'analytics': {
        'model_retrain_days': 1
    }
}


def create_config_file(environment: str = 'development', output_file: str = 'config.yaml'):
    """Create configuration file for specific environment"""
    templates = {
        'development': DEVELOPMENT_CONFIG,
        'production': PRODUCTION_CONFIG,
        'testing': TESTING_CONFIG
    }
    
    template = templates.get(environment, DEVELOPMENT_CONFIG)
    
    with open(output_file, 'w') as f:
        yaml.dump(template, f, default_flow_style=False)
        
    print(f"✅ Configuration file created: {output_file}")
    print(f"🎯 Environment: {environment}")
    return output_file


if __name__ == "__main__":
    # Example usage
    config = ProductionConfig()
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration validated successfully")
    
    # Save example configurations
    for env in ['development', 'production', 'testing']:
        create_config_file(env, f'config_{env}.yaml')
    
    print("\n🎯 FAANG-Ready Configuration System:")
    print("  ✅ Environment-specific configurations")
    print("  ✅ Environment variable overrides")
    print("  ✅ Validation and error checking")
    print("  ✅ Production security features")
    print("  ✅ Comprehensive monitoring setup")
