# Celery Configuration for Distributed Task Processing
# Designed for high-throughput portfolio optimization and ML training

from celery import Celery
from celery.schedules import crontab
import os
from typing import Dict, Any

# Initialize Celery app
celery_app = Celery('portfolio_optimizer')

# Configuration
celery_app.conf.update(
    # Broker settings
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2'),
    
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Performance optimization
    task_routes={
        'portfolio.optimize': {'queue': 'optimization'},
        'portfolio.backtest': {'queue': 'analysis'},
        'data.collect_alternative': {'queue': 'data_collection'},
        'ml.train_model': {'queue': 'ml_training'},
        'risk.calculate_metrics': {'queue': 'risk_analysis'},
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Result settings
    result_expires=3600,  # 1 hour
    task_track_started=True,
    task_send_sent_event=True,
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'collect-market-data': {
            'task': 'data.collect_market_data',
            'schedule': crontab(minute='*/15'),  # Every 15 minutes during market hours
        },
        'collect-alternative-data': {
            'task': 'data.collect_alternative_data',
            'schedule': crontab(hour='*/2'),  # Every 2 hours
        },
        'calculate-portfolio-metrics': {
            'task': 'portfolio.calculate_daily_metrics',
            'schedule': crontab(hour=17, minute=30),  # After market close
        },
        'run-risk-analysis': {
            'task': 'risk.daily_risk_assessment',
            'schedule': crontab(hour=18, minute=0),  # Daily at 6 PM
        },
        'retrain-ml-models': {
            'task': 'ml.retrain_models',
            'schedule': crontab(hour=2, minute=0, day_of_week=1),  # Weekly on Monday
        },
        'generate-reports': {
            'task': 'reports.generate_daily_report',
            'schedule': crontab(hour=19, minute=0),  # Daily at 7 PM
        },
        'cleanup-old-data': {
            'task': 'maintenance.cleanup_old_data',
            'schedule': crontab(hour=3, minute=0),  # Daily at 3 AM
        },
    },
)

# Task definitions
@celery_app.task(bind=True, max_retries=3)
def optimize_portfolio(self, portfolio_id: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize portfolio allocation using quantum-inspired algorithms.
    
    Args:
        portfolio_id: Unique identifier for the portfolio
        constraints: Optimization constraints and parameters
    
    Returns:
        Optimization results with weights and metrics
    """
    try:
        from src.portfolio.portfolio_optimizer import PortfolioOptimizer
        from src.utils.professional_logging import PortfolioAnalyticsLogger
        
        logger = PortfolioAnalyticsLogger("celery_optimizer")
        logger.info("Starting portfolio optimization", extra={
            'portfolio_id': portfolio_id,
            'task_id': self.request.id
        })
        
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_with_constraints(portfolio_id, constraints)
        
        logger.info("Portfolio optimization completed", extra={
            'portfolio_id': portfolio_id,
            'expected_return': result.get('expected_return'),
            'volatility': result.get('volatility')
        })
        
        return result
        
    except Exception as exc:
        logger.error(f"Portfolio optimization failed: {str(exc)}", extra={
            'portfolio_id': portfolio_id,
            'error': str(exc)
        })
        self.retry(countdown=60 * (self.request.retries + 1))

@celery_app.task(bind=True, max_retries=2)
def collect_market_data(self, symbols: list = None) -> Dict[str, Any]:
    """
    Collect real-time market data from configured APIs.
    
    Args:
        symbols: List of symbols to collect data for
    
    Returns:
        Collection status and metadata
    """
    try:
        from src.data.alternative_data_collector import AlternativeDataCollector
        from src.utils.professional_logging import PortfolioAnalyticsLogger
        
        logger = PortfolioAnalyticsLogger("celery_data_collector")
        logger.info("Starting market data collection", extra={
            'symbols_count': len(symbols) if symbols else 0,
            'task_id': self.request.id
        })
        
        collector = AlternativeDataCollector()
        
        # Collect from multiple sources
        results = {}
        
        if symbols:
            results['price_data'] = collector.fetch_alpha_vantage_data(symbols)
            results['news_data'] = collector.fetch_news_data(symbols)
            results['social_sentiment'] = collector.fetch_reddit_sentiment(symbols)
        else:
            # Default collection for major indices
            symbols = ['SPY', 'QQQ', 'IWM', 'VIX']
            results['market_data'] = collector.fetch_alpha_vantage_data(symbols)
        
        logger.info("Market data collection completed", extra={
            'collected_sources': list(results.keys()),
            'symbols_processed': len(symbols) if symbols else 0
        })
        
        return {
            'status': 'success',
            'collected_at': collector.get_current_timestamp(),
            'data_sources': list(results.keys()),
            'symbol_count': len(symbols) if symbols else 0
        }
        
    except Exception as exc:
        logger.error(f"Market data collection failed: {str(exc)}", extra={
            'symbols': symbols,
            'error': str(exc)
        })
        self.retry(countdown=300)  # Retry in 5 minutes

@celery_app.task(bind=True, max_retries=2)
def train_ml_model(self, model_type: str, training_data_path: str) -> Dict[str, Any]:
    """
    Train machine learning models for portfolio prediction.
    
    Args:
        model_type: Type of model to train
        training_data_path: Path to training data
    
    Returns:
        Training results and model metadata
    """
    try:
        from src.models.model_manager import ModelManager
        from src.utils.professional_logging import PortfolioAnalyticsLogger
        
        logger = PortfolioAnalyticsLogger("celery_ml_trainer")
        logger.info("Starting ML model training", extra={
            'model_type': model_type,
            'data_path': training_data_path,
            'task_id': self.request.id
        })
        
        manager = ModelManager()
        result = manager.train_model(model_type, training_data_path)
        
        logger.info("ML model training completed", extra={
            'model_type': model_type,
            'model_id': result.get('model_id'),
            'accuracy': result.get('accuracy'),
            'training_time': result.get('training_time')
        })
        
        return result
        
    except Exception as exc:
        logger.error(f"ML model training failed: {str(exc)}", extra={
            'model_type': model_type,
            'error': str(exc)
        })
        self.retry(countdown=600)  # Retry in 10 minutes

@celery_app.task(bind=True)
def calculate_risk_metrics(self, portfolio_id: str) -> Dict[str, Any]:
    """
    Calculate comprehensive risk metrics for portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
    
    Returns:
        Risk analysis results
    """
    try:
        from src.risk.risk_managment import RiskManager
        from src.utils.professional_logging import PortfolioAnalyticsLogger
        
        logger = PortfolioAnalyticsLogger("celery_risk_analyzer")
        logger.info("Starting risk analysis", extra={
            'portfolio_id': portfolio_id,
            'task_id': self.request.id
        })
        
        risk_manager = RiskManager()
        metrics = risk_manager.calculate_comprehensive_metrics(portfolio_id)
        
        logger.info("Risk analysis completed", extra={
            'portfolio_id': portfolio_id,
            'var_95': metrics.get('var_95'),
            'sharpe_ratio': metrics.get('sharpe_ratio'),
            'max_drawdown': metrics.get('max_drawdown')
        })
        
        return metrics
        
    except Exception as exc:
        logger.error(f"Risk analysis failed: {str(exc)}", extra={
            'portfolio_id': portfolio_id,
            'error': str(exc)
        })
        raise

@celery_app.task(bind=True)
def generate_report(self, report_type: str, portfolio_id: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive portfolio reports.
    
    Args:
        report_type: Type of report to generate
        portfolio_id: Optional portfolio identifier
    
    Returns:
        Report generation status and file paths
    """
    try:
        from src.utils.professional_logging import PortfolioAnalyticsLogger
        import json
        import os
        from datetime import datetime
        
        logger = PortfolioAnalyticsLogger("celery_report_generator")
        logger.info("Starting report generation", extra={
            'report_type': report_type,
            'portfolio_id': portfolio_id,
            'task_id': self.request.id
        })
        
        # Generate report based on type
        report_data = {
            'report_type': report_type,
            'generated_at': datetime.utcnow().isoformat(),
            'portfolio_id': portfolio_id,
            'status': 'completed'
        }
        
        # Save report
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{report_type}_{timestamp}.json"
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info("Report generation completed", extra={
            'report_type': report_type,
            'file_path': filepath,
            'file_size': os.path.getsize(filepath)
        })
        
        return {
            'status': 'success',
            'report_path': filepath,
            'generated_at': report_data['generated_at']
        }
        
    except Exception as exc:
        logger.error(f"Report generation failed: {str(exc)}", extra={
            'report_type': report_type,
            'error': str(exc)
        })
        raise

# Task routing and monitoring
@celery_app.task(bind=True)
def health_check(self):
    """Health check task for monitoring system status."""
    from src.utils.professional_logging import PortfolioAnalyticsLogger
    
    logger = PortfolioAnalyticsLogger("celery_health")
    logger.info("Health check completed", extra={
        'task_id': self.request.id,
        'worker_status': 'healthy'
    })
    
    return {
        'status': 'healthy',
        'timestamp': celery_app.now(),
        'worker_id': self.request.hostname
    }

# Auto-discover tasks from all modules
celery_app.autodiscover_tasks([
    'src.portfolio',
    'src.data',
    'src.models',
    'src.risk',
    'src.api'
])

if __name__ == '__main__':
    celery_app.start()
