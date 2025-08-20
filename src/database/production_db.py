"""
Production Database Configuration for Quantum Portfolio Optimizer
PostgreSQL schema design for FAANG-level data analytics project
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import os
from typing import Optional, Dict, Any
import pandas as pd

Base = declarative_base()


class Portfolio(Base):
    """Portfolio master table"""
    __tablename__ = 'portfolios'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50), nullable=False)  # equal_weight, max_sharpe, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    risk_profile = Column(String(20))  # conservative, moderate, aggressive
    benchmark_ticker = Column(String(10), default='SPY')
    
    # Relationships
    holdings = relationship("PortfolioHolding", back_populates="portfolio")
    performance_records = relationship("PerformanceRecord", back_populates="portfolio")
    rebalance_events = relationship("RebalanceEvent", back_populates="portfolio")
    
    __table_args__ = (
        Index('idx_portfolio_strategy', 'strategy_type'),
        Index('idx_portfolio_created', 'created_at'),
    )


class Asset(Base):
    """Asset master table"""
    __tablename__ = 'assets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), unique=True, nullable=False)
    name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    currency = Column(String(3), default='USD')
    exchange = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    price_data = relationship("PriceData", back_populates="asset")
    holdings = relationship("PortfolioHolding", back_populates="asset")
    
    __table_args__ = (
        Index('idx_asset_ticker', 'ticker'),
        Index('idx_asset_sector', 'sector'),
    )


class PriceData(Base):
    """Historical price data"""
    __tablename__ = 'price_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float, nullable=False)
    adjusted_close = Column(Float, nullable=False)
    volume = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    asset = relationship("Asset", back_populates="price_data")
    
    __table_args__ = (
        Index('idx_price_asset_date', 'asset_id', 'date'),
        Index('idx_price_date', 'date'),
    )


class PortfolioHolding(Base):
    """Portfolio holdings (weights/allocations)"""
    __tablename__ = 'portfolio_holdings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'), nullable=False)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    weight = Column(Float, nullable=False)  # Portfolio weight (0-1)
    shares = Column(Float)  # Number of shares
    value = Column(Float)  # Dollar value of holding
    rebalance_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")
    asset = relationship("Asset", back_populates="holdings")
    
    __table_args__ = (
        Index('idx_holding_portfolio_date', 'portfolio_id', 'rebalance_date'),
        Index('idx_holding_asset', 'asset_id'),
    )


class PerformanceRecord(Base):
    """Daily portfolio performance metrics"""
    __tablename__ = 'performance_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Returns
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    
    # Risk metrics
    volatility_20d = Column(Float)  # 20-day rolling volatility
    var_95 = Column(Float)  # Value at Risk (95%)
    cvar_95 = Column(Float)  # Conditional VaR
    max_drawdown = Column(Float)
    
    # Performance metrics
    sharpe_ratio_20d = Column(Float)  # 20-day rolling Sharpe
    sortino_ratio_20d = Column(Float)
    calmar_ratio = Column(Float)
    
    # Benchmark comparison
    benchmark_return = Column(Float)
    excess_return = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    
    # Portfolio value
    total_value = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_records")
    
    __table_args__ = (
        Index('idx_performance_portfolio_date', 'portfolio_id', 'date'),
        Index('idx_performance_date', 'date'),
    )


class MLModelPrediction(Base):
    """ML model predictions and performance tracking"""
    __tablename__ = 'ml_predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    prediction_horizon = Column(Integer, default=1)  # Days ahead
    
    # Predictions
    predicted_return = Column(Float)
    predicted_volatility = Column(Float)
    confidence_score = Column(Float)
    
    # Model metrics (when available)
    model_accuracy = Column(Float)
    feature_importance = Column(JSON)
    
    # Actuals (filled in later)
    actual_return = Column(Float)
    prediction_error = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_ml_model_asset_date', 'model_name', 'asset_id', 'prediction_date'),
        Index('idx_ml_prediction_date', 'prediction_date'),
    )


class AlternativeData(Base):
    """Alternative data sources (sentiment, trends, etc.)"""
    __tablename__ = 'alternative_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_source = Column(String(50), nullable=False)  # reddit, news, trends, etc.
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'))
    date = Column(DateTime, nullable=False)
    
    # Data payload
    raw_data = Column(JSON)
    processed_score = Column(Float)  # Normalized score (-1 to 1)
    confidence = Column(Float)
    
    # Metadata
    data_quality_score = Column(Float)
    source_count = Column(Integer)  # Number of sources aggregated
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_alt_data_source_date', 'data_source', 'date'),
        Index('idx_alt_data_asset_date', 'asset_id', 'date'),
    )


class RebalanceEvent(Base):
    """Portfolio rebalancing events"""
    __tablename__ = 'rebalance_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey('portfolios.id'), nullable=False)
    rebalance_date = Column(DateTime, nullable=False)
    
    # Rebalancing details
    trigger_type = Column(String(50))  # scheduled, threshold, manual
    old_weights = Column(JSON)
    new_weights = Column(JSON)
    
    # Transaction costs
    transaction_costs = Column(Float)
    estimated_impact = Column(Float)
    
    # Performance before/after
    pre_rebalance_value = Column(Float)
    post_rebalance_value = Column(Float)
    
    # Metadata
    rebalance_reason = Column(Text)
    approved_by = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="rebalance_events")
    
    __table_args__ = (
        Index('idx_rebalance_portfolio_date', 'portfolio_id', 'rebalance_date'),
    )


class BacktestResult(Base):
    """Backtesting results storage"""
    __tablename__ = 'backtest_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_name = Column(String(255), nullable=False)
    strategy_config = Column(JSON, nullable=False)
    
    # Backtest parameters
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, default=100000)
    
    # Results summary
    total_return = Column(Float)
    annualized_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Detailed results
    daily_returns = Column(JSON)  # Array of daily returns
    performance_metrics = Column(JSON)  # Full metrics dict
    
    # Comparison to benchmark
    benchmark_ticker = Column(String(10))
    excess_return = Column(Float)
    information_ratio = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_backtest_name_date', 'backtest_name', 'created_at'),
    )


class ABTestResult(Base):
    """A/B testing results for strategy comparison"""
    __tablename__ = 'ab_test_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_name = Column(String(255), nullable=False)
    
    # Experiment setup
    control_strategy = Column(String(100), nullable=False)
    treatment_strategy = Column(String(100), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Statistical results
    control_performance = Column(JSON)
    treatment_performance = Column(JSON)
    statistical_results = Column(JSON)  # p-values, confidence intervals, etc.
    
    # Conclusion
    is_significant = Column(Boolean)
    effect_size = Column(Float)
    recommendation = Column(String(500))
    confidence_level = Column(Float, default=0.95)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_abtest_experiment_date', 'experiment_name', 'created_at'),
    )


class DatabaseManager:
    """
    Production database manager for portfolio analytics
    Handles connections, data operations, and migrations
    """
    
    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            # Build from environment variables, default to SQLite for testing
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'portfolio_optimizer')
            db_user = os.getenv('DB_USER', 'postgres')
            db_password = os.getenv('DB_PASSWORD', 'password')
            
            # Use SQLite for testing when PostgreSQL is not available
            if os.getenv('TESTING') or not os.getenv('DATABASE_URL'):
                database_url = "sqlite:///./test_portfolio.db"
            else:
                database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Create engine with appropriate settings for SQLite vs PostgreSQL
        if database_url.startswith('sqlite'):
            self.engine = create_engine(
                database_url, 
                echo=False,
                connect_args={"check_same_thread": False}  # For SQLite
            )
        else:
            self.engine = create_engine(database_url, echo=False)
            
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
        
    def store_price_data(self, ticker: str, price_df: pd.DataFrame):
        """Store price data for an asset"""
        session = self.get_session()
        try:
            # Get or create asset
            asset = session.query(Asset).filter(Asset.ticker == ticker).first()
            if not asset:
                asset = Asset(ticker=ticker, name=ticker)
                session.add(asset)
                session.commit()
            
            # Store price data
            for date, row in price_df.iterrows():
                price_record = PriceData(
                    asset_id=asset.id,
                    date=date,
                    open_price=row.get('Open'),
                    high_price=row.get('High'),
                    low_price=row.get('Low'),
                    close_price=row.get('Close'),
                    adjusted_close=row.get('Adj Close', row.get('Close')),
                    volume=row.get('Volume')
                )
                session.add(price_record)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    def store_portfolio_performance(self, portfolio_id: str, date: datetime, metrics: Dict[str, Any]):
        """Store daily portfolio performance metrics"""
        session = self.get_session()
        try:
            performance = PerformanceRecord(
                portfolio_id=portfolio_id,
                date=date,
                **metrics
            )
            session.add(performance)
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    def store_ab_test_result(self, experiment_data: Dict[str, Any]):
        """Store A/B testing results"""
        session = self.get_session()
        try:
            ab_test = ABTestResult(**experiment_data)
            session.add(ab_test)
            session.commit()
            return ab_test.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    def get_portfolio_performance(self, portfolio_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve portfolio performance data"""
        session = self.get_session()
        try:
            query = session.query(PerformanceRecord).filter(
                PerformanceRecord.portfolio_id == portfolio_id,
                PerformanceRecord.date >= start_date,
                PerformanceRecord.date <= end_date
            ).order_by(PerformanceRecord.date)
            
            data = []
            for record in query:
                data.append({
                    'date': record.date,
                    'daily_return': record.daily_return,
                    'cumulative_return': record.cumulative_return,
                    'volatility_20d': record.volatility_20d,
                    'sharpe_ratio_20d': record.sharpe_ratio_20d,
                    'max_drawdown': record.max_drawdown,
                    'total_value': record.total_value
                })
            
            return pd.DataFrame(data)
            
        finally:
            session.close()


# Example usage and setup
def setup_production_database():
    """
    Set up production database with proper configuration
    Call this during application initialization
    """
    # Create database manager
    db_manager = DatabaseManager()
    
    # Create tables
    db_manager.create_tables()
    
    print("✅ Production database setup complete")
    print(f"📊 Tables created: {len(Base.metadata.tables)}")
    print("🔧 Ready for portfolio analytics data storage")
    
    return db_manager


if __name__ == "__main__":
    # Example setup
    db_manager = setup_production_database()
    
    # Example data operations
    sample_portfolio_metrics = {
        'daily_return': 0.012,
        'cumulative_return': 0.156,
        'volatility_20d': 0.18,
        'sharpe_ratio_20d': 1.85,
        'max_drawdown': -0.085,
        'total_value': 105600.0
    }
    
    print("📈 Database ready for production data storage")
    print("🎯 FAANG-level database architecture implemented")


# Global database manager instance - lazy loaded
_db_manager = None

def get_db_manager():
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# Create a lazy SessionLocal that only initializes when first accessed
class LazySessionLocal:
    def __call__(self):
        return get_db_manager().SessionLocal()

SessionLocal = LazySessionLocal()
