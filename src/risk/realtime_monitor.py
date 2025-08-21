"""
Real-time Risk Monitoring Module
Extends existing RiskManager with real-time monitoring capabilities
"""

import asyncio
import redis
import json
import websockets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from threading import Thread
import pandas as pd
import numpy as np

from .risk_managment import RiskManager
from ..utils.professional_logging import log_risk_event

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RiskMetricsSnapshot:
    """Container for real-time risk metrics snapshot"""
    timestamp: datetime
    portfolio_id: str
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annual_volatility: float
    daily_volatility: float
    correlation_matrix: Dict[str, Dict[str, float]]
    concentration_risk: Dict[str, float]  # Asset concentration percentages
    leverage_ratio: float


@dataclass
class WebSocketClient:
    """Container for WebSocket client connection"""
    websocket: Any
    user_id: str
    subscribed_portfolios: List[str]
    last_heartbeat: datetime


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring service with 30-second refresh cycles
    Extends existing RiskManager capabilities
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
        """
        Initialize real-time risk monitor
        
        Args:
            redis_host: Redis server host for caching
            redis_port: Redis server port
            redis_db: Redis database number
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.websocket_clients: Dict[str, WebSocketClient] = {}
        self.monitoring_active = False
        self.refresh_interval = 30  # seconds
        self.risk_callbacks: List[Callable] = []
        
        # Cache settings
        self.cache_ttl = 60  # Cache TTL in seconds
        self.cache_prefix = "risk_metrics"
        
        logger.info("RealTimeRiskMonitor initialized")
    
    def add_risk_callback(self, callback: Callable[[RiskMetricsSnapshot], None]):
        """Add callback function for risk metric updates"""
        self.risk_callbacks.append(callback)
    
    def remove_risk_callback(self, callback: Callable[[RiskMetricsSnapshot], None]):
        """Remove callback function"""
        if callback in self.risk_callbacks:
            self.risk_callbacks.remove(callback)
    
    async def calculate_realtime_metrics(self, 
                                       portfolio_id: str,
                                       returns_data: pd.DataFrame,
                                       weights: np.ndarray,
                                       asset_prices: Optional[pd.DataFrame] = None) -> RiskMetricsSnapshot:
        """
        Calculate real-time risk metrics for a portfolio
        
        Args:
            portfolio_id: Unique portfolio identifier
            returns_data: Historical returns data
            weights: Current portfolio weights
            asset_prices: Optional current asset prices for concentration analysis
            
        Returns:
            RiskMetricsSnapshot with all calculated metrics
        """
        # Use existing RiskManager for core calculations
        risk_manager = RiskManager(returns_data, weights)
        
        # Get comprehensive risk metrics
        metrics = risk_manager.get_risk_metrics()
        
        # Calculate additional real-time specific metrics
        correlation_matrix = self._calculate_correlation_matrix(returns_data)
        concentration_risk = self._calculate_concentration_risk(weights, returns_data.columns)
        leverage_ratio = self._calculate_leverage_ratio(weights, asset_prices)
        
        snapshot = RiskMetricsSnapshot(
            timestamp=datetime.now(),
            portfolio_id=portfolio_id,
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            var_99=metrics['var_99'],
            cvar_99=metrics['cvar_99'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            annual_volatility=metrics['annual_volatility'],
            daily_volatility=metrics['daily_return_std'],
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio
        )
        
        # Cache the metrics
        await self._cache_metrics(snapshot)
        
        # Log risk event
        log_risk_event(
            portfolio_id=portfolio_id,
            event_type="risk_metrics_calculated",
            event_data={
                "var_95": snapshot.var_95,
                "cvar_95": snapshot.cvar_95,
                "max_drawdown": snapshot.max_drawdown,
                "leverage_ratio": snapshot.leverage_ratio
            }
        )
        
        return snapshot
    
    def _calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for portfolio assets"""
        corr_matrix = returns_data.corr()
        return {
            asset1: {
                asset2: float(corr_matrix.loc[asset1, asset2])
                for asset2 in corr_matrix.columns
            }
            for asset1 in corr_matrix.index
        }
    
    def _calculate_concentration_risk(self, weights: np.ndarray, asset_names: List[str]) -> Dict[str, float]:
        """Calculate concentration risk metrics"""
        concentration = {}
        for i, asset in enumerate(asset_names):
            concentration[asset] = float(weights[i])
        
        # Add concentration metrics
        concentration['max_position'] = float(weights.max())
        concentration['top_3_concentration'] = float(np.sort(weights)[-3:].sum())
        concentration['herfindahl_index'] = float((weights ** 2).sum())
        
        return concentration
    
    def _calculate_leverage_ratio(self, weights: np.ndarray, asset_prices: Optional[pd.DataFrame]) -> float:
        """Calculate portfolio leverage ratio"""
        # Simple leverage calculation - sum of absolute weights
        return float(np.abs(weights).sum())
    
    async def _cache_metrics(self, snapshot: RiskMetricsSnapshot):
        """Cache risk metrics in Redis"""
        cache_key = f"{self.cache_prefix}:{snapshot.portfolio_id}"
        
        # Convert snapshot to JSON
        snapshot_dict = {
            'timestamp': snapshot.timestamp.isoformat(),
            'portfolio_id': snapshot.portfolio_id,
            'var_95': snapshot.var_95,
            'cvar_95': snapshot.cvar_95,
            'var_99': snapshot.var_99,
            'cvar_99': snapshot.cvar_99,
            'max_drawdown': snapshot.max_drawdown,
            'sharpe_ratio': snapshot.sharpe_ratio,
            'sortino_ratio': snapshot.sortino_ratio,
            'calmar_ratio': snapshot.calmar_ratio,
            'annual_volatility': snapshot.annual_volatility,
            'daily_volatility': snapshot.daily_volatility,
            'correlation_matrix': snapshot.correlation_matrix,
            'concentration_risk': snapshot.concentration_risk,
            'leverage_ratio': snapshot.leverage_ratio
        }
        
        # Store in Redis with TTL
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(snapshot_dict))
        
        # Also store in time series for historical tracking
        ts_key = f"{self.cache_prefix}_ts:{snapshot.portfolio_id}"
        self.redis_client.zadd(ts_key, {json.dumps(snapshot_dict): snapshot.timestamp.timestamp()})
        
        # Keep only last 24 hours of data
        cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
        self.redis_client.zremrangebyscore(ts_key, 0, cutoff_time)
    
    async def get_cached_metrics(self, portfolio_id: str) -> Optional[RiskMetricsSnapshot]:
        """Get cached risk metrics for a portfolio"""
        cache_key = f"{self.cache_prefix}:{portfolio_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            return RiskMetricsSnapshot(
                timestamp=datetime.fromisoformat(data['timestamp']),
                portfolio_id=data['portfolio_id'],
                var_95=data['var_95'],
                cvar_95=data['cvar_95'],
                var_99=data['var_99'],
                cvar_99=data['cvar_99'],
                max_drawdown=data['max_drawdown'],
                sharpe_ratio=data['sharpe_ratio'],
                sortino_ratio=data['sortino_ratio'],
                calmar_ratio=data['calmar_ratio'],
                annual_volatility=data['annual_volatility'],
                daily_volatility=data['daily_volatility'],
                correlation_matrix=data['correlation_matrix'],
                concentration_risk=data['concentration_risk'],
                leverage_ratio=data['leverage_ratio']
            )
        return None
    
    async def get_historical_metrics(self, portfolio_id: str, hours: int = 24) -> List[RiskMetricsSnapshot]:
        """Get historical risk metrics for a portfolio"""
        ts_key = f"{self.cache_prefix}_ts:{portfolio_id}"
        
        # Get data from last N hours
        start_time = (datetime.now() - timedelta(hours=hours)).timestamp()
        raw_data = self.redis_client.zrangebyscore(ts_key, start_time, '+inf', withscores=True)
        
        snapshots = []
        for data_json, timestamp in raw_data:
            data = json.loads(data_json)
            snapshots.append(RiskMetricsSnapshot(
                timestamp=datetime.fromisoformat(data['timestamp']),
                portfolio_id=data['portfolio_id'],
                var_95=data['var_95'],
                cvar_95=data['cvar_95'],
                var_99=data['var_99'],
                cvar_99=data['cvar_99'],
                max_drawdown=data['max_drawdown'],
                sharpe_ratio=data['sharpe_ratio'],
                sortino_ratio=data['sortino_ratio'],
                calmar_ratio=data['calmar_ratio'],
                annual_volatility=data['annual_volatility'],
                daily_volatility=data['daily_volatility'],
                correlation_matrix=data['correlation_matrix'],
                concentration_risk=data['concentration_risk'],
                leverage_ratio=data['leverage_ratio']
            ))
        
        return sorted(snapshots, key=lambda x: x.timestamp)
    
    async def start_monitoring(self, portfolio_configs: List[Dict[str, Any]]):
        """
        Start real-time monitoring for specified portfolios
        
        Args:
            portfolio_configs: List of portfolio configurations with data sources
        """
        self.monitoring_active = True
        logger.info(f"Starting real-time risk monitoring for {len(portfolio_configs)} portfolios")
        
        while self.monitoring_active:
            try:
                # Calculate metrics for all portfolios
                for config in portfolio_configs:
                    portfolio_id = config['portfolio_id']
                    returns_data = config['returns_data']
                    weights = config['weights']
                    asset_prices = config.get('asset_prices')
                    
                    # Calculate real-time metrics
                    snapshot = await self.calculate_realtime_metrics(
                        portfolio_id, returns_data, weights, asset_prices
                    )
                    
                    # Trigger callbacks
                    for callback in self.risk_callbacks:
                        try:
                            await callback(snapshot)
                        except Exception as e:
                            logger.error(f"Error in risk callback: {e}")
                    
                    # Broadcast to WebSocket clients
                    await self._broadcast_to_websockets(snapshot)
                
                # Wait for next refresh cycle
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        logger.info("Real-time risk monitoring stopped")
    
    async def register_websocket_client(self, websocket, user_id: str, subscribed_portfolios: List[str]):
        """Register a new WebSocket client for real-time updates"""
        client_id = f"{user_id}_{id(websocket)}"
        self.websocket_clients[client_id] = WebSocketClient(
            websocket=websocket,
            user_id=user_id,
            subscribed_portfolios=subscribed_portfolios,
            last_heartbeat=datetime.now()
        )
        logger.info(f"Registered WebSocket client: {client_id}")
    
    async def unregister_websocket_client(self, websocket):
        """Unregister a WebSocket client"""
        client_id = None
        for cid, client in self.websocket_clients.items():
            if client.websocket == websocket:
                client_id = cid
                break
        
        if client_id:
            del self.websocket_clients[client_id]
            logger.info(f"Unregistered WebSocket client: {client_id}")
    
    async def _broadcast_to_websockets(self, snapshot: RiskMetricsSnapshot):
        """Broadcast risk metrics to subscribed WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = {
            'type': 'risk_metrics_update',
            'portfolio_id': snapshot.portfolio_id,
            'timestamp': snapshot.timestamp.isoformat(),
            'metrics': {
                'var_95': snapshot.var_95,
                'cvar_95': snapshot.cvar_95,
                'var_99': snapshot.var_99,
                'cvar_99': snapshot.cvar_99,
                'max_drawdown': snapshot.max_drawdown,
                'sharpe_ratio': snapshot.sharpe_ratio,
                'sortino_ratio': snapshot.sortino_ratio,
                'annual_volatility': snapshot.annual_volatility,
                'daily_volatility': snapshot.daily_volatility,
                'concentration_risk': snapshot.concentration_risk,
                'leverage_ratio': snapshot.leverage_ratio
            }
        }
        
        # Send to clients subscribed to this portfolio
        disconnected_clients = []
        for client_id, client in self.websocket_clients.items():
            if snapshot.portfolio_id in client.subscribed_portfolios:
                try:
                    await client.websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.append(client_id)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket client {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.websocket_clients:
                del self.websocket_clients[client_id]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the monitoring system"""
        return {
            'monitoring_active': self.monitoring_active,
            'connected_websocket_clients': len(self.websocket_clients),
            'redis_connected': self.redis_client.ping(),
            'cache_prefix': self.cache_prefix,
            'refresh_interval': self.refresh_interval,
            'last_check': datetime.now().isoformat()
        }


# Example usage and testing
async def test_realtime_monitor():
    """Test the real-time risk monitor"""
    # Initialize monitor
    monitor = RealTimeRiskMonitor()
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    # Generate deterministic sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    # Create deterministic returns data
    returns_matrix = []
    for i, date in enumerate(dates):
        day_returns = []
        for j, asset in enumerate(assets):
            return_hash = hash(f"{date.strftime('%Y%m%d')}_{asset}") % 10000
            base_return = 0.0005
            noise = ((return_hash - 5000) / 10000.0) * 0.03  # ±1.5% daily volatility
            day_returns.append(base_return + noise)
        returns_matrix.append(day_returns)
    
    returns_data = pd.DataFrame(
        returns_matrix,
        index=dates,
        columns=assets
    )
    
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Calculate metrics
    snapshot = await monitor.calculate_realtime_metrics(
        'test_portfolio_001',
        returns_data,
        weights
    )
    
    print(f"Risk Metrics Snapshot for {snapshot.portfolio_id}:")
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  VaR (95%): {snapshot.var_95:.3%}")
    print(f"  CVaR (95%): {snapshot.cvar_95:.3%}")
    print(f"  Max Drawdown: {snapshot.max_drawdown:.3%}")
    print(f"  Sharpe Ratio: {snapshot.sharpe_ratio:.2f}")
    print(f"  Leverage Ratio: {snapshot.leverage_ratio:.2f}")
    print(f"  Max Position: {snapshot.concentration_risk['max_position']:.2%}")


if __name__ == "__main__":
    asyncio.run(test_realtime_monitor())
