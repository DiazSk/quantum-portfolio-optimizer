"""
WebSocket API for Real-time Risk Data
Provides WebSocket endpoints for real-time risk metric broadcasting
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    # Mock JWT for development
    class MockJWT:
        @staticmethod
        def decode(token, secret, algorithms=None):
            # Simple mock - in production use real JWT
            return {"user_id": "mock_user", "role": "portfolio_manager"}
    jwt = MockJWT()

from ..risk.realtime_monitor import RealTimeRiskMonitor, RiskMetricsSnapshot
from ..risk.alert_engine import AlertEngine, RiskAlert
from ..risk.escalation_manager import EscalationManager, EscalationEvent
from ..utils.professional_logging import log_risk_event

# Configure logging
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.portfolio_subscriptions: Dict[str, Set[str]] = {}  # portfolio_id -> connection_ids
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str, portfolio_ids: List[str] = None):
        """Register a new WebSocket connection"""
        await websocket.accept()
        
        self.active_connections[connection_id] = websocket
        
        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Track portfolio subscriptions
        if portfolio_ids:
            for portfolio_id in portfolio_ids:
                if portfolio_id not in self.portfolio_subscriptions:
                    self.portfolio_subscriptions[portfolio_id] = set()
                self.portfolio_subscriptions[portfolio_id].add(connection_id)
        
        # Store metadata
        self.connection_metadata[connection_id] = {
            'user_id': user_id,
            'portfolio_ids': portfolio_ids or [],
            'connected_at': datetime.now(),
            'last_ping': datetime.now()
        }
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get('user_id')
            portfolio_ids = metadata.get('portfolio_ids', [])
            
            # Remove from active connections
            del self.active_connections[connection_id]
            
            # Remove from user tracking
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove from portfolio subscriptions
            for portfolio_id in portfolio_ids:
                if portfolio_id in self.portfolio_subscriptions:
                    self.portfolio_subscriptions[portfolio_id].discard(connection_id)
                    if not self.portfolio_subscriptions[portfolio_id]:
                        del self.portfolio_subscriptions[portfolio_id]
            
            # Remove metadata
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection_id)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def send_to_user(self, message: str, user_id: str):
        """Send message to all connections for a user"""
        if user_id in self.user_connections:
            connection_ids = list(self.user_connections[user_id])
            for connection_id in connection_ids:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_portfolio(self, message: str, portfolio_id: str):
        """Broadcast message to all subscribers of a portfolio"""
        if portfolio_id in self.portfolio_subscriptions:
            connection_ids = list(self.portfolio_subscriptions[portfolio_id])
            for connection_id in connection_ids:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_all(self, message: str):
        """Broadcast message to all active connections"""
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'unique_users': len(self.user_connections),
            'portfolio_subscriptions': len(self.portfolio_subscriptions),
            'connections_by_user': {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            }
        }


class WebSocketRiskAPI:
    """WebSocket API for real-time risk monitoring"""
    
    def __init__(self, 
                 risk_monitor: RealTimeRiskMonitor,
                 alert_engine: AlertEngine,
                 escalation_manager: EscalationManager):
        """
        Initialize WebSocket Risk API
        
        Args:
            risk_monitor: Real-time risk monitor instance
            alert_engine: Alert engine instance
            escalation_manager: Escalation manager instance
        """
        self.risk_monitor = risk_monitor
        self.alert_engine = alert_engine
        self.escalation_manager = escalation_manager
        self.connection_manager = ConnectionManager()
        
        # Setup callbacks
        self.risk_monitor.add_risk_callback(self._handle_risk_update)
        self.alert_engine.add_alert_callback(self._handle_alert)
        self.escalation_manager.add_escalation_callback(self._handle_escalation)
        
        # Configuration
        self.jwt_secret = "your-secret-key"  # Should be from environment
        self.jwt_algorithm = "HS256"
        
        logger.info("WebSocketRiskAPI initialized")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and extract user info"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.InvalidTokenError:
            return None
    
    async def websocket_endpoint(self, websocket: WebSocket, token: str, portfolio_ids: str = ""):
        """Main WebSocket endpoint for risk monitoring"""
        # Verify authentication
        user_payload = self.verify_token(token)
        if not user_payload:
            await websocket.close(code=4001, reason="Invalid token")
            return
        
        user_id = user_payload.get('user_id')
        if not user_id:
            await websocket.close(code=4002, reason="Missing user_id in token")
            return
        
        # Parse portfolio IDs
        portfolio_list = [p.strip() for p in portfolio_ids.split(',') if p.strip()] if portfolio_ids else []
        
        # Generate connection ID
        connection_id = str(uuid.uuid4())
        
        try:
            # Accept connection
            await self.connection_manager.connect(websocket, connection_id, user_id, portfolio_list)
            
            # Send welcome message
            welcome_message = {
                'type': 'connection_established',
                'connection_id': connection_id,
                'subscribed_portfolios': portfolio_list,
                'timestamp': datetime.now().isoformat()
            }
            await self.connection_manager.send_personal_message(json.dumps(welcome_message), connection_id)
            
            # Send current risk metrics for subscribed portfolios
            for portfolio_id in portfolio_list:
                cached_metrics = await self.risk_monitor.get_cached_metrics(portfolio_id)
                if cached_metrics:
                    await self._send_risk_metrics(cached_metrics, connection_id)
            
            # Handle incoming messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_websocket_message(message, connection_id, user_id)
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    error_msg = {'type': 'error', 'message': 'Invalid JSON format'}
                    await self.connection_manager.send_personal_message(json.dumps(error_msg), connection_id)
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    error_msg = {'type': 'error', 'message': str(e)}
                    await self.connection_manager.send_personal_message(json.dumps(error_msg), connection_id)
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        
        finally:
            self.connection_manager.disconnect(connection_id)
    
    async def _handle_websocket_message(self, message: Dict[str, Any], connection_id: str, user_id: str):
        """Handle incoming WebSocket messages"""
        message_type = message.get('type')
        
        if message_type == 'ping':
            # Handle heartbeat
            pong_message = {
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }
            await self.connection_manager.send_personal_message(json.dumps(pong_message), connection_id)
            
            # Update last ping time
            if connection_id in self.connection_manager.connection_metadata:
                self.connection_manager.connection_metadata[connection_id]['last_ping'] = datetime.now()
        
        elif message_type == 'subscribe_portfolio':
            # Subscribe to additional portfolio
            portfolio_id = message.get('portfolio_id')
            if portfolio_id:
                if portfolio_id not in self.connection_manager.portfolio_subscriptions:
                    self.connection_manager.portfolio_subscriptions[portfolio_id] = set()
                self.connection_manager.portfolio_subscriptions[portfolio_id].add(connection_id)
                
                # Update metadata
                metadata = self.connection_manager.connection_metadata.get(connection_id, {})
                portfolio_ids = metadata.get('portfolio_ids', [])
                if portfolio_id not in portfolio_ids:
                    portfolio_ids.append(portfolio_id)
                    metadata['portfolio_ids'] = portfolio_ids
                
                # Send current metrics
                cached_metrics = await self.risk_monitor.get_cached_metrics(portfolio_id)
                if cached_metrics:
                    await self._send_risk_metrics(cached_metrics, connection_id)
                
                response = {'type': 'subscription_confirmed', 'portfolio_id': portfolio_id}
                await self.connection_manager.send_personal_message(json.dumps(response), connection_id)
        
        elif message_type == 'unsubscribe_portfolio':
            # Unsubscribe from portfolio
            portfolio_id = message.get('portfolio_id')
            if portfolio_id and portfolio_id in self.connection_manager.portfolio_subscriptions:
                self.connection_manager.portfolio_subscriptions[portfolio_id].discard(connection_id)
                
                # Update metadata
                metadata = self.connection_manager.connection_metadata.get(connection_id, {})
                portfolio_ids = metadata.get('portfolio_ids', [])
                if portfolio_id in portfolio_ids:
                    portfolio_ids.remove(portfolio_id)
                    metadata['portfolio_ids'] = portfolio_ids
                
                response = {'type': 'unsubscription_confirmed', 'portfolio_id': portfolio_id}
                await self.connection_manager.send_personal_message(json.dumps(response), connection_id)
        
        elif message_type == 'get_alerts':
            # Get current alerts
            portfolio_id = message.get('portfolio_id')
            alerts = self.alert_engine.get_active_alerts(portfolio_id)
            
            alerts_data = []
            for alert in alerts:
                alerts_data.append({
                    'id': alert.id,
                    'portfolio_id': alert.portfolio_id,
                    'metric_type': alert.metric_type.value,
                    'severity': alert.severity.value,
                    'description': alert.description,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'status': alert.status.value
                })
            
            response = {
                'type': 'alerts_data',
                'portfolio_id': portfolio_id,
                'alerts': alerts_data,
                'timestamp': datetime.now().isoformat()
            }
            await self.connection_manager.send_personal_message(json.dumps(response), connection_id)
        
        elif message_type == 'acknowledge_alert':
            # Acknowledge an alert
            alert_id = message.get('alert_id')
            if alert_id:
                success = self.alert_engine.acknowledge_alert(alert_id, user_id)
                response = {
                    'type': 'alert_acknowledgment',
                    'alert_id': alert_id,
                    'success': success,
                    'acknowledged_by': user_id,
                    'timestamp': datetime.now().isoformat()
                }
                await self.connection_manager.send_personal_message(json.dumps(response), connection_id)
    
    async def _handle_risk_update(self, snapshot: RiskMetricsSnapshot):
        """Handle risk metrics update from monitor"""
        # Broadcast to portfolio subscribers
        await self.connection_manager.broadcast_to_portfolio(
            json.dumps(self._format_risk_metrics(snapshot)),
            snapshot.portfolio_id
        )
    
    async def _handle_alert(self, alert: RiskAlert):
        """Handle new alert from alert engine"""
        alert_message = {
            'type': 'new_alert',
            'alert': {
                'id': alert.id,
                'portfolio_id': alert.portfolio_id,
                'metric_type': alert.metric_type.value,
                'severity': alert.severity.value,
                'description': alert.description,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'triggered_at': alert.triggered_at.isoformat(),
                'status': alert.status.value
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast to portfolio subscribers
        await self.connection_manager.broadcast_to_portfolio(
            json.dumps(alert_message),
            alert.portfolio_id
        )
    
    async def _handle_escalation(self, escalation: EscalationEvent):
        """Handle escalation event"""
        escalation_message = {
            'type': 'alert_escalated',
            'escalation': {
                'id': escalation.id,
                'alert_id': escalation.alert_id,
                'from_level': escalation.from_level.value,
                'to_level': escalation.to_level.value,
                'escalated_at': escalation.escalated_at.isoformat(),
                'escalated_by': escalation.escalated_by,
                'reason': escalation.reason
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast to all connections (escalations are important)
        await self.connection_manager.broadcast_to_all(json.dumps(escalation_message))
    
    async def _send_risk_metrics(self, snapshot: RiskMetricsSnapshot, connection_id: str):
        """Send risk metrics to specific connection"""
        message = self._format_risk_metrics(snapshot)
        await self.connection_manager.send_personal_message(json.dumps(message), connection_id)
    
    def _format_risk_metrics(self, snapshot: RiskMetricsSnapshot) -> Dict[str, Any]:
        """Format risk metrics for WebSocket transmission"""
        return {
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
                'calmar_ratio': snapshot.calmar_ratio,
                'annual_volatility': snapshot.annual_volatility,
                'daily_volatility': snapshot.daily_volatility,
                'leverage_ratio': snapshot.leverage_ratio,
                'concentration_risk': snapshot.concentration_risk
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for WebSocket API"""
        risk_health = await self.risk_monitor.health_check()
        connection_stats = self.connection_manager.get_connection_stats()
        
        return {
            'websocket_api_status': 'healthy',
            'risk_monitor_status': risk_health,
            'connection_stats': connection_stats,
            'timestamp': datetime.now().isoformat()
        }


# FastAPI integration
def create_websocket_routes(app: FastAPI, websocket_api: WebSocketRiskAPI):
    """Add WebSocket routes to FastAPI app"""
    
    @app.websocket("/ws/risk")
    async def websocket_risk_endpoint(websocket: WebSocket, token: str, portfolio_ids: str = ""):
        """WebSocket endpoint for risk monitoring"""
        await websocket_api.websocket_endpoint(websocket, token, portfolio_ids)
    
    @app.get("/api/risk/websocket/health")
    async def websocket_health():
        """Health check endpoint for WebSocket API"""
        return await websocket_api.health_check()
    
    @app.get("/api/risk/websocket/stats")
    async def websocket_stats():
        """Get WebSocket connection statistics"""
        return websocket_api.connection_manager.get_connection_stats()


# Example usage and testing
async def test_websocket_api():
    """Test the WebSocket API"""
    from ..risk.realtime_monitor import RealTimeRiskMonitor
    from ..risk.alert_engine import AlertEngine
    from ..risk.escalation_manager import EscalationManager
    
    # Initialize components
    risk_monitor = RealTimeRiskMonitor()
    alert_engine = AlertEngine()
    escalation_manager = EscalationManager()
    
    # Initialize WebSocket API
    websocket_api = WebSocketRiskAPI(risk_monitor, alert_engine, escalation_manager)
    
    # Get health check
    health = await websocket_api.health_check()
    print("WebSocket API Health Check:")
    print(json.dumps(health, indent=2))
    
    # Test connection manager
    manager = websocket_api.connection_manager
    stats = manager.get_connection_stats()
    print(f"\nConnection Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_websocket_api())
