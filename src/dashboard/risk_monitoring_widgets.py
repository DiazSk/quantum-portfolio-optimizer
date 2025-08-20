"""
Risk Monitoring Dashboard Components
Extends existing Streamlit dashboard with real-time risk monitoring widgets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import websocket
import threading

from ..risk.realtime_monitor import RealTimeRiskMonitor, RiskMetricsSnapshot
from ..risk.alert_engine import AlertEngine, AlertSeverity, RiskAlert
from ..risk.escalation_manager import EscalationManager
from ..utils.professional_logging import log_risk_event

# Cache for real-time data
if 'risk_metrics_cache' not in st.session_state:
    st.session_state.risk_metrics_cache = {}

if 'alert_cache' not in st.session_state:
    st.session_state.alert_cache = []

if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False


class RiskTrafficLightWidget:
    """Traffic light indicator widget for risk status"""
    
    @staticmethod
    def render(portfolio_id: str, risk_metrics: Optional[RiskMetricsSnapshot] = None):
        """Render traffic light indicators for risk metrics"""
        st.subheader("🚦 Risk Status Dashboard")
        
        if not risk_metrics:
            st.warning("No risk metrics available")
            return
        
        # Create columns for traffic lights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_status = RiskTrafficLightWidget._get_traffic_light_status(
                risk_metrics.var_95, -0.05, -0.03, is_negative=True
            )
            RiskTrafficLightWidget._render_traffic_light(
                "VaR (95%)", 
                f"{risk_metrics.var_95:.3%}",
                var_status
            )
        
        with col2:
            drawdown_status = RiskTrafficLightWidget._get_traffic_light_status(
                risk_metrics.max_drawdown, -0.20, -0.10, is_negative=True
            )
            RiskTrafficLightWidget._render_traffic_light(
                "Max Drawdown",
                f"{risk_metrics.max_drawdown:.3%}",
                drawdown_status
            )
        
        with col3:
            concentration_status = RiskTrafficLightWidget._get_traffic_light_status(
                risk_metrics.concentration_risk.get('max_position', 0), 0.30, 0.20
            )
            RiskTrafficLightWidget._render_traffic_light(
                "Concentration",
                f"{risk_metrics.concentration_risk.get('max_position', 0):.1%}",
                concentration_status
            )
        
        with col4:
            leverage_status = RiskTrafficLightWidget._get_traffic_light_status(
                risk_metrics.leverage_ratio, 1.5, 1.2
            )
            RiskTrafficLightWidget._render_traffic_light(
                "Leverage",
                f"{risk_metrics.leverage_ratio:.2f}x",
                leverage_status
            )
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            volatility_status = RiskTrafficLightWidget._get_traffic_light_status(
                risk_metrics.annual_volatility, 0.25, 0.18
            )
            RiskTrafficLightWidget._render_traffic_light(
                "Volatility",
                f"{risk_metrics.annual_volatility:.1%}",
                volatility_status
            )
        
        with col6:
            sharpe_status = RiskTrafficLightWidget._get_traffic_light_status(
                risk_metrics.sharpe_ratio, 0.5, 1.0, is_negative=False, reverse=True
            )
            RiskTrafficLightWidget._render_traffic_light(
                "Sharpe Ratio",
                f"{risk_metrics.sharpe_ratio:.2f}",
                sharpe_status
            )
        
        with col7:
            cvar_status = RiskTrafficLightWidget._get_traffic_light_status(
                risk_metrics.cvar_95, -0.08, -0.05, is_negative=True
            )
            RiskTrafficLightWidget._render_traffic_light(
                "CVaR (95%)",
                f"{risk_metrics.cvar_95:.3%}",
                cvar_status
            )
        
        with col8:
            # Overall risk score (composite)
            overall_score = RiskTrafficLightWidget._calculate_overall_risk_score(risk_metrics)
            overall_status = RiskTrafficLightWidget._get_traffic_light_status(
                overall_score, 0.7, 0.4, reverse=True
            )
            RiskTrafficLightWidget._render_traffic_light(
                "Overall Risk",
                f"{overall_score:.1%}",
                overall_status
            )
    
    @staticmethod
    def _get_traffic_light_status(value: float, red_threshold: float, yellow_threshold: float, 
                                 is_negative: bool = False, reverse: bool = False) -> str:
        """Determine traffic light status based on thresholds"""
        if is_negative:
            # For negative values (VaR, drawdown), lower is worse
            if value <= red_threshold:
                return "red" if not reverse else "green"
            elif value <= yellow_threshold:
                return "yellow"
            else:
                return "green" if not reverse else "red"
        else:
            # For positive values, higher might be worse (depends on reverse)
            if reverse:
                # For metrics where higher is better (Sharpe ratio)
                if value >= yellow_threshold:
                    return "green"
                elif value >= red_threshold:
                    return "yellow"
                else:
                    return "red"
            else:
                # For metrics where higher is worse (concentration, leverage)
                if value >= red_threshold:
                    return "red"
                elif value >= yellow_threshold:
                    return "yellow"
                else:
                    return "green"
    
    @staticmethod
    def _render_traffic_light(title: str, value: str, status: str):
        """Render individual traffic light"""
        color_map = {
            "green": "#28a745",
            "yellow": "#ffc107", 
            "red": "#dc3545"
        }
        
        emoji_map = {
            "green": "🟢",
            "yellow": "🟡",
            "red": "🔴"
        }
        
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 10px; 
                    background-color: {color_map[status]}20; border: 2px solid {color_map[status]};">
            <h4 style="margin: 0; color: {color_map[status]};">{emoji_map[status]} {title}</h4>
            <h3 style="margin: 5px 0; color: {color_map[status]};">{value}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _calculate_overall_risk_score(risk_metrics: RiskMetricsSnapshot) -> float:
        """Calculate overall risk score (0 = low risk, 1 = high risk)"""
        # Normalize various risk metrics to 0-1 scale
        var_score = min(abs(risk_metrics.var_95) / 0.10, 1.0)  # Normalize to 10% VaR
        drawdown_score = min(abs(risk_metrics.max_drawdown) / 0.30, 1.0)  # Normalize to 30% drawdown
        concentration_score = min(risk_metrics.concentration_risk.get('max_position', 0) / 0.50, 1.0)  # Normalize to 50%
        leverage_score = min(risk_metrics.leverage_ratio / 2.0, 1.0)  # Normalize to 2x leverage
        volatility_score = min(risk_metrics.annual_volatility / 0.40, 1.0)  # Normalize to 40% volatility
        
        # Weighted average (can be customized)
        weights = [0.25, 0.25, 0.15, 0.20, 0.15]  # Sum to 1.0
        scores = [var_score, drawdown_score, concentration_score, leverage_score, volatility_score]
        
        return sum(w * s for w, s in zip(weights, scores))


class RealTimeRiskCharts:
    """Real-time trend charts for risk metrics"""
    
    @staticmethod
    def render(portfolio_id: str, historical_data: List[RiskMetricsSnapshot] = None):
        """Render real-time risk trend charts"""
        st.subheader("📈 Real-time Risk Trends")
        
        if not historical_data or len(historical_data) < 2:
            st.info("Collecting historical data... Please wait for more data points.")
            return
        
        # Prepare data
        df = RealTimeRiskCharts._prepare_chart_data(historical_data)
        
        # Create tabs for different chart views
        tab1, tab2, tab3, tab4 = st.tabs(["VaR & CVaR", "Portfolio Metrics", "Risk Ratios", "Concentration"])
        
        with tab1:
            RealTimeRiskCharts._render_var_charts(df)
        
        with tab2:
            RealTimeRiskCharts._render_portfolio_charts(df)
        
        with tab3:
            RealTimeRiskCharts._render_ratio_charts(df)
        
        with tab4:
            RealTimeRiskCharts._render_concentration_charts(df)
    
    @staticmethod
    def _prepare_chart_data(historical_data: List[RiskMetricsSnapshot]) -> pd.DataFrame:
        """Prepare data for charting"""
        data = []
        for snapshot in historical_data:
            data.append({
                'timestamp': snapshot.timestamp,
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
                'max_concentration': snapshot.concentration_risk.get('max_position', 0)
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    @staticmethod
    def _render_var_charts(df: pd.DataFrame):
        """Render VaR and CVaR charts"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Value at Risk (VaR)", "Conditional Value at Risk (CVaR)"),
            vertical_spacing=0.1
        )
        
        # VaR chart
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['var_95'] * 100, 
                      name='VaR 95%', line=dict(color='red', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['var_99'] * 100,
                      name='VaR 99%', line=dict(color='darkred', width=2)),
            row=1, col=1
        )
        
        # CVaR chart
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cvar_95'] * 100,
                      name='CVaR 95%', line=dict(color='orange', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cvar_99'] * 100,
                      name='CVaR 99%', line=dict(color='darkorange', width=2)),
            row=2, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=-5, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-8, line_dash="dash", line_color="orange", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, title="Risk Metrics Over Time")
        fig.update_yaxes(title_text="VaR (%)", row=1, col=1)
        fig.update_yaxes(title_text="CVaR (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_portfolio_charts(df: pd.DataFrame):
        """Render portfolio-level metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Max Drawdown", "Volatility", "Leverage Ratio", "Overall Trend"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Max Drawdown
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['max_drawdown'] * 100,
                      name='Max Drawdown', line=dict(color='red', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=1, col=1)
        
        # Volatility
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['annual_volatility'] * 100,
                      name='Annual Volatility', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=25, line_dash="dash", line_color="orange", row=1, col=2)
        
        # Leverage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['leverage_ratio'],
                      name='Leverage Ratio', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", row=2, col=1)
        
        # Combined trend (normalized)
        normalized_var = (df['var_95'].abs() / 0.10).clip(0, 2)
        normalized_vol = (df['annual_volatility'] / 0.30).clip(0, 2)
        combined_risk = (normalized_var + normalized_vol) / 2
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=combined_risk,
                      name='Combined Risk Score', line=dict(color='darkred', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title="Portfolio Risk Metrics")
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Leverage", row=2, col=1)
        fig.update_yaxes(title_text="Risk Score", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_ratio_charts(df: pd.DataFrame):
        """Render risk-adjusted ratio charts"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"),
            horizontal_spacing=0.1
        )
        
        # Sharpe Ratio
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['sharpe_ratio'],
                      name='Sharpe Ratio', line=dict(color='green', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", row=1, col=1)
        
        # Sortino Ratio
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['sortino_ratio'],
                      name='Sortino Ratio', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=1.5, line_dash="dash", line_color="blue", row=1, col=2)
        
        # Calmar Ratio
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['calmar_ratio'],
                      name='Calmar Ratio', line=dict(color='purple', width=2)),
            row=1, col=3
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="purple", row=1, col=3)
        
        fig.update_layout(height=400, showlegend=False, title="Risk-Adjusted Performance Ratios")
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_concentration_charts(df: pd.DataFrame):
        """Render concentration and correlation charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Concentration over time
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['max_concentration'] * 100,
                          name='Max Position %', line=dict(color='orange', width=3))
            )
            fig.add_hline(y=30, line_dash="dash", line_color="red")
            fig.update_layout(title="Position Concentration", yaxis_title="Max Position (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk distribution pie chart (latest data)
            if len(df) > 0:
                latest = df.iloc[-1]
                
                fig = go.Figure(data=[go.Pie(
                    labels=['VaR Risk', 'Volatility Risk', 'Concentration Risk', 'Leverage Risk'],
                    values=[
                        abs(latest['var_95']) * 100,
                        latest['annual_volatility'] * 100,
                        latest['max_concentration'] * 100,
                        (latest['leverage_ratio'] - 1) * 50  # Normalize leverage contribution
                    ],
                    hole=0.3
                )])
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(title="Current Risk Composition")
                st.plotly_chart(fig, use_container_width=True)


class AlertHistoryPanel:
    """Alert history and status display panel"""
    
    @staticmethod
    def render(portfolio_id: str = None):
        """Render alert history panel"""
        st.subheader("🚨 Alert Management")
        
        # Get alerts from cache
        alerts = st.session_state.alert_cache
        
        if portfolio_id:
            alerts = [a for a in alerts if a.get('portfolio_id') == portfolio_id]
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.selectbox(
                "Filter by Severity",
                ["All", "Critical", "High", "Medium", "Low"],
                key="alert_severity_filter"
            )
        
        with col2:
            status_filter = st.selectbox(
                "Filter by Status", 
                ["All", "Triggered", "Acknowledged", "Resolved", "Escalated"],
                key="alert_status_filter"
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Period",
                ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
                key="alert_time_filter"
            )
        
        # Apply filters
        filtered_alerts = AlertHistoryPanel._filter_alerts(alerts, severity_filter, status_filter, time_filter)
        
        if not filtered_alerts:
            st.info("No alerts match the current filters.")
            return
        
        # Display alerts
        for alert in filtered_alerts[:20]:  # Limit to 20 most recent
            AlertHistoryPanel._render_alert_card(alert)
        
        # Alert statistics
        if filtered_alerts:
            AlertHistoryPanel._render_alert_statistics(filtered_alerts)
    
    @staticmethod
    def _filter_alerts(alerts: List[Dict], severity_filter: str, status_filter: str, time_filter: str) -> List[Dict]:
        """Filter alerts based on user selection"""
        filtered = alerts.copy()
        
        # Severity filter
        if severity_filter != "All":
            filtered = [a for a in filtered if a.get('severity', '').lower() == severity_filter.lower()]
        
        # Status filter
        if status_filter != "All":
            filtered = [a for a in filtered if a.get('status', '').lower() == status_filter.lower()]
        
        # Time filter
        now = datetime.now()
        time_deltas = {
            "Last Hour": timedelta(hours=1),
            "Last 6 Hours": timedelta(hours=6),
            "Last 24 Hours": timedelta(hours=24),
            "Last Week": timedelta(days=7)
        }
        
        if time_filter in time_deltas:
            cutoff_time = now - time_deltas[time_filter]
            filtered = [
                a for a in filtered 
                if datetime.fromisoformat(a.get('triggered_at', '').replace('Z', '+00:00')) >= cutoff_time
            ]
        
        return sorted(filtered, key=lambda x: x.get('triggered_at', ''), reverse=True)
    
    @staticmethod
    def _render_alert_card(alert: Dict):
        """Render individual alert card"""
        severity = alert.get('severity', 'unknown').lower()
        status = alert.get('status', 'unknown').lower()
        
        # Color coding
        severity_colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#6c757d'
        }
        
        status_colors = {
            'triggered': '#dc3545',
            'acknowledged': '#ffc107',
            'resolved': '#28a745',
            'escalated': '#6f42c1'
        }
        
        severity_color = severity_colors.get(severity, '#6c757d')
        status_color = status_colors.get(status, '#6c757d')
        
        # Format timestamp
        triggered_at = alert.get('triggered_at', '')
        if triggered_at:
            try:
                dt = datetime.fromisoformat(triggered_at.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = triggered_at
        else:
            time_str = 'Unknown'
        
        with st.container():
            st.markdown(f"""
            <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 10px 0; 
                        background-color: #f8f9fa; border-radius: 5px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: {severity_color};">
                        {alert.get('portfolio_id', 'Unknown')} - {alert.get('metric_type', 'Unknown')}
                    </h4>
                    <span style="background-color: {status_color}; color: white; padding: 4px 8px; 
                                 border-radius: 12px; font-size: 12px; font-weight: bold;">
                        {status.upper()}
                    </span>
                </div>
                <p style="margin: 5px 0; color: #495057;">{alert.get('description', 'No description')}</p>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #6c757d;">
                    <span>Current: {alert.get('current_value', 'N/A'):.4f}</span>
                    <span>Threshold: {alert.get('threshold_value', 'N/A'):.4f}</span>
                    <span>Triggered: {time_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if status == 'triggered' and st.button(f"Acknowledge", key=f"ack_{alert.get('id')}"):
                    # Implement acknowledge logic
                    st.success("Alert acknowledged")
            
            with col2:
                if status in ['triggered', 'acknowledged'] and st.button(f"Resolve", key=f"resolve_{alert.get('id')}"):
                    # Implement resolve logic
                    st.success("Alert resolved")
    
    @staticmethod
    def _render_alert_statistics(alerts: List[Dict]):
        """Render alert statistics"""
        st.subheader("📊 Alert Statistics")
        
        # Count by severity
        severity_counts = {}
        status_counts = {}
        
        for alert in alerts:
            severity = alert.get('severity', 'unknown').lower()
            status = alert.get('status', 'unknown').lower()
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            if severity_counts:
                fig = px.pie(
                    values=list(severity_counts.values()),
                    names=list(severity_counts.keys()),
                    title="Alerts by Severity"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Status distribution
            if status_counts:
                fig = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="Alerts by Status"
                )
                st.plotly_chart(fig, use_container_width=True)


class WebSocketDashboardConnector:
    """WebSocket connector for real-time dashboard updates"""
    
    def __init__(self, websocket_url: str, auth_token: str):
        """Initialize WebSocket connector"""
        self.websocket_url = websocket_url
        self.auth_token = auth_token
        self.ws = None
        self.connected = False
        self.thread = None
    
    def connect(self):
        """Connect to WebSocket server"""
        if not self.connected:
            self.thread = threading.Thread(target=self._websocket_thread, daemon=True)
            self.thread.start()
    
    def _websocket_thread(self):
        """WebSocket connection thread"""
        try:
            url_with_auth = f"{self.websocket_url}?token={self.auth_token}"
            self.ws = websocket.WebSocketApp(
                url_with_auth,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self.ws.run_forever()
        except Exception as e:
            st.error(f"WebSocket connection error: {e}")
    
    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        self.connected = True
        st.session_state.websocket_connected = True
        st.success("✅ Real-time connection established")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'risk_metrics_update':
                # Update risk metrics cache
                portfolio_id = data.get('portfolio_id')
                metrics = data.get('metrics', {})
                timestamp = data.get('timestamp')
                
                st.session_state.risk_metrics_cache[portfolio_id] = {
                    'metrics': metrics,
                    'timestamp': timestamp
                }
            
            elif message_type == 'new_alert':
                # Add to alert cache
                alert_data = data.get('alert', {})
                st.session_state.alert_cache.append(alert_data)
                
                # Keep only last 100 alerts
                if len(st.session_state.alert_cache) > 100:
                    st.session_state.alert_cache = st.session_state.alert_cache[-100:]
                
                # Show notification
                severity = alert_data.get('severity', 'unknown')
                if severity in ['critical', 'high']:
                    st.error(f"🚨 {severity.upper()} ALERT: {alert_data.get('description', 'Unknown alert')}")
            
            elif message_type == 'alert_escalated':
                # Handle escalation notification
                escalation = data.get('escalation', {})
                st.warning(f"⚠️ Alert escalated to {escalation.get('to_level', 'unknown')} level")
            
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON
        except Exception as e:
            st.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        st.error(f"WebSocket error: {error}")
        self.connected = False
        st.session_state.websocket_connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        self.connected = False
        st.session_state.websocket_connected = False
        st.warning("⚠️ Real-time connection lost")


def render_risk_monitoring_page():
    """Render the complete risk monitoring page"""
    st.set_page_config(page_title="Risk Monitoring", page_icon="📊", layout="wide")
    
    st.title("🎯 Real-time Risk Monitoring Dashboard")
    st.markdown("Monitor portfolio risk metrics in real-time with configurable alerts and escalation workflows.")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Risk Monitoring Controls")
        
        # Portfolio selection
        portfolio_id = st.selectbox(
            "Select Portfolio",
            ["portfolio_001", "portfolio_002", "portfolio_003", "test_portfolio"],
            key="risk_portfolio_selector"
        )
        
        # Auto-refresh settings
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 30)
        
        # WebSocket connection
        st.subheader("📡 Real-time Connection")
        
        if st.session_state.websocket_connected:
            st.success("✅ Connected")
            if st.button("Disconnect"):
                st.session_state.websocket_connected = False
        else:
            if st.button("Connect WebSocket"):
                # Initialize WebSocket connection
                connector = WebSocketDashboardConnector(
                    "ws://localhost:8000/ws/risk",
                    "dummy_token"  # In real app, get from auth
                )
                connector.connect()
        
        # Manual refresh
        if st.button("🔄 Manual Refresh"):
            st.experimental_rerun()
    
    # Main dashboard content
    if portfolio_id:
        # Get cached risk metrics
        cached_data = st.session_state.risk_metrics_cache.get(portfolio_id)
        
        if cached_data:
            # Mock RiskMetricsSnapshot from cached data
            from datetime import datetime
            
            # Convert cached data to RiskMetricsSnapshot-like object
            class MockSnapshot:
                def __init__(self, data):
                    self.portfolio_id = portfolio_id
                    self.timestamp = datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now()
                    metrics = data.get('metrics', {})
                    self.var_95 = metrics.get('var_95', 0)
                    self.cvar_95 = metrics.get('cvar_95', 0) 
                    self.var_99 = metrics.get('var_99', 0)
                    self.cvar_99 = metrics.get('cvar_99', 0)
                    self.max_drawdown = metrics.get('max_drawdown', 0)
                    self.sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    self.sortino_ratio = metrics.get('sortino_ratio', 0)
                    self.calmar_ratio = metrics.get('calmar_ratio', 0)
                    self.annual_volatility = metrics.get('annual_volatility', 0)
                    self.daily_volatility = metrics.get('daily_volatility', 0)
                    self.leverage_ratio = metrics.get('leverage_ratio', 1.0)
                    self.concentration_risk = metrics.get('concentration_risk', {'max_position': 0})
            
            risk_metrics = MockSnapshot(cached_data)
        else:
            risk_metrics = None
        
        # Render dashboard components
        RiskTrafficLightWidget.render(portfolio_id, risk_metrics)
        
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Risk Trends", "🚨 Alert Management", "⚙️ Settings"])
        
        with tab1:
            # Generate some mock historical data if no real data
            if not cached_data:
                mock_historical = []
                for i in range(20):
                    timestamp = datetime.now() - timedelta(minutes=30*i)
                    mock_historical.append(MockSnapshot({
                        'timestamp': timestamp.isoformat(),
                        'metrics': {
                            'var_95': -0.03 + np.random.normal(0, 0.01),
                            'cvar_95': -0.05 + np.random.normal(0, 0.015),
                            'max_drawdown': -0.12 + np.random.normal(0, 0.02),
                            'annual_volatility': 0.18 + np.random.normal(0, 0.02),
                            'sharpe_ratio': 1.2 + np.random.normal(0, 0.1),
                            'leverage_ratio': 1.1 + np.random.normal(0, 0.05),
                            'concentration_risk': {'max_position': 0.25 + np.random.normal(0, 0.02)}
                        }
                    }))
                RealTimeRiskCharts.render(portfolio_id, mock_historical)
            else:
                RealTimeRiskCharts.render(portfolio_id, [risk_metrics])
        
        with tab2:
            AlertHistoryPanel.render(portfolio_id)
        
        with tab3:
            st.subheader("⚙️ Risk Monitoring Settings")
            
            # Alert threshold configuration
            st.markdown("### Alert Thresholds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                var_threshold = st.number_input("VaR (95%) Threshold", value=-0.05, format="%.4f")
                drawdown_threshold = st.number_input("Max Drawdown Threshold", value=-0.20, format="%.4f")
                concentration_threshold = st.number_input("Concentration Threshold", value=0.30, format="%.4f")
            
            with col2:
                leverage_threshold = st.number_input("Leverage Threshold", value=1.5, format="%.2f")
                volatility_threshold = st.number_input("Volatility Threshold", value=0.25, format="%.4f")
                sharpe_threshold = st.number_input("Min Sharpe Ratio", value=0.5, format="%.2f")
            
            if st.button("Update Thresholds"):
                st.success("Alert thresholds updated successfully!")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()


if __name__ == "__main__":
    render_risk_monitoring_page()
