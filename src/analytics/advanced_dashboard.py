"""
Advanced Analytics Dashboard
Epic 7: Interactive analytics with real-time data streaming and statistical analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    """Analytics dashboard configuration"""
    refresh_interval: int = 30  # seconds
    max_historical_days: int = 365
    default_tickers: List[str] = None
    enable_real_time: bool = True
    
    def __post_init__(self):
        if self.default_tickers is None:
            self.default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

class RealTimeDataStream:
    """Real-time market data streaming"""
    
    def __init__(self, tickers: List[str], callback=None):
        self.tickers = tickers
        self.callback = callback
        self.running = False
        self.data_cache = {}
        self.thread = None
        
    def start_stream(self):
        """Start real-time data streaming"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()
        logger.info(f"Started real-time stream for {len(self.tickers)} tickers")
    
    def stop_stream(self):
        """Stop real-time data streaming"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Stopped real-time data stream")
    
    def _stream_worker(self):
        """Background worker for data streaming"""
        while self.running:
            try:
                # Fetch latest data for all tickers
                for ticker in self.tickers:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest_data = {
                            'ticker': ticker,
                            'price': hist['Close'].iloc[-1],
                            'volume': hist['Volume'].iloc[-1],
                            'timestamp': datetime.now(),
                            'change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                        }
                        
                        self.data_cache[ticker] = latest_data
                        
                        if self.callback:
                            self.callback(ticker, latest_data)
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in data stream: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get latest cached data"""
        return self.data_cache.copy()

class StatisticalAnalyzer:
    """Advanced statistical analysis engine"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + StatisticalAnalyzer.calculate_returns(prices)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    @staticmethod
    def monte_carlo_simulation(initial_price: float, 
                             returns: pd.Series, 
                             days: int = 252, 
                             simulations: int = 1000) -> np.ndarray:
        """Monte Carlo price simulation"""
        mu = returns.mean()
        sigma = returns.std()
        
        results = np.zeros((simulations, days))
        results[:, 0] = initial_price
        
        for sim in range(simulations):
            for day in range(1, days):
                random_shock = np.random.normal(mu, sigma)
                results[sim, day] = results[sim, day-1] * (1 + random_shock)
        
        return results

class AdvancedAnalyticsDashboard:
    """Main analytics dashboard class"""
    
    def __init__(self):
        self.config = AnalyticsConfig()
        self.data_stream = None
        self.analyzer = StatisticalAnalyzer()
        
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Advanced Portfolio Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Dashboard header
        st.title("🚀 Advanced Portfolio Analytics Dashboard")
        st.markdown("*Real-time market data, statistical analysis, and risk metrics*")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main dashboard content
        if st.session_state.get('selected_tickers'):
            self._render_main_dashboard()
        else:
            st.info("Please select tickers in the sidebar to begin analysis")
    
    def _render_sidebar(self):
        """Render sidebar configuration"""
        st.sidebar.header("📈 Configuration")
        
        # Ticker selection
        st.sidebar.subheader("Select Tickers")
        default_tickers = st.sidebar.multiselect(
            "Choose securities to analyze:",
            options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'SPY', 'QQQ'],
            default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            key='ticker_selector'
        )
        
        # Custom ticker input
        custom_ticker = st.sidebar.text_input("Add custom ticker:", key='custom_ticker')
        if custom_ticker and st.sidebar.button("Add Ticker"):
            if custom_ticker.upper() not in default_tickers:
                default_tickers.append(custom_ticker.upper())
                st.session_state.ticker_selector = default_tickers
                st.experimental_rerun()
        
        st.session_state.selected_tickers = default_tickers
        
        # Time period selection
        st.sidebar.subheader("📅 Analysis Period")
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        
        selected_period = st.sidebar.selectbox(
            "Select time period:",
            options=list(period_options.keys()),
            index=3,  # Default to 1 year
            key='period_selector'
        )
        st.session_state.analysis_period = period_options[selected_period]
        
        # Real-time data toggle
        st.sidebar.subheader("⚡ Real-Time Features")
        enable_realtime = st.sidebar.checkbox(
            "Enable real-time updates",
            value=True,
            key='realtime_toggle'
        )
        
        if enable_realtime and not st.session_state.get('stream_active'):
            self._start_realtime_stream()
        elif not enable_realtime and st.session_state.get('stream_active'):
            self._stop_realtime_stream()
        
        # Analysis options
        st.sidebar.subheader("🔧 Analysis Options")
        st.session_state.show_correlations = st.sidebar.checkbox("Show correlations", True)
        st.session_state.show_monte_carlo = st.sidebar.checkbox("Monte Carlo simulation", True)
        st.session_state.show_risk_metrics = st.sidebar.checkbox("Risk metrics", True)
        st.session_state.confidence_level = st.sidebar.slider("VaR Confidence Level", 0.01, 0.10, 0.05, 0.01)
        
    def _render_main_dashboard(self):
        """Render main dashboard content"""
        tickers = st.session_state.selected_tickers
        period = st.session_state.analysis_period
        
        # Load data
        with st.spinner("Loading market data..."):
            data = self._load_data(tickers, period)
            
        if data.empty:
            st.error("Could not load data for selected tickers")
            return
        
        # Real-time updates section
        if st.session_state.get('realtime_toggle'):
            self._render_realtime_section()
        
        # Main analytics tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Price Analysis", 
            "📈 Returns & Volatility", 
            "🔍 Risk Metrics",
            "🎯 Correlations",
            "🎲 Monte Carlo"
        ])
        
        with tab1:
            self._render_price_analysis(data)
            
        with tab2:
            self._render_returns_analysis(data)
            
        with tab3:
            self._render_risk_analysis(data)
            
        with tab4:
            if st.session_state.show_correlations:
                self._render_correlation_analysis(data)
            else:
                st.info("Correlation analysis disabled in sidebar")
                
        with tab5:
            if st.session_state.show_monte_carlo:
                self._render_monte_carlo(data)
            else:
                st.info("Monte Carlo simulation disabled in sidebar")
    
    def _render_realtime_section(self):
        """Render real-time data section"""
        st.subheader("⚡ Real-Time Market Data")
        
        if hasattr(self, 'data_stream') and self.data_stream:
            latest_data = self.data_stream.get_latest_data()
            
            if latest_data:
                cols = st.columns(len(latest_data))
                
                for idx, (ticker, data) in enumerate(latest_data.items()):
                    with cols[idx]:
                        change_color = "green" if data['change'] >= 0 else "red"
                        st.metric(
                            label=ticker,
                            value=f"${data['price']:.2f}",
                            delta=f"{data['change']:.2f}%"
                        )
                        st.caption(f"Vol: {data['volume']:,}")
            else:
                st.info("Waiting for real-time data...")
        else:
            st.warning("Real-time stream not active")
    
    def _render_price_analysis(self, data: pd.DataFrame):
        """Render price analysis charts"""
        st.subheader("📊 Price Performance Analysis")
        
        # Normalized price chart
        fig_normalized = go.Figure()
        
        for ticker in data.columns:
            normalized_prices = data[ticker] / data[ticker].iloc[0] * 100
            fig_normalized.add_trace(go.Scatter(
                x=data.index,
                y=normalized_prices,
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
        
        fig_normalized.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_normalized, use_container_width=True)
        
        # Volume analysis
        st.subheader("📈 Volume Analysis")
        
        # Load volume data
        volume_data = pd.DataFrame()
        for ticker in data.columns:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=st.session_state.analysis_period)
                volume_data[ticker] = hist['Volume']
            except:
                continue
        
        if not volume_data.empty:
            fig_volume = px.line(
                volume_data,
                title="Trading Volume Over Time",
                labels={'value': 'Volume', 'index': 'Date'}
            )
            fig_volume.update_layout(height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
    
    def _render_returns_analysis(self, data: pd.DataFrame):
        """Render returns and volatility analysis"""
        st.subheader("📈 Returns & Volatility Analysis")
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Returns distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Return Distributions")
            fig_hist = make_subplots(
                rows=len(data.columns), cols=1,
                subplot_titles=[f"{ticker} Returns" for ticker in data.columns],
                vertical_spacing=0.1
            )
            
            for idx, ticker in enumerate(data.columns):
                fig_hist.add_trace(
                    go.Histogram(
                        x=returns[ticker],
                        name=ticker,
                        nbinsx=50,
                        showlegend=False
                    ),
                    row=idx+1, col=1
                )
            
            fig_hist.update_layout(height=200*len(data.columns))
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Rolling Volatility")
            volatility = returns.rolling(window=30).std() * np.sqrt(252)
            
            fig_vol = px.line(
                volatility,
                title="30-Day Rolling Volatility (Annualized)",
                labels={'value': 'Volatility', 'index': 'Date'}
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # Returns summary statistics
        st.subheader("📊 Summary Statistics")
        
        summary_stats = pd.DataFrame({
            'Mean Return (Daily)': returns.mean(),
            'Std Dev (Daily)': returns.std(),
            'Annualized Return': returns.mean() * 252,
            'Annualized Volatility': returns.std() * np.sqrt(252),
            'Sharpe Ratio': [self.analyzer.calculate_sharpe_ratio(returns[col]) for col in returns.columns],
            'Max Drawdown': [self.analyzer.calculate_max_drawdown(data[col]) for col in data.columns]
        })
        
        st.dataframe(summary_stats.round(4), use_container_width=True)
    
    def _render_risk_analysis(self, data: pd.DataFrame):
        """Render risk metrics analysis"""
        st.subheader("🔍 Risk Metrics Analysis")
        
        returns = data.pct_change().dropna()
        confidence_level = st.session_state.confidence_level
        
        # Calculate risk metrics
        risk_metrics = {}
        for ticker in data.columns:
            ticker_returns = returns[ticker]
            risk_metrics[ticker] = {
                'VaR (Daily)': self.analyzer.calculate_var(ticker_returns, confidence_level),
                'CVaR (Daily)': ticker_returns[ticker_returns <= self.analyzer.calculate_var(ticker_returns, confidence_level)].mean(),
                'Volatility (Ann.)': ticker_returns.std() * np.sqrt(252),
                'Max Drawdown': self.analyzer.calculate_max_drawdown(data[ticker]),
                'Sharpe Ratio': self.analyzer.calculate_sharpe_ratio(ticker_returns)
            }
        
        # Display risk metrics table
        risk_df = pd.DataFrame(risk_metrics).T
        st.dataframe(risk_df.round(4), use_container_width=True)
        
        # VaR visualization
        col1, col2 = st.columns(2)
        
        with col1:
            var_values = [risk_metrics[ticker]['VaR (Daily)'] for ticker in data.columns]
            fig_var = px.bar(
                x=data.columns,
                y=var_values,
                title=f"Value at Risk ({int(confidence_level*100)}% confidence)",
                labels={'x': 'Ticker', 'y': 'VaR (Daily)'}
            )
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            sharpe_values = [risk_metrics[ticker]['Sharpe Ratio'] for ticker in data.columns]
            fig_sharpe = px.bar(
                x=data.columns,
                y=sharpe_values,
                title="Sharpe Ratios",
                labels={'x': 'Ticker', 'y': 'Sharpe Ratio'}
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
    
    def _render_correlation_analysis(self, data: pd.DataFrame):
        """Render correlation analysis"""
        st.subheader("🎯 Correlation Analysis")
        
        returns = data.pct_change().dropna()
        correlation_matrix = returns.corr()
        
        # Correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title="Returns Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Rolling correlations
        if len(data.columns) >= 2:
            st.subheader("📈 Rolling Correlations")
            
            ticker1 = st.selectbox("First ticker:", data.columns, key='corr_ticker1')
            ticker2 = st.selectbox("Second ticker:", data.columns, index=1, key='corr_ticker2')
            
            if ticker1 != ticker2:
                window = st.slider("Rolling window (days):", 30, 252, 60)
                
                rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2])
                
                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr,
                    mode='lines',
                    name=f'{ticker1} vs {ticker2}',
                    line=dict(width=2)
                ))
                
                fig_rolling.update_layout(
                    title=f"Rolling {window}-Day Correlation: {ticker1} vs {ticker2}",
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    height=400
                )
                
                st.plotly_chart(fig_rolling, use_container_width=True)
    
    def _render_monte_carlo(self, data: pd.DataFrame):
        """Render Monte Carlo simulation"""
        st.subheader("🎲 Monte Carlo Price Simulation")
        
        ticker = st.selectbox("Select ticker for simulation:", data.columns, key='mc_ticker')
        days = st.slider("Simulation period (days):", 30, 365, 252)
        simulations = st.slider("Number of simulations:", 100, 2000, 1000)
        
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                returns = data[ticker].pct_change().dropna()
                initial_price = data[ticker].iloc[-1]
                
                # Run simulation
                sim_results = self.analyzer.monte_carlo_simulation(
                    initial_price, returns, days, simulations
                )
                
                # Plot results
                fig_mc = go.Figure()
                
                # Add sample paths
                for i in range(min(50, simulations)):  # Show max 50 paths
                    fig_mc.add_trace(go.Scatter(
                        x=list(range(days)),
                        y=sim_results[i],
                        mode='lines',
                        line=dict(width=0.5, color='lightblue'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add percentile bands
                percentiles = np.percentile(sim_results, [5, 25, 50, 75, 95], axis=0)
                
                fig_mc.add_trace(go.Scatter(
                    x=list(range(days)),
                    y=percentiles[2],  # Median
                    mode='lines',
                    name='Median',
                    line=dict(width=3, color='red')
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=list(range(days)),
                    y=percentiles[0],
                    mode='lines',
                    name='5th Percentile',
                    line=dict(width=2, color='orange', dash='dash')
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=list(range(days)),
                    y=percentiles[4],
                    mode='lines',
                    name='95th Percentile',
                    line=dict(width=2, color='green', dash='dash')
                ))
                
                fig_mc.update_layout(
                    title=f"Monte Carlo Price Simulation: {ticker}",
                    xaxis_title="Days",
                    yaxis_title="Price ($)",
                    height=600
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Summary statistics
                final_prices = sim_results[:, -1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
                with col2:
                    st.metric("5th Percentile", f"${np.percentile(final_prices, 5):.2f}")
                with col3:
                    st.metric("95th Percentile", f"${np.percentile(final_prices, 95):.2f}")
                with col4:
                    prob_profit = (final_prices > initial_price).mean() * 100
                    st.metric("Prob. of Profit", f"{prob_profit:.1f}%")
    
    def _load_data(self, tickers: List[str], period: str) -> pd.DataFrame:
        """Load historical price data"""
        data = pd.DataFrame()
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                data[ticker] = hist['Close']
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
                continue
        
        return data.dropna()
    
    def _start_realtime_stream(self):
        """Start real-time data stream"""
        if st.session_state.selected_tickers:
            self.data_stream = RealTimeDataStream(st.session_state.selected_tickers)
            self.data_stream.start_stream()
            st.session_state.stream_active = True
    
    def _stop_realtime_stream(self):
        """Stop real-time data stream"""
        if hasattr(self, 'data_stream') and self.data_stream:
            self.data_stream.stop_stream()
        st.session_state.stream_active = False

# Main entry point
def main():
    """Main function to run the dashboard"""
    dashboard = AdvancedAnalyticsDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
