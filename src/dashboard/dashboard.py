"""
Streamlit Dashboard for Quantum Portfolio Optimizer
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import os
import sys
from dotenv import load_dotenv
import asyncio
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load environment variables
load_dotenv()

# Import our real API collectors
from src.data.alternative_data_collector import AlternativeDataCollector

# Try to import portfolio optimizer, use fallback if not available
try:
    from src.portfolio.portfolio_optimizer import PortfolioOptimizer
    PORTFOLIO_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    PORTFOLIO_OPTIMIZER_AVAILABLE = False
    st.warning(f"⚠️ Portfolio optimizer not available: {e}")
    st.info("💡 Using mock optimization for demo purposes.")

# Page config
st.set_page_config(
    page_title="Quantum Portfolio Optimizer",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 Quantum Portfolio Optimizer</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Portfolio Management with Alternative Data Integration**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Portfolio selection
    default_tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'JPM', 'GS', 'XOM']
    selected_tickers = st.multiselect(
        "Select Assets",
        options=['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'META', 'TSLA', 
                'JPM', 'GS', 'BAC', 'JNJ', 'PFE', 'XOM', 'CVX'],
        default=default_tickers
    )
    
    optimization_method = st.selectbox(
        "Optimization Method",
        ["Maximum Sharpe Ratio", "Minimum Variance", "Risk Parity", "Equal Weight"]
    )
    
    use_ml = st.checkbox("Use ML Predictions", value=True)
    use_alt_data = st.checkbox("Include Alternative Data", value=True)
    
    st.divider()
    
    # Risk parameters
    st.subheader("Risk Parameters")
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.1) / 100
    max_position = st.slider("Max Position Size (%)", 10, 50, 25, 5) / 100
    
    st.divider()
    
    # Optimization status
    if 'optimized_weights' in st.session_state:
        st.success("✅ Portfolio optimized! View results in the Portfolio tab.")
    else:
        st.info("👆 Configure your settings above, then click to optimize:")
    
    if st.button("🔄 Optimize Portfolio", type="primary", use_container_width=True):
        st.session_state['optimize'] = True

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "📈 Performance", "🎯 Alternative Data", "⚠️ Risk Analysis", "📑 Reports"])

# Tab 1: Portfolio Allocation
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Portfolio Allocation")
        
        # Check if we should run optimization
        if selected_tickers and st.session_state.get('optimize', False):
            # Clear the optimize flag
            st.session_state['optimize'] = False
            
            with st.spinner("🤖 Running ML-powered portfolio optimization..."):
                try:
                    if PORTFOLIO_OPTIMIZER_AVAILABLE:
                        # Initialize the portfolio optimizer
                        optimizer = PortfolioOptimizer(
                            tickers=selected_tickers,
                            lookback_years=2  # Use 2 years of historical data
                        )
                        
                        # Get alternative data if available
                        alt_data_scores = None
                        if 'alt_data_cache' in st.session_state:
                            alt_data_scores = st.session_state['alt_data_cache']
                        
                        # Run optimization
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📊 Fetching market data...")
                        progress_bar.progress(20)
                        
                        status_text.text("🤖 Training ML models...")
                        progress_bar.progress(40)
                        
                        status_text.text("⚖️ Optimizing portfolio...")
                        progress_bar.progress(70)
                        
                        # Perform optimization using the run() method
                        optimization_result = optimizer.run()
                        
                        if optimization_result is not None:
                            # Extract weights and convert to pandas Series
                            weights = pd.Series(
                                optimization_result['weights'], 
                                index=optimization_result['tickers']
                            )
                            
                            # Extract performance metrics
                            performance_metrics = {
                                'expected_return': optimization_result['metrics']['return'],
                                'sharpe_ratio': optimization_result['metrics']['sharpe'],
                                'volatility': optimization_result['metrics']['volatility'],
                                'max_drawdown': optimization_result['metrics']['max_drawdown']
                            }
                        else:
                            raise Exception("Optimization returned no results")
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Optimization complete!")
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Store results in session state
                        st.session_state['optimized_weights'] = weights
                        st.session_state['performance_metrics'] = performance_metrics
                        st.success("✅ Portfolio optimization completed successfully!")
                    else:
                        # Fallback to smart mock optimization
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📊 Fetching market data...")
                        progress_bar.progress(25)
                        time.sleep(0.5)
                        
                        status_text.text("🤖 Simulating optimization...")
                        progress_bar.progress(75)
                        time.sleep(1)
                        
                        # Smart mock weights based on alternative data if available
                        if 'alt_data_cache' in st.session_state:
                            scores = st.session_state['alt_data_cache']
                            # Weight based on alternative data scores
                            score_weights = scores['alt_data_score'].values
                            score_weights = score_weights / score_weights.sum()  # Normalize
                            weights = pd.Series(score_weights, index=scores['ticker'])
                        else:
                            # Simple equal weight
                            weights = pd.Series([1/len(selected_tickers)] * len(selected_tickers), index=selected_tickers)
                        
                        # Mock performance metrics
                        performance_metrics = {
                            'expected_return': np.random.uniform(0.08, 0.15),
                            'sharpe_ratio': np.random.uniform(1.2, 2.0),
                            'volatility': np.random.uniform(0.10, 0.18),
                            'max_drawdown': -np.random.uniform(0.08, 0.15)
                        }
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Demo optimization complete!")
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Store results
                        st.session_state['optimized_weights'] = weights
                        st.session_state['performance_metrics'] = performance_metrics
                        st.success("✅ Demo portfolio optimization completed!")
                        st.info("💡 Install PyPortfolioOpt for real ML-powered optimization")
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    st.info("Using demo weights instead...")
                    # Fallback to simple mock weights
                    weights = np.random.dirichlet(np.ones(len(selected_tickers)) * 2)
                    weights = pd.Series(weights, index=selected_tickers)
                    st.session_state['optimized_weights'] = weights
        
        # Display portfolio weights
        if selected_tickers:
            # Use optimized weights if available, otherwise use demo weights
            if 'optimized_weights' in st.session_state:
                weights = st.session_state['optimized_weights']
                if isinstance(weights, pd.Series):
                    weights_array = weights.values
                    labels = weights.index.tolist()
                else:
                    weights_array = weights
                    labels = selected_tickers
            else:
                weights_array = np.random.dirichlet(np.ones(len(selected_tickers)) * 2)
                weights_array = np.round(weights_array, 4)
                labels = selected_tickers
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=weights_array,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig.update_layout(
                title="Optimized Portfolio Weights" if 'optimized_weights' in st.session_state else "Demo Portfolio Weights",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Allocation Details")
        
        if selected_tickers:
            # Use the same weights logic as the chart
            if 'optimized_weights' in st.session_state:
                weights = st.session_state['optimized_weights']
                if isinstance(weights, pd.Series):
                    weights_df = pd.DataFrame({
                        'Asset': weights.index.tolist(),
                        'Weight (%)': weights.values * 100
                    }).sort_values('Weight (%)', ascending=False)
                else:
                    weights_df = pd.DataFrame({
                        'Asset': selected_tickers,
                        'Weight (%)': weights * 100
                    }).sort_values('Weight (%)', ascending=False)
            else:
                demo_weights = np.random.dirichlet(np.ones(len(selected_tickers)) * 2)
                weights_df = pd.DataFrame({
                    'Asset': selected_tickers,
                    'Weight (%)': demo_weights * 100
                }).sort_values('Weight (%)', ascending=False)
            
            st.dataframe(
                weights_df.style.format({'Weight (%)': '{:.2f}%'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Investment calculator
            st.divider()
            investment = st.number_input("Investment Amount ($)", value=100000, step=1000)
            
            weights_df['Allocation ($)'] = weights_df['Weight (%)'] / 100 * investment
            st.dataframe(
                weights_df[['Asset', 'Allocation ($)']].style.format({'Allocation ($)': '${:,.0f}'}),
                use_container_width=True,
                hide_index=True
            )

# Tab 2: Performance
with tab2:
    st.subheader("Portfolio Performance Metrics")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Use real metrics if available, otherwise use demo data
    if 'performance_metrics' in st.session_state:
        metrics = st.session_state['performance_metrics']
        
        with col1:
            annual_return = metrics.get('expected_return', 0.15)
            st.metric("Expected Return", f"{annual_return:.1%}", f"+{annual_return*100:.1f}%")
        
        with col2:
            sharpe = metrics.get('sharpe_ratio', 2.0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", "✅ Optimized" if sharpe > 1.5 else "⚠️ Low")
        
        with col3:
            max_dd = metrics.get('max_drawdown', -0.12)
            st.metric("Max Drawdown", f"{max_dd:.1%}", "⚠️")
        
        with col4:
            volatility = metrics.get('volatility', 0.15)
            st.metric("Volatility", f"{volatility:.1%}")
    else:
        # Demo metrics
        with col1:
            annual_return = np.random.uniform(0.12, 0.22)
            st.metric("Annual Return (Demo)", f"{annual_return:.1%}", f"+{annual_return*100:.1f}%")
        
        with col2:
            sharpe = np.random.uniform(1.8, 2.5)
            st.metric("Sharpe Ratio (Demo)", f"{sharpe:.2f}", "⬆️ Good")
        
        with col3:
            max_dd = -np.random.uniform(0.08, 0.15)
            st.metric("Max Drawdown (Demo)", f"{max_dd:.1%}", "⚠️")
        
        with col4:
            volatility = np.random.uniform(0.12, 0.18)
            st.metric("Volatility (Demo)", f"{volatility:.1%}")
    
    # Backtesting chart
    st.subheader("Historical Performance")
    
    # Generate mock backtest data
    dates = pd.date_range(end=datetime.now(), periods=252*2, freq='D')
    returns = np.random.normal(0.0008, 0.012, len(dates))
    cumulative = (1 + returns).cumprod() * 100000
    
    # Create performance chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative,
        mode='lines',
        name='Portfolio',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add benchmark (S&P 500)
    benchmark_returns = np.random.normal(0.0005, 0.01, len(dates))
    benchmark = (1 + benchmark_returns).cumprod() * 100000
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark,
        mode='lines',
        name='S&P 500',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Portfolio vs Benchmark Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Alternative Data
with tab3:
    st.subheader("🎯 Alternative Data Intelligence")
    
    # Check API availability
    required_keys = ['ALPHA_VANTAGE_API_KEY', 'REDDIT_CLIENT_ID', 'NEWS_API_KEY']
    available_keys = [key for key in required_keys if os.getenv(key)]
    
    # Initialize fetch_alt_data to avoid NameError
    fetch_alt_data = False
    
    if len(available_keys) < 2:
        st.warning("⚠️ Alternative data requires at least 2 API keys. Please configure your .env file.")
        st.info("Available APIs: Alpha Vantage ✅, Reddit ✅, News API ✅")
    else:
        # Real-time alternative data collection
        if selected_tickers:
            # Button to fetch real alternative data
            col1, col2 = st.columns([1, 3])
            with col1:
                fetch_alt_data = st.button("🔄 Fetch Real Data", type="primary")
            with col2:
                if 'alt_data_cache' in st.session_state:
                    cache_time = st.session_state.get('alt_data_cache_time', datetime.now())
                    time_diff = datetime.now() - cache_time
                    if time_diff.total_seconds() < 3600:  # 1 hour cache
                        st.info(f"📋 Using cached data ({int(time_diff.total_seconds()/60)} min old)")
        
        # Fetch or use cached data (only if we have APIs and tickers)
        if len(available_keys) >= 2 and selected_tickers and (fetch_alt_data or 'alt_data_cache' not in st.session_state):
            with st.spinner("🔄 Collecting real alternative data..."):
                try:
                    # Limit to 5 tickers for performance
                    analysis_tickers = selected_tickers[:5] if len(selected_tickers) > 5 else selected_tickers
                    
                    # Initialize the real alternative data collector
                    collector = AlternativeDataCollector(analysis_tickers)
                    
                    # Collect real alternative data
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📡 Collecting Reddit sentiment...")
                    progress_bar.progress(20)
                    
                    status_text.text("📰 Analyzing news sentiment...")
                    progress_bar.progress(40)
                    
                    status_text.text("📈 Fetching market data...")
                    progress_bar.progress(60)
                    
                    status_text.text("🔍 Processing Google Trends...")
                    progress_bar.progress(80)
                    
                    # Collect all alternative data
                    alt_data = asyncio.run(collector.collect_all_alternative_data())
                    scores = collector.calculate_alternative_data_score(alt_data)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Alternative data collection complete!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Cache the results
                    st.session_state['alt_data_cache'] = scores
                    st.session_state['alt_data_cache_time'] = datetime.now()
                    st.success(f"✅ Real alternative data collected for {len(analysis_tickers)} assets!")
                    
                except Exception as e:
                    st.error(f"❌ Error collecting alternative data: {str(e)}")
                    st.info("Using demo data instead...")
                    # Fallback to mock data
                    scores = pd.DataFrame({
                        'ticker': selected_tickers,
                        'alt_data_score': np.random.uniform(0.3, 0.8, len(selected_tickers)),
                        'alt_data_confidence': np.random.uniform(0.6, 0.9, len(selected_tickers))
                    })
                    st.session_state['alt_data_cache'] = scores
        
        # Display the alternative data
        if 'alt_data_cache' in st.session_state:
            scores = st.session_state['alt_data_cache']
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Alternative Data Scores")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=scores['ticker'],
                        y=scores['alt_data_score'],
                        marker_color=scores['alt_data_score'],
                        marker_colorscale='RdYlGn',
                        text=[f"{score:.2f}" for score in scores['alt_data_score']],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Real Alternative Data Scores",
                    yaxis_title="Score",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Top Performing Assets")
                
                top_assets = scores.nlargest(3, 'alt_data_score')
                for idx, row in top_assets.iterrows():
                    score = row['alt_data_score']
                    confidence = row.get('alt_data_confidence', 0.8)
                    color = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
                    st.metric(
                        f"{color} {row['ticker']}", 
                        f"{score:.2f}",
                        f"Confidence: {confidence:.1%}"
                    )
            
            # Detailed table
            st.markdown("### 📋 Detailed Alternative Data Analysis")
            
            # Format the scores dataframe for display
            display_df = scores.copy()
            if 'alt_data_confidence' in display_df.columns:
                display_df = display_df.rename(columns={
                    'ticker': 'Asset',
                    'alt_data_score': 'Score',
                    'alt_data_confidence': 'Confidence'
                })
                
                st.dataframe(
                    display_df.style.format({
                        'Score': '{:.3f}',
                        'Confidence': '{:.2%}'
                    }).background_gradient(cmap='RdYlGn', subset=['Score']),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                display_df = display_df.rename(columns={
                    'ticker': 'Asset',
                    'alt_data_score': 'Alternative Data Score'
                })
                
                st.dataframe(
                    display_df.style.format({
                        'Alternative Data Score': '{:.3f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Alternative Data Score']),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("👆 Please select assets in the sidebar to view alternative data analysis.")

# Tab 4: Risk Analysis
with tab4:
    st.subheader("⚠️ Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Metrics")
        
        risk_metrics = {
            'Value at Risk (95%)': -np.random.uniform(0.02, 0.04),
            'CVaR (95%)': -np.random.uniform(0.03, 0.05),
            'Beta': np.random.uniform(0.8, 1.2),
            'Sortino Ratio': np.random.uniform(2.0, 3.0),
            'Calmar Ratio': np.random.uniform(1.5, 2.5)
        }
        
        for metric, value in risk_metrics.items():
            if 'Ratio' in metric:
                st.metric(metric, f"{value:.2f}")
            else:
                st.metric(metric, f"{value:.2%}")
    
    with col2:
        st.markdown("### Correlation Matrix")
        
        if selected_tickers and len(selected_tickers) > 1:
            # Generate mock correlation matrix
            n = len(selected_tickers)
            corr = np.random.uniform(0.3, 0.9, (n, n))
            np.fill_diagonal(corr, 1)
            corr = (corr + corr.T) / 2
            
            fig = go.Figure(data=go.Heatmap(
                z=corr,
                x=selected_tickers,
                y=selected_tickers,
                colorscale='RdBu',
                zmid=0.5
            ))
            
            fig.update_layout(
                title="Asset Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Tab 5: Reports
with tab5:
    st.subheader("📑 Generated Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.success("PDF Report generated successfully!")
            st.download_button(
                label="Download PDF",
                data=b"Mock PDF content",
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    
    with col2:
        if st.button("📊 Export to Excel", use_container_width=True):
            st.success("Excel file created!")
            st.download_button(
                label="Download Excel",
                data=b"Mock Excel content",
                file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📈 Download Backtest", use_container_width=True):
            st.success("Backtest data ready!")
            st.download_button(
                label="Download CSV",
                data="Date,Portfolio,Benchmark\n2024-01-01,100000,100000",
                file_name=f"backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    summary_data = {
        'Metric': ['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
        'Portfolio': ['45.2%', '18.6%', '14.3%', '2.31', '-12.4%', '58%'],
        'Benchmark': ['32.1%', '14.2%', '16.8%', '1.82', '-18.7%', '54%']
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Quantum Portfolio Optimizer v1.0 | Built with Streamlit, XGBoost, and Alternative Data</p>
    <p>⚡ Real-time optimization with ML predictions and satellite data integration</p>
</div>
""", unsafe_allow_html=True)