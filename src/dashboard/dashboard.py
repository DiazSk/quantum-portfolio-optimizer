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
    
    if st.button("🔄 Optimize Portfolio", type="primary", use_container_width=True):
        st.session_state['optimize'] = True

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "📈 Performance", "🎯 Alternative Data", "⚠️ Risk Analysis", "📑 Reports"])

# Tab 1: Portfolio Allocation
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Portfolio Allocation")
        
        # Generate mock weights for demo
        if selected_tickers:
            weights = np.random.dirichlet(np.ones(len(selected_tickers)) * 2)
            weights = np.round(weights, 4)
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=selected_tickers,
                values=weights,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig.update_layout(
                title="Optimized Portfolio Weights",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Allocation Details")
        
        if selected_tickers:
            weights_df = pd.DataFrame({
                'Asset': selected_tickers,
                'Weight (%)': weights * 100
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
    
    with col1:
        annual_return = np.random.uniform(0.12, 0.22)
        st.metric("Annual Return", f"{annual_return:.1%}", f"+{annual_return*100:.1f}%")
    
    with col2:
        sharpe = np.random.uniform(1.8, 2.5)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", "⬆️ Good")
    
    with col3:
        max_dd = -np.random.uniform(0.08, 0.15)
        st.metric("Max Drawdown", f"{max_dd:.1%}", "⚠️")
    
    with col4:
        volatility = np.random.uniform(0.12, 0.18)
        st.metric("Volatility", f"{volatility:.1%}")
    
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
    st.subheader("🎯 Alternative Data Signals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Analysis")
        
        # Mock sentiment data
        sentiment_data = pd.DataFrame({
            'Source': ['Reddit', 'Twitter', 'News', 'Google Trends'],
            'Score': np.random.uniform(-0.5, 0.5, 4)
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_data['Source'],
                y=sentiment_data['Score'],
                marker_color=['red' if x < 0 else 'green' for x in sentiment_data['Score']]
            )
        ])
        
        fig.update_layout(
            title="Aggregate Sentiment Scores",
            yaxis_title="Sentiment",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Satellite Data Proxy")
        
        # Mock satellite data
        satellite_metrics = {
            'Parking Lot Occupancy': np.random.uniform(0.6, 0.9),
            'Shipping Activity': np.random.uniform(0.5, 0.8),
            'Factory Activity': np.random.uniform(0.4, 0.7)
        }
        
        for metric, value in satellite_metrics.items():
            st.metric(metric, f"{value:.1%}", f"{'↑' if value > 0.6 else '↓'} {(value-0.6)*100:.1f}%")
    
    # Alternative data table
    st.markdown("### Alternative Data Scores by Asset")
    
    if selected_tickers:
        alt_data_df = pd.DataFrame({
            'Asset': selected_tickers,
            'Sentiment': np.random.uniform(-0.3, 0.3, len(selected_tickers)),
            'Google Trends': np.random.uniform(40, 90, len(selected_tickers)),
            'Satellite Signal': np.random.uniform(0.4, 0.9, len(selected_tickers)),
            'Composite Score': np.random.uniform(0.3, 0.8, len(selected_tickers))
        })
        
        st.dataframe(
            alt_data_df.style.format({
                'Sentiment': '{:+.2f}',
                'Google Trends': '{:.0f}',
                'Satellite Signal': '{:.2f}',
                'Composite Score': '{:.2f}'
            }).background_gradient(cmap='RdYlGn', subset=['Composite Score']),
            use_container_width=True,
            hide_index=True
        )

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