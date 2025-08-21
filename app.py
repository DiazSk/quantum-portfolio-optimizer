"""
🚀 Quantum Portfolio Optimizer - Live Demo
Professional Portfolio Management Platform

LIVE DEMO for Fall 2025 Job Applications
Enterprise-grade portfolio optimization with ML and AI insights
"""

import streamlit as st
import sys
import os

# Set page config first
st.set_page_config(
    page_title="Quantum Portfolio Optimizer - Live Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/DiazSk/quantum-portfolio-optimizer',
        'Report a bug': 'https://github.com/DiazSk/quantum-portfolio-optimizer/issues',
        'About': "Professional Portfolio Optimization Platform - Live Demo for Recruiters"
    }
)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Import and run the main dashboard
    from src.dashboard.dashboard import main
    
    # Professional header for recruiters
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79, #2e86ab); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🚀 Quantum Portfolio Optimizer
        </h1>
        <h3 style="color: #a8dadc; text-align: center; margin: 0.5rem 0;">
            Enterprise Portfolio Management Platform - Live Demo
        </h3>
        <p style="color: #f1faee; text-align: center; margin: 0;">
            Advanced ML/AI-powered portfolio optimization with real-time analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add professional notice for recruiters
    st.info("""
    **👋 Welcome Recruiters & Hiring Managers!**
    
    This is a **live demonstration** of my portfolio optimization platform. Features include:
    - **Real-time portfolio optimization** with 6+ ML algorithms
    - **Risk analytics** with VaR, Sharpe ratios, and Monte Carlo simulations  
    - **AI-powered insights** with market sentiment analysis
    - **Professional reporting** with PDF generation and performance tracking
    
    **Tech Stack:** Python, Streamlit, Plotly, scikit-learn, yfinance, pandas, numpy
    **GitHub:** https://github.com/DiazSk/quantum-portfolio-optimizer
    """)
    
    # Run the main dashboard
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"""
    **🚨 Deployment Error:** {str(e)}
    
    **For Recruiters:** This platform is still being optimized for cloud deployment.
    Please visit the GitHub repository for full source code and documentation:
    
    **GitHub:** https://github.com/DiazSk/quantum-portfolio-optimizer
    """)
    
    # Show basic demo content as fallback
    st.markdown("## 📊 Portfolio Optimization Demo (Fallback)")
    
    import pandas as pd
    import numpy as np
    import plotly.express as px
    
    # Create sample portfolio visualization
    sample_data = {
        'Asset': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Weight': [0.25, 0.20, 0.20, 0.20, 0.15],
        'Expected Return': [0.12, 0.11, 0.13, 0.10, 0.15],
        'Risk': [0.22, 0.20, 0.25, 0.24, 0.35]
    }
    
    df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portfolio Allocation")
        fig = px.pie(df, values='Weight', names='Asset', 
                     title="Optimized Portfolio Weights")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚖️ Risk vs Return")
        fig = px.scatter(df, x='Risk', y='Expected Return', 
                        size='Weight', text='Asset',
                        title="Risk-Return Profile")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 🎯 Key Features Demonstrated
    - **Multi-asset portfolio optimization** with risk constraints
    - **Interactive visualizations** with Plotly
    - **Real-time data integration** with financial APIs
    - **Professional UI/UX** suitable for enterprise clients
    
    **Full source code and advanced features available on GitHub!**
    """)
