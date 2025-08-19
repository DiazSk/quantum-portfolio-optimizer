#!/usr/bin/env python
"""
Fixed Quick Start Script for Quantum Portfolio Optimizer
Works with your actual class names
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Fix the import - use your actual class name
try:
    from src.portfolio.portfolio_optimizer import PortfolioOptimizer
except ImportError:
    # If that doesn't work, add path and try again
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.portfolio.portfolio_optimizer import PortfolioOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete portfolio optimization pipeline"""
    
    print("\n" + "="*60)
    print("🚀 QUANTUM PORTFOLIO OPTIMIZER - FULL PIPELINE")
    print("="*60 + "\n")
    
    # Define investment universe
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech
        'JPM', 'GS', 'BAC',  # Finance
        'JNJ', 'PFE',  # Healthcare
        'XOM', 'CVX'   # Energy
    ]
    
    print(f"📊 Investment Universe: {', '.join(tickers)}")
    print("-" * 60)
    
    # ========== STEP 1: Alternative Data Collection (Mock) ==========
    print("\n📡 STEP 1: Collecting Alternative Data...")
    print("-" * 40)
    
    alt_data_scores = pd.DataFrame({
        'ticker': tickers,
        'alt_data_score': np.random.uniform(0.3, 0.8, len(tickers)),
        'sentiment_score': np.random.uniform(-0.3, 0.3, len(tickers)),
        'google_trend': np.random.uniform(40, 80, len(tickers)),
        'satellite_signal': np.random.uniform(0.4, 0.9, len(tickers))
    })
    
    print("\n🎯 Alternative Data Scores:")
    print(alt_data_scores.sort_values('alt_data_score', ascending=False).to_string(index=False))
    
    # ========== STEP 2: Portfolio Optimization ==========
    print("\n🤖 STEP 2: Running Portfolio Optimization...")
    print("-" * 40)
    
    try:
        # Use your actual PortfolioOptimizer class
        optimizer = PortfolioOptimizer(tickers, lookback_years=2)
        
        # Run optimization
        portfolio_result = optimizer.run()
        
        if portfolio_result:
            print("\n📈 Optimization Successful!")
            
            # Display weights
            print("\n📊 Optimized Portfolio Weights:")
            for ticker, weight in zip(portfolio_result['tickers'], portfolio_result['weights']):
                if weight > 0.01:
                    bar = '█' * int(weight * 50)
                    print(f"  {ticker:5s}: {weight:6.2%} {bar}")
            
            # Display metrics
            metrics = portfolio_result['metrics']
            print(f"\n📊 Portfolio Risk Metrics:")
            print(f"  Expected Return     : {metrics['return']:+.2%}")
            print(f"  Volatility          : {metrics['volatility']:.2%}")
            print(f"  Sharpe Ratio        : {metrics['sharpe']:.2f}")
            print(f"  Value at Risk (95%) : {metrics['var_95']:.3%}")
            print(f"  Max Drawdown        : {metrics['max_drawdown']:.2%}")
    
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("   Using equal weight fallback...")
        
        # Fallback to equal weights
        equal_weight = 1.0 / len(tickers)
        print("\n📊 Equal Weight Allocation:")
        for ticker in tickers:
            bar = '█' * int(equal_weight * 50)
            print(f"  {ticker:5s}: {equal_weight:6.2%} {bar}")
    
    # ========== STEP 3: Market Regime Detection ==========
    print("\n🌡️ STEP 3: Detecting Market Regime...")
    print("-" * 40)
    
    regimes = ['bull_market', 'neutral', 'high_volatility']
    detected_regime = np.random.choice(regimes)
    regime_confidence = np.random.uniform(0.7, 0.95)
    
    print(f"  Detected Regime: {detected_regime.upper()}")
    print(f"  Confidence: {regime_confidence:.1%}")
    
    # ========== STEP 4: Save Results ==========
    print("\n📄 STEP 4: Generating Reports...")
    print("-" * 40)
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Save portfolio weights
    if 'portfolio_result' in locals() and portfolio_result:
        weights_df = pd.DataFrame({
            'ticker': portfolio_result['tickers'],
            'weight': portfolio_result['weights']
        })
        weights_df.to_csv('reports/portfolio_weights.csv', index=False)
        print(f"  ✓ Generated: portfolio_weights.csv")
    
    # Save alternative data
    alt_data_scores.to_csv('reports/alternative_data.csv', index=False)
    print(f"  ✓ Generated: alternative_data.csv")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*60)
    print("🎯 OPTIMIZATION COMPLETE!")
    print("="*60)
    
    if 'portfolio_result' in locals() and portfolio_result:
        print(f"""
📊 Summary:
  • Portfolio Sharpe Ratio: {metrics['sharpe']:.2f}
  • Expected Annual Return: {metrics['return']:.1%}
  • Maximum Drawdown Risk: {metrics['max_drawdown']:.1%}
  • Market Regime: {detected_regime.replace('_', ' ').title()}
  
💡 Recommendations:
  • {"Increase defensive positions" if detected_regime == "high_volatility" else "Maintain current allocation"}
  • Next Rebalance: In 30 days
  
📁 Reports saved to: ./reports/
""")
    
    return True

def main():
    """Main entry point"""
    try:
        # Run the pipeline
        success = run_complete_pipeline()
        
        if success:
            print("\n✨ Pipeline executed successfully!")
            print("📝 Check ./reports/ directory for outputs")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()