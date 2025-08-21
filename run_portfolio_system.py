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
    
    # ========== STEP 1: Alternative Data Collection (Real APIs) ==========
    print("\n📡 STEP 1: Collecting Alternative Data...")
    print("-" * 40)
    
    # Import real alternative data collector
    from src.data.alternative_data_collector import AlternativeDataCollector
    
    # Initialize collector with real API keys from .env
    alt_collector = AlternativeDataCollector(tickers)
    
    # Collect real alternative data for all tickers
    print("🔍 Collecting real alternative data from multiple sources...")
    try:
        # Use asyncio to run the async method
        import asyncio
        alt_data_raw = asyncio.run(alt_collector.collect_all_alternative_data())
        
        # Process the raw data into the expected format
        # Calculate real composite alternative data score from multiple sources
        sentiment_scores = alt_data_raw.get('reddit_sentiment', [0.0] * len(alt_data_raw))
        google_trends = alt_data_raw.get('google_search_volume', [50.0] * len(alt_data_raw))
        satellite_signals = alt_data_raw.get('satellite_retail_activity', [0.5] * len(alt_data_raw))
        
        # Real composite scoring algorithm (weighted average of normalized signals)
        composite_scores = []
        for i in range(len(alt_data_raw)):
            # Normalize each signal to 0-1 range
            norm_sentiment = (sentiment_scores[i] + 1.0) / 2.0  # -1 to 1 → 0 to 1
            norm_google = google_trends[i] / 100.0  # 0 to 100 → 0 to 1
            norm_satellite = satellite_signals[i]  # Already 0 to 1
            
            # Weighted composite score (sentiment: 40%, google: 30%, satellite: 30%)
            composite_score = (0.4 * norm_sentiment + 0.3 * norm_google + 0.3 * norm_satellite)
            composite_scores.append(composite_score)
        
        alt_data_scores = pd.DataFrame({
            'ticker': alt_data_raw['ticker'],
            'alt_data_score': composite_scores,  # Real calculated composite score
            'sentiment_score': sentiment_scores,
            'google_trend': google_trends,
            'satellite_signal': satellite_signals
        })
        print(f"✅ Successfully collected alternative data for {len(alt_data_scores)} securities")
        
    except Exception as e:
        print(f"⚠️  Warning: Alternative data collection failed ({e}), using neutral fallback")
        alt_data_scores = pd.DataFrame({
            'ticker': tickers,
            'alt_data_score': [0.5] * len(tickers),  # Neutral baseline
            'sentiment_score': [0.0] * len(tickers),  # Neutral sentiment
            'google_trend': [50.0] * len(tickers),   # Baseline trend
            'satellite_signal': [0.5] * len(tickers)  # Baseline signal
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
    
    # Real regime detection based on actual market data
    try:
        # Import real regime detection module
        from src.models.regime_detection import RegimeDetector
        
        # Initialize regime detector with market data
        regime_detector = RegimeDetector()
        
        # Detect current market regime based on VIX, yield curve, and market momentum
        regime_result = regime_detector.detect_current_regime()
        detected_regime = regime_result['regime']
        regime_confidence = regime_result['confidence']
        
        print(f"  Detected Regime: {detected_regime.upper()}")
        print(f"  Confidence: {regime_confidence:.1%}")
        print(f"  Key Indicators: VIX={regime_result.get('vix', 'N/A')}, Yield Spread={regime_result.get('yield_spread', 'N/A')}")
        
    except ImportError:
        # Fallback: Simple VIX-based regime detection using real market data
        print("  Using VIX-based regime detection...")
        try:
            import yfinance as yf
            vix_data = yf.download("^VIX", period="5d", interval="1d")
            
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                
                # Real regime classification based on VIX levels
                if current_vix < 15:
                    detected_regime = "bull_market"
                    regime_confidence = 0.85
                elif current_vix < 25:
                    detected_regime = "neutral"
                    regime_confidence = 0.75
                else:
                    detected_regime = "high_volatility"
                    regime_confidence = 0.90
                    
                print(f"  Detected Regime: {detected_regime.upper()} (VIX: {current_vix:.1f})")
                print(f"  Confidence: {regime_confidence:.1%}")
            else:
                # Conservative fallback when no data available
                detected_regime = "neutral"
                regime_confidence = 0.50
                print(f"  Detected Regime: NEUTRAL (default - no VIX data)")
                print(f"  Confidence: {regime_confidence:.1%}")
                
        except Exception as e:
            # Conservative fallback
            detected_regime = "neutral"
            regime_confidence = 0.50
            print(f"  Detected Regime: NEUTRAL (default - data unavailable)")
            print(f"  Confidence: {regime_confidence:.1%}")
    
    except Exception as e:
        # Conservative fallback for any other errors
        detected_regime = "neutral"
        regime_confidence = 0.50
        print(f"  Detected Regime: NEUTRAL (default - regime detection failed)")
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