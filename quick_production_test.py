#!/usr/bin/env python3
"""
Quick production test with real APIs
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, '.')

def quick_production_test():
    """Run a quick production test with 2 tickers"""
    try:
        from src.data.alternative_data_collector import EnhancedAlternativeDataCollector
        from src.portfolio.portfolio_optimizer import PortfolioOptimizer
        
        print('🚀 Quick Production Test with Real APIs')
        print('='*50)
        
        # Test with just 2 tickers for speed
        tickers = ['AAPL', 'GOOGL']
        
        # Step 1: Alternative Data Collection
        print('\n📡 Step 1: Collecting Real Alternative Data...')
        collector = EnhancedAlternativeDataCollector(tickers)
        
        alt_data = collector.collect_all_alternative_data()
        alt_scores = collector.calculate_alternative_data_score(alt_data)
        
        print('✅ Alternative data collected!')
        print(alt_scores[['ticker', 'alt_data_score', 'alt_data_confidence']].to_string(index=False))
        
        # Step 2: Portfolio Optimization  
        print('\n🤖 Step 2: Running Portfolio Optimization...')
        optimizer = PortfolioOptimizer(tickers, lookback_years=1, risk_tolerance=5, optimization_method='Maximum Sharpe Ratio')
        portfolio_result = optimizer.run()
        
        if portfolio_result:
            print('✅ Portfolio optimization complete!')
            print(f"Expected Return: {portfolio_result['metrics']['return']:.2%}")
            print(f"Sharpe Ratio: {portfolio_result['metrics']['sharpe']:.2f}")
            
            # Show weights
            for ticker, weight in zip(portfolio_result['tickers'], portfolio_result['weights']):
                print(f"{ticker}: {weight:.2%}")
        
        print('\n🎉 Quick production test SUCCESS!')
        return True
        
    except Exception as e:
        print(f'❌ Production test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_production_test()
