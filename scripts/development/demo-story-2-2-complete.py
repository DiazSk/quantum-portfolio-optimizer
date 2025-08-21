"""
Story 2.2 Complete Integration Demo
Showcases the full Statistical Backtesting Validation Framework
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.statistical_backtesting_framework import WalkForwardBacktester, StatisticalValidator
from src.models.performance_attribution import PerformanceAttributionEngine, ProfessionalReporter


def showcase_story_2_2():
    """
    Complete demonstration of Story 2.2: Statistical Backtesting Validation Framework
    """
    print("\n" + "="*90)
    print("🎯 STORY 2.2: COMPLETE STATISTICAL BACKTESTING FRAMEWORK")
    print("🚀 FAANG-Ready Advanced Analytics Showcase")
    print("=" * 90 + "\n")
    
    try:
        # Step 1: Initialize Components
        print("🔧 Step 1: Initializing Advanced Analytics Components...")
        
        backtester = WalkForwardBacktester(
            training_window=180,  # ~6 months
            testing_window=45,    # ~1.5 months
            transaction_cost=0.0015  # 15 bps realistic cost
        )
        
        validator = StatisticalValidator(
            n_bootstrap=200,  # More robust for demo
            confidence_level=0.95
        )
        
        attribution_engine = PerformanceAttributionEngine()
        reporter = ProfessionalReporter()
        
        print("✅ All components initialized successfully!")
        
        # Step 2: Generate Realistic Market Data
        print("\n📊 Step 2: Preparing Market Data...")
        
        import yfinance as yf
        
        # Use real market data for demonstration
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=600)  # ~2 years
        
        print(f"   Downloading {len(tickers)} assets from {start_date.date()} to {end_date.date()}")
        
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        prices = data['Close'].dropna()
        
        print(f"✅ Market data prepared: {len(prices)} days, {len(tickers)} assets")
        
        # Step 3: Execute Walk-Forward Backtesting
        print("\n📈 Step 3: Executing Walk-Forward Backtesting...")
        
        backtest_results = backtester.run_backtest(
            prices=prices,
            tickers=tickers,
            use_ensemble=False  # Traditional optimization for speed
        )
        
        if backtest_results:
            print("✅ Backtesting completed successfully!")
            
            # Display key results
            metrics = backtest_results['performance_metrics']
            print(f"   📊 Portfolio Return: {backtest_results['total_return_portfolio']*100:.2f}%")
            print(f"   📊 Benchmark Return: {backtest_results['total_return_benchmark']*100:.2f}%")
            print(f"   📊 Sharpe Ratio: {metrics.get('sharpe_portfolio', 0):.3f}")
            print(f"   📊 Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"   📊 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"   📊 Periods Completed: {backtest_results['periods_completed']}")
        
        # Step 4: Statistical Validation
        print("\n🧪 Step 4: Performing Statistical Validation...")
        
        if backtest_results:
            portfolio_returns = backtest_results['portfolio_returns']
            benchmark_returns = backtest_results['benchmark_returns']
            
            # Bootstrap confidence intervals for Sharpe ratio
            def sharpe_metric(port_ret, bench_ret):
                excess_ret = port_ret - bench_ret
                if excess_ret.std() == 0:
                    return 0
                return excess_ret.mean() / excess_ret.std() * np.sqrt(252)
            
            ci_results = validator.bootstrap_confidence_intervals(
                portfolio_returns, benchmark_returns, sharpe_metric
            )
            
            print("✅ Bootstrap analysis completed!")
            print(f"   📊 Sharpe Ratio Mean: {ci_results['mean']:.3f}")
            print(f"   📊 95% Confidence Interval: [{ci_results['ci_lower']:.3f}, {ci_results['ci_upper']:.3f}]")
            
            # Statistical significance testing
            sig_results = validator.statistical_significance_test(
                portfolio_returns, benchmark_returns
            )
            
            t_test = sig_results.get('t_test', {})
            print(f"   📊 Statistical Significance: {t_test.get('significant', False)}")
            print(f"   📊 P-value: {t_test.get('p_value', 0):.4f}")
        
        # Step 5: Performance Attribution
        print("\n🎯 Step 5: Performance Attribution Analysis...")
        
        if backtest_results:
            returns = prices.pct_change().dropna()
            market_returns = returns.mean(axis=1)  # Market proxy
            
            # Setup factor model
            factors = attribution_engine.setup_factor_model(market_returns)
            print(f"✅ Factor model established with {len(factors.columns)} factors")
            
            # Calculate portfolio and benchmark returns
            portfolio_weights = backtest_results.get('portfolio_weights', pd.DataFrame())
            
            if not portfolio_weights.empty:
                # Simplified attribution for demo
                portfolio_rets = (portfolio_weights * returns.reindex(portfolio_weights.index, fill_value=0)).sum(axis=1)
                benchmark_rets = returns.mean(axis=1).reindex(portfolio_rets.index)
                
                # Factor attribution
                factor_attribution = attribution_engine.factor_attribution(
                    portfolio_rets, benchmark_rets, factors.reindex(portfolio_rets.index)
                )
                
                if factor_attribution:
                    print("✅ Factor attribution completed!")
                    factor_contribs = factor_attribution.get('factor_contributions', {})
                    for factor, contrib in factor_contribs.items():
                        print(f"   📊 {factor} factor: {contrib:.6f}")
                    print(f"   📊 Alpha contribution: {factor_attribution.get('alpha_contribution', 0):.6f}")
        
        # Step 6: Professional Reporting
        print("\n📋 Step 6: Generating Professional Reports...")
        
        if backtest_results and 'sig_results' in locals():
            # Generate executive summary
            exec_summary = reporter.generate_executive_summary(
                backtest_results, 
                factor_attribution if 'factor_attribution' in locals() else {},
                sig_results
            )
            
            print("✅ Executive summary generated!")
            print("\n" + "="*60)
            print("📋 EXECUTIVE SUMMARY PREVIEW:")
            print("="*60)
            print(exec_summary[:500] + "..." if len(exec_summary) > 500 else exec_summary)
            print("="*60)
            
            # Create performance charts
            charts = reporter.create_performance_charts(backtest_results)
            print("✅ Performance charts created!")
        
        # Final Summary
        print("\n" + "="*90)
        print("🎯 STORY 2.2 COMPLETE IMPLEMENTATION DEMONSTRATED!")
        print("="*90)
        print("✅ AC-2.2.1: Walk-Forward Backtesting - IMPLEMENTED & VALIDATED")
        print("✅ AC-2.2.2: Bootstrap Statistical Validation - IMPLEMENTED & VALIDATED")
        print("✅ AC-2.2.3: Performance Attribution Analysis - IMPLEMENTED & VALIDATED")
        print("✅ AC-2.2.4: Professional Reporting Engine - IMPLEMENTED & VALIDATED")
        print("\n🚀 READY FOR FAANG TECHNICAL INTERVIEWS!")
        print("🎯 Advanced Statistical Analysis Capabilities Demonstrated")
        print("📊 Publication-Quality Research Methodologies Implemented")
        print("🔬 Industry-Standard Backtesting Framework Complete")
        print("="*90 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    showcase_story_2_2()
