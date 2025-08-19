"""
Backtesting Engine for Portfolio Optimization
Implements walk-forward analysis and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResults:
    """Container for backtest results"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    calmar_ratio: float
    sortino_ratio: float
    portfolio_values: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics_by_period: pd.DataFrame

class PortfolioBacktester:
    """
    Professional backtesting engine with walk-forward analysis
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 rebalance_frequency: str = 'monthly',
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting portfolio value
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            transaction_cost: Cost per trade as fraction
            slippage: Slippage factor for market impact
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = 0.04  # 4% annual
        
    def fetch_historical_data(self, 
                            tickers: List[str], 
                            start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data
        """
        print(f"📊 Fetching historical data from {start_date} to {end_date}...")
        
        # Download data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            prices = pd.DataFrame(data['Close'])
            prices.columns = tickers
        else:
            prices = data['Close']
            
        # Fill missing data
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Loaded {len(prices)} days of data for {len(prices.columns)} assets")
        return prices
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive portfolio metrics
        """
        # Annual metrics
        trading_days = 252
        annual_return = returns.mean() * trading_days
        annual_vol = returns.std() * np.sqrt(trading_days)
        
        # Sharpe ratio
        excess_returns = returns.mean() - (self.risk_free_rate / trading_days)
        sharpe = (excess_returns * np.sqrt(trading_days)) / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days)
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'annual_return': annual_return,
            'volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def walk_forward_backtest(self,
                            optimizer,
                            tickers: List[str],
                            start_date: str,
                            end_date: str,
                            lookback_window: int = 252,
                            optimization_window: int = 63,
                            use_ml: bool = True) -> BacktestResults:
        """
        Walk-forward backtesting with periodic reoptimization
        
        Args:
            optimizer: Portfolio optimizer instance
            tickers: List of tickers to trade
            start_date: Backtest start date
            end_date: Backtest end date
            lookback_window: Days of data for training (252 = 1 year)
            optimization_window: Days between rebalancing (63 = quarterly)
            use_ml: Whether to use ML predictions
        """
        print("\n🚀 Starting Walk-Forward Backtest")
        print("=" * 60)
        
        # Fetch all required data (including lookback period)
        data_start = pd.to_datetime(start_date) - timedelta(days=lookback_window * 2)
        prices = self.fetch_historical_data(tickers, 
                                           data_start.strftime('%Y-%m-%d'), 
                                           end_date)
        
        # Initialize portfolio
        portfolio_values = []
        portfolio_dates = []
        weights_history = []
        trades = []
        current_weights = np.array([1/len(tickers)] * len(tickers))  # Equal weight initially
        current_value = self.initial_capital
        
        # Convert dates
        backtest_start = pd.to_datetime(start_date)
        backtest_end = pd.to_datetime(end_date)
        
        # Get rebalance dates based on frequency
        rebalance_dates = self._get_rebalance_dates(
            prices.index, 
            backtest_start, 
            backtest_end
        )
        
        print(f"📅 Rebalancing on {len(rebalance_dates)} dates")
        
        # Iterate through backtest period
        for i, current_date in enumerate(prices.loc[backtest_start:backtest_end].index):
            # Check if we need to rebalance
            if current_date in rebalance_dates:
                print(f"\n🔄 Rebalancing on {current_date.strftime('%Y-%m-%d')}")
                
                # Get training data (lookback window)
                train_end = current_date
                train_start = train_end - timedelta(days=lookback_window)
                train_prices = prices.loc[train_start:train_end, tickers]
                
                if len(train_prices) < 50:
                    print(f"  ⚠️ Insufficient data, keeping current weights")
                    continue
                
                # Optimize portfolio
                try:
                    if use_ml:
                        # Train ML models and optimize
                        optimizer.prices = train_prices
                        optimizer.returns = train_prices.pct_change().dropna()
                        ml_predictions = optimizer.train_ml_models()
                        
                        # Get expected returns
                        expected_returns = np.array([
                            ml_predictions.get(ticker, 0) for ticker in tickers
                        ]) * 252
                    else:
                        # Use historical mean returns
                        expected_returns = train_prices.pct_change().mean().values * 252
                    
                    # Calculate covariance
                    cov_matrix = train_prices.pct_change().cov().values * 252
                    
                    # Optimize weights
                    new_weights = optimizer.optimize_portfolio(expected_returns, cov_matrix)
                    
                    # Apply transaction costs
                    weight_changes = np.abs(new_weights - current_weights)
                    transaction_costs = weight_changes.sum() * self.transaction_cost
                    current_value *= (1 - transaction_costs)
                    
                    # Record trade
                    for j, ticker in enumerate(tickers):
                        if abs(new_weights[j] - current_weights[j]) > 0.01:
                            trades.append({
                                'date': current_date,
                                'ticker': ticker,
                                'old_weight': current_weights[j],
                                'new_weight': new_weights[j],
                                'change': new_weights[j] - current_weights[j]
                            })
                    
                    current_weights = new_weights
                    weights_history.append({
                        'date': current_date,
                        **{ticker: weight for ticker, weight in zip(tickers, current_weights)}
                    })
                    
                    print(f"  ✅ New weights: {dict(zip(tickers, np.round(current_weights, 3)))}")
                    
                except Exception as e:
                    print(f"  ❌ Optimization failed: {e}")
                    continue
            
            # Calculate daily returns
            if i > 0:
                prev_prices = prices.iloc[prices.index.get_loc(current_date) - 1][tickers].values
                curr_prices = prices.loc[current_date, tickers].values
                daily_returns = (curr_prices - prev_prices) / prev_prices
                
                # Apply slippage
                daily_returns = daily_returns * (1 - self.slippage)
                
                # Calculate portfolio return
                portfolio_return = np.dot(current_weights, daily_returns)
                current_value *= (1 + portfolio_return)
            
            portfolio_values.append(current_value)
            portfolio_dates.append(current_date)
        
        # Create results DataFrame
        portfolio_df = pd.DataFrame({
            'date': portfolio_dates,
            'value': portfolio_values
        }).set_index('date')
        
        # Calculate returns
        portfolio_returns = portfolio_df['value'].pct_change().dropna()
        
        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(portfolio_returns)
        
        # Calculate total return
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Create period metrics (monthly)
        period_metrics = self._calculate_period_metrics(portfolio_df, portfolio_returns)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Create results
        results = BacktestResults(
            total_return=total_return,
            annual_return=metrics['annual_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            volatility=metrics['volatility'],
            win_rate=metrics['win_rate'],
            calmar_ratio=metrics['calmar_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            portfolio_values=portfolio_df['value'],
            returns=portfolio_returns,
            trades=trades_df,
            metrics_by_period=period_metrics
        )
        
        # Print summary
        self._print_backtest_summary(results, start_date, end_date)
        
        return results
    
    def _get_rebalance_dates(self, 
                            index: pd.DatetimeIndex, 
                            start: pd.Timestamp, 
                            end: pd.Timestamp) -> List[pd.Timestamp]:
        """
        Get rebalancing dates based on frequency
        """
        dates = index[(index >= start) & (index <= end)]
        
        if self.rebalance_frequency == 'daily':
            return dates.tolist()
        elif self.rebalance_frequency == 'weekly':
            return dates[dates.weekday == 0].tolist()  # Mondays
        elif self.rebalance_frequency == 'monthly':
            return dates.to_series().groupby(pd.Grouper(freq='M')).first().tolist()
        elif self.rebalance_frequency == 'quarterly':
            return dates.to_series().groupby(pd.Grouper(freq='Q')).first().tolist()
        else:
            return dates[::21].tolist()  # Default to monthly (21 trading days)
    
    def _calculate_period_metrics(self, 
                                 portfolio_df: pd.DataFrame, 
                                 returns: pd.Series) -> pd.DataFrame:
        """
        Calculate metrics by period (monthly)
        """
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_metrics = []
        
        for month in monthly_returns.index:
            month_rets = returns[returns.index.to_period('M') == month.to_period('M')]
            if len(month_rets) > 0:
                monthly_metrics.append({
                    'period': month.strftime('%Y-%m'),
                    'return': monthly_returns[month],
                    'volatility': month_rets.std() * np.sqrt(21),
                    'sharpe': (month_rets.mean() * 21) / (month_rets.std() * np.sqrt(21)) 
                              if month_rets.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(month_rets)
                })
        
        return pd.DataFrame(monthly_metrics)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown for a series of returns
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _print_backtest_summary(self, results: BacktestResults, start_date: str, end_date: str):
        """
        Print formatted backtest summary
        """
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${results.portfolio_values.iloc[-1]:,.2f}")
        print("-" * 60)
        print(f"Total Return: {results.total_return:+.2%}")
        print(f"Annual Return: {results.annual_return:+.2%}")
        print(f"Volatility: {results.volatility:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {results.calmar_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print("=" * 60)
    
    def plot_results(self, results: BacktestResults, save_path: str = None):
        """
        Create comprehensive visualization of backtest results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Portfolio Value
        ax1 = axes[0, 0]
        results.portfolio_values.plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        cumulative = (1 + results.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        drawdown.plot(ax=ax2, color='red', linewidth=2)
        ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        ax2.set_title('Drawdown Chart', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap
        ax3 = axes[1, 0]
        monthly_returns = results.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        monthly_pivot = monthly_pivot.pivot(index='Month', columns='Year', values='Return')
        sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax3)
        ax3.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        # 4. Returns Distribution
        ax4 = axes[1, 1]
        results.returns.hist(bins=50, ax=ax4, color='skyblue', edgecolor='black')
        ax4.axvline(x=results.returns.mean(), color='red', linestyle='--', label='Mean')
        ax4.axvline(x=results.returns.median(), color='green', linestyle='--', label='Median')
        ax4.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Daily Return')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Backtest Results - Sharpe: {results.sharpe_ratio:.2f}, Return: {results.total_return:.1%}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        return fig
    
    def export_results(self, results: BacktestResults, filepath: str):
        """
        Export backtest results to JSON
        """
        export_data = {
            'summary': {
                'total_return': results.total_return,
                'annual_return': results.annual_return,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'calmar_ratio': results.calmar_ratio,
                'max_drawdown': results.max_drawdown,
                'volatility': results.volatility,
                'win_rate': results.win_rate
            },
            'portfolio_values': {str(k): v for k, v in results.portfolio_values.to_dict().items()},
            'trades': results.trades.to_dict('records') if not results.trades.empty else [],
            'period_metrics': results.metrics_by_period.to_dict('records')
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Results exported to {filepath}")

# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.portfolio.portfolio_optimizer import PortfolioOptimizer
    
    # Initialize components
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'JPM', 'XOM', 'JNJ']
    optimizer = PortfolioOptimizer(tickers)
    backtester = PortfolioBacktester(
        initial_capital=100000,
        rebalance_frequency='monthly',
        transaction_cost=0.001
    )
    
    # Run backtest
    results = backtester.walk_forward_backtest(
        optimizer=optimizer,
        tickers=tickers,
        start_date='2022-01-01',
        end_date='2024-01-01',
        lookback_window=252,  # 1 year for training
        optimization_window=21,  # Monthly rebalancing
        use_ml=True
    )
    
    # Visualize results
    fig = backtester.plot_results(results, save_path='reports/backtest_results.png')
    
    # Export results
    backtester.export_results(results, 'reports/backtest_results.json')