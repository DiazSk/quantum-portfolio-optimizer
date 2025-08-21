"""
STORY 7.1: PYTHON DATA ANALYSIS FUNDAMENTALS
Core Python Skills for Entry-Level Data Analyst Positions
================================================================================

Comprehensive Python data analysis capabilities demonstrating proficiency in
data manipulation, statistical analysis, and visualization for financial data.

AC-7.1.1: Data Loading & Cleaning
AC-7.1.2: Pandas Data Manipulation
AC-7.1.3: Statistical Analysis
AC-7.1.4: Data Visualization
AC-7.1.5: Documentation & Code Quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.professional_logging import get_logger

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = get_logger(__name__)

class FinancialDataAnalyzer:
    """
    Professional financial data analysis toolkit
    Demonstrates core Python skills for data analyst positions
    """
    
    def __init__(self):
        """Initialize the analyzer with default configurations"""
        self.data_cache = {}
        self.analysis_results = {}
        
        # Common stock symbols for analysis
        self.sample_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
            'JPM', 'BAC', 'WFC', 'GS', 'MS',          # Financials
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO',       # Healthcare
            'SPY', 'QQQ', 'IWM', 'EFA', 'EEM'         # ETFs
        ]
        
        # Portfolio for analysis
        self.sample_portfolio = {
            'AAPL': 0.15, 'MSFT': 0.12, 'GOOGL': 0.10, 'AMZN': 0.08,
            'JPM': 0.08, 'JNJ': 0.08, 'SPY': 0.20, 'QQQ': 0.10,
            'EFA': 0.05, 'EEM': 0.04
        }
        
        logger.info("FinancialDataAnalyzer initialized")
    
    def load_stock_data(self, symbols: List[str], period: str = "2y", 
                       handle_missing: str = "drop") -> pd.DataFrame:
        """
        Load and clean stock price data from multiple sources
        Implements AC-7.1.1: Data Loading & Cleaning
        
        Args:
            symbols: List of stock symbols to fetch
            period: Time period for data (1y, 2y, 5y, max)
            handle_missing: How to handle missing data ('drop', 'forward_fill', 'interpolate')
        
        Returns:
            DataFrame with cleaned stock price data
        """
        try:
            logger.info(f"Loading data for {len(symbols)} symbols over {period}")
            
            # Initialize empty DataFrame
            stock_data = pd.DataFrame()
            
            for symbol in symbols:
                try:
                    # Fetch data using yfinance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    # Basic data validation
                    if data.empty:
                        logger.warning(f"No data found for {symbol}")
                        continue
                    
                    # Use adjusted close price
                    stock_data[symbol] = data['Adj Close']
                    
                    logger.debug(f"Loaded {len(data)} records for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load data for {symbol}: {e}")
                    continue
            
            # Data cleaning operations
            initial_rows = len(stock_data)
            
            # Handle missing values based on strategy
            if handle_missing == "drop":
                stock_data = stock_data.dropna()
            elif handle_missing == "forward_fill":
                stock_data = stock_data.fillna(method='ffill')
            elif handle_missing == "interpolate":
                stock_data = stock_data.interpolate(method='linear')
            
            # Remove any remaining rows with all NaN values
            stock_data = stock_data.dropna(how='all')
            
            # Data quality validation
            if len(stock_data) < initial_rows * 0.9:
                logger.warning(f"Significant data loss: {initial_rows} -> {len(stock_data)} rows")
            
            # Sort by date
            stock_data = stock_data.sort_index()
            
            # Cache the data
            cache_key = f"{'-'.join(symbols)}_{period}_{handle_missing}"
            self.data_cache[cache_key] = stock_data
            
            logger.info(f"Successfully loaded clean data: {stock_data.shape}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to load stock data: {e}")
            raise
    
    def calculate_returns_and_metrics(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive return metrics and statistics
        Implements AC-7.1.2: Pandas Data Manipulation & AC-7.1.3: Statistical Analysis
        
        Args:
            price_data: DataFrame with stock prices
        
        Returns:
            Dictionary containing various return calculations and metrics
        """
        try:
            results = {}
            
            # Daily returns
            daily_returns = price_data.pct_change().dropna()
            results['daily_returns'] = daily_returns
            
            # Cumulative returns
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            results['cumulative_returns'] = cumulative_returns
            
            # Monthly returns (end-of-month)
            monthly_prices = price_data.resample('M').last()
            monthly_returns = monthly_prices.pct_change().dropna()
            results['monthly_returns'] = monthly_returns
            
            # Rolling statistics (30-day windows)
            rolling_mean = daily_returns.rolling(window=30).mean()
            rolling_std = daily_returns.rolling(window=30).std()
            rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)  # Annualized
            
            results['rolling_30d_mean'] = rolling_mean
            results['rolling_30d_volatility'] = rolling_std * np.sqrt(252)  # Annualized
            results['rolling_30d_sharpe'] = rolling_sharpe
            
            # Summary statistics
            summary_stats = self._calculate_summary_statistics(daily_returns)
            results['summary_statistics'] = summary_stats
            
            # Correlation matrix
            correlation_matrix = daily_returns.corr()
            results['correlation_matrix'] = correlation_matrix
            
            logger.info(f"Calculated returns and metrics for {len(price_data.columns)} securities")
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate returns and metrics: {e}")
            raise
    
    def _calculate_summary_statistics(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive summary statistics"""
        
        stats = pd.DataFrame(index=returns_data.columns)
        
        # Basic statistics
        stats['Mean_Daily_Return'] = returns_data.mean()
        stats['Std_Daily_Return'] = returns_data.std()
        stats['Annualized_Return'] = returns_data.mean() * 252
        stats['Annualized_Volatility'] = returns_data.std() * np.sqrt(252)
        
        # Risk metrics
        stats['Sharpe_Ratio'] = stats['Annualized_Return'] / stats['Annualized_Volatility']
        stats['Max_Drawdown'] = self._calculate_max_drawdown(returns_data)
        stats['VaR_95'] = returns_data.quantile(0.05)  # 5% VaR
        stats['CVaR_95'] = returns_data[returns_data <= stats['VaR_95']].mean()  # Conditional VaR
        
        # Distribution statistics
        stats['Skewness'] = returns_data.skew()
        stats['Kurtosis'] = returns_data.kurtosis()
        stats['Min_Return'] = returns_data.min()
        stats['Max_Return'] = returns_data.max()
        
        # Performance metrics
        positive_returns = returns_data > 0
        stats['Win_Rate'] = positive_returns.mean()
        stats['Avg_Win'] = returns_data[positive_returns].mean()
        stats['Avg_Loss'] = returns_data[~positive_returns].mean()
        stats['Win_Loss_Ratio'] = stats['Avg_Win'] / abs(stats['Avg_Loss'])
        
        return stats.round(4)
    
    def _calculate_max_drawdown(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate maximum drawdown for each asset"""
        cumulative = (1 + returns_data).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def analyze_portfolio_performance(self, weights: Dict[str, float], 
                                    returns_data: pd.DataFrame,
                                    benchmark: str = 'SPY') -> Dict[str, Any]:
        """
        Analyze portfolio performance with benchmark comparison
        Demonstrates advanced pandas operations and portfolio analytics
        
        Args:
            weights: Dictionary of asset weights
            returns_data: DataFrame with asset returns
            benchmark: Benchmark symbol for comparison
        
        Returns:
            Dictionary with comprehensive portfolio analysis
        """
        try:
            # Align weights with available data
            available_assets = [asset for asset in weights.keys() if asset in returns_data.columns]
            aligned_weights = {asset: weights[asset] for asset in available_assets}
            
            # Normalize weights to sum to 1
            total_weight = sum(aligned_weights.values())
            normalized_weights = {asset: weight/total_weight for asset, weight in aligned_weights.items()}
            
            # Calculate portfolio returns
            portfolio_weights = pd.Series(normalized_weights)
            portfolio_returns = (returns_data[available_assets] * portfolio_weights).sum(axis=1)
            
            # Benchmark returns
            benchmark_returns = returns_data[benchmark] if benchmark in returns_data.columns else returns_data.iloc[:, 0]
            
            # Performance metrics
            portfolio_stats = self._calculate_portfolio_metrics(portfolio_returns, benchmark_returns)
            
            # Attribution analysis
            contribution_analysis = self._calculate_contribution_analysis(
                returns_data[available_assets], portfolio_weights
            )
            
            # Risk decomposition
            risk_analysis = self._calculate_risk_decomposition(
                returns_data[available_assets], portfolio_weights
            )
            
            results = {
                'portfolio_weights': normalized_weights,
                'portfolio_returns': portfolio_returns,
                'benchmark_returns': benchmark_returns,
                'performance_metrics': portfolio_stats,
                'contribution_analysis': contribution_analysis,
                'risk_analysis': risk_analysis,
                'total_return': (1 + portfolio_returns).cumprod().iloc[-1] - 1,
                'benchmark_total_return': (1 + benchmark_returns).cumprod().iloc[-1] - 1
            }
            
            logger.info(f"Portfolio analysis completed for {len(available_assets)} assets")
            return results
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            raise
    
    def _calculate_portfolio_metrics(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics"""
        
        # Basic performance
        portfolio_ann_return = portfolio_returns.mean() * 252
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        benchmark_ann_return = benchmark_returns.mean() * 252
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        portfolio_sharpe = portfolio_ann_return / portfolio_volatility
        benchmark_sharpe = benchmark_ann_return / benchmark_volatility
        
        # Relative performance
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Beta and alpha
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        alpha = portfolio_ann_return - (beta * benchmark_ann_return)
        
        # Drawdown analysis
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = portfolio_cumulative.expanding().max()
        drawdowns = (portfolio_cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'annualized_return': portfolio_ann_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'benchmark_return': benchmark_ann_return,
            'benchmark_volatility': benchmark_volatility,
            'benchmark_sharpe': benchmark_sharpe,
            'excess_return': portfolio_ann_return - benchmark_ann_return
        }
    
    def _calculate_contribution_analysis(self, asset_returns: pd.DataFrame, 
                                       weights: pd.Series) -> pd.DataFrame:
        """Calculate asset contribution to portfolio performance"""
        
        # Asset contributions to return
        weighted_returns = asset_returns.multiply(weights, axis=1)
        contribution_to_return = weighted_returns.mean() * 252  # Annualized
        
        # Asset contributions to risk
        correlation_matrix = asset_returns.corr()
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix * asset_returns.cov() * 252, weights))
        
        # Marginal contribution to risk
        marginal_contrib = np.dot(correlation_matrix * asset_returns.cov() * 252, weights) / np.sqrt(portfolio_variance)
        contribution_to_risk = weights * marginal_contrib
        
        contribution_df = pd.DataFrame({
            'Weight': weights,
            'Return_Contribution': contribution_to_return,
            'Risk_Contribution': contribution_to_risk,
            'Return_Per_Risk': contribution_to_return / contribution_to_risk
        })
        
        return contribution_df.round(4)
    
    def _calculate_risk_decomposition(self, asset_returns: pd.DataFrame, 
                                    weights: pd.Series) -> Dict[str, Any]:
        """Calculate portfolio risk decomposition"""
        
        # Correlation matrix
        correlation_matrix = asset_returns.corr()
        
        # Covariance matrix (annualized)
        cov_matrix = asset_returns.cov() * 252
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk decomposition
        risk_decomp = {
            'portfolio_volatility': portfolio_volatility,
            'correlation_matrix': correlation_matrix,
            'covariance_matrix': cov_matrix,
            'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
            'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
        }
        
        return risk_decomp
    
    def create_performance_visualizations(self, analysis_results: Dict[str, Any], 
                                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive performance visualization charts
        Implements AC-7.1.4: Data Visualization
        
        Args:
            analysis_results: Results from portfolio analysis
            save_path: Optional path to save figures
        """
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Cumulative Returns Comparison
            ax1 = plt.subplot(3, 3, 1)
            portfolio_cum = (1 + analysis_results['portfolio_returns']).cumprod()
            benchmark_cum = (1 + analysis_results['benchmark_returns']).cumprod()
            
            portfolio_cum.plot(label='Portfolio', linewidth=2, color='blue')
            benchmark_cum.plot(label='Benchmark', linewidth=2, color='red')
            plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Rolling Sharpe Ratio
            ax2 = plt.subplot(3, 3, 2)
            portfolio_rolling_sharpe = analysis_results['portfolio_returns'].rolling(60).mean() / \
                                     analysis_results['portfolio_returns'].rolling(60).std() * np.sqrt(252)
            portfolio_rolling_sharpe.plot(color='blue', linewidth=2)
            plt.title('Rolling 60-Day Sharpe Ratio', fontsize=14, fontweight='bold')
            plt.ylabel('Sharpe Ratio')
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. Portfolio Weights
            ax3 = plt.subplot(3, 3, 3)
            weights_data = pd.Series(analysis_results['portfolio_weights'])
            colors = plt.cm.Set3(np.linspace(0, 1, len(weights_data)))
            wedges, texts, autotexts = ax3.pie(weights_data.values, labels=weights_data.index, 
                                              autopct='%1.1f%%', colors=colors)
            plt.title('Portfolio Allocation', fontsize=14, fontweight='bold')
            
            # 4. Return Contribution Analysis
            ax4 = plt.subplot(3, 3, 4)
            contrib_data = analysis_results['contribution_analysis']['Return_Contribution'].sort_values(ascending=True)
            contrib_data.plot(kind='barh', color=['red' if x < 0 else 'green' for x in contrib_data.values])
            plt.title('Asset Return Contributions', fontsize=14, fontweight='bold')
            plt.xlabel('Annualized Return Contribution')
            plt.grid(True, alpha=0.3)
            
            # 5. Risk Contribution Analysis
            ax5 = plt.subplot(3, 3, 5)
            risk_contrib = analysis_results['contribution_analysis']['Risk_Contribution'].sort_values(ascending=False)
            risk_contrib.plot(kind='bar', color='orange', alpha=0.7)
            plt.title('Asset Risk Contributions', fontsize=14, fontweight='bold')
            plt.ylabel('Risk Contribution')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 6. Correlation Heatmap
            ax6 = plt.subplot(3, 3, 6)
            correlation_matrix = analysis_results['risk_analysis']['correlation_matrix']
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            plt.title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
            
            # 7. Monthly Returns Distribution
            ax7 = plt.subplot(3, 3, 7)
            monthly_returns = analysis_results['portfolio_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.hist(bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(monthly_returns.mean(), color='red', linestyle='--', 
                       label=f'Mean: {monthly_returns.mean():.2%}')
            plt.title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Monthly Return')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 8. Drawdown Analysis
            ax8 = plt.subplot(3, 3, 8)
            portfolio_cumulative = (1 + analysis_results['portfolio_returns']).cumprod()
            rolling_max = portfolio_cumulative.expanding().max()
            drawdowns = (portfolio_cumulative - rolling_max) / rolling_max
            drawdowns.plot(color='red', alpha=0.7, linewidth=2)
            plt.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color='red')
            plt.title('Portfolio Drawdowns', fontsize=14, fontweight='bold')
            plt.ylabel('Drawdown')
            plt.grid(True, alpha=0.3)
            
            # 9. Performance Metrics Summary
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            metrics = analysis_results['performance_metrics']
            
            summary_text = f"""
Portfolio Performance Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return Metrics:
  Annual Return: {metrics['annualized_return']:.2%}
  Volatility: {metrics['volatility']:.2%}
  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
  Max Drawdown: {metrics['max_drawdown']:.2%}

Risk Metrics:
  Beta: {metrics['beta']:.2f}
  Alpha: {metrics['alpha']:.2%}
  Tracking Error: {metrics['tracking_error']:.2%}
  Information Ratio: {metrics['information_ratio']:.2f}

Benchmark Comparison:
  Excess Return: {metrics['excess_return']:.2%}
  Benchmark Return: {metrics['benchmark_return']:.2%}
  Benchmark Sharpe: {metrics['benchmark_sharpe']:.2f}
"""
            
            ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Chart saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            raise

def main():
    """
    Main function demonstrating comprehensive Python data analysis skills
    This serves as a portfolio piece for data analyst interviews
    """
    print("🐍 Python Data Analysis Fundamentals - Portfolio Demonstration")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = FinancialDataAnalyzer()
    
    # 1. Data Loading and Cleaning
    print("\n📊 Step 1: Loading and Cleaning Financial Data")
    stock_data = analyzer.load_stock_data(analyzer.sample_symbols[:10], period="2y")
    print(f"✅ Loaded clean data for {len(stock_data.columns)} stocks over {len(stock_data)} days")
    print(f"📈 Data range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
    
    # 2. Returns and Statistical Analysis
    print("\n📈 Step 2: Calculating Returns and Statistical Metrics")
    returns_analysis = analyzer.calculate_returns_and_metrics(stock_data)
    print(f"✅ Calculated returns for {len(returns_analysis['daily_returns'].columns)} assets")
    
    # Display summary statistics for top performers
    print("\n🏆 Top 3 Performers by Sharpe Ratio:")
    summary_stats = returns_analysis['summary_statistics'].sort_values('Sharpe_Ratio', ascending=False)
    top_performers = summary_stats.head(3)[['Annualized_Return', 'Annualized_Volatility', 'Sharpe_Ratio']]
    print(top_performers.to_string())
    
    # 3. Portfolio Analysis
    print("\n💼 Step 3: Portfolio Performance Analysis")
    portfolio_analysis = analyzer.analyze_portfolio_performance(
        analyzer.sample_portfolio, 
        returns_analysis['daily_returns']
    )
    
    # Display key portfolio metrics
    metrics = portfolio_analysis['performance_metrics']
    print(f"✅ Portfolio Analysis Complete:")
    print(f"   📊 Annual Return: {metrics['annualized_return']:.2%}")
    print(f"   📉 Volatility: {metrics['volatility']:.2%}")
    print(f"   ⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   🔻 Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"   🎯 Alpha vs Benchmark: {metrics['alpha']:.2%}")
    
    # 4. Data Visualization
    print("\n📊 Step 4: Creating Performance Visualizations")
    try:
        analyzer.create_performance_visualizations(portfolio_analysis)
        print("✅ Interactive charts displayed")
    except Exception as e:
        print(f"⚠️  Chart display failed: {e}")
    
    # 5. Generate Summary Report
    print("\n📋 Step 5: Analysis Summary & Key Insights")
    print("=" * 50)
    
    # Risk-Return Analysis
    contrib_analysis = portfolio_analysis['contribution_analysis']
    best_contributor = contrib_analysis['Return_Contribution'].idxmax()
    worst_contributor = contrib_analysis['Return_Contribution'].idxmin()
    
    print(f"🎯 Portfolio Insights:")
    print(f"   • Best Performer: {best_contributor} (+{contrib_analysis.loc[best_contributor, 'Return_Contribution']:.2%})")
    print(f"   • Worst Performer: {worst_contributor} ({contrib_analysis.loc[worst_contributor, 'Return_Contribution']:.2%})")
    
    # Risk Analysis
    risk_analysis = portfolio_analysis['risk_analysis']
    print(f"   • Average Correlation: {risk_analysis['average_correlation']:.2f}")
    print(f"   • Portfolio Volatility: {risk_analysis['portfolio_volatility']:.2%}")
    
    # Recommendations
    print(f"\n💡 Data-Driven Recommendations:")
    if metrics['sharpe_ratio'] > 1.0:
        print("   ✅ Portfolio shows strong risk-adjusted performance")
    else:
        print("   ⚠️  Consider optimizing risk-return profile")
    
    if metrics['alpha'] > 0:
        print(f"   ✅ Generating alpha of {metrics['alpha']:.2%} vs benchmark")
    else:
        print("   ⚠️  Portfolio underperforming benchmark")
    
    print("\n🎓 Analysis demonstrates proficiency in:")
    print("   • Data loading and cleaning with error handling")
    print("   • Advanced pandas operations and statistical analysis")
    print("   • Financial metrics calculation and risk assessment")
    print("   • Professional data visualization and reporting")
    print("   • Clean, documented, and maintainable code structure")
    
    print("\n" + "=" * 70)
    print("📊 Python Data Analysis Portfolio - Demonstration Complete ✅")

if __name__ == "__main__":
    main()
