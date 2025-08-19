"""
Risk Management Module - Works without PyPortfolioOpt
Implements VaR, CVaR, Maximum Drawdown, and other risk metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """Risk management for portfolio optimization"""
    
    def __init__(self, returns: pd.DataFrame, weights: np.ndarray = None):
        """
        Initialize risk manager
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights (optional)
        """
        self.returns = returns
        self.weights = weights if weights is not None else np.array([1/len(returns.columns)] * len(returns.columns))
        self.portfolio_returns = self._calculate_portfolio_returns()
        
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns from weights"""
        return (self.returns * self.weights).sum(axis=1)
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        Args:
            confidence_level: Confidence level (default 95%)
            
        Returns:
            VaR value (negative number representing potential loss)
        """
        return np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            confidence_level: Confidence level (default 95%)
            
        Returns:
            CVaR value (expected loss beyond VaR)
        """
        var = self.calculate_var(confidence_level)
        return self.portfolio_returns[self.portfolio_returns <= var].mean()
    
    def calculate_max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate Maximum Drawdown
        
        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        trough_date = drawdown.idxmin()
        
        # Find the peak before the trough
        peak_date = cumulative[:trough_date].idxmax()
        
        return max_drawdown, peak_date, trough_date
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
            
        Returns:
            Sharpe ratio
        """
        annual_return = self.portfolio_returns.mean() * 252
        annual_vol = self.portfolio_returns.std() * np.sqrt(252)
        
        if annual_vol == 0:
            return 0
        
        return (annual_return - risk_free_rate) / annual_vol
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation)
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
            
        Returns:
            Sortino ratio
        """
        annual_return = self.portfolio_returns.mean() * 252
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std() * np.sqrt(252)
        
        if downside_std == 0:
            return float('inf')
        
        return (annual_return - risk_free_rate) / downside_std
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown)
        
        Returns:
            Calmar ratio
        """
        annual_return = self.portfolio_returns.mean() * 252
        max_drawdown, _, _ = self.calculate_max_drawdown()
        
        if max_drawdown == 0:
            return float('inf')
        
        return annual_return / abs(max_drawdown)
    
    def calculate_information_ratio(self, benchmark_returns: pd.Series = None) -> float:
        """
        Calculate Information Ratio
        
        Args:
            benchmark_returns: Benchmark returns (default: equal-weight portfolio)
            
        Returns:
            Information ratio
        """
        if benchmark_returns is None:
            # Use equal-weight as benchmark
            benchmark_returns = self.returns.mean(axis=1)
        
        active_returns = self.portfolio_returns - benchmark_returns
        
        if active_returns.std() == 0:
            return 0
        
        return (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))
    
    def calculate_beta(self, market_returns: pd.Series = None) -> float:
        """
        Calculate Beta relative to market
        
        Args:
            market_returns: Market returns (default: average of all assets)
            
        Returns:
            Beta value
        """
        if market_returns is None:
            market_returns = self.returns.mean(axis=1)
        
        covariance = np.cov(self.portfolio_returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 1
        
        return covariance / market_variance
    
    def calculate_skewness(self) -> float:
        """Calculate return distribution skewness"""
        return stats.skew(self.portfolio_returns)
    
    def calculate_kurtosis(self) -> float:
        """Calculate return distribution kurtosis"""
        return stats.kurtosis(self.portfolio_returns)
    
    def get_risk_metrics(self) -> Dict:
        """
        Calculate all risk metrics
        
        Returns:
            Dictionary of all risk metrics
        """
        max_dd, peak_date, trough_date = self.calculate_max_drawdown()
        
        return {
            'annual_return': self.portfolio_returns.mean() * 252,
            'annual_volatility': self.portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'information_ratio': self.calculate_information_ratio(),
            'beta': self.calculate_beta(),
            'var_95': self.calculate_var(0.95),
            'cvar_95': self.calculate_cvar(0.95),
            'var_99': self.calculate_var(0.99),
            'cvar_99': self.calculate_cvar(0.99),
            'max_drawdown': max_dd,
            'max_drawdown_peak': peak_date,
            'max_drawdown_trough': trough_date,
            'skewness': self.calculate_skewness(),
            'kurtosis': self.calculate_kurtosis(),
            'daily_return_mean': self.portfolio_returns.mean(),
            'daily_return_std': self.portfolio_returns.std(),
            'winning_days': (self.portfolio_returns > 0).sum() / len(self.portfolio_returns),
            'best_day': self.portfolio_returns.max(),
            'worst_day': self.portfolio_returns.min()
        }
    
    def print_risk_report(self):
        """Print formatted risk report"""
        metrics = self.get_risk_metrics()
        
        print("\n" + "="*60)
        print("📊 PORTFOLIO RISK REPORT")
        print("="*60)
        
        print("\n📈 RETURN METRICS:")
        print(f"  Annual Return:        {metrics['annual_return']:+.2%}")
        print(f"  Annual Volatility:    {metrics['annual_volatility']:.2%}")
        print(f"  Daily Mean Return:    {metrics['daily_return_mean']:+.4%}")
        print(f"  Winning Days:         {metrics['winning_days']:.1%}")
        
        print("\n⚖️ RISK-ADJUSTED METRICS:")
        print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:        {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:         {metrics['calmar_ratio']:.2f}")
        print(f"  Information Ratio:    {metrics['information_ratio']:.2f}")
        print(f"  Beta:                 {metrics['beta']:.2f}")
        
        print("\n⚠️ RISK METRICS:")
        print(f"  VaR (95%):           {metrics['var_95']:.3%}")
        print(f"  CVaR (95%):          {metrics['cvar_95']:.3%}")
        print(f"  VaR (99%):           {metrics['var_99']:.3%}")
        print(f"  CVaR (99%):          {metrics['cvar_99']:.3%}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:.2%}")
        print(f"  Best Day:            {metrics['best_day']:+.3%}")
        print(f"  Worst Day:           {metrics['worst_day']:.3%}")
        
        print("\n📊 DISTRIBUTION METRICS:")
        print(f"  Skewness:            {metrics['skewness']:.3f}")
        print(f"  Kurtosis:            {metrics['kurtosis']:.3f}")
        
        print("\n" + "="*60)

# Example usage
def main():
    """Test the risk management module"""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # Simulate returns for 5 assets
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
    returns_data = {}
    
    for asset in assets:
        # Simulate daily returns (mean=0.05% daily, std=1.5% daily)
        returns_data[asset] = np.random.normal(0.0005, 0.015, len(dates))
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Example portfolio weights
    weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
    
    # Initialize risk manager
    risk_mgr = RiskManager(returns_df, weights)
    
    # Print risk report
    risk_mgr.print_risk_report()
    
    # Get specific metrics
    print("\n🎯 Key Risk Metrics:")
    print(f"  Sharpe Ratio: {risk_mgr.calculate_sharpe_ratio():.2f}")
    print(f"  VaR (95%): {risk_mgr.calculate_var():.3%}")
    print(f"  Max Drawdown: {risk_mgr.calculate_max_drawdown()[0]:.2%}")

if __name__ == "__main__":
    main()