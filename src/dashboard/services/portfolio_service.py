"""
Portfolio service integration for Streamlit dashboard
Task 2.1: Core Streamlit Pages Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PortfolioDataService:
    """Centralized service for portfolio data integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_timeout = timedelta(minutes=5)
    
    def get_portfolio_summary(self, tenant_id: str, user_context: Dict) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            holdings = self.get_holdings(tenant_id, user_context)
            performance = self.get_performance_data(tenant_id)
            risk_metrics = self.get_risk_summary(tenant_id, holdings)
            
            return {
                'holdings': holdings,
                'performance': performance,
                'risk_metrics': risk_metrics,
                'last_updated': datetime.now(),
                'status': 'success'
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {
                'holdings': pd.DataFrame(),
                'performance': pd.DataFrame(),
                'risk_metrics': {},
                'last_updated': datetime.now(),
                'status': 'error',
                'error_message': str(e)
            }
    
    def get_portfolio_holdings(self, tenant_id: str = 'demo_tenant', user_context: Dict = None) -> pd.DataFrame:
        """Alias for get_holdings - for backward compatibility"""
        return self.get_holdings(tenant_id, user_context or {})
    
    def get_holdings(self, tenant_id: str, user_context: Dict) -> pd.DataFrame:
        """Get current portfolio holdings"""
        cache_key = f"holdings_{tenant_id}"
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]['data']
        
        try:
            # Try to get real data from portfolio optimizer
            holdings = self._get_real_holdings(tenant_id)
            
            if holdings is None or holdings.empty:
                holdings = self._get_demo_holdings()
            
            self._cache_data(cache_key, holdings)
            return holdings
            
        except Exception as e:
            self.logger.warning(f"Error getting holdings, using demo data: {e}")
            return self._get_demo_holdings()
    
    def _get_real_holdings(self, tenant_id: str) -> Optional[pd.DataFrame]:
        """Attempt to get real holdings from portfolio optimizer"""
        try:
            from src.portfolio.portfolio_optimizer import PortfolioOptimizer
            
            optimizer = PortfolioOptimizer()
            current_portfolio = optimizer.get_current_portfolio()
            
            if current_portfolio is not None and not current_portfolio.empty:
                # Standardize column names
                if 'ticker' in current_portfolio.columns:
                    current_portfolio = current_portfolio.rename(columns={'ticker': 'Symbol'})
                if 'weight' in current_portfolio.columns:
                    current_portfolio = current_portfolio.rename(columns={'weight': 'Weight'})
                
                # Calculate values if not present
                if 'Value' not in current_portfolio.columns and 'Weight' in current_portfolio.columns:
                    total_portfolio_value = 1000000  # Default $1M portfolio
                    current_portfolio['Value'] = current_portfolio['Weight'] * total_portfolio_value
                
                return current_portfolio
            
            return None
            
        except ImportError:
            self.logger.info("Portfolio optimizer not available, using demo data")
            return None
        except Exception as e:
            self.logger.warning(f"Error accessing portfolio optimizer: {e}")
            return None
    
    def _get_demo_holdings(self) -> pd.DataFrame:
        """Generate demo holdings with realistic data"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK.B', 'JNJ', 'V']
        
        # Use date-based seed for consistent but changing data
        date_seed = int(datetime.now().strftime('%Y%m%d'))
        np.random.seed(date_seed)
        
        # Generate realistic weights
        raw_weights = np.random.exponential(1, len(symbols))
        weights = raw_weights / raw_weights.sum()
        
        portfolio_value = 1000000  # $1M demo portfolio
        
        holdings_data = []
        for i, symbol in enumerate(symbols):
            weight = weights[i]
            value = portfolio_value * weight
            
            # Realistic price data
            base_price = 100 + i * 20 + np.random.normal(0, 10)
            shares = int(value / base_price)
            day_change = np.random.normal(0, 0.02)  # ±2% daily volatility
            
            holdings_data.append({
                'Symbol': symbol,
                'Company': self._get_company_name(symbol),
                'Sector': self._get_sector(symbol),
                'Weight': weight,
                'Value': value,
                'Shares': shares,
                'LastPrice': base_price,
                'DayChange': day_change,
                'PrevClose': base_price / (1 + day_change),
                'MarketCap': base_price * shares * np.random.uniform(50, 500),
                'PE_Ratio': np.random.uniform(15, 35),
                'DividendYield': np.random.uniform(0, 0.04)
            })
        
        return pd.DataFrame(holdings_data)
    
    def get_performance_data(self, tenant_id: str, period_days: int = 30) -> pd.DataFrame:
        """Get portfolio performance time series"""
        cache_key = f"performance_{tenant_id}_{period_days}"
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]['data']
        
        try:
            performance = self._get_real_performance(tenant_id, period_days)
            
            if performance is None or performance.empty:
                performance = self._generate_demo_performance(period_days)
            
            self._cache_data(cache_key, performance)
            return performance
            
        except Exception as e:
            self.logger.warning(f"Error getting performance data: {e}")
            return self._generate_demo_performance(period_days)
    
    def _get_real_performance(self, tenant_id: str, period_days: int) -> Optional[pd.DataFrame]:
        """Attempt to get real performance data"""
        try:
            from src.portfolio.portfolio_optimizer import PortfolioOptimizer
            
            optimizer = PortfolioOptimizer()
            performance = optimizer.get_performance_history(days=period_days)
            
            return performance
            
        except ImportError:
            return None
        except Exception as e:
            self.logger.warning(f"Error accessing performance data: {e}")
            return None
    
    def _generate_demo_performance(self, period_days: int) -> pd.DataFrame:
        """Generate realistic demo performance data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate realistic returns
        np.random.seed(42)  # Consistent demo data
        
        # Portfolio returns with realistic parameters
        daily_returns = np.random.normal(0.0008, 0.015, len(dates))  # 0.2% daily return, 1.5% volatility
        portfolio_values = 1000000 * (1 + daily_returns).cumprod()
        
        # Benchmark returns (slightly lower return and volatility)
        benchmark_returns = np.random.normal(0.0005, 0.012, len(dates))
        benchmark_values = 1000000 * (1 + benchmark_returns).cumprod()
        
        # Calculate cumulative returns
        portfolio_cumret = (portfolio_values / portfolio_values[0] - 1) * 100
        benchmark_cumret = (benchmark_values / benchmark_values[0] - 1) * 100
        
        return pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': portfolio_values,
            'Benchmark_Value': benchmark_values,
            'Portfolio_Return': daily_returns * 100,  # Convert to percentage
            'Benchmark_Return': benchmark_returns * 100,
            'Portfolio_CumReturn': portfolio_cumret,
            'Benchmark_CumReturn': benchmark_cumret,
            'Active_Return': (daily_returns - benchmark_returns) * 100,
            'Drawdown': self._calculate_drawdown(portfolio_values)
        })
    
    def get_risk_summary(self, tenant_id: str, holdings_df: pd.DataFrame) -> Dict[str, float]:
        """Get portfolio risk summary"""
        cache_key = f"risk_{tenant_id}"
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]['data']
        
        try:
            risk_metrics = self._calculate_risk_metrics(holdings_df)
            self._cache_data(cache_key, risk_metrics)
            return risk_metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics()
    
    def _calculate_risk_metrics(self, holdings_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic risk metrics from holdings"""
        if holdings_df.empty:
            return self._get_default_risk_metrics()
        
        weights = holdings_df['Weight'].values
        n_assets = len(holdings_df)
        
        # Concentration metrics
        max_weight = weights.max()
        hhi = (weights ** 2).sum()  # Herfindahl-Hirschman Index
        
        # Diversification metrics
        effective_assets = 1 / hhi
        diversification_ratio = effective_assets / n_assets
        
        # Estimated portfolio volatility
        avg_vol = 0.20  # Assume 20% average asset volatility
        correlation = 0.6  # Assume 60% average correlation
        portfolio_vol = avg_vol * np.sqrt(np.sum(weights ** 2) + 2 * correlation * np.sum(np.outer(weights, weights)))
        
        # VaR estimation
        var_95 = portfolio_vol / np.sqrt(252) * 1.645  # Daily VaR at 95% confidence
        
        return {
            'portfolio_volatility': portfolio_vol,
            'value_at_risk_95': var_95,
            'max_weight': max_weight,
            'concentration_hhi': hhi,
            'effective_assets': effective_assets,
            'diversification_ratio': diversification_ratio,
            'expected_return': 0.08,  # 8% expected annual return
            'sharpe_ratio': 0.08 / portfolio_vol,
            'tracking_error': 0.04
        }
    
    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """Default risk metrics when calculation fails"""
        return {
            'portfolio_volatility': 0.15,
            'value_at_risk_95': 0.015,
            'max_weight': 0.15,
            'concentration_hhi': 0.12,
            'effective_assets': 8.0,
            'diversification_ratio': 0.8,
            'expected_return': 0.08,
            'sharpe_ratio': 0.53,
            'tracking_error': 0.04
        }
    
    def _calculate_drawdown(self, values: np.ndarray) -> np.ndarray:
        """Calculate drawdown series"""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        return drawdown
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol"""
        names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation', 
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms Inc.',
            'BRK.B': 'Berkshire Hathaway Inc.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.'
        }
        return names.get(symbol, f"{symbol} Corp.")
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'NVDA': 'Technology',
            'META': 'Technology',
            'BRK.B': 'Financial Services',
            'JNJ': 'Healthcare',
            'V': 'Financial Services'
        }
        return sectors.get(symbol, 'Technology')
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid"""
        if key not in self._cache:
            return False
        
        cache_time = self._cache[key]['timestamp']
        return datetime.now() - cache_time < self._cache_timeout
    
    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self._cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()


# Global instance for import
portfolio_service = PortfolioDataService()
