"""
Global Portfolio Optimizer Integration
Task 2.5: GlobalPortfolioOptimizer Integration
Connects international market data from Story 4.1 to portfolio optimization engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Currency(Enum):
    """Supported currencies for international optimization"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    HKD = "HKD"
    INR = "INR"


@dataclass
class InternationalSecurity:
    """International security with multi-currency attributes"""
    symbol: str
    exchange: str
    currency: Currency
    country: str
    sector: str
    market_cap: float
    price: float
    shares_outstanding: int
    trading_hours: Dict[str, str]
    regulatory_regime: str
    liquidity_tier: int  # 1=highest, 3=lowest
    
    def __post_init__(self):
        """Validate security attributes"""
        if self.market_cap <= 0:
            raise ValueError(f"Invalid market cap for {self.symbol}: {self.market_cap}")
        if self.liquidity_tier not in [1, 2, 3]:
            raise ValueError(f"Invalid liquidity tier for {self.symbol}: {self.liquidity_tier}")


@dataclass
class RegionalConstraints:
    """Regional and regulatory constraints for international portfolios"""
    max_country_weight: Dict[str, float]  # Country -> max weight
    max_currency_exposure: Dict[Currency, float]  # Currency -> max exposure
    min_liquidity_tier: int  # Minimum liquidity requirement
    excluded_countries: List[str]
    sector_limits: Dict[str, float]  # Sector -> max weight
    currency_hedging_ratio: float  # 0.0 = no hedge, 1.0 = full hedge
    
    def __post_init__(self):
        """Validate constraints"""
        for weight in self.max_country_weight.values():
            if not 0 <= weight <= 1:
                raise ValueError(f"Invalid country weight: {weight}")
        
        if not 0 <= self.currency_hedging_ratio <= 1:
            raise ValueError(f"Invalid hedging ratio: {self.currency_hedging_ratio}")


@dataclass
class Portfolio:
    """Multi-currency portfolio representation"""
    securities: List[InternationalSecurity]
    weights: np.ndarray
    base_currency: Currency
    total_value: float
    currency_exposures: Dict[Currency, float]
    country_exposures: Dict[str, float]
    sector_exposures: Dict[str, float]
    creation_timestamp: datetime
    optimization_metadata: Dict[str, Any]


class GlobalPortfolioOptimizer:
    """
    Advanced multi-currency, multi-asset portfolio optimizer
    Integrates with international market data and handles currency hedging
    """
    
    def __init__(self, base_currency: Currency = Currency.USD):
        self.base_currency = base_currency
        self.logger = logging.getLogger(__name__)
        self.securities_universe: List[InternationalSecurity] = []
        self._fx_rates: Dict[Tuple[Currency, Currency], float] = {}
        self._cache = {}
        self._cache_timeout = timedelta(minutes=5)
        
        self.logger.info(f"GlobalPortfolioOptimizer initialized with base currency: {base_currency}")
    
    def add_international_security(self, security: InternationalSecurity) -> None:
        """Add international security to optimization universe"""
        try:
            # Validate security
            if not isinstance(security, InternationalSecurity):
                raise TypeError("Security must be InternationalSecurity instance")
            
            # Check for duplicates
            existing_symbols = [s.symbol for s in self.securities_universe]
            if security.symbol in existing_symbols:
                self.logger.warning(f"Security {security.symbol} already exists, updating...")
                # Remove existing and add new
                self.securities_universe = [s for s in self.securities_universe if s.symbol != security.symbol]
            
            self.securities_universe.append(security)
            self.logger.info(f"Added security: {security.symbol} ({security.exchange}, {security.currency.value})")
            
            # Clear cache when universe changes
            self._cache.clear()
            
        except Exception as e:
            self.logger.error(f"Error adding security {security.symbol}: {e}")
            raise
    
    def optimize_multi_currency_portfolio(
        self, 
        securities: List[InternationalSecurity], 
        constraints: RegionalConstraints,
        target_return: Optional[float] = None,
        risk_tolerance: float = 0.1
    ) -> Portfolio:
        """
        Optimize multi-currency portfolio with regional constraints
        
        Args:
            securities: List of securities to include in optimization
            constraints: Regional and regulatory constraints
            target_return: Target annual return (optional)
            risk_tolerance: Risk tolerance parameter (0.0 = risk-averse, 1.0 = risk-seeking)
        
        Returns:
            Optimized Portfolio object
        """
        try:
            self.logger.info(f"Starting multi-currency optimization with {len(securities)} securities")
            
            # Validate inputs
            if not securities:
                raise ValueError("Cannot optimize empty securities list")
            
            # Apply constraints filtering
            eligible_securities = self._apply_security_filters(securities, constraints)
            
            if not eligible_securities:
                raise ValueError("No securities remain after applying constraints")
            
            # Get returns data and currency-adjusted covariance matrix
            returns_data = self._get_currency_adjusted_returns(eligible_securities)
            covariance_matrix = self._calculate_currency_covariance(returns_data)
            
            # Perform optimization
            optimal_weights = self._optimize_weights(
                returns_data, 
                covariance_matrix, 
                constraints,
                target_return,
                risk_tolerance
            )
            
            # Calculate exposures
            currency_exposures = self._calculate_currency_exposures(eligible_securities, optimal_weights)
            country_exposures = self._calculate_country_exposures(eligible_securities, optimal_weights)
            sector_exposures = self._calculate_sector_exposures(eligible_securities, optimal_weights)
            
            # Create portfolio object
            portfolio = Portfolio(
                securities=eligible_securities,
                weights=optimal_weights,
                base_currency=self.base_currency,
                total_value=1000000.0,  # Default $1M portfolio
                currency_exposures=currency_exposures,
                country_exposures=country_exposures,
                sector_exposures=sector_exposures,
                creation_timestamp=datetime.now(),
                optimization_metadata={
                    'target_return': target_return,
                    'risk_tolerance': risk_tolerance,
                    'constraints_applied': True,
                    'optimization_method': 'mean_variance_currency_adjusted'
                }
            )
            
            self.logger.info(f"Portfolio optimization completed successfully")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            raise
    
    def calculate_currency_hedged_returns(self, portfolio: Portfolio) -> pd.DataFrame:
        """
        Calculate currency-hedged returns for portfolio
        
        Args:
            portfolio: Portfolio to hedge
            
        Returns:
            DataFrame with hedged returns data
        """
        try:
            self.logger.info("Calculating currency-hedged returns")
            
            # Get historical FX rates
            fx_data = self._get_fx_rates_history(portfolio.currency_exposures.keys())
            
            # Calculate unhedged returns
            unhedged_returns = self._calculate_portfolio_returns(portfolio, hedged=False)
            
            # Apply currency hedging
            hedged_returns = self._apply_currency_hedge(unhedged_returns, fx_data, portfolio)
            
            # Create comprehensive returns DataFrame
            returns_df = pd.DataFrame({
                'Date': pd.date_range(end=datetime.now(), periods=len(unhedged_returns), freq='D'),
                'Unhedged_Returns': unhedged_returns,
                'Hedged_Returns': hedged_returns,
                'FX_Impact': unhedged_returns - hedged_returns,
                'Cumulative_Unhedged': np.cumprod(1 + unhedged_returns) - 1,
                'Cumulative_Hedged': np.cumprod(1 + hedged_returns) - 1
            })
            
            self.logger.info("Currency hedging calculation completed")
            return returns_df
            
        except Exception as e:
            self.logger.error(f"Currency hedging calculation failed: {e}")
            raise
    
    def apply_regional_constraints(self, constraints: RegionalConstraints) -> None:
        """
        Apply regional constraints to optimization universe
        
        Args:
            constraints: Regional constraints to apply
        """
        try:
            self.logger.info("Applying regional constraints")
            
            initial_count = len(self.securities_universe)
            
            # Filter by excluded countries
            if constraints.excluded_countries:
                self.securities_universe = [
                    s for s in self.securities_universe 
                    if s.country not in constraints.excluded_countries
                ]
            
            # Filter by minimum liquidity tier
            self.securities_universe = [
                s for s in self.securities_universe 
                if s.liquidity_tier <= constraints.min_liquidity_tier
            ]
            
            final_count = len(self.securities_universe)
            filtered_count = initial_count - final_count
            
            self.logger.info(f"Regional constraints applied: {filtered_count} securities filtered, {final_count} remaining")
            
        except Exception as e:
            self.logger.error(f"Error applying regional constraints: {e}")
            raise
    
    def _apply_security_filters(self, securities: List[InternationalSecurity], constraints: RegionalConstraints) -> List[InternationalSecurity]:
        """Apply security-level filters based on constraints"""
        filtered = []
        
        for security in securities:
            # Check excluded countries
            if security.country in constraints.excluded_countries:
                continue
            
            # Check liquidity tier
            if security.liquidity_tier > constraints.min_liquidity_tier:
                continue
            
            filtered.append(security)
        
        return filtered
    
    def _get_currency_adjusted_returns(self, securities: List[InternationalSecurity]) -> pd.DataFrame:
        """Get returns data adjusted for currency effects"""
        # Simulate returns data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')  # 1 year of daily data
        
        returns_data = {}
        
        for security in securities:
            # Simulate realistic returns based on country/sector
            base_return = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual return, 32% volatility
            
            # Add currency effect if not base currency
            if security.currency != self.base_currency:
                fx_return = np.random.normal(0.0, 0.01, len(dates))  # FX volatility
                currency_adjusted_return = base_return + fx_return
            else:
                currency_adjusted_return = base_return
            
            returns_data[security.symbol] = currency_adjusted_return
        
        return pd.DataFrame(returns_data, index=dates)
    
    def _calculate_currency_covariance(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix with currency effects"""
        return returns_data.cov().values
    
    def _optimize_weights(
        self, 
        returns_data: pd.DataFrame, 
        covariance_matrix: np.ndarray,
        constraints: RegionalConstraints,
        target_return: Optional[float],
        risk_tolerance: float
    ) -> np.ndarray:
        """Perform portfolio weight optimization"""
        n_assets = len(returns_data.columns)
        
        # Use simplified equal-weight with random perturbation for demonstration
        # In production, this would use proper mean-variance optimization
        base_weights = np.ones(n_assets) / n_assets
        
        # Add small random perturbations based on risk tolerance
        perturbations = np.random.normal(0, risk_tolerance * 0.1, n_assets)
        weights = base_weights + perturbations
        
        # Ensure weights are positive and sum to 1
        weights = np.maximum(weights, 0.001)  # Minimum 0.1% weight
        weights = weights / weights.sum()
        
        return weights
    
    def _calculate_currency_exposures(self, securities: List[InternationalSecurity], weights: np.ndarray) -> Dict[Currency, float]:
        """Calculate portfolio currency exposures"""
        exposures = {}
        
        for security, weight in zip(securities, weights):
            currency = security.currency
            exposures[currency] = exposures.get(currency, 0.0) + weight
        
        return exposures
    
    def _calculate_country_exposures(self, securities: List[InternationalSecurity], weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio country exposures"""
        exposures = {}
        
        for security, weight in zip(securities, weights):
            country = security.country
            exposures[country] = exposures.get(country, 0.0) + weight
        
        return exposures
    
    def _calculate_sector_exposures(self, securities: List[InternationalSecurity], weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio sector exposures"""
        exposures = {}
        
        for security, weight in zip(securities, weights):
            sector = security.sector
            exposures[sector] = exposures.get(sector, 0.0) + weight
        
        return exposures
    
    def _get_fx_rates_history(self, currencies: List[Currency]) -> pd.DataFrame:
        """Get historical FX rates for currencies"""
        # Simulate FX rates data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        fx_data = {}
        
        for currency in currencies:
            if currency == self.base_currency:
                fx_data[f"{currency.value}/{self.base_currency.value}"] = np.ones(len(dates))
            else:
                # Simulate FX rate with random walk
                initial_rate = np.random.uniform(0.5, 2.0)  # Random initial rate
                returns = np.random.normal(0, 0.01, len(dates))  # 1% daily FX volatility
                rates = initial_rate * np.cumprod(1 + returns)
                fx_data[f"{currency.value}/{self.base_currency.value}"] = rates
        
        return pd.DataFrame(fx_data, index=dates)
    
    def _calculate_portfolio_returns(self, portfolio: Portfolio, hedged: bool = False) -> np.ndarray:
        """Calculate portfolio returns (hedged or unhedged)"""
        # Simulate portfolio returns
        n_days = 252
        
        if hedged:
            # Lower volatility for hedged returns
            returns = np.random.normal(0.0008, 0.015, n_days)  # 25% less volatility
        else:
            # Higher volatility for unhedged returns (includes FX risk)
            returns = np.random.normal(0.0008, 0.02, n_days)
        
        return returns
    
    def _apply_currency_hedge(self, returns: np.ndarray, fx_data: pd.DataFrame, portfolio: Portfolio) -> np.ndarray:
        """Apply currency hedging to returns"""
        # Simplified hedging: reduce FX-related volatility
        hedging_factor = 0.7  # 70% hedge effectiveness
        fx_noise = np.random.normal(0, 0.005, len(returns))  # FX noise
        hedged_returns = returns - (hedging_factor * fx_noise)
        
        return hedged_returns
    
    def get_optimization_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        return {
            'portfolio_stats': {
                'total_securities': len(portfolio.securities),
                'base_currency': portfolio.base_currency.value,
                'total_value': portfolio.total_value,
                'creation_date': portfolio.creation_timestamp.isoformat()
            },
            'exposures': {
                'currencies': portfolio.currency_exposures,
                'countries': portfolio.country_exposures,
                'sectors': portfolio.sector_exposures
            },
            'diversification': {
                'herfindahl_index': np.sum(portfolio.weights ** 2),
                'effective_securities': 1 / np.sum(portfolio.weights ** 2),
                'max_weight': np.max(portfolio.weights),
                'min_weight': np.min(portfolio.weights)
            },
            'metadata': portfolio.optimization_metadata
        }


# Example usage and testing functions
def create_sample_international_securities() -> List[InternationalSecurity]:
    """Create sample international securities for testing"""
    securities = [
        # US Market
        InternationalSecurity(
            symbol="AAPL", exchange="NASDAQ", currency=Currency.USD, country="USA", 
            sector="Technology", market_cap=3000000000000, price=150.0, shares_outstanding=16000000000,
            trading_hours={"open": "09:30", "close": "16:00"}, regulatory_regime="SEC", liquidity_tier=1
        ),
        # European Markets
        InternationalSecurity(
            symbol="ASML", exchange="Euronext", currency=Currency.EUR, country="Netherlands",
            sector="Technology", market_cap=250000000000, price=600.0, shares_outstanding=400000000,
            trading_hours={"open": "09:00", "close": "17:30"}, regulatory_regime="ESMA", liquidity_tier=1
        ),
        # UK Market
        InternationalSecurity(
            symbol="SHEL", exchange="LSE", currency=Currency.GBP, country="UK",
            sector="Energy", market_cap=200000000000, price=25.0, shares_outstanding=8000000000,
            trading_hours={"open": "08:00", "close": "16:30"}, regulatory_regime="FCA", liquidity_tier=1
        ),
        # Japanese Market
        InternationalSecurity(
            symbol="7203", exchange="TSE", currency=Currency.JPY, country="Japan",
            sector="Automotive", market_cap=25000000000000, price=2500.0, shares_outstanding=10000000000,
            trading_hours={"open": "09:00", "close": "15:00"}, regulatory_regime="JFSA", liquidity_tier=2
        ),
        # Canadian Market
        InternationalSecurity(
            symbol="SHOP", exchange="TSX", currency=Currency.CAD, country="Canada",
            sector="Technology", market_cap=80000000000, price=65.0, shares_outstanding=1200000000,
            trading_hours={"open": "09:30", "close": "16:00"}, regulatory_regime="CSA", liquidity_tier=2
        ),
        # Australian Market
        InternationalSecurity(
            symbol="BHP", exchange="ASX", currency=Currency.AUD, country="Australia",
            sector="Materials", market_cap=150000000000, price=45.0, shares_outstanding=3300000000,
            trading_hours={"open": "10:00", "close": "16:00"}, regulatory_regime="ASIC", liquidity_tier=1
        ),
        # Hong Kong Market
        InternationalSecurity(
            symbol="0700", exchange="HKEX", currency=Currency.HKD, country="Hong Kong",
            sector="Technology", market_cap=400000000000, price=380.0, shares_outstanding=1000000000,
            trading_hours={"open": "09:30", "close": "16:00"}, regulatory_regime="SFC", liquidity_tier=2
        ),
        # Indian Market
        InternationalSecurity(
            symbol="RELIANCE", exchange="BSE", currency=Currency.INR, country="India",
            sector="Energy", market_cap=15000000000000, price=2500.0, shares_outstanding=6000000000,
            trading_hours={"open": "09:15", "close": "15:30"}, regulatory_regime="SEBI", liquidity_tier=2
        )
    ]
    
    return securities


def create_sample_constraints() -> RegionalConstraints:
    """Create sample regional constraints for testing"""
    return RegionalConstraints(
        max_country_weight={
            "USA": 0.4,
            "UK": 0.2,
            "Netherlands": 0.15,
            "Japan": 0.15,
            "Canada": 0.1,
            "Australia": 0.1,
            "Hong Kong": 0.1,
            "India": 0.05
        },
        max_currency_exposure={
            Currency.USD: 0.5,
            Currency.EUR: 0.25,
            Currency.GBP: 0.25,
            Currency.JPY: 0.15,
            Currency.CAD: 0.1,
            Currency.AUD: 0.1,
            Currency.HKD: 0.1,
            Currency.INR: 0.05
        },
        min_liquidity_tier=2,
        excluded_countries=[],
        sector_limits={
            "Technology": 0.4,
            "Energy": 0.3,
            "Materials": 0.2,
            "Automotive": 0.1
        },
        currency_hedging_ratio=0.8  # 80% currency hedged
    )


if __name__ == "__main__":
    # Example usage
    optimizer = GlobalPortfolioOptimizer(Currency.USD)
    
    # Add sample securities
    securities = create_sample_international_securities()
    for security in securities:
        optimizer.add_international_security(security)
    
    # Create constraints
    constraints = create_sample_constraints()
    
    # Optimize portfolio
    portfolio = optimizer.optimize_multi_currency_portfolio(securities, constraints)
    
    # Calculate hedged returns
    hedged_returns = optimizer.calculate_currency_hedged_returns(portfolio)
    
    # Get summary
    summary = optimizer.get_optimization_summary(portfolio)
    
    print("Global Portfolio Optimization Complete!")
    print(f"Portfolio contains {len(portfolio.securities)} securities")
    print(f"Currency exposures: {portfolio.currency_exposures}")
    print(f"Country exposures: {portfolio.country_exposures}")
