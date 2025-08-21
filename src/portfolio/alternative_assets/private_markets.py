"""
Private markets and alternative strategy data models for institutional portfolio management.

This module provides comprehensive data structures for private equity, venture capital,
hedge funds, infrastructure, and other private market investment strategies.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
import math


class PrivateMarketType(Enum):
    """Types of private market investments"""
    PRIVATE_EQUITY = "private_equity"
    VENTURE_CAPITAL = "venture_capital"
    HEDGE_FUND = "hedge_fund"
    REAL_ESTATE_PRIVATE = "real_estate_private"
    INFRASTRUCTURE = "infrastructure"
    NATURAL_RESOURCES = "natural_resources"
    DISTRESSED_DEBT = "distressed_debt"
    MEZZANINE_CAPITAL = "mezzanine_capital"
    FUND_OF_FUNDS = "fund_of_funds"


class InvestmentStage(Enum):
    """Investment stage classifications"""
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    GROWTH = "growth"
    BUYOUT = "buyout"
    DISTRESSED = "distressed"
    TURNAROUND = "turnaround"
    EXPANSION = "expansion"
    LATE_STAGE = "late_stage"


class FundStatus(Enum):
    """Fund lifecycle status"""
    FUNDRAISING = "fundraising"
    INVESTING = "investing"
    HARVESTING = "harvesting"
    LIQUIDATING = "liquidating"
    CLOSED = "closed"


class GeographicFocus(Enum):
    """Geographic investment focus"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    EMERGING_MARKETS = "emerging_markets"
    GLOBAL = "global"
    CHINA = "china"
    INDIA = "india"


class HedgeFundStrategyType(Enum):
    """Hedge fund strategy types"""
    GLOBAL_MACRO = "global_macro"
    EQUITY_LONG_SHORT = "equity_long_short"
    QUANTITATIVE = "quantitative"
    MULTI_STRATEGY = "multi_strategy"
    EVENT_DRIVEN = "event_driven"
    RELATIVE_VALUE = "relative_value"
    DISTRESSED = "distressed"
    MERGER_ARBITRAGE = "merger_arbitrage"
    CONVERTIBLE_ARBITRAGE = "convertible_arbitrage"
    FIXED_INCOME = "fixed_income"


@dataclass
class PrivateMarketInvestment:
    """
    Comprehensive private market investment data model for institutional portfolios.
    
    Includes private market-specific metrics like IRR, J-curve effects, illiquidity
    premiums, and valuation methodologies essential for alternative investment analysis.
    """
    
    # Fund Identification
    fund_id: str
    fund_name: str
    fund_manager: str
    strategy: PrivateMarketType
    
    # Investment Details
    vintage_year: int
    fund_size: float
    committed_capital: float
    called_capital: float
    distributed_capital: float
    nav: float
    
    # Performance Metrics
    irr: Optional[float] = None  # Internal rate of return
    tvpi: Optional[float] = None  # Total value to paid-in capital
    dpi: Optional[float] = None  # Distributed to paid-in capital
    rvpi: Optional[float] = None  # Residual value to paid-in capital
    
    # Private Equity Specific
    portfolio_companies: Optional[int] = None
    sector_focus: Optional[List[str]] = None
    geographic_focus: Optional[GeographicFocus] = None
    investment_stage: Optional[InvestmentStage] = None
    
    # Hedge Fund Specific
    management_fee: Optional[float] = None
    performance_fee: Optional[float] = None
    high_water_mark: Optional[bool] = None
    lock_up_period: Optional[int] = None  # months
    redemption_frequency: Optional[str] = None
    
    # Risk and Liquidity
    illiquidity_factor: float = 0.9  # 0.8-1.0 for most private investments
    j_curve_adjustment: float = 0.0  # early-year performance adjustment
    risk_score: float = 5.0  # 1-10 scale
    
    # Valuation
    valuation_method: str = "nav"  # market, income, cost, nav
    last_valuation_date: Optional[datetime] = None
    valuation_frequency: str = "quarterly"  # quarterly, annual
    independent_valuation: bool = False
    
    # Fund Terms
    fund_life: int = 10  # years
    investment_period: int = 5  # years
    extension_options: int = 2  # years of possible extension
    
    # ESG and Impact
    esg_rating: Optional[str] = None
    impact_metrics: Optional[Dict[str, float]] = None
    
    # Data Quality
    last_updated: Optional[datetime] = None
    data_completeness_score: Optional[float] = None
    
    # Status and Lifecycle
    fund_status: FundStatus = FundStatus.INVESTING
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.sector_focus is None:
            self.sector_focus = []
        if self.impact_metrics is None:
            self.impact_metrics = {}
    
    def calculate_tvpi(self) -> float:
        """Calculate Total Value to Paid-In capital ratio"""
        if self.called_capital <= 0:
            return 0.0
        
        total_value = self.distributed_capital + self.nav
        return total_value / self.called_capital
    
    def calculate_dpi(self) -> float:
        """Calculate Distributed to Paid-In capital ratio"""
        if self.called_capital <= 0:
            return 0.0
        
        return self.distributed_capital / self.called_capital
    
    def calculate_rvpi(self) -> float:
        """Calculate Residual Value to Paid-In capital ratio"""
        if self.called_capital <= 0:
            return 0.0
        
        return self.nav / self.called_capital
    
    def estimate_j_curve_impact(self, years_since_vintage: int) -> float:
        """Estimate J-curve impact on returns"""
        if years_since_vintage <= 0:
            return 0.0
        
        # Typical J-curve pattern for PE/VC
        if self.strategy in [PrivateMarketType.PRIVATE_EQUITY, PrivateMarketType.VENTURE_CAPITAL]:
            if years_since_vintage <= 3:
                # Negative cash flows in early years
                return -0.1 * (4 - years_since_vintage)
            elif years_since_vintage <= 7:
                # Positive returns as investments mature
                return 0.05 * (years_since_vintage - 3)
            else:
                # Declining returns in later years - ensure proper decline
                return max(0.0, 0.2 - 0.03 * (years_since_vintage - 7))
        
        return 0.0
    
    def calculate_commitment_pacing(self) -> Dict[str, float]:
        """Calculate capital commitment pacing"""
        if self.fund_size <= 0:
            return {}
        
        total_commitment_rate = self.called_capital / self.fund_size
        remaining_commitment = 1.0 - total_commitment_rate
        
        return {
            "committed_percentage": total_commitment_rate,
            "remaining_commitment": remaining_commitment,
            "call_rate_per_year": total_commitment_rate / max(1, 
                datetime.now().year - self.vintage_year)
        }
    
    def get_fund_maturity_stage(self) -> str:
        """Determine fund maturity stage"""
        current_year = datetime.now().year
        fund_age = current_year - self.vintage_year
        
        if fund_age <= 2:
            return "EARLY"
        elif fund_age <= 6:
            return "MIDDLE"
        elif fund_age <= self.fund_life:
            return "LATE"
        else:
            return "EXTENDED"
    
    def calculate_illiquidity_premium(self) -> float:
        """Calculate expected illiquidity premium"""
        base_premium = 0.03  # 3% base illiquidity premium
        
        # Adjust based on strategy
        strategy_adjustments = {
            PrivateMarketType.VENTURE_CAPITAL: 0.02,  # Higher premium for VC
            PrivateMarketType.DISTRESSED_DEBT: 0.015,
            PrivateMarketType.INFRASTRUCTURE: -0.005,  # Lower for infrastructure
            PrivateMarketType.HEDGE_FUND: -0.02  # Lower for liquid hedge funds
        }
        
        strategy_adjustment = strategy_adjustments.get(self.strategy, 0.0)
        
        # Adjust based on fund size (larger funds may have lower premiums)
        size_adjustment = 0.0
        if self.fund_size > 5000000000:  # $5B+
            size_adjustment = -0.005
        elif self.fund_size < 500000000:  # <$500M
            size_adjustment = 0.01
        
        return base_premium + strategy_adjustment + size_adjustment
    
    def assess_valuation_reliability(self) -> str:
        """Assess reliability of current valuation"""
        score = 0
        
        # Independent valuation
        if self.independent_valuation:
            score += 3
        
        # Recent valuation
        if self.last_valuation_date:
            days_since_valuation = (datetime.now() - self.last_valuation_date).days
            if days_since_valuation <= 90:
                score += 2
            elif days_since_valuation <= 180:
                score += 1
        
        # Valuation method
        if self.valuation_method in ["market", "income"]:
            score += 2
        elif self.valuation_method == "nav":
            score += 1
        
        # Fund maturity
        maturity = self.get_fund_maturity_stage()
        if maturity in ["MIDDLE", "LATE"]:
            score += 1
        
        if score >= 7:
            return "HIGH"
        elif score >= 4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def validate_private_market_data(self) -> List[str]:
        """Validate private market investment data"""
        issues = []
        
        if not self.fund_id or not self.fund_name:
            issues.append("Missing fund identification")
        
        if self.fund_size <= 0:
            issues.append("Invalid fund size")
        
        if self.called_capital > self.fund_size:
            issues.append("Called capital exceeds fund size")
        
        if self.called_capital < 0 or self.distributed_capital < 0:
            issues.append("Negative capital amounts")
        
        if self.vintage_year < 1990 or self.vintage_year > datetime.now().year:
            issues.append("Invalid vintage year")
        
        if self.irr and (self.irr < -1.0 or self.irr > 5.0):
            issues.append("IRR outside reasonable range")
        
        if self.illiquidity_factor < 0 or self.illiquidity_factor > 1:
            issues.append("Illiquidity factor outside valid range")
        
        return issues


@dataclass
class HedgeFundStrategy:
    """
    Hedge fund strategy-specific data model.
    """
    
    strategy_name: str
    strategy_type: str  # long_short, market_neutral, event_driven, etc.
    
    # Performance Metrics
    monthly_returns: List[float]
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk Metrics
    beta_to_market: float
    correlation_to_market: float
    var_95: float  # Value at Risk
    
    # Strategy Specifics
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    
    # Leverage and Risk Management
    leverage_ratio: float
    risk_limit_utilization: float
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if self.max_drawdown <= 0:
            return 0.0
        
        return self.annual_return / abs(self.max_drawdown)
    
    def calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not self.monthly_returns:
            return 0.0
        
        downside_returns = [r for r in self.monthly_returns if r < target_return]
        if not downside_returns:
            return float('inf')
        
        downside_deviation = math.sqrt(sum((r - target_return) ** 2 
                                         for r in downside_returns) / len(downside_returns))
        
        if downside_deviation == 0:
            return float('inf')
        
        excess_return = self.annual_return - target_return
        return excess_return / (downside_deviation * math.sqrt(12))


@dataclass
class PrivateMarketPortfolioMetrics:
    """
    Portfolio-level metrics for private market allocation analysis.
    """
    
    total_private_market_allocation: float
    strategy_diversification: Dict[PrivateMarketType, float]
    vintage_year_diversification: Dict[int, float]
    
    # Performance Metrics
    portfolio_irr: float
    portfolio_tvpi: float
    portfolio_dpi: float
    
    # Risk and Liquidity
    weighted_illiquidity_score: float
    j_curve_exposure: float
    capital_call_risk: float
    
    # Commitment and Funding
    unfunded_commitments: float
    committed_vs_invested_ratio: float
    pacing_model_forecast: Dict[str, float]
    
    def calculate_private_market_diversification(self) -> float:
        """Calculate private market diversification score"""
        strategy_entropy = self._calculate_entropy(self.strategy_diversification)
        vintage_entropy = self._calculate_entropy(self.vintage_year_diversification)
        
        # Weight strategy diversification more heavily
        return 0.7 * strategy_entropy + 0.3 * vintage_entropy
    
    def _calculate_entropy(self, allocation_dict: Dict) -> float:
        """Calculate Shannon entropy for diversification"""
        if not allocation_dict:
            return 0.0
        
        total = sum(allocation_dict.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for weight in allocation_dict.values():
            if weight > 0:
                proportion = weight / total
                entropy -= proportion * math.log2(proportion)
        
        max_entropy = math.log2(len(allocation_dict))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def assess_liquidity_risk(self) -> str:
        """Assess overall private market liquidity risk"""
        if self.weighted_illiquidity_score > 0.9:
            return "VERY_HIGH"
        elif self.weighted_illiquidity_score > 0.8:
            return "HIGH"
        elif self.weighted_illiquidity_score > 0.6:
            return "MODERATE"
        else:
            return "LOW"
    
    def calculate_commitment_pacing_score(self) -> float:
        """Calculate quality of commitment pacing"""
        # Ideal pacing spreads commitments over 3-5 years
        vintage_years = list(self.vintage_year_diversification.keys())
        if not vintage_years:
            return 0.0
        
        year_range = max(vintage_years) - min(vintage_years) + 1
        
        if 3 <= year_range <= 5:
            return 1.0
        elif year_range == 2 or year_range == 6:
            return 0.8
        elif year_range == 1 or year_range == 7:
            return 0.6
        else:
            return 0.4


@dataclass
class InfrastructureInvestment:
    """
    Infrastructure investment data model.
    """
    
    asset_name: str
    infrastructure_type: str  # transportation, energy, utilities, social
    asset_category: str  # brownfield, greenfield, core, core+
    
    # Financial Metrics
    enterprise_value: float
    debt_to_equity_ratio: float
    revenue_multiple: float
    ebitda_multiple: float
    
    # Infrastructure Specifics
    regulatory_environment: str
    concession_period: Optional[int] = None  # years
    inflation_protection: bool = False
    contracted_revenue_percentage: float = 0.0
    
    # ESG and Impact
    environmental_impact_score: float = 0.0
    social_impact_score: float = 0.0
    governance_score: float = 0.0
    
    def calculate_esg_composite_score(self) -> float:
        """Calculate composite ESG score"""
        return (self.environmental_impact_score + 
                self.social_impact_score + 
                self.governance_score) / 3.0
