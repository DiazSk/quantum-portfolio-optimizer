"""
Real Estate Investment Trust (REIT) data models for alternative asset portfolio optimization.

This module provides comprehensive REIT-specific data structures for portfolio
construction, risk management, and performance attribution analysis.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum


class PropertyType(Enum):
    """REIT property sector classifications"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    OFFICE = "office"
    DATA_CENTER = "data_center"
    SELF_STORAGE = "self_storage"
    HOTEL = "hotel"
    MIXED_USE = "mixed_use"


class FundStatus(Enum):
    """REIT fund structure classifications"""
    PUBLIC = "public"
    PRIVATE = "private"
    NON_TRADED = "non_traded"
    LISTED = "listed"


class RedemptionFrequency(Enum):
    """REIT redemption timing options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    NONE = "none"


@dataclass
class REITSecurity:
    """
    Comprehensive REIT security data model for institutional portfolio management.
    
    Includes REIT-specific metrics like FFO, NAV premiums, property fundamentals,
    and liquidity characteristics essential for alternative asset allocation.
    """
    
    # Basic Identification
    symbol: str
    name: str
    isin: str
    exchange: str
    
    # Real Estate Specifics
    property_type: PropertyType
    geographic_focus: str  # US, international, regional focus
    property_locations: List[str]  # primary geographic markets
    
    # Financial Metrics
    market_cap: float
    nav_per_share: float
    market_price: float
    nav_premium_discount: float  # (market_price - nav) / nav
    dividend_yield: float
    funds_from_operations: float  # FFO - REIT-specific earnings metric
    
    # Risk and Liquidity Factors
    illiquidity_factor: float  # 0.0 (liquid) to 1.0 (illiquid)
    average_daily_volume: float
    bid_ask_spread: float
    beta_to_reit_index: float
    correlation_to_equities: float
    
    # Valuation Metrics
    price_to_nav: float
    price_to_ffo: float  # P/FFO ratio
    debt_to_total_capital: float
    occupancy_rate: float
    
    # Alternative Asset Specifics
    vintage_year: Optional[int] = None  # For private REITs
    fund_status: FundStatus = FundStatus.PUBLIC
    redemption_frequency: Optional[RedemptionFrequency] = None
    lock_up_period: Optional[int] = None  # months
    
    # Performance and Risk Data
    annual_return: Optional[float] = None
    annual_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # Property Fundamentals
    cap_rate: Optional[float] = None  # Capitalization rate
    noi_growth_rate: Optional[float] = None  # Net operating income growth
    lease_duration_avg: Optional[int] = None  # Average lease duration in months
    
    # ESG and Sustainability
    green_building_percentage: Optional[float] = None
    esg_score: Optional[float] = None
    
    # Data Quality and Timestamps
    last_updated: Optional[datetime] = None
    data_quality_score: Optional[float] = None
    
    def calculate_nav_premium_discount(self) -> float:
        """Calculate NAV premium/discount percentage"""
        if self.nav_per_share and self.nav_per_share > 0:
            return (self.market_price - self.nav_per_share) / self.nav_per_share
        return 0.0
    
    def calculate_price_to_ffo(self) -> float:
        """Calculate price-to-FFO ratio"""
        if self.funds_from_operations and self.funds_from_operations > 0:
            return self.market_price / self.funds_from_operations
        return 0.0
    
    def get_liquidity_tier(self) -> str:
        """Classify liquidity tier based on trading characteristics"""
        if self.illiquidity_factor <= 0.1 and self.average_daily_volume > 100000:
            return "HIGHLY_LIQUID"
        elif self.illiquidity_factor <= 0.3 and self.average_daily_volume > 10000:
            return "LIQUID"
        elif self.illiquidity_factor <= 0.6:
            return "MODERATELY_LIQUID"
        else:
            return "ILLIQUID"
    
    def get_property_diversification_score(self) -> float:
        """Calculate diversification score based on property locations"""
        if not self.property_locations:
            return 0.0
        
        # Simple diversification score based on number of locations
        location_count = len(self.property_locations)
        max_score = 1.0
        return min(location_count / 10.0, max_score)  # Normalize to 0-1 scale
    
    def validate_data_quality(self) -> List[str]:
        """Validate REIT data quality and return list of issues"""
        issues = []
        
        if not self.symbol or len(self.symbol) < 1:
            issues.append("Missing or invalid symbol")
        
        if self.market_cap <= 0:
            issues.append("Invalid market cap")
        
        if self.dividend_yield < 0 or self.dividend_yield > 1:
            issues.append("Dividend yield outside reasonable range")
        
        if self.illiquidity_factor < 0 or self.illiquidity_factor > 1:
            issues.append("Illiquidity factor outside 0-1 range")
        
        if self.occupancy_rate and (self.occupancy_rate < 0 or self.occupancy_rate > 1):
            issues.append("Occupancy rate outside 0-100% range")
        
        if self.debt_to_total_capital < 0 or self.debt_to_total_capital > 1:
            issues.append("Debt-to-capital ratio outside reasonable range")
        
        return issues


@dataclass
class REITPortfolioMetrics:
    """
    Portfolio-level metrics for REIT allocation analysis and risk management.
    """
    
    total_reit_allocation: float  # Percentage of portfolio in REITs
    property_type_diversification: dict  # Allocation by property type
    geographic_diversification: dict  # Allocation by region
    
    # Risk Metrics
    portfolio_reit_beta: float
    reit_correlation_to_equities: float
    reit_correlation_to_bonds: float
    
    # Performance Metrics
    weighted_ffo_yield: float
    weighted_dividend_yield: float
    weighted_occupancy_rate: float
    
    # Liquidity Analysis
    illiquid_reit_percentage: float
    redemption_constraint_analysis: dict
    
    def calculate_diversification_score(self) -> float:
        """Calculate overall REIT diversification score"""
        property_score = self._calculate_entropy(self.property_type_diversification)
        geographic_score = self._calculate_entropy(self.geographic_diversification)
        
        # Weighted average of diversification dimensions
        return 0.6 * property_score + 0.4 * geographic_score
    
    def _calculate_entropy(self, allocation_dict: dict) -> float:
        """Calculate Shannon entropy for diversification measurement"""
        import math
        
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
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(allocation_dict))
        return entropy / max_entropy if max_entropy > 0 else 0.0
