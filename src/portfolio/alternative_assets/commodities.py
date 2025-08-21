"""
Commodity futures and physical asset data models for alternative asset portfolio optimization.

This module provides comprehensive commodity-specific data structures for futures trading,
physical asset management, and commodities-based portfolio construction and risk analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
import math


class CommodityType(Enum):
    """Primary commodity classifications"""
    ENERGY = "energy"
    PRECIOUS_METALS = "precious_metals"
    BASE_METALS = "base_metals"
    AGRICULTURE = "agriculture"
    LIVESTOCK = "livestock"
    SOFT_COMMODITIES = "soft_commodities"


class CommoditySubcategory(Enum):
    """Detailed commodity subcategories"""
    # Energy
    CRUDE_OIL = "crude_oil"
    NATURAL_GAS = "natural_gas"
    HEATING_OIL = "heating_oil"
    GASOLINE = "gasoline"
    
    # Precious Metals
    GOLD = "gold"
    SILVER = "silver"
    PLATINUM = "platinum"
    PALLADIUM = "palladium"
    
    # Base Metals
    COPPER = "copper"
    ALUMINUM = "aluminum"
    ZINC = "zinc"
    NICKEL = "nickel"
    LEAD = "lead"
    
    # Agriculture
    WHEAT = "wheat"
    CORN = "corn"
    SOYBEANS = "soybeans"
    RICE = "rice"
    COFFEE = "coffee"
    SUGAR = "sugar"
    COTTON = "cotton"


class Exchange(Enum):
    """Major commodity exchanges"""
    NYMEX = "NYMEX"  # New York Mercantile Exchange
    COMEX = "COMEX"  # Commodity Exchange
    CBOT = "CBOT"    # Chicago Board of Trade
    CME = "CME"      # Chicago Mercantile Exchange
    ICE = "ICE"      # Intercontinental Exchange
    LME = "LME"      # London Metal Exchange


@dataclass
class CommodityFuture:
    """
    Comprehensive commodity futures contract data model for institutional trading.
    
    Includes futures-specific metrics like basis, open interest, convenience yield,
    and storage costs essential for commodity investment strategies.
    """
    
    # Contract Identification
    symbol: str
    commodity_name: str
    exchange: Exchange
    contract_month: str
    expiration_date: datetime
    
    # Commodity Classification
    commodity_type: CommodityType
    subcategory: CommoditySubcategory
    underlying_asset: str
    
    # Contract Specifications
    contract_size: float  # units per contract
    price_unit: str  # $/barrel, $/ounce, $/bushel
    minimum_tick: float
    tick_value: float
    
    # Physical Commodity Factors
    storage_cost: float  # annual storage cost as % of value
    convenience_yield: float  # benefit of physical ownership
    seasonal_factor: float  # seasonal price variation coefficient
    
    # Market Data
    spot_price: float
    futures_price: float
    basis: float  # futures - spot price
    open_interest: int
    volume: int
    
    # Risk Metrics
    volatility_30d: float
    correlation_to_dollar: float
    correlation_to_equities: float
    beta_to_commodity_index: float
    
    # Supply/Demand Fundamentals
    global_production: Optional[float] = None
    global_consumption: Optional[float] = None
    inventory_levels: Optional[float] = None
    geopolitical_risk_score: float = 0.0  # 0-10 scale
    
    # Technical Analysis
    days_to_expiration: Optional[int] = None
    contango_backwardation: Optional[str] = None  # "contango", "backwardation", "flat"
    
    # Environmental and ESG
    carbon_footprint: Optional[float] = None
    sustainability_score: Optional[float] = None
    
    # Data Quality
    last_updated: Optional[datetime] = None
    data_source: Optional[str] = None
    
    def calculate_basis(self) -> float:
        """Calculate basis (futures - spot price)"""
        return self.futures_price - self.spot_price
    
    def calculate_days_to_expiration(self, current_date: datetime = None) -> int:
        """Calculate days until contract expiration"""
        if current_date is None:
            current_date = datetime.now()
        
        delta = self.expiration_date - current_date
        return max(0, delta.days)
    
    def determine_curve_structure(self) -> str:
        """Determine if futures curve is in contango or backwardation"""
        if self.basis > 0:
            return "contango"
        elif self.basis < 0:
            return "backwardation"
        else:
            return "flat"
    
    def calculate_annualized_storage_cost(self) -> float:
        """Calculate annualized storage cost"""
        return self.storage_cost * self.spot_price
    
    def get_seasonal_adjustment(self, month: int) -> float:
        """Get seasonal price adjustment factor for given month"""
        # Simplified seasonal patterns - would be more sophisticated in production
        seasonal_patterns = {
            CommoditySubcategory.NATURAL_GAS: {
                1: 1.15, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.90, 6: 0.85,
                7: 0.85, 8: 0.90, 9: 0.95, 10: 1.05, 11: 1.10, 12: 1.15
            },
            CommoditySubcategory.WHEAT: {
                1: 1.02, 2: 1.01, 3: 1.00, 4: 0.98, 5: 0.96, 6: 0.95,
                7: 0.98, 8: 1.02, 9: 1.05, 10: 1.03, 11: 1.02, 12: 1.02
            }
        }
        
        pattern = seasonal_patterns.get(self.subcategory)
        if pattern and month in pattern:
            return pattern[month]
        return 1.0  # No seasonal adjustment
    
    def calculate_roll_yield(self, next_contract_price: float) -> float:
        """Calculate roll yield between current and next contract"""
        if next_contract_price <= 0:
            return 0.0
        
        return (next_contract_price - self.futures_price) / self.futures_price
    
    def validate_contract_data(self) -> List[str]:
        """Validate commodity futures data quality"""
        issues = []
        
        if not self.symbol:
            issues.append("Missing contract symbol")
        
        if self.contract_size <= 0:
            issues.append("Invalid contract size")
        
        if self.futures_price <= 0 or self.spot_price <= 0:
            issues.append("Invalid price data")
        
        if self.expiration_date <= datetime.now():
            issues.append("Contract already expired")
        
        if self.volatility_30d < 0 or self.volatility_30d > 5:
            issues.append("Volatility outside reasonable range")
        
        if self.geopolitical_risk_score < 0 or self.geopolitical_risk_score > 10:
            issues.append("Geopolitical risk score outside valid range")
        
        return issues


@dataclass
class PhysicalCommodityPosition:
    """
    Physical commodity holding data model for direct commodity investments.
    """
    
    commodity: CommoditySubcategory
    quantity: float
    unit: str  # barrels, ounces, tons, etc.
    
    # Storage and Logistics
    storage_location: str
    storage_facility: str
    storage_cost_per_period: float
    insurance_cost: float
    
    # Quality and Certification
    quality_grade: str
    
    # Valuation
    acquisition_price: float
    current_market_value: float
    acquisition_date: datetime
    
    # Risk Factors
    spoilage_risk: float  # 0-1 scale
    theft_risk: float
    weather_exposure: float
    transportation_cost: float
    
    # Optional fields with defaults
    certification: Optional[str] = None  # organic, fair trade, etc.
    assay_report: Optional[str] = None
    
    def calculate_total_carrying_cost(self, holding_period_months: int) -> float:
        """Calculate total carrying costs for holding period"""
        monthly_storage = self.storage_cost_per_period / 12
        monthly_insurance = self.insurance_cost / 12
        
        return (monthly_storage + monthly_insurance) * holding_period_months
    
    def calculate_holding_period_return(self) -> float:
        """Calculate holding period return excluding carrying costs"""
        if self.acquisition_price <= 0:
            return 0.0
        
        return (self.current_market_value - self.acquisition_price) / self.acquisition_price


@dataclass
class CommodityPortfolioMetrics:
    """
    Portfolio-level metrics for commodity allocation analysis.
    """
    
    total_commodity_allocation: float  # Percentage of portfolio
    commodity_type_diversification: Dict[CommodityType, float]
    futures_vs_physical_allocation: Dict[str, float]
    
    # Risk Metrics
    commodity_portfolio_volatility: float
    correlation_to_inflation: float
    correlation_to_dollar_index: float
    
    # Futures-Specific Metrics
    average_roll_yield: float
    contango_exposure: float  # Percentage in contango markets
    backwardation_exposure: float
    
    # Physical Commodity Metrics
    storage_cost_burden: float  # Annual storage costs as % of value
    liquidity_constraint_score: float
    
    def calculate_commodity_diversification(self) -> float:
        """Calculate commodity type diversification score"""
        if not self.commodity_type_diversification:
            return 0.0
        
        # Shannon entropy calculation
        total = sum(self.commodity_type_diversification.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for weight in self.commodity_type_diversification.values():
            if weight > 0:
                proportion = weight / total
                entropy -= proportion * math.log2(proportion)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(self.commodity_type_diversification))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def get_inflation_hedge_effectiveness(self) -> str:
        """Assess commodity portfolio as inflation hedge"""
        if self.correlation_to_inflation > 0.7:
            return "EXCELLENT"
        elif self.correlation_to_inflation > 0.5:
            return "GOOD"
        elif self.correlation_to_inflation > 0.3:
            return "MODERATE"
        else:
            return "POOR"


@dataclass
class CommoditySupplyDemandData:
    """
    Fundamental supply and demand data for commodity analysis.
    """
    
    commodity: CommoditySubcategory
    
    # Supply Data
    global_production: float
    major_producers: Dict[str, float]  # Country/region -> production
    production_capacity: float
    capacity_utilization: float
    
    # Demand Data
    global_consumption: float
    major_consumers: Dict[str, float]  # Country/region -> consumption
    demand_growth_rate: float
    
    # Inventory Data
    global_inventory: float
    strategic_reserves: float
    commercial_inventory: float
    inventory_to_consumption_ratio: float
    
    # Market Structure
    market_concentration_hhi: float  # Herfindahl-Hirschman Index
    cartel_influence: bool
    trade_restrictions: List[str]
    
    def calculate_supply_demand_balance(self) -> float:
        """Calculate supply-demand balance ratio"""
        if self.global_consumption <= 0:
            return float('inf')
        
        return self.global_production / self.global_consumption
    
    def get_inventory_adequacy(self) -> str:
        """Assess inventory level adequacy"""
        ratio = self.inventory_to_consumption_ratio
        
        if ratio > 0.3:
            return "HIGH"
        elif ratio > 0.15:
            return "ADEQUATE"
        elif ratio > 0.05:
            return "LOW"
        else:
            return "CRITICAL"
