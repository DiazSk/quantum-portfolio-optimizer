"""
Commodity Market Analysis and Integration Module

Provides comprehensive commodity market analysis including:
- Physical commodity pricing and storage cost modeling
- Agricultural commodity analysis with seasonal patterns
- Energy commodity integration (oil, gas, renewables)
- Precious metals and industrial metals analysis
- Commodity correlation analysis with inflation and currency factors

Business Value:
- Comprehensive commodity market coverage
- Inflation hedging strategy capabilities
- Natural resource investment modeling
- Advanced commodity risk analytics
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.portfolio.alternative_assets.commodities import (
    CommodityFuture, CommodityType, CommoditySubcategory, Exchange
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeasonalPattern(Enum):
    """Seasonal price patterns for different commodities"""
    AGRICULTURAL_HARVEST = "agricultural_harvest"  # Peak harvest season
    ENERGY_WINTER = "energy_winter"  # Winter heating demand
    ENERGY_SUMMER = "energy_summer"  # Summer cooling demand
    NO_PATTERN = "no_pattern"  # Minimal seasonal variation


@dataclass
class CommoditySupplyDemandData:
    """Supply and demand fundamentals for commodity analysis"""
    
    commodity: CommoditySubcategory
    
    # Supply Metrics
    global_production: float  # Annual production in metric tons/barrels
    major_producers: Dict[str, float]  # Country/region production shares
    reserve_levels: float  # Proven reserves
    capacity_utilization: float  # Production capacity utilization (0-1)
    
    # Demand Metrics
    global_consumption: float  # Annual consumption
    major_consumers: Dict[str, float]  # Country/region consumption shares
    industrial_demand: float  # Industrial usage percentage
    investment_demand: float  # Investment/financial demand percentage
    
    # Inventory and Flow
    inventory_levels: float  # Current inventory levels
    days_of_supply: float  # Inventory as days of consumption
    inventory_to_usage_ratio: float  # Strategic inventory ratios
    
    # Market Structure
    market_concentration: float  # HHI index for market concentration
    geopolitical_risk_score: float  # 0-1 scale geopolitical risk
    substitution_risk: float  # Risk of commodity substitution


class CommodityMarketAnalyzer:
    """
    Advanced commodity market analysis and modeling system
    
    Provides comprehensive analysis of physical commodities, futures markets,
    and commodity investment strategies with inflation hedging capabilities.
    """
    
    def __init__(self):
        """Initialize commodity market analyzer"""
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        self.storage_costs = self._initialize_storage_costs()
        self.correlation_matrix = None
        
        logger.info("Commodity market analyzer initialized")
    
    def _initialize_seasonal_patterns(self) -> Dict[CommoditySubcategory, SeasonalPattern]:
        """Initialize seasonal patterns for different commodities"""
        return {
            # Energy - Winter heating demand
            CommoditySubcategory.NATURAL_GAS: SeasonalPattern.ENERGY_WINTER,
            CommoditySubcategory.HEATING_OIL: SeasonalPattern.ENERGY_WINTER,
            
            # Energy - Summer driving season
            CommoditySubcategory.GASOLINE: SeasonalPattern.ENERGY_SUMMER,
            
            # Agriculture - Harvest cycles
            CommoditySubcategory.CORN: SeasonalPattern.AGRICULTURAL_HARVEST,
            CommoditySubcategory.WHEAT: SeasonalPattern.AGRICULTURAL_HARVEST,
            CommoditySubcategory.SOYBEANS: SeasonalPattern.AGRICULTURAL_HARVEST,
            CommoditySubcategory.COTTON: SeasonalPattern.AGRICULTURAL_HARVEST,
            
            # Metals - Less seasonal
            CommoditySubcategory.GOLD: SeasonalPattern.NO_PATTERN,
            CommoditySubcategory.SILVER: SeasonalPattern.NO_PATTERN,
            CommoditySubcategory.COPPER: SeasonalPattern.NO_PATTERN,
            CommoditySubcategory.CRUDE_OIL: SeasonalPattern.NO_PATTERN,
        }
    
    def _initialize_storage_costs(self) -> Dict[CommoditySubcategory, float]:
        """Initialize annual storage costs as percentage of commodity value"""
        return {
            # Energy - High storage costs
            CommoditySubcategory.CRUDE_OIL: 0.05,  # 5% annual storage cost
            CommoditySubcategory.NATURAL_GAS: 0.08,  # Higher for gas storage
            CommoditySubcategory.GASOLINE: 0.06,
            CommoditySubcategory.HEATING_OIL: 0.05,
            
            # Precious Metals - Low storage costs
            CommoditySubcategory.GOLD: 0.01,  # 1% for secure storage
            CommoditySubcategory.SILVER: 0.015,
            CommoditySubcategory.PLATINUM: 0.012,
            CommoditySubcategory.PALLADIUM: 0.015,
            
            # Base Metals - Moderate storage costs
            CommoditySubcategory.COPPER: 0.03,
            CommoditySubcategory.ALUMINUM: 0.025,
            CommoditySubcategory.ZINC: 0.03,
            CommoditySubcategory.NICKEL: 0.035,
            
            # Agriculture - Variable storage costs
            CommoditySubcategory.WHEAT: 0.12,  # 12% due to spoilage risk
            CommoditySubcategory.CORN: 0.10,
            CommoditySubcategory.SOYBEANS: 0.11,
            CommoditySubcategory.RICE: 0.09,
            CommoditySubcategory.COFFEE: 0.08,
            CommoditySubcategory.SUGAR: 0.07,
            CommoditySubcategory.COTTON: 0.06,
        }
    
    def analyze_commodity_seasonality(self, 
                                    commodity: CommoditySubcategory,
                                    price_history: pd.Series) -> Dict[str, float]:
        """
        Analyze seasonal price patterns for a commodity
        
        Args:
            commodity: Commodity to analyze
            price_history: Historical price data with datetime index
            
        Returns:
            Dict with seasonal analysis results
        """
        if len(price_history) < 24:  # Need at least 2 years
            return {'seasonal_strength': 0.0, 'peak_months': [], 'trough_months': []}
        
        # Calculate monthly returns
        monthly_data = price_history.resample('M').last().pct_change().dropna()
        
        # Group by month
        monthly_averages = monthly_data.groupby(monthly_data.index.month).mean()
        monthly_std = monthly_data.groupby(monthly_data.index.month).std()
        
        # Calculate seasonal strength
        seasonal_strength = monthly_std.mean() / abs(monthly_averages.mean()) if monthly_averages.mean() != 0 else 0
        
        # Identify peak and trough months
        peak_months = monthly_averages.nlargest(3).index.tolist()
        trough_months = monthly_averages.nsmallest(3).index.tolist()
        
        # Expected pattern based on commodity type
        expected_pattern = self.seasonal_patterns.get(commodity, SeasonalPattern.NO_PATTERN)
        pattern_consistency = self._validate_seasonal_pattern(
            monthly_averages, expected_pattern
        )
        
        return {
            'seasonal_strength': float(seasonal_strength),
            'peak_months': peak_months,
            'trough_months': trough_months,
            'monthly_averages': monthly_averages.to_dict(),
            'expected_pattern': expected_pattern.value,
            'pattern_consistency': pattern_consistency
        }
    
    def _validate_seasonal_pattern(self, 
                                 monthly_averages: pd.Series, 
                                 expected_pattern: SeasonalPattern) -> float:
        """Validate if observed pattern matches expected seasonal pattern"""
        
        if expected_pattern == SeasonalPattern.NO_PATTERN:
            # For no pattern, consistency is inverse of variation
            return 1.0 - (monthly_averages.std() / abs(monthly_averages.mean()) 
                         if monthly_averages.mean() != 0 else 1.0)
        
        elif expected_pattern == SeasonalPattern.ENERGY_WINTER:
            # Expect higher prices in winter months (Nov-Mar)
            winter_months = [11, 12, 1, 2, 3]
            winter_avg = monthly_averages[winter_months].mean()
            summer_avg = monthly_averages[[6, 7, 8]].mean()
            return max(0.0, (winter_avg - summer_avg) / abs(winter_avg) if winter_avg != 0 else 0)
        
        elif expected_pattern == SeasonalPattern.ENERGY_SUMMER:
            # Expect higher prices in summer months (Jun-Aug)
            summer_months = [6, 7, 8]
            summer_avg = monthly_averages[summer_months].mean()
            winter_avg = monthly_averages[[12, 1, 2]].mean()
            return max(0.0, (summer_avg - winter_avg) / abs(summer_avg) if summer_avg != 0 else 0)
        
        elif expected_pattern == SeasonalPattern.AGRICULTURAL_HARVEST:
            # Expect lower prices during harvest (Sep-Nov)
            harvest_months = [9, 10, 11]
            harvest_avg = monthly_averages[harvest_months].mean()
            pre_harvest_avg = monthly_averages[[6, 7, 8]].mean()
            return max(0.0, (pre_harvest_avg - harvest_avg) / abs(pre_harvest_avg) 
                      if pre_harvest_avg != 0 else 0)
        
        return 0.0
    
    def calculate_convenience_yield(self, 
                                  spot_price: float,
                                  futures_price: float,
                                  time_to_expiry: float,
                                  storage_cost: float,
                                  risk_free_rate: float = 0.02) -> float:
        """
        Calculate convenience yield for physical commodity
        
        Convenience yield = r + storage_cost - (ln(F/S) / T)
        where F = futures price, S = spot price, T = time to expiry
        """
        if spot_price <= 0 or time_to_expiry <= 0:
            return 0.0
        
        try:
            implied_rate = np.log(futures_price / spot_price) / time_to_expiry
            convenience_yield = risk_free_rate + storage_cost - implied_rate
            return max(-0.5, min(0.5, convenience_yield))  # Cap at reasonable bounds
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def analyze_commodity_curve_structure(self, 
                                        futures_contracts: List[CommodityFuture]) -> Dict[str, float]:
        """
        Analyze futures curve structure (contango vs backwardation)
        
        Args:
            futures_contracts: List of futures contracts for same commodity, different expiries
            
        Returns:
            Dict with curve analysis results
        """
        if len(futures_contracts) < 2:
            return {'curve_slope': 0.0, 'structure': 'insufficient_data'}
        
        # Sort by expiration date
        contracts = sorted(futures_contracts, key=lambda x: x.expiration_date)
        
        # Calculate slopes between consecutive contracts
        slopes = []
        for i in range(1, len(contracts)):
            prev_contract = contracts[i-1]
            curr_contract = contracts[i]
            
            days_diff = (curr_contract.expiration_date - prev_contract.expiration_date).days
            if days_diff > 0:
                price_diff = curr_contract.futures_price - prev_contract.futures_price
                slope = price_diff / prev_contract.futures_price * (365 / days_diff)  # Annualized
                slopes.append(slope)
        
        if not slopes:
            return {'curve_slope': 0.0, 'structure': 'insufficient_data'}
        
        avg_slope = np.mean(slopes)
        slope_consistency = 1.0 - np.std(slopes) / (abs(avg_slope) + 0.01)  # Consistency measure
        
        # Classify curve structure
        if avg_slope > 0.05:  # 5% annualized upward slope
            structure = 'contango'
        elif avg_slope < -0.05:  # 5% annualized downward slope
            structure = 'backwardation'
        else:
            structure = 'flat'
        
        return {
            'curve_slope': float(avg_slope),
            'slope_consistency': float(slope_consistency),
            'structure': structure,
            'contracts_analyzed': len(contracts),
            'individual_slopes': slopes
        }
    
    def calculate_commodity_inflation_correlation(self,
                                                commodity_returns: pd.Series,
                                                inflation_data: pd.Series,
                                                rolling_window: int = 252) -> Dict[str, float]:
        """
        Calculate correlation between commodity and inflation
        
        Args:
            commodity_returns: Daily commodity returns
            inflation_data: Daily inflation data (or proxy like TIPS spreads)
            rolling_window: Window for rolling correlation calculation
            
        Returns:
            Dict with correlation analysis
        """
        if len(commodity_returns) < rolling_window or len(inflation_data) < rolling_window:
            return {'correlation': 0.0, 'correlation_stability': 0.0}
        
        # Align data
        aligned_data = pd.concat([commodity_returns, inflation_data], axis=1).dropna()
        if len(aligned_data) < rolling_window:
            return {'correlation': 0.0, 'correlation_stability': 0.0}
        
        commodity_col = aligned_data.columns[0]
        inflation_col = aligned_data.columns[1]
        
        # Calculate overall correlation
        overall_correlation = aligned_data[commodity_col].corr(aligned_data[inflation_col])
        
        # Calculate rolling correlation for stability analysis
        rolling_corr = aligned_data[commodity_col].rolling(rolling_window).corr(
            aligned_data[inflation_col]
        ).dropna()
        
        correlation_stability = 1.0 - rolling_corr.std() if len(rolling_corr) > 0 else 0.0
        
        # Calculate correlation in different market regimes
        high_inflation_periods = aligned_data[aligned_data[inflation_col] > 
                                           aligned_data[inflation_col].quantile(0.75)]
        low_inflation_periods = aligned_data[aligned_data[inflation_col] < 
                                          aligned_data[inflation_col].quantile(0.25)]
        
        high_inflation_corr = (high_inflation_periods[commodity_col]
                              .corr(high_inflation_periods[inflation_col]) 
                              if len(high_inflation_periods) > 10 else np.nan)
        
        low_inflation_corr = (low_inflation_periods[commodity_col]
                             .corr(low_inflation_periods[inflation_col]) 
                             if len(low_inflation_periods) > 10 else np.nan)
        
        return {
            'correlation': float(overall_correlation) if not np.isnan(overall_correlation) else 0.0,
            'correlation_stability': float(correlation_stability),
            'high_inflation_correlation': float(high_inflation_corr) if not np.isnan(high_inflation_corr) else None,
            'low_inflation_correlation': float(low_inflation_corr) if not np.isnan(low_inflation_corr) else None,
            'rolling_correlation_mean': float(rolling_corr.mean()) if len(rolling_corr) > 0 else 0.0,
            'rolling_correlation_std': float(rolling_corr.std()) if len(rolling_corr) > 0 else 0.0
        }
    
    def calculate_commodity_dollar_sensitivity(self,
                                             commodity_returns: pd.Series,
                                             dollar_index_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate sensitivity of commodity to US Dollar movements
        
        Most commodities are priced in USD and have negative correlation with USD strength
        """
        if len(commodity_returns) < 30 or len(dollar_index_returns) < 30:
            return {'dollar_beta': 0.0, 'dollar_correlation': 0.0}
        
        # Align data
        aligned_data = pd.concat([commodity_returns, dollar_index_returns], axis=1).dropna()
        if len(aligned_data) < 30:
            return {'dollar_beta': 0.0, 'dollar_correlation': 0.0}
        
        commodity_col = aligned_data.columns[0]
        dollar_col = aligned_data.columns[1]
        
        # Calculate correlation
        correlation = aligned_data[commodity_col].corr(aligned_data[dollar_col])
        
        # Calculate beta (sensitivity)
        covariance = aligned_data[commodity_col].cov(aligned_data[dollar_col])
        dollar_variance = aligned_data[dollar_col].var()
        
        beta = covariance / dollar_variance if dollar_variance > 0 else 0.0
        
        return {
            'dollar_beta': float(beta),
            'dollar_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'sensitivity_interpretation': self._interpret_dollar_sensitivity(beta)
        }
    
    def _interpret_dollar_sensitivity(self, beta: float) -> str:
        """Interpret dollar sensitivity beta coefficient"""
        if beta < -1.5:
            return 'highly_dollar_sensitive_negative'
        elif beta < -0.5:
            return 'moderately_dollar_sensitive_negative'
        elif beta < 0.5:
            return 'low_dollar_sensitivity'
        elif beta < 1.5:
            return 'moderately_dollar_sensitive_positive'
        else:
            return 'highly_dollar_sensitive_positive'
    
    def build_commodity_correlation_matrix(self, 
                                         commodity_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Build correlation matrix for multiple commodities
        
        Args:
            commodity_data: Dict mapping commodity names to return series
            
        Returns:
            Correlation matrix as DataFrame
        """
        # Combine all commodity data
        combined_data = pd.concat(commodity_data, axis=1).dropna()
        
        if combined_data.empty:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = combined_data.corr()
        
        # Store for later use
        self.correlation_matrix = correlation_matrix
        
        return correlation_matrix
    
    def analyze_commodity_diversification_benefit(self, 
                                                 correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze diversification benefits within commodity portfolio
        
        Returns metrics showing how well commodities diversify each other
        """
        if correlation_matrix.empty:
            return {'diversification_ratio': 0.0, 'average_correlation': 1.0}
        
        # Calculate average pairwise correlation
        mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
        upper_triangle = correlation_matrix.where(mask)
        avg_correlation = upper_triangle.stack().mean()
        
        # Calculate diversification ratio
        # Higher correlation = lower diversification benefit
        diversification_ratio = 1.0 - avg_correlation if not np.isnan(avg_correlation) else 0.0
        
        # Identify most and least correlated pairs
        upper_triangle_stacked = upper_triangle.stack()
        highest_corr_pair = upper_triangle_stacked.idxmax()
        lowest_corr_pair = upper_triangle_stacked.idxmin()
        
        return {
            'diversification_ratio': float(diversification_ratio),
            'average_correlation': float(avg_correlation) if not np.isnan(avg_correlation) else 1.0,
            'highest_correlation': float(upper_triangle_stacked.max()),
            'lowest_correlation': float(upper_triangle_stacked.min()),
            'highest_corr_pair': highest_corr_pair,
            'lowest_corr_pair': lowest_corr_pair,
            'num_commodities': len(correlation_matrix)
        }
    
    def generate_commodity_market_summary(self) -> Dict[str, any]:
        """Generate comprehensive commodity market analysis summary"""
        return {
            'analyzer_status': 'active',
            'supported_commodities': len(self.storage_costs),
            'seasonal_patterns_configured': len(self.seasonal_patterns),
            'correlation_matrix_available': self.correlation_matrix is not None,
            'analysis_capabilities': [
                'seasonal_analysis',
                'convenience_yield_calculation',
                'curve_structure_analysis',
                'inflation_correlation',
                'dollar_sensitivity',
                'diversification_analysis'
            ],
            'last_updated': datetime.now().isoformat()
        }


# Demo usage function
def demo_commodity_analysis():
    """Demonstrate commodity market analysis capabilities"""
    analyzer = CommodityMarketAnalyzer()
    
    print("Commodity Market Analyzer Demo")
    print("=" * 50)
    
    # Generate sample price data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Sample commodity returns with seasonal pattern
    base_returns = np.random.normal(0, 0.02, len(dates))
    seasonal_component = 0.1 * np.sin(2 * np.pi * dates.dayofyear / 365)
    commodity_returns = pd.Series(base_returns + seasonal_component * 0.01, index=dates)
    
    # Analyze seasonality
    seasonality = analyzer.analyze_commodity_seasonality(
        CommoditySubcategory.NATURAL_GAS, 
        (1 + commodity_returns).cumprod() * 100
    )
    
    print(f"Seasonal Analysis for Natural Gas:")
    print(f"- Seasonal Strength: {seasonality['seasonal_strength']:.3f}")
    print(f"- Peak Months: {seasonality['peak_months']}")
    print(f"- Pattern Consistency: {seasonality['pattern_consistency']:.3f}")
    
    # Sample correlation analysis
    inflation_data = pd.Series(np.random.normal(0, 0.005, len(dates)), index=dates)
    inflation_corr = analyzer.calculate_commodity_inflation_correlation(
        commodity_returns, inflation_data
    )
    
    print(f"\nInflation Correlation Analysis:")
    print(f"- Correlation: {inflation_corr['correlation']:.3f}")
    print(f"- Correlation Stability: {inflation_corr['correlation_stability']:.3f}")
    
    # Summary
    summary = analyzer.generate_commodity_market_summary()
    print(f"\nAnalyzer Summary:")
    print(f"- Supported Commodities: {summary['supported_commodities']}")
    print(f"- Analysis Capabilities: {len(summary['analysis_capabilities'])}")


if __name__ == "__main__":
    demo_commodity_analysis()
