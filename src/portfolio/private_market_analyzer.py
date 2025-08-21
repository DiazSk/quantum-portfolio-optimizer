"""
Private Markets and Hedge Fund Modeling Module

Comprehensive private market investment modeling including:
- Private equity fund modeling with J-curve effects
- Venture capital investment analysis with stage-based returns
- Hedge fund strategies with alternative risk-return profiles
- Illiquidity adjustment models for private market valuations
- Advanced portfolio allocation and commitment pacing

Business Value:
- Access to $13T+ private markets industry
- Sophisticated institutional investment strategies
- Alternative investment portfolio optimization
- Risk-adjusted return enhancement through illiquid assets
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import math

from src.portfolio.alternative_assets.private_markets import (
    PrivateMarketInvestment, PrivateMarketType, InvestmentStage, 
    FundStatus, HedgeFundStrategy, PrivateMarketPortfolioMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivateMarketValuationMethod(Enum):
    """Valuation methodologies for private market assets"""
    MARKET_MULTIPLE = "market_multiple"
    DISCOUNTED_CASH_FLOW = "discounted_cash_flow"
    NET_ASSET_VALUE = "net_asset_value"
    COMPARABLE_TRANSACTIONS = "comparable_transactions"
    OPTION_PRICING = "option_pricing"  # For VC investments


@dataclass
class PrivateMarketCommitment:
    """Private market commitment and capital call modeling"""
    
    fund_id: str
    commitment_amount: float
    commitment_date: datetime
    
    # Capital Call Schedule
    total_called: float
    total_distributed: float
    remaining_commitment: float
    
    # Pacing Model Parameters
    expected_call_schedule: List[Tuple[datetime, float]]  # (date, amount)
    call_uncertainty: float  # Standard deviation of call timing
    
    # Performance Tracking
    called_to_date_irr: float
    distributed_to_paid_in: float  # DPI ratio
    total_value_to_paid_in: float  # TVPI ratio


class PrivateMarketAnalyzer:
    """
    Advanced private market investment analysis and modeling system
    
    Provides comprehensive analysis of private equity, venture capital,
    and hedge fund investments with sophisticated risk and return modeling.
    """
    
    def __init__(self):
        """Initialize private market analyzer"""
        self.j_curve_models = self._initialize_j_curve_models()
        self.stage_return_models = self._initialize_stage_return_models()
        self.hedge_fund_strategies = self._initialize_hedge_fund_strategies()
        
        logger.info("Private market analyzer initialized")
    
    def _initialize_j_curve_models(self) -> Dict[PrivateMarketType, Dict]:
        """Initialize J-curve models for different private market strategies"""
        return {
            PrivateMarketType.PRIVATE_EQUITY: {
                'investment_period': 3,  # Years for initial investments
                'peak_negative_impact': -0.15,  # Maximum negative cash flow impact
                'maturity_period': 7,  # Years to maturity
                'peak_positive_impact': 0.20,  # Maximum positive return
                'decline_rate': 0.03  # Annual decline after peak
            },
            PrivateMarketType.VENTURE_CAPITAL: {
                'investment_period': 4,  # Longer investment period for VC
                'peak_negative_impact': -0.20,  # Higher early negative impact
                'maturity_period': 8,  # Longer maturity for VC
                'peak_positive_impact': 0.35,  # Higher potential returns
                'decline_rate': 0.05  # Faster decline due to concentration
            },
            PrivateMarketType.HEDGE_FUND: {
                'investment_period': 2,  # Shorter investment period
                'peak_negative_impact': -0.08,  # Lower negative impact
                'maturity_period': 5,  # Shorter holding periods
                'peak_positive_impact': 0.15,  # Moderate returns
                'decline_rate': 0.02  # Gradual decline
            },
            PrivateMarketType.DISTRESSED_DEBT: {
                'investment_period': 1,  # Quick deployment
                'peak_negative_impact': -0.05,  # Minimal negative impact
                'maturity_period': 3,  # Short holding periods
                'peak_positive_impact': 0.12,  # Steady returns
                'decline_rate': 0.01  # Minimal decline
            }
        }
    
    def _initialize_stage_return_models(self) -> Dict[InvestmentStage, Dict]:
        """Initialize return models for different VC investment stages"""
        return {
            InvestmentStage.SEED: {
                'expected_return': 0.45,  # 45% expected annual return
                'volatility': 0.80,  # 80% volatility
                'success_rate': 0.10,  # 10% success rate
                'failure_rate': 0.70,  # 70% total loss rate
                'holding_period': 8  # Years to exit
            },
            InvestmentStage.SERIES_A: {
                'expected_return': 0.35,  # 35% expected annual return
                'volatility': 0.65,  # 65% volatility
                'success_rate': 0.15,  # 15% success rate
                'failure_rate': 0.50,  # 50% total loss rate
                'holding_period': 6  # Years to exit
            },
            InvestmentStage.SERIES_B: {
                'expected_return': 0.25,  # 25% expected annual return
                'volatility': 0.50,  # 50% volatility
                'success_rate': 0.25,  # 25% success rate
                'failure_rate': 0.30,  # 30% total loss rate
                'holding_period': 5  # Years to exit
            },
            InvestmentStage.GROWTH: {
                'expected_return': 0.18,  # 18% expected annual return
                'volatility': 0.35,  # 35% volatility
                'success_rate': 0.40,  # 40% success rate
                'failure_rate': 0.15,  # 15% total loss rate
                'holding_period': 4  # Years to exit
            },
            InvestmentStage.BUYOUT: {
                'expected_return': 0.12,  # 12% expected annual return
                'volatility': 0.25,  # 25% volatility
                'success_rate': 0.60,  # 60% success rate
                'failure_rate': 0.05,  # 5% total loss rate
                'holding_period': 5  # Years to exit
            }
        }
    
    def _initialize_hedge_fund_strategies(self) -> Dict[str, Dict]:
        """Initialize hedge fund strategy characteristics"""
        return {
            'long_short_equity': {
                'expected_return': 0.08,
                'volatility': 0.12,
                'market_correlation': 0.6,
                'max_leverage': 3.0,
                'liquidity': 'monthly'
            },
            'market_neutral': {
                'expected_return': 0.05,
                'volatility': 0.05,
                'market_correlation': 0.1,
                'max_leverage': 5.0,
                'liquidity': 'monthly'
            },
            'event_driven': {
                'expected_return': 0.10,
                'volatility': 0.08,
                'market_correlation': 0.4,
                'max_leverage': 2.0,
                'liquidity': 'quarterly'
            },
            'global_macro': {
                'expected_return': 0.09,
                'volatility': 0.15,
                'market_correlation': 0.2,
                'max_leverage': 4.0,
                'liquidity': 'monthly'
            },
            'relative_value': {
                'expected_return': 0.06,
                'volatility': 0.04,
                'market_correlation': 0.3,
                'max_leverage': 8.0,
                'liquidity': 'quarterly'
            }
        }
    
    def model_j_curve_cash_flows(self, 
                                fund: PrivateMarketInvestment,
                                projection_years: int = 12) -> Dict[str, List[float]]:
        """
        Model J-curve cash flow pattern for private market fund
        
        Args:
            fund: Private market fund investment
            projection_years: Number of years to project
            
        Returns:
            Dict with projected cash flows, NAV, and cumulative returns
        """
        if fund.strategy not in self.j_curve_models:
            logger.warning(f"No J-curve model for strategy {fund.strategy}")
            return self._generate_generic_cash_flows(projection_years)
        
        model = self.j_curve_models[fund.strategy]
        
        # Initialize projections
        years = list(range(1, projection_years + 1))
        capital_calls = []
        distributions = []
        nav_progression = []
        cumulative_returns = []
        
        current_nav = fund.nav
        total_called = fund.called_capital
        total_distributed = fund.distributed_capital
        
        for year in years:
            # Model capital calls (declining over investment period)
            if year <= model['investment_period']:
                call_intensity = 1.0 - (year - 1) / model['investment_period']
                annual_call = fund.committed_capital * 0.25 * call_intensity
                annual_call = min(annual_call, fund.committed_capital - total_called)
            else:
                annual_call = 0.0
            
            capital_calls.append(annual_call)
            total_called += annual_call
            
            # Model distributions (J-curve pattern)
            j_curve_impact = self._calculate_j_curve_impact(year, model)
            
            if year <= model['investment_period']:
                # Early years: minimal distributions, possible negative impact
                annual_distribution = total_called * max(0, j_curve_impact * 0.1)
            elif year <= model['maturity_period']:
                # Growth years: increasing distributions
                base_distribution = total_called * 0.15
                annual_distribution = base_distribution * (1 + j_curve_impact)
            else:
                # Mature years: declining distributions
                base_distribution = total_called * 0.10
                annual_distribution = base_distribution * (1 + j_curve_impact)
            
            distributions.append(annual_distribution)
            total_distributed += annual_distribution
            
            # Model NAV progression
            nav_growth_rate = 0.12 + j_curve_impact  # Base 12% growth + J-curve impact
            current_nav = current_nav * (1 + nav_growth_rate) - annual_distribution + annual_call
            nav_progression.append(max(0, current_nav))
            
            # Calculate cumulative returns
            if total_called > 0:
                cumulative_return = (total_distributed + current_nav) / total_called - 1.0
            else:
                cumulative_return = 0.0
            cumulative_returns.append(cumulative_return)
        
        return {
            'years': years,
            'capital_calls': capital_calls,
            'distributions': distributions,
            'nav_progression': nav_progression,
            'cumulative_returns': cumulative_returns,
            'total_called_projected': total_called,
            'total_distributed_projected': total_distributed,
            'final_irr': self._calculate_irr(capital_calls, distributions, nav_progression[-1])
        }
    
    def _calculate_j_curve_impact(self, year: int, model: Dict) -> float:
        """Calculate J-curve impact for given year based on model parameters"""
        investment_period = model['investment_period']
        peak_negative = model['peak_negative_impact']
        maturity_period = model['maturity_period']
        peak_positive = model['peak_positive_impact']
        decline_rate = model['decline_rate']
        
        if year <= investment_period:
            # Negative impact during investment period
            intensity = year / investment_period
            return peak_negative * (1 - intensity)
        elif year <= maturity_period:
            # Positive impact building to peak
            progress = (year - investment_period) / (maturity_period - investment_period)
            return peak_positive * progress
        else:
            # Declining impact after maturity
            decline_years = year - maturity_period
            return max(0, peak_positive * (1 - decline_rate * decline_years))
    
    def _generate_generic_cash_flows(self, years: int) -> Dict[str, List[float]]:
        """Generate generic cash flow pattern for unknown strategies"""
        years_list = list(range(1, years + 1))
        calls = [max(0, 100_000 * (4 - year) / 4) for year in years_list[:4]] + [0] * (years - 4)
        distributions = [0] * 2 + [year * 50_000 for year in range(1, years - 1)]
        nav = [sum(calls[:i]) - sum(distributions[:i]) for i in range(len(years_list))]
        returns = [(sum(distributions[:i]) + nav[i]) / max(1, sum(calls[:i])) - 1 for i in range(len(years_list))]
        
        return {
            'years': years_list,
            'capital_calls': calls,
            'distributions': distributions,
            'nav_progression': nav,
            'cumulative_returns': returns,
            'total_called_projected': sum(calls),
            'total_distributed_projected': sum(distributions),
            'final_irr': 0.12
        }
    
    def analyze_venture_capital_portfolio(self, 
                                        vc_investments: List[PrivateMarketInvestment]) -> Dict[str, float]:
        """
        Analyze venture capital portfolio with stage-based return modeling
        
        Args:
            vc_investments: List of VC investments across different stages
            
        Returns:
            Portfolio-level analysis with risk and return metrics
        """
        if not vc_investments:
            return {'portfolio_expected_return': 0.0, 'portfolio_risk': 0.0}
        
        # Calculate stage distribution
        stage_weights = {}
        total_commitment = sum(inv.committed_capital for inv in vc_investments)
        
        for investment in vc_investments:
            stage = investment.investment_stage
            weight = investment.committed_capital / total_commitment
            stage_weights[stage] = stage_weights.get(stage, 0) + weight
        
        # Calculate portfolio metrics
        portfolio_expected_return = 0.0
        portfolio_variance = 0.0
        
        for stage, weight in stage_weights.items():
            if stage in self.stage_return_models:
                stage_model = self.stage_return_models[stage]
                stage_return = stage_model['expected_return']
                stage_vol = stage_model['volatility']
                
                portfolio_expected_return += weight * stage_return
                portfolio_variance += (weight ** 2) * (stage_vol ** 2)
        
        portfolio_volatility = math.sqrt(portfolio_variance)
        
        # Calculate diversification metrics
        stage_concentration = max(stage_weights.values()) if stage_weights else 1.0
        stage_diversity_score = 1.0 - stage_concentration
        
        # Calculate success probability
        weighted_success_rate = 0.0
        weighted_failure_rate = 0.0
        
        for stage, weight in stage_weights.items():
            if stage in self.stage_return_models:
                stage_model = self.stage_return_models[stage]
                weighted_success_rate += weight * stage_model['success_rate']
                weighted_failure_rate += weight * stage_model['failure_rate']
        
        return {
            'portfolio_expected_return': portfolio_expected_return,
            'portfolio_volatility': portfolio_volatility,
            'stage_diversification_score': stage_diversity_score,
            'stage_concentration': stage_concentration,
            'weighted_success_rate': weighted_success_rate,
            'weighted_failure_rate': weighted_failure_rate,
            'number_of_investments': len(vc_investments),
            'total_commitment': total_commitment,
            'stage_distribution': stage_weights
        }
    
    def model_hedge_fund_strategy_allocation(self, 
                                           allocations: Dict[str, float],
                                           market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Model hedge fund strategy allocation with dynamic adjustments
        
        Args:
            allocations: Strategy allocation weights
            market_conditions: Current market environment metrics
            
        Returns:
            Optimized allocation and performance projections
        """
        # Validate allocations
        total_allocation = sum(allocations.values())
        if abs(total_allocation - 1.0) > 0.01:
            logger.warning(f"Allocations sum to {total_allocation}, normalizing")
            allocations = {k: v/total_allocation for k, v in allocations.items()}
        
        # Calculate portfolio metrics
        portfolio_return = 0.0
        portfolio_variance = 0.0
        portfolio_market_correlation = 0.0
        weighted_max_leverage = 0.0
        
        for strategy, weight in allocations.items():
            if strategy in self.hedge_fund_strategies:
                strategy_data = self.hedge_fund_strategies[strategy]
                
                # Adjust returns based on market conditions
                adjusted_return = self._adjust_strategy_return(
                    strategy, strategy_data, market_conditions
                )
                
                portfolio_return += weight * adjusted_return
                portfolio_variance += (weight ** 2) * (strategy_data['volatility'] ** 2)
                portfolio_market_correlation += weight * strategy_data['market_correlation']
                weighted_max_leverage += weight * strategy_data['max_leverage']
        
        # Add correlation adjustments (simplified)
        correlation_adjustment = 0.7  # Assume 70% correlation between strategies
        portfolio_variance *= correlation_adjustment
        portfolio_volatility = math.sqrt(portfolio_variance)
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification benefit
        individual_vol = sum(allocations[s] * self.hedge_fund_strategies[s]['volatility'] 
                           for s in allocations if s in self.hedge_fund_strategies)
        diversification_benefit = (individual_vol - portfolio_volatility) / individual_vol if individual_vol > 0 else 0
        
        return {
            'portfolio_expected_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_sharpe_ratio': sharpe_ratio,
            'portfolio_market_correlation': portfolio_market_correlation,
            'weighted_max_leverage': weighted_max_leverage,
            'diversification_benefit': diversification_benefit,
            'strategy_count': len(allocations),
            'allocation_concentration': max(allocations.values())
        }
    
    def _adjust_strategy_return(self, 
                              strategy: str, 
                              strategy_data: Dict, 
                              market_conditions: Dict) -> float:
        """Adjust strategy expected returns based on market conditions"""
        base_return = strategy_data['expected_return']
        
        # Market volatility adjustment
        market_vol = market_conditions.get('market_volatility', 0.15)
        if market_vol > 0.25:  # High volatility environment
            if strategy in ['market_neutral', 'relative_value']:
                return base_return * 1.2  # Benefit from volatility
            elif strategy == 'long_short_equity':
                return base_return * 0.8  # Challenged by volatility
        
        # Interest rate environment
        interest_rates = market_conditions.get('interest_rates', 0.02)
        if interest_rates > 0.05:  # High rate environment
            if strategy == 'relative_value':
                return base_return * 1.3  # Benefits from rate differentials
        
        # Market direction
        market_trend = market_conditions.get('market_trend', 0.0)  # -1 to 1
        if strategy == 'long_short_equity':
            # Adjust based on market skill in different environments
            if market_trend < -0.5:  # Bear market
                return base_return * 1.1  # Short-selling opportunities
            elif market_trend > 0.5:  # Bull market
                return base_return * 0.9  # Harder to generate alpha
        
        return base_return
    
    def calculate_illiquidity_premium(self, 
                                    liquid_return: float,
                                    illiquid_return: float,
                                    illiquidity_period: float,
                                    liquidity_risk: float = 0.02) -> float:
        """
        Calculate illiquidity premium for private market investments
        
        Args:
            liquid_return: Expected return of liquid equivalent
            illiquid_return: Expected return of illiquid investment
            illiquidity_period: Years of illiquidity
            liquidity_risk: Additional risk premium for liquidity
            
        Returns:
            Annualized illiquidity premium
        """
        # Base premium from return difference
        base_premium = illiquid_return - liquid_return
        
        # Time-based premium (compound effect)
        time_premium = liquidity_risk * math.sqrt(illiquidity_period)
        
        # Uncertainty premium (increases with time)
        uncertainty_premium = 0.005 * illiquidity_period  # 0.5% per year
        
        total_premium = base_premium + time_premium + uncertainty_premium
        
        return max(0, total_premium)  # Premium should be non-negative
    
    def _calculate_irr(self, cash_flows_out: List[float], cash_flows_in: List[float], terminal_value: float) -> float:
        """Calculate IRR from cash flow projections"""
        # Combine cash flows (negative for calls, positive for distributions)
        net_cash_flows = [-out + in_flow for out, in_flow in zip(cash_flows_out, cash_flows_in)]
        net_cash_flows[-1] += terminal_value  # Add terminal NAV to final period
        
        # Simple IRR approximation (Newton's method would be more accurate)
        if not net_cash_flows or all(cf <= 0 for cf in net_cash_flows):
            return 0.0
        
        # Use NPV = 0 to solve for IRR (simplified)
        total_outflows = sum(cash_flows_out)
        total_inflows = sum(cash_flows_in) + terminal_value
        
        if total_outflows <= 0:
            return 0.0
        
        simple_return = (total_inflows / total_outflows) - 1.0
        holding_period = len(cash_flows_out)
        
        if holding_period <= 0:
            return simple_return
        
        # Annualize
        irr = (1 + simple_return) ** (1.0 / holding_period) - 1.0
        return max(-0.5, min(2.0, irr))  # Cap at reasonable bounds
    
    def generate_private_market_summary(self) -> Dict[str, any]:
        """Generate private market analysis summary"""
        return {
            'analyzer_status': 'active',
            'supported_strategies': list(self.j_curve_models.keys()),
            'vc_stages_supported': list(self.stage_return_models.keys()),
            'hedge_fund_strategies': list(self.hedge_fund_strategies.keys()),
            'modeling_capabilities': [
                'j_curve_modeling',
                'cash_flow_projection',
                'vc_stage_analysis',
                'hedge_fund_optimization',
                'illiquidity_premium_calculation',
                'irr_calculation'
            ],
            'last_updated': datetime.now().isoformat()
        }


# Demo usage function
def demo_private_market_analysis():
    """Demonstrate private market analysis capabilities"""
    analyzer = PrivateMarketAnalyzer()
    
    print("Private Market Analyzer Demo")
    print("=" * 50)
    
    # Create sample PE fund
    pe_fund = PrivateMarketInvestment(
        fund_id="PE_FUND_001",
        fund_name="Sample PE Fund",
        fund_manager="Top Tier Manager",
        strategy=PrivateMarketType.PRIVATE_EQUITY,
        vintage_year=2020,
        fund_size=1_000_000_000,
        committed_capital=50_000_000,
        called_capital=30_000_000,
        distributed_capital=5_000_000,
        nav=35_000_000
    )
    
    # Model J-curve cash flows
    cash_flows = analyzer.model_j_curve_cash_flows(pe_fund, 10)
    
    print(f"J-Curve Analysis for {pe_fund.fund_name}:")
    print(f"- Projected Final IRR: {cash_flows['final_irr']:.1%}")
    print(f"- Total Projected Calls: ${cash_flows['total_called_projected']:,.0f}")
    print(f"- Total Projected Distributions: ${cash_flows['total_distributed_projected']:,.0f}")
    
    # Hedge fund allocation analysis
    hf_allocation = {
        'long_short_equity': 0.4,
        'market_neutral': 0.2,
        'event_driven': 0.2,
        'global_macro': 0.2
    }
    
    market_conditions = {
        'market_volatility': 0.20,  # 20% volatility
        'interest_rates': 0.03,     # 3% rates
        'market_trend': 0.1         # Slightly positive
    }
    
    hf_analysis = analyzer.model_hedge_fund_strategy_allocation(hf_allocation, market_conditions)
    
    print(f"\nHedge Fund Portfolio Analysis:")
    print(f"- Expected Return: {hf_analysis['portfolio_expected_return']:.1%}")
    print(f"- Portfolio Volatility: {hf_analysis['portfolio_volatility']:.1%}")
    print(f"- Sharpe Ratio: {hf_analysis['portfolio_sharpe_ratio']:.2f}")
    print(f"- Diversification Benefit: {hf_analysis['diversification_benefit']:.1%}")
    
    # Summary
    summary = analyzer.generate_private_market_summary()
    print(f"\nPrivate Market Coverage:")
    print(f"- Supported Strategies: {len(summary['supported_strategies'])}")
    print(f"- VC Stages: {len(summary['vc_stages_supported'])}")
    print(f"- Hedge Fund Strategies: {len(summary['hedge_fund_strategies'])}")


if __name__ == "__main__":
    demo_private_market_analysis()
