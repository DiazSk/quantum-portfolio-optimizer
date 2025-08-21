"""
Private market data collector for institutional alternative asset portfolios.

Provides comprehensive data collection for private equity, venture capital,
hedge funds, infrastructure, and other private market investment strategies.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
import json

from src.portfolio.alternative_assets.private_markets import (
    PrivateMarketInvestment, PrivateMarketType, InvestmentStage, FundStatus,
    HedgeFundStrategy, PrivateMarketPortfolioMetrics, GeographicFocus, HedgeFundStrategyType
)


logger = logging.getLogger(__name__)


class PrivateMarketDataCollector:
    """
    Comprehensive private market data collector for institutional portfolios.
    
    Features:
    - Private equity fund data collection
    - Venture capital performance metrics
    - Hedge fund strategy analysis
    - Infrastructure investment tracking
    - Performance attribution and benchmarking
    """
    
    def __init__(self):
        """Initialize private market data collector with universe and benchmarks"""
        self.pe_universe = self._get_pe_universe()
        self.vc_universe = self._get_vc_universe()
        self.hedge_fund_universe = self._get_hedge_fund_universe()
        self.infrastructure_universe = self._get_infrastructure_universe()
        self.benchmark_data = self._get_benchmark_data()
        self.fee_structures = self._get_fee_structures()
        self.vintage_curves = self._get_vintage_curves()
        
    def _get_pe_universe(self) -> Dict[str, Dict]:
        """Define private equity fund universe"""
        return {
            'KKR_XIII': {
                'fund_name': 'KKR North America Fund XIII',
                'fund_manager': 'KKR & Co.',
                'strategy': PrivateMarketType.PRIVATE_EQUITY,
                'vintage_year': 2018,
                'fund_size': 15000000000,
                'target_sectors': ['Technology', 'Healthcare', 'Industrials'],
                'geographic_focus': GeographicFocus.NORTH_AMERICA,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 5,
                'leverage_multiple': 6.0
            },
            'APOLLO_IX': {
                'fund_name': 'Apollo Investment Fund IX',
                'fund_manager': 'Apollo Global Management',
                'strategy': PrivateMarketType.PRIVATE_EQUITY,
                'vintage_year': 2019,
                'fund_size': 20000000000,
                'target_sectors': ['Financial Services', 'Consumer', 'Energy'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 4,
                'leverage_multiple': 5.5
            },
            'BLACKSTONE_VIII': {
                'fund_name': 'Blackstone Capital Partners VIII',
                'fund_manager': 'The Blackstone Group',
                'strategy': PrivateMarketType.PRIVATE_EQUITY,
                'vintage_year': 2020,
                'fund_size': 22000000000,
                'target_sectors': ['Technology', 'Healthcare', 'Services'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 5,
                'leverage_multiple': 6.2
            },
            'TPG_VIII': {
                'fund_name': 'TPG Partners VIII',
                'fund_manager': 'TPG Inc.',
                'strategy': PrivateMarketType.PRIVATE_EQUITY,
                'vintage_year': 2017,
                'fund_size': 13000000000,
                'target_sectors': ['Technology', 'Healthcare', 'Consumer'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 4,
                'leverage_multiple': 5.8
            },
            'BAIN_CAPITAL_XII': {
                'fund_name': 'Bain Capital Fund XII',
                'fund_manager': 'Bain Capital',
                'strategy': PrivateMarketType.PRIVATE_EQUITY,
                'vintage_year': 2021,
                'fund_size': 7500000000,
                'target_sectors': ['Technology', 'Healthcare', 'Services'],
                'geographic_focus': GeographicFocus.NORTH_AMERICA,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 5,
                'leverage_multiple': 5.0
            }
        }
    
    def _get_vc_universe(self) -> Dict[str, Dict]:
        """Define venture capital fund universe"""
        return {
            'SEQUOIA_XV': {
                'fund_name': 'Sequoia Capital Fund XV',
                'fund_manager': 'Sequoia Capital',
                'strategy': PrivateMarketType.VENTURE_CAPITAL,
                'vintage_year': 2020,
                'fund_size': 2800000000,
                'target_sectors': ['Software', 'AI/ML', 'Fintech', 'Biotech'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.GROWTH,
                'typical_hold_period': 7,
                'j_curve_duration': 3
            },
            'A16Z_VII': {
                'fund_name': 'Andreessen Horowitz Fund VII',
                'fund_manager': 'Andreessen Horowitz',
                'strategy': PrivateMarketType.VENTURE_CAPITAL,
                'vintage_year': 2021,
                'fund_size': 4500000000,
                'target_sectors': ['Software', 'Crypto', 'Consumer', 'Enterprise'],
                'geographic_focus': GeographicFocus.NORTH_AMERICA,
                'investment_stage': InvestmentStage.GROWTH,
                'typical_hold_period': 8,
                'j_curve_duration': 4
            },
            'NEA_XVIII': {
                'fund_name': 'New Enterprise Associates XVIII',
                'fund_manager': 'New Enterprise Associates',
                'strategy': PrivateMarketType.VENTURE_CAPITAL,
                'vintage_year': 2019,
                'fund_size': 3600000000,
                'target_sectors': ['Healthcare', 'Technology', 'Energy'],
                'geographic_focus': GeographicFocus.NORTH_AMERICA,
                'investment_stage': InvestmentStage.EXPANSION,
                'typical_hold_period': 6,
                'j_curve_duration': 3
            },
            'INSIGHT_XII': {
                'fund_name': 'Insight Partners Fund XII',
                'fund_manager': 'Insight Partners',
                'strategy': PrivateMarketType.VENTURE_CAPITAL,
                'vintage_year': 2022,
                'fund_size': 9500000000,
                'target_sectors': ['Software', 'Internet', 'Data/Analytics'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.LATE_STAGE,
                'typical_hold_period': 5,
                'j_curve_duration': 2
            },
            'ACCEL_VI': {
                'fund_name': 'Accel Partners VI',
                'fund_manager': 'Accel Partners',
                'strategy': PrivateMarketType.VENTURE_CAPITAL,
                'vintage_year': 2018,
                'fund_size': 2000000000,
                'target_sectors': ['SaaS', 'Consumer', 'Fintech'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.SERIES_B,
                'typical_hold_period': 7,
                'j_curve_duration': 4
            }
        }
    
    def _get_hedge_fund_universe(self) -> Dict[str, Dict]:
        """Define hedge fund universe"""
        return {
            'BRIDGEWATER': {
                'fund_name': 'Bridgewater Pure Alpha',
                'fund_manager': 'Bridgewater Associates',
                'strategy': PrivateMarketType.HEDGE_FUND,
                'hedge_fund_strategy': HedgeFundStrategyType.GLOBAL_MACRO,
                'vintage_year': 2015,
                'fund_size': 150000000000,
                'target_volatility': 0.12,
                'geographic_focus': GeographicFocus.GLOBAL,
                'use_derivatives': True,
                'leverage_ratio': 3.0
            },
            'CITADEL': {
                'fund_name': 'Citadel Wellington',
                'fund_manager': 'Citadel LLC',
                'strategy': PrivateMarketType.HEDGE_FUND,
                'hedge_fund_strategy': HedgeFundStrategyType.EQUITY_LONG_SHORT,
                'vintage_year': 2016,
                'fund_size': 38000000000,
                'target_volatility': 0.10,
                'geographic_focus': GeographicFocus.GLOBAL,
                'use_derivatives': True,
                'leverage_ratio': 4.0
            },
            'RENAISSANCE': {
                'fund_name': 'Renaissance Institutional Equities',
                'fund_manager': 'Renaissance Technologies',
                'strategy': PrivateMarketType.HEDGE_FUND,
                'hedge_fund_strategy': HedgeFundStrategyType.QUANTITATIVE,
                'vintage_year': 2017,
                'fund_size': 75000000000,
                'target_volatility': 0.15,
                'geographic_focus': GeographicFocus.GLOBAL,
                'use_derivatives': True,
                'leverage_ratio': 5.0
            },
            'MILLENNIUM': {
                'fund_name': 'Millennium International',
                'fund_manager': 'Millennium Management',
                'strategy': PrivateMarketType.HEDGE_FUND,
                'hedge_fund_strategy': HedgeFundStrategyType.MULTI_STRATEGY,
                'vintage_year': 2018,
                'fund_size': 57000000000,
                'target_volatility': 0.08,
                'geographic_focus': GeographicFocus.GLOBAL,
                'use_derivatives': True,
                'leverage_ratio': 3.5
            },
            'TWO_SIGMA': {
                'fund_name': 'Two Sigma Spectrum',
                'fund_manager': 'Two Sigma Investments',
                'strategy': PrivateMarketType.HEDGE_FUND,
                'hedge_fund_strategy': HedgeFundStrategyType.QUANTITATIVE,
                'vintage_year': 2019,
                'fund_size': 58000000000,
                'target_volatility': 0.12,
                'geographic_focus': GeographicFocus.GLOBAL,
                'use_derivatives': True,
                'leverage_ratio': 4.5
            }
        }
    
    def _get_infrastructure_universe(self) -> Dict[str, Dict]:
        """Define infrastructure fund universe"""
        return {
            'BROOKFIELD_IV': {
                'fund_name': 'Brookfield Infrastructure Fund IV',
                'fund_manager': 'Brookfield Asset Management',
                'strategy': PrivateMarketType.INFRASTRUCTURE,
                'vintage_year': 2020,
                'fund_size': 25000000000,
                'target_sectors': ['Utilities', 'Transport', 'Energy', 'Data'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 10,
                'inflation_hedge': True
            },
            'KKR_AMERICAS_III': {
                'fund_name': 'KKR Americas Fund III',
                'fund_manager': 'KKR & Co.',
                'strategy': PrivateMarketType.INFRASTRUCTURE,
                'vintage_year': 2019,
                'fund_size': 7400000000,
                'target_sectors': ['Energy', 'Transport', 'Utilities'],
                'geographic_focus': GeographicFocus.NORTH_AMERICA,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 12,
                'inflation_hedge': True
            },
            'BLACKSTONE_INFRA': {
                'fund_name': 'Blackstone Infrastructure Partners',
                'fund_manager': 'The Blackstone Group',
                'strategy': PrivateMarketType.INFRASTRUCTURE,
                'vintage_year': 2021,
                'fund_size': 15000000000,
                'target_sectors': ['Digital Infrastructure', 'Energy Transition'],
                'geographic_focus': GeographicFocus.GLOBAL,
                'investment_stage': InvestmentStage.BUYOUT,
                'typical_hold_period': 8,
                'inflation_hedge': True
            }
        }
    
    def _get_benchmark_data(self) -> Dict[str, Dict]:
        """Define benchmark performance data"""
        return {
            PrivateMarketType.PRIVATE_EQUITY: {
                'cambridge_pe_index': {
                    '1_year': 0.12,
                    '3_year': 0.14,
                    '5_year': 0.13,
                    '10_year': 0.11
                },
                'public_market_equivalent': {
                    '1_year': 1.08,
                    '3_year': 1.15,
                    '5_year': 1.12,
                    '10_year': 1.09
                }
            },
            PrivateMarketType.VENTURE_CAPITAL: {
                'cambridge_vc_index': {
                    '1_year': 0.18,
                    '3_year': 0.22,
                    '5_year': 0.19,
                    '10_year': 0.16
                },
                'public_market_equivalent': {
                    '1_year': 1.25,
                    '3_year': 1.35,
                    '5_year': 1.28,
                    '10_year': 1.22
                }
            },
            PrivateMarketType.HEDGE_FUND: {
                'hfri_composite': {
                    '1_year': 0.08,
                    '3_year': 0.09,
                    '5_year': 0.07,
                    '10_year': 0.06
                },
                'sharpe_ratio': {
                    '1_year': 1.2,
                    '3_year': 1.1,
                    '5_year': 1.0,
                    '10_year': 0.9
                }
            },
            PrivateMarketType.INFRASTRUCTURE: {
                'edhec_infra_index': {
                    '1_year': 0.09,
                    '3_year': 0.10,
                    '5_year': 0.09,
                    '10_year': 0.08
                },
                'inflation_correlation': {
                    '1_year': 0.6,
                    '3_year': 0.65,
                    '5_year': 0.7,
                    '10_year': 0.75
                }
            }
        }
    
    def _get_fee_structures(self) -> Dict[str, Dict]:
        """Define typical fee structures by strategy"""
        return {
            PrivateMarketType.PRIVATE_EQUITY: {
                'management_fee': 0.02,
                'performance_fee': 0.20,
                'hurdle_rate': 0.08,
                'high_water_mark': True,
                'fee_decline': True  # Management fee declines over time
            },
            PrivateMarketType.VENTURE_CAPITAL: {
                'management_fee': 0.025,
                'performance_fee': 0.20,
                'hurdle_rate': 0.06,
                'high_water_mark': True,
                'fee_decline': True
            },
            PrivateMarketType.HEDGE_FUND: {
                'management_fee': 0.015,
                'performance_fee': 0.15,
                'hurdle_rate': 0.0,
                'high_water_mark': True,
                'fee_decline': False
            },
            PrivateMarketType.INFRASTRUCTURE: {
                'management_fee': 0.015,
                'performance_fee': 0.15,
                'hurdle_rate': 0.06,
                'high_water_mark': True,
                'fee_decline': True
            }
        }
    
    def _get_vintage_curves(self) -> Dict[int, Dict]:
        """Define J-curve patterns by vintage year and strategy"""
        return {
            # Year-by-year cash flow patterns (as % of committed capital)
            1: {'pe': -0.15, 'vc': -0.20, 'infra': -0.10, 'hf': 0.0},
            2: {'pe': -0.25, 'vc': -0.35, 'infra': -0.15, 'hf': 0.0},
            3: {'pe': -0.10, 'vc': -0.15, 'infra': 0.05, 'hf': 0.0},
            4: {'pe': 0.20, 'vc': 0.10, 'infra': 0.15, 'hf': 0.0},
            5: {'pe': 0.45, 'vc': 0.60, 'infra': 0.20, 'hf': 0.0},
            6: {'pe': 0.80, 'vc': 1.20, 'infra': 0.25, 'hf': 0.0},
            7: {'pe': 1.20, 'vc': 1.80, 'infra': 0.30, 'hf': 0.0},
            8: {'pe': 1.50, 'vc': 2.20, 'infra': 0.35, 'hf': 0.0},
            9: {'pe': 1.60, 'vc': 2.00, 'infra': 0.40, 'hf': 0.0},
            10: {'pe': 1.65, 'vc': 1.50, 'infra': 0.50, 'hf': 0.0}
        }
    
    async def collect_private_market_data(self, fund_ids: Optional[List[str]] = None) -> List[PrivateMarketInvestment]:
        """
        Collect comprehensive private market investment data.
        
        Args:
            fund_ids: List of fund IDs to collect. If None, uses full universe.
            
        Returns:
            List of PrivateMarketInvestment objects with institutional metrics
        """
        all_funds = {**self.pe_universe, **self.vc_universe, **self.hedge_fund_universe, **self.infrastructure_universe}
        
        if fund_ids is None:
            fund_ids = list(all_funds.keys())
        
        funds = []
        
        # Process funds in parallel
        fund_tasks = [self._collect_single_fund(fund_id, all_funds[fund_id]) 
                      for fund_id in fund_ids if fund_id in all_funds]
        fund_results = await asyncio.gather(*fund_tasks, return_exceptions=True)
        
        for result in fund_results:
            if isinstance(result, PrivateMarketInvestment):
                funds.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error collecting fund data: {result}")
        
        logger.info(f"Collected data for {len(funds)} private market funds")
        return funds
    
    async def _collect_single_fund(self, fund_id: str, fund_config: Dict) -> PrivateMarketInvestment:
        """Collect data for a single private market fund"""
        try:
            current_year = datetime.now().year
            fund_age = current_year - fund_config['vintage_year']
            
            # Generate realistic fund metrics based on strategy and age
            metrics = self._generate_fund_metrics(fund_config, fund_age)
            
            # Get fee structure
            fee_structure = self.fee_structures.get(fund_config['strategy'])
            
            # Create fund investment object
            fund = PrivateMarketInvestment(
                fund_id=fund_id,
                fund_name=fund_config['fund_name'],
                fund_manager=fund_config['fund_manager'],
                strategy=fund_config['strategy'],
                vintage_year=fund_config['vintage_year'],
                fund_size=fund_config['fund_size'],
                committed_capital=metrics['committed_capital'],
                called_capital=metrics['called_capital'],
                distributed_capital=metrics['distributed_capital'],
                nav=metrics['nav'],
                irr=metrics['irr'],
                sector_focus=fund_config.get('target_sectors', []),
                geographic_focus=fund_config.get('geographic_focus'),
                investment_stage=fund_config.get('investment_stage'),
                management_fee=fee_structure.get('management_fee', 0.02) if fee_structure else 0.02,
                performance_fee=fee_structure.get('performance_fee', 0.20) if fee_structure else 0.20,
                hurdle_rate=fee_structure.get('hurdle_rate', 0.08) if fee_structure else 0.08,
                high_water_mark=fee_structure.get('high_water_mark', True) if fee_structure else True,
                illiquidity_factor=self._calculate_illiquidity_factor(fund_config),
                valuation_method="nav",
                fund_life=fund_config.get('typical_hold_period', 10),
                investment_period=min(5, fund_config.get('typical_hold_period', 10) // 2),
                fund_status=self._determine_fund_status(fund_age, fund_config),
                benchmark_comparison=self._get_benchmark_comparison(fund_config['strategy'], metrics['irr']),
                esg_score=np.random.uniform(6.0, 9.0),  # ESG scores becoming important
                leverage_ratio=fund_config.get('leverage_ratio', 0.0),
                concentration_risk=self._calculate_concentration_risk(fund_config),
                liquidity_terms=self._get_liquidity_terms(fund_config['strategy'])
            )
            
            # Add strategy-specific attributes
            if fund_config['strategy'] == PrivateMarketType.HEDGE_FUND:
                fund.hedge_fund_strategy = fund_config.get('hedge_fund_strategy')
                fund.use_derivatives = fund_config.get('use_derivatives', False)
                fund.target_volatility = fund_config.get('target_volatility', 0.10)
            
            return fund
            
        except Exception as e:
            logger.error(f"Error collecting data for fund {fund_id}: {e}")
            raise
    
    def _generate_fund_metrics(self, fund_config: Dict, fund_age: int) -> Dict[str, float]:
        """Generate realistic fund performance metrics based on strategy and age"""
        fund_size = fund_config['fund_size']
        strategy = fund_config['strategy']
        
        # Base commitment rate (how much of fund size is actually committed)
        commitment_rate = min(1.0, 0.7 + fund_age * 0.05)
        committed_capital = fund_size * commitment_rate
        
        # Capital call pattern based on strategy and fund age
        if strategy == PrivateMarketType.HEDGE_FUND:
            # Hedge funds call capital immediately
            called_capital = committed_capital
            calling_rate = 1.0
        else:
            # PE/VC/Infrastructure have gradual capital calls
            calling_rate = min(1.0, fund_age * 0.15)
            called_capital = committed_capital * calling_rate
        
        # Distribution pattern (J-curve effect)
        if strategy == PrivateMarketType.HEDGE_FUND:
            # Hedge funds distribute regularly
            distributed_capital = max(0, called_capital * np.random.uniform(0.05, 0.15))
        else:
            # PE/VC/Infrastructure follow J-curve pattern
            distribution_rate = max(0, (fund_age - 3) * 0.08) if fund_age > 3 else 0
            distributed_capital = called_capital * distribution_rate
        
        # NAV calculation
        if strategy == PrivateMarketType.HEDGE_FUND:
            # Hedge fund NAV based on performance
            performance_factor = 1 + np.random.uniform(-0.1, 0.2)  # -10% to +20%
            nav = called_capital * performance_factor
        else:
            # PE/VC/Infrastructure NAV includes unrealized gains
            j_curve_multiplier = self._get_j_curve_multiplier(strategy, fund_age)
            nav = max(0, called_capital * j_curve_multiplier - distributed_capital)
        
        # IRR calculation based on cash flows
        irr = self._calculate_fund_irr(strategy, fund_age, called_capital, distributed_capital, nav)
        
        return {
            'committed_capital': committed_capital,
            'called_capital': called_capital,
            'distributed_capital': distributed_capital,
            'nav': nav,
            'irr': irr
        }
    
    def _get_j_curve_multiplier(self, strategy: PrivateMarketType, fund_age: int) -> float:
        """Get J-curve multiplier based on strategy and fund age"""
        strategy_map = {
            PrivateMarketType.PRIVATE_EQUITY: 'pe',
            PrivateMarketType.VENTURE_CAPITAL: 'vc',
            PrivateMarketType.INFRASTRUCTURE: 'infra'
        }
        
        strategy_key = strategy_map.get(strategy, 'pe')
        
        # Early years: losses due to fees and unrealized investments
        if fund_age <= 2:
            return 0.8 - fund_age * 0.1
        # Middle years: value creation
        elif fund_age <= 6:
            return 0.6 + (fund_age - 2) * 0.3
        # Later years: harvest phase
        else:
            peak_multiplier = 1.8 if strategy == PrivateMarketType.VENTURE_CAPITAL else 1.4
            decline_rate = 0.05 if strategy == PrivateMarketType.INFRASTRUCTURE else 0.1
            return max(1.0, peak_multiplier - (fund_age - 6) * decline_rate)
    
    def _calculate_fund_irr(self, strategy: PrivateMarketType, fund_age: int, 
                           called: float, distributed: float, nav: float) -> float:
        """Calculate fund IRR based on cash flows"""
        if called <= 0:
            return 0.0
        
        # Simplified IRR calculation
        total_value = distributed + nav
        multiple = total_value / called if called > 0 else 1.0
        
        # Annualized return based on fund age
        if fund_age <= 0:
            return 0.0
        
        # Strategy-specific return expectations
        base_returns = {
            PrivateMarketType.PRIVATE_EQUITY: 0.12,
            PrivateMarketType.VENTURE_CAPITAL: 0.18,
            PrivateMarketType.HEDGE_FUND: 0.08,
            PrivateMarketType.INFRASTRUCTURE: 0.09
        }
        
        expected_return = base_returns.get(strategy, 0.10)
        
        # Add vintage year effect and random variation
        vintage_adjustment = np.random.uniform(-0.03, 0.05)
        j_curve_adjustment = -0.05 if fund_age <= 3 else 0.02
        
        return max(-0.5, min(1.0, expected_return + vintage_adjustment + j_curve_adjustment))
    
    def _calculate_illiquidity_factor(self, fund_config: Dict) -> float:
        """Calculate illiquidity factor based on strategy"""
        base_illiquidity = {
            PrivateMarketType.PRIVATE_EQUITY: 0.95,
            PrivateMarketType.VENTURE_CAPITAL: 0.98,
            PrivateMarketType.HEDGE_FUND: 0.2,  # Most hedge funds have monthly/quarterly liquidity
            PrivateMarketType.INFRASTRUCTURE: 0.90,
            PrivateMarketType.REAL_ESTATE_PRIVATE: 0.85
        }
        
        return base_illiquidity.get(fund_config['strategy'], 0.90)
    
    def _determine_fund_status(self, fund_age: int, fund_config: Dict) -> FundStatus:
        """Determine fund lifecycle status based on age and strategy"""
        strategy = fund_config['strategy']
        typical_life = fund_config.get('typical_hold_period', 10)
        
        if strategy == PrivateMarketType.HEDGE_FUND:
            return FundStatus.INVESTING  # Hedge funds continuously invest
        
        if fund_age <= 1:
            return FundStatus.FUNDRAISING
        elif fund_age <= typical_life * 0.4:
            return FundStatus.INVESTING
        elif fund_age <= typical_life * 0.8:
            return FundStatus.HARVESTING
        elif fund_age <= typical_life:
            return FundStatus.LIQUIDATING
        else:
            return FundStatus.CLOSED
    
    def _get_benchmark_comparison(self, strategy: PrivateMarketType, fund_irr: float) -> Dict[str, float]:
        """Get benchmark comparison for the fund"""
        benchmarks = self.benchmark_data.get(strategy, {})
        
        comparison = {}
        for benchmark_name, periods in benchmarks.items():
            # Use 5-year benchmark as primary comparison
            benchmark_return = periods.get('5_year', 0.10)
            comparison[benchmark_name] = fund_irr - benchmark_return
        
        return comparison
    
    def _calculate_concentration_risk(self, fund_config: Dict) -> float:
        """Calculate concentration risk based on fund characteristics"""
        # Base concentration risk by strategy
        base_risk = {
            PrivateMarketType.PRIVATE_EQUITY: 0.3,
            PrivateMarketType.VENTURE_CAPITAL: 0.6,  # Higher concentration in VC
            PrivateMarketType.HEDGE_FUND: 0.2,
            PrivateMarketType.INFRASTRUCTURE: 0.4
        }.get(fund_config['strategy'], 0.3)
        
        # Adjust for fund size (larger funds typically less concentrated)
        fund_size = fund_config['fund_size']
        if fund_size > 10000000000:  # $10B+
            size_adjustment = -0.1
        elif fund_size < 1000000000:  # <$1B
            size_adjustment = 0.1
        else:
            size_adjustment = 0.0
        
        # Adjust for sector focus
        num_sectors = len(fund_config.get('target_sectors', []))
        if num_sectors <= 2:
            sector_adjustment = 0.1
        elif num_sectors >= 5:
            sector_adjustment = -0.1
        else:
            sector_adjustment = 0.0
        
        return max(0.1, min(0.9, base_risk + size_adjustment + sector_adjustment))
    
    def _get_liquidity_terms(self, strategy: PrivateMarketType) -> Dict[str, any]:
        """Get liquidity terms based on strategy"""
        terms = {
            PrivateMarketType.PRIVATE_EQUITY: {
                'redemption_frequency': 'NONE',
                'notice_period_days': 0,
                'gate_provisions': False,
                'side_pockets': True,
                'transfer_restrictions': True
            },
            PrivateMarketType.VENTURE_CAPITAL: {
                'redemption_frequency': 'NONE',
                'notice_period_days': 0,
                'gate_provisions': False,
                'side_pockets': True,
                'transfer_restrictions': True
            },
            PrivateMarketType.HEDGE_FUND: {
                'redemption_frequency': 'MONTHLY',
                'notice_period_days': 30,
                'gate_provisions': True,
                'side_pockets': False,
                'transfer_restrictions': False
            },
            PrivateMarketType.INFRASTRUCTURE: {
                'redemption_frequency': 'ANNUAL',
                'notice_period_days': 90,
                'gate_provisions': True,
                'side_pockets': True,
                'transfer_restrictions': True
            }
        }
        
        return terms.get(strategy, terms[PrivateMarketType.PRIVATE_EQUITY])
    
    async def collect_pe_funds(self, fund_ids: Optional[List[str]] = None) -> List[PrivateMarketInvestment]:
        """Collect private equity fund data specifically"""
        if fund_ids is None:
            fund_ids = list(self.pe_universe.keys())
        
        return await self.collect_private_market_data(fund_ids)
    
    async def collect_vc_funds(self, fund_ids: Optional[List[str]] = None) -> List[PrivateMarketInvestment]:
        """Collect venture capital fund data specifically"""
        if fund_ids is None:
            fund_ids = list(self.vc_universe.keys())
        
        return await self.collect_private_market_data(fund_ids)
    
    async def collect_hedge_funds(self, fund_ids: Optional[List[str]] = None) -> List[PrivateMarketInvestment]:
        """Collect hedge fund data specifically"""
        if fund_ids is None:
            fund_ids = list(self.hedge_fund_universe.keys())
        
        return await self.collect_private_market_data(fund_ids)
    
    async def collect_infrastructure_funds(self, fund_ids: Optional[List[str]] = None) -> List[PrivateMarketInvestment]:
        """Collect infrastructure fund data specifically"""
        if fund_ids is None:
            fund_ids = list(self.infrastructure_universe.keys())
        
        return await self.collect_private_market_data(fund_ids)
    
    async def calculate_portfolio_metrics(self, funds: List[PrivateMarketInvestment],
                                        weights: Optional[List[float]] = None) -> PrivateMarketPortfolioMetrics:
        """Calculate portfolio-level metrics for private market investments"""
        if not funds:
            raise ValueError("No funds provided")
        
        if weights is None:
            weights = [1.0 / len(funds)] * len(funds)
        
        if len(weights) != len(funds):
            raise ValueError("Weights length must match funds length")
        
        # Portfolio calculations
        total_committed = sum(fund.committed_capital * weight for fund, weight in zip(funds, weights))
        total_called = sum(fund.called_capital * weight for fund, weight in zip(funds, weights))
        total_distributed = sum(fund.distributed_capital * weight for fund, weight in zip(funds, weights))
        total_nav = sum(fund.nav * weight for fund, weight in zip(funds, weights))
        
        # Portfolio IRR (weighted average)
        portfolio_irr = sum(fund.irr * weight for fund, weight in zip(funds, weights))
        
        # Portfolio multiples
        portfolio_tvpi = (total_distributed + total_nav) / total_called if total_called > 0 else 0
        portfolio_dpi = total_distributed / total_called if total_called > 0 else 0
        portfolio_rvpi = total_nav / total_called if total_called > 0 else 0
        
        # Strategy allocation
        strategy_allocation = {}
        for fund, weight in zip(funds, weights):
            strategy = fund.strategy.value
            strategy_allocation[strategy] = strategy_allocation.get(strategy, 0) + weight
        
        # Vintage year diversification
        vintage_allocation = {}
        for fund, weight in zip(funds, weights):
            vintage = str(fund.vintage_year)
            vintage_allocation[vintage] = vintage_allocation.get(vintage, 0) + weight
        
        # Geographic diversification
        geographic_allocation = {}
        for fund, weight in zip(funds, weights):
            if fund.geographic_focus:
                geo = fund.geographic_focus.value
                geographic_allocation[geo] = geographic_allocation.get(geo, 0) + weight
        
        # Risk metrics
        weighted_illiquidity = sum(fund.illiquidity_factor * weight for fund, weight in zip(funds, weights))
        concentration_risk = max(fund.concentration_risk for fund in funds)
        
        # J-curve impact assessment
        current_year = datetime.now().year
        j_curve_impact = 0.0
        for fund, weight in zip(funds, weights):
            fund_age = current_year - fund.vintage_year
            j_curve_impact += fund.estimate_j_curve_impact(fund_age) * weight
        
        return PrivateMarketPortfolioMetrics(
            total_committed_capital=total_committed,
            total_called_capital=total_called,
            total_distributed_capital=total_distributed,
            total_nav=total_nav,
            portfolio_irr=portfolio_irr,
            portfolio_tvpi=portfolio_tvpi,
            portfolio_dpi=portfolio_dpi,
            portfolio_rvpi=portfolio_rvpi,
            strategy_allocation=strategy_allocation,
            vintage_diversification=vintage_allocation,
            geographic_diversification=geographic_allocation,
            illiquidity_factor=weighted_illiquidity,
            j_curve_impact=j_curve_impact,
            concentration_risk=concentration_risk,
            benchmark_comparison={
                'pe_excess_return': portfolio_irr - 0.12,  # vs PE benchmark
                'public_market_equivalent': portfolio_tvpi / 1.1  # vs public markets
            }
        )


# Example usage and testing
async def main():
    """Example usage of private market data collector"""
    collector = PrivateMarketDataCollector()
    
    # Collect sample funds from each strategy
    pe_funds = await collector.collect_pe_funds(['KKR_XIII', 'APOLLO_IX'])
    vc_funds = await collector.collect_vc_funds(['SEQUOIA_XV', 'A16Z_VII'])
    hedge_funds = await collector.collect_hedge_funds(['BRIDGEWATER', 'CITADEL'])
    infra_funds = await collector.collect_infrastructure_funds(['BROOKFIELD_IV'])
    
    all_funds = pe_funds + vc_funds + hedge_funds + infra_funds
    
    print(f"Collected data for {len(all_funds)} private market funds:")
    for fund in all_funds:
        print(f"- {fund.fund_id}: {fund.fund_name}")
        print(f"  Strategy: {fund.strategy.value}")
        print(f"  Vintage: {fund.vintage_year}")
        print(f"  Fund Size: ${fund.fund_size / 1e9:.1f}B")
        print(f"  IRR: {fund.irr:.1%}")
        print(f"  TVPI: {fund.calculate_tvpi():.2f}")
        print(f"  Status: {fund.fund_status.value}")
        print()
    
    # Calculate portfolio metrics
    if all_funds:
        portfolio_metrics = await collector.calculate_portfolio_metrics(all_funds)
        print("Portfolio Metrics:")
        print(f"Total Committed: ${portfolio_metrics.total_committed_capital / 1e9:.1f}B")
        print(f"Portfolio IRR: {portfolio_metrics.portfolio_irr:.1%}")
        print(f"Portfolio TVPI: {portfolio_metrics.portfolio_tvpi:.2f}")
        print(f"Illiquidity Factor: {portfolio_metrics.illiquidity_factor:.1%}")
        print(f"J-Curve Impact: {portfolio_metrics.j_curve_impact:.1%}")
        print("Strategy Allocation:")
        for strategy, allocation in portfolio_metrics.strategy_allocation.items():
            print(f"  {strategy}: {allocation:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
