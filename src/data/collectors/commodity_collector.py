"""
Commodity futures and physical asset data collector for alternative asset portfolio optimization.

This module provides comprehensive commodity data collection including futures contracts,
spot prices, storage costs, convenience yields, and supply/demand fundamentals essential
for institutional commodity investment strategies.
"""

import logging
import asyncio
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import asdict

from src.portfolio.alternative_assets.commodities import (
    CommodityFuture, CommodityType, CommoditySubcategory, Exchange,
    PhysicalCommodityPosition, CommoditySupplyDemandData
)


class CommodityDataCollector:
    """
    Comprehensive commodity data collection system for institutional portfolio management.
    
    Supports futures contracts, spot prices, physical commodities, storage costs,
    convenience yields, and fundamental supply/demand analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize data sources and API clients
        self._init_api_clients()
        
        # Commodity universe definitions
        self.futures_universe = self._get_futures_universe()
        self.spot_universe = self._get_spot_universe()
        self.etf_proxies = self._get_etf_proxies()
        
        # Commodity mappings and data
        self.commodity_mapping = self._get_commodity_mapping()
        self.storage_costs = self._get_storage_cost_data()
        self.seasonal_patterns = self._get_seasonal_patterns()
        
    def _init_api_clients(self):
        """Initialize API connections for commodity data sources"""
        try:
            # Yahoo Finance for ETF and futures data - no session needed
            # yfinance handles sessions internally
            
            # Additional APIs would be initialized in production:
            # self.quandl_client = QuandlClient(api_key=os.getenv('QUANDL_API_KEY'))
            # self.eia_client = EIAClient(api_key=os.getenv('EIA_API_KEY'))  # Energy Information Administration
            # self.usda_client = USDAClient(api_key=os.getenv('USDA_API_KEY'))  # Agricultural data
            # self.lme_client = LMEClient(api_key=os.getenv('LME_API_KEY'))  # London Metal Exchange
            
            self.logger.info("Commodity data collector APIs initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"API initialization failed, using fallback data: {e}")
    
    def _get_futures_universe(self) -> Dict[str, Dict]:
        """Get comprehensive futures contract universe"""
        return {
            # Energy Futures
            'CL=F': {
                'name': 'Crude Oil WTI',
                'type': CommodityType.ENERGY,
                'subcategory': CommoditySubcategory.CRUDE_OIL,
                'exchange': Exchange.NYMEX,
                'contract_size': 1000,  # barrels
                'price_unit': '$/barrel',
                'tick_size': 0.01,
                'tick_value': 10.0
            },
            'NG=F': {
                'name': 'Natural Gas',
                'type': CommodityType.ENERGY,
                'subcategory': CommoditySubcategory.NATURAL_GAS,
                'exchange': Exchange.NYMEX,
                'contract_size': 10000,  # MMBtu
                'price_unit': '$/MMBtu',
                'tick_size': 0.001,
                'tick_value': 10.0
            },
            'RB=F': {
                'name': 'RBOB Gasoline',
                'type': CommodityType.ENERGY,
                'subcategory': CommoditySubcategory.GASOLINE,
                'exchange': Exchange.NYMEX,
                'contract_size': 42000,  # gallons
                'price_unit': '$/gallon',
                'tick_size': 0.0001,
                'tick_value': 4.20
            },
            
            # Precious Metals
            'GC=F': {
                'name': 'Gold',
                'type': CommodityType.PRECIOUS_METALS,
                'subcategory': CommoditySubcategory.GOLD,
                'exchange': Exchange.COMEX,
                'contract_size': 100,  # troy ounces
                'price_unit': '$/ounce',
                'tick_size': 0.10,
                'tick_value': 10.0
            },
            'SI=F': {
                'name': 'Silver',
                'type': CommodityType.PRECIOUS_METALS,
                'subcategory': CommoditySubcategory.SILVER,
                'exchange': Exchange.COMEX,
                'contract_size': 5000,  # troy ounces
                'price_unit': '$/ounce',
                'tick_size': 0.001,
                'tick_value': 5.0
            },
            'PL=F': {
                'name': 'Platinum',
                'type': CommodityType.PRECIOUS_METALS,
                'subcategory': CommoditySubcategory.PLATINUM,
                'exchange': Exchange.NYMEX,
                'contract_size': 50,  # troy ounces
                'price_unit': '$/ounce',
                'tick_size': 0.10,
                'tick_value': 5.0
            },
            
            # Base Metals
            'HG=F': {
                'name': 'Copper',
                'type': CommodityType.BASE_METALS,
                'subcategory': CommoditySubcategory.COPPER,
                'exchange': Exchange.COMEX,
                'contract_size': 25000,  # pounds
                'price_unit': '$/pound',
                'tick_size': 0.0005,
                'tick_value': 12.50
            },
            
            # Agricultural
            'ZC=F': {
                'name': 'Corn',
                'type': CommodityType.AGRICULTURE,
                'subcategory': CommoditySubcategory.CORN,
                'exchange': Exchange.CBOT,
                'contract_size': 5000,  # bushels
                'price_unit': '$/bushel',
                'tick_size': 0.0025,
                'tick_value': 12.50
            },
            'ZS=F': {
                'name': 'Soybeans',
                'type': CommodityType.AGRICULTURE,
                'subcategory': CommoditySubcategory.SOYBEANS,
                'exchange': Exchange.CBOT,
                'contract_size': 5000,  # bushels
                'price_unit': '$/bushel',
                'tick_size': 0.0025,
                'tick_value': 12.50
            },
            'ZW=F': {
                'name': 'Wheat',
                'type': CommodityType.AGRICULTURE,
                'subcategory': CommoditySubcategory.WHEAT,
                'exchange': Exchange.CBOT,
                'contract_size': 5000,  # bushels
                'price_unit': '$/bushel',
                'tick_size': 0.0025,
                'tick_value': 12.50
            },
        }
    
    def _get_spot_universe(self) -> Dict[str, str]:
        """Get spot price symbols for major commodities"""
        return {
            'Gold': 'GLD',      # SPDR Gold Trust
            'Silver': 'SLV',    # iShares Silver Trust
            'Oil': 'USO',       # United States Oil Fund
            'Natural_Gas': 'UNG',  # United States Natural Gas Fund
            'Copper': 'CPER',   # United States Copper Index Fund
            'Agriculture': 'DBA', # Invesco DB Agriculture Fund
            'Base_Metals': 'DBB', # Invesco DB Base Metals Fund
            'Energy': 'DBE',    # Invesco DB Energy Fund
        }
    
    def _get_etf_proxies(self) -> Dict[str, str]:
        """Get ETF proxies for commodity exposure"""
        return {
            'DJP': 'iPath Bloomberg Commodity Index',
            'DBC': 'Invesco DB Commodity Index',
            'PDBC': 'Invesco Optimum Yield Diversified Commodity',
            'GLD': 'SPDR Gold Shares',
            'SLV': 'iShares Silver Trust',
            'USO': 'United States Oil Fund',
            'UNG': 'United States Natural Gas Fund',
            'DBA': 'Invesco DB Agriculture Fund',
            'DBB': 'Invesco DB Base Metals Fund',
            'DBE': 'Invesco DB Energy Fund',
            'CORN': 'Teucrium Corn Fund',
            'WEAT': 'Teucrium Wheat Fund',
            'SOYB': 'Teucrium Soybean Fund'
        }
    
    def _get_commodity_mapping(self) -> Dict[str, Dict]:
        """Get commodity classification and properties"""
        return {
            'Gold': {
                'type': CommodityType.PRECIOUS_METALS,
                'subcategory': CommoditySubcategory.GOLD,
                'storage_intensive': True,
                'seasonal': False,
                'geopolitical_sensitive': True
            },
            'Silver': {
                'type': CommodityType.PRECIOUS_METALS,
                'subcategory': CommoditySubcategory.SILVER,
                'storage_intensive': True,
                'seasonal': False,
                'geopolitical_sensitive': True
            },
            'Crude_Oil': {
                'type': CommodityType.ENERGY,
                'subcategory': CommoditySubcategory.CRUDE_OIL,
                'storage_intensive': True,
                'seasonal': True,
                'geopolitical_sensitive': True
            },
            'Natural_Gas': {
                'type': CommodityType.ENERGY,
                'subcategory': CommoditySubcategory.NATURAL_GAS,
                'storage_intensive': True,
                'seasonal': True,
                'geopolitical_sensitive': True
            },
            'Copper': {
                'type': CommodityType.BASE_METALS,
                'subcategory': CommoditySubcategory.COPPER,
                'storage_intensive': False,
                'seasonal': False,
                'geopolitical_sensitive': False
            },
            'Corn': {
                'type': CommodityType.AGRICULTURE,
                'subcategory': CommoditySubcategory.CORN,
                'storage_intensive': True,
                'seasonal': True,
                'geopolitical_sensitive': False
            },
            'Wheat': {
                'type': CommodityType.AGRICULTURE,
                'subcategory': CommoditySubcategory.WHEAT,
                'storage_intensive': True,
                'seasonal': True,
                'geopolitical_sensitive': True
            },
            'Soybeans': {
                'type': CommodityType.AGRICULTURE,
                'subcategory': CommoditySubcategory.SOYBEANS,
                'storage_intensive': True,
                'seasonal': True,
                'geopolitical_sensitive': False
            }
        }
    
    def _get_storage_cost_data(self) -> Dict[str, float]:
        """Get annual storage costs as percentage of commodity value"""
        return {
            CommoditySubcategory.GOLD: 0.005,      # 0.5% per year
            CommoditySubcategory.SILVER: 0.008,    # 0.8% per year
            CommoditySubcategory.CRUDE_OIL: 0.15,  # 15% per year (high storage costs)
            CommoditySubcategory.NATURAL_GAS: 0.20, # 20% per year (very high)
            CommoditySubcategory.COPPER: 0.02,     # 2% per year
            CommoditySubcategory.CORN: 0.08,       # 8% per year
            CommoditySubcategory.WHEAT: 0.10,      # 10% per year
            CommoditySubcategory.SOYBEANS: 0.12,   # 12% per year
            CommoditySubcategory.PLATINUM: 0.006,  # 0.6% per year
            CommoditySubcategory.PALLADIUM: 0.006, # 0.6% per year
        }
    
    def _get_seasonal_patterns(self) -> Dict[CommoditySubcategory, Dict[int, float]]:
        """Get seasonal adjustment factors by month"""
        return {
            CommoditySubcategory.NATURAL_GAS: {
                1: 1.15, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.90, 6: 0.85,
                7: 0.85, 8: 0.90, 9: 0.95, 10: 1.05, 11: 1.10, 12: 1.15
            },
            CommoditySubcategory.CORN: {
                1: 1.02, 2: 1.01, 3: 1.00, 4: 0.98, 5: 0.96, 6: 0.95,
                7: 0.98, 8: 1.02, 9: 1.05, 10: 1.03, 11: 1.02, 12: 1.02
            },
            CommoditySubcategory.WHEAT: {
                1: 1.01, 2: 1.00, 3: 0.99, 4: 0.97, 5: 0.95, 6: 0.94,
                7: 0.96, 8: 1.00, 9: 1.03, 10: 1.02, 11: 1.01, 12: 1.01
            },
            CommoditySubcategory.SOYBEANS: {
                1: 1.00, 2: 0.99, 3: 0.98, 4: 0.97, 5: 0.96, 6: 0.95,
                7: 0.97, 8: 1.01, 9: 1.05, 10: 1.08, 11: 1.03, 12: 1.01
            },
            CommoditySubcategory.CRUDE_OIL: {
                1: 1.03, 2: 1.02, 3: 1.00, 4: 0.98, 5: 0.97, 6: 0.96,
                7: 0.97, 8: 0.98, 9: 1.00, 10: 1.02, 11: 1.03, 12: 1.04
            }
        }
    
    async def collect_futures_data(self, contracts: List[str] = None) -> List[CommodityFuture]:
        """
        Collect comprehensive futures contract data including prices, open interest, and volume.
        
        Args:
            contracts: List of futures symbols. If None, uses full universe.
            
        Returns:
            List of CommodityFuture objects with complete contract data
        """
        if contracts is None:
            contracts = list(self.futures_universe.keys())
        
        self.logger.info(f"Collecting futures data for {len(contracts)} contracts")
        
        futures_data = []
        
        for contract_symbol in contracts:
            try:
                future_data = await self._collect_single_futures_contract(contract_symbol)
                if future_data:
                    futures_data.append(future_data)
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect futures data for {contract_symbol}: {e}")
                continue
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        self.logger.info(f"Successfully collected data for {len(futures_data)} futures contracts")
        return futures_data
    
    async def _collect_single_futures_contract(self, symbol: str) -> Optional[CommodityFuture]:
        """Collect data for a single futures contract"""
        try:
            if symbol not in self.futures_universe:
                self.logger.warning(f"Unknown futures symbol: {symbol}")
                return None
            
            contract_info = self.futures_universe[symbol]
            
            # Get market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            info = ticker.info
            
            if len(hist) < 30:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate metrics
            returns = hist['Close'].pct_change().dropna()
            current_price = hist['Close'][-1]
            volume = hist['Volume'][-1] if not hist['Volume'].empty else 0
            
            # Get spot price (using ETF proxy)
            spot_price = self._get_spot_price(contract_info['subcategory'])
            
            # Calculate basis and other metrics
            basis = current_price - spot_price
            volatility_30d = returns[-30:].std() * np.sqrt(252) if len(returns) >= 30 else 0
            
            # Get storage costs and convenience yield
            storage_cost = self.storage_costs.get(contract_info['subcategory'], 0.05)
            convenience_yield = self._calculate_convenience_yield(symbol, current_price, spot_price)
            seasonal_factor = self._get_seasonal_factor(contract_info['subcategory'])
            
            # Estimate contract expiration (simplified)
            expiration_date = self._estimate_expiration_date(symbol)
            
            # Get supply/demand data
            supply_demand = await self._get_supply_demand_data(contract_info['subcategory'])
            
            # Create CommodityFuture object
            commodity_future = CommodityFuture(
                symbol=symbol,
                commodity_name=contract_info['name'],
                exchange=contract_info['exchange'],
                contract_month=self._get_contract_month(symbol),
                expiration_date=expiration_date,
                
                commodity_type=contract_info['type'],
                subcategory=contract_info['subcategory'],
                underlying_asset=contract_info['name'],
                
                contract_size=contract_info['contract_size'],
                price_unit=contract_info['price_unit'],
                minimum_tick=contract_info['tick_size'],
                tick_value=contract_info['tick_value'],
                
                storage_cost=storage_cost,
                convenience_yield=convenience_yield,
                seasonal_factor=seasonal_factor,
                
                spot_price=spot_price,
                futures_price=current_price,
                basis=basis,
                open_interest=int(info.get('openInterest', 0)),
                volume=int(volume),
                
                volatility_30d=volatility_30d,
                correlation_to_dollar=self._estimate_dollar_correlation(contract_info['subcategory']),
                correlation_to_equities=self._estimate_equity_correlation(contract_info['subcategory']),
                beta_to_commodity_index=self._estimate_commodity_beta(returns),
                
                global_production=supply_demand.get('production', 0),
                global_consumption=supply_demand.get('consumption', 0),
                inventory_levels=supply_demand.get('inventory', 0),
                geopolitical_risk_score=self._assess_geopolitical_risk(contract_info['subcategory'])
            )
            
            return commodity_future
            
        except Exception as e:
            self.logger.error(f"Error collecting futures data for {symbol}: {e}")
            return None
    
    def _get_spot_price(self, subcategory: CommoditySubcategory) -> float:
        """Get spot price for commodity using ETF proxy"""
        try:
            # Map subcategory to ETF symbol
            etf_mapping = {
                CommoditySubcategory.GOLD: 'GLD',
                CommoditySubcategory.SILVER: 'SLV',
                CommoditySubcategory.CRUDE_OIL: 'USO',
                CommoditySubcategory.NATURAL_GAS: 'UNG',
                CommoditySubcategory.COPPER: 'CPER',
                CommoditySubcategory.CORN: 'CORN',
                CommoditySubcategory.WHEAT: 'WEAT',
                CommoditySubcategory.SOYBEANS: 'SOYB'
            }
            
            etf_symbol = etf_mapping.get(subcategory)
            if not etf_symbol:
                return 0.0
            
            ticker = yf.Ticker(etf_symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                return float(hist['Close'][-1])
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error getting spot price for {subcategory}: {e}")
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Could not get spot price for {subcategory}: {e}")
        
        return 0.0
    
    def _calculate_convenience_yield(self, symbol: str, futures_price: float, spot_price: float) -> float:
        """Calculate convenience yield from futures curve"""
        if spot_price <= 0 or futures_price <= 0:
            return 0.0
        
        # Simplified convenience yield calculation
        # In practice, would use full futures curve and risk-free rate
        time_to_expiry = 0.25  # Assume 3 months
        
        if futures_price < spot_price:  # Backwardation
            return (spot_price - futures_price) / spot_price / time_to_expiry
        else:  # Contango
            return 0.0  # No convenience yield in contango
    
    def _get_seasonal_factor(self, subcategory: CommoditySubcategory) -> float:
        """Get current seasonal adjustment factor"""
        current_month = datetime.now().month
        seasonal_pattern = self.seasonal_patterns.get(subcategory, {})
        return seasonal_pattern.get(current_month, 1.0)
    
    def _estimate_expiration_date(self, symbol: str) -> datetime:
        """Estimate futures contract expiration date"""
        # Simplified: assume next quarter end
        now = datetime.now()
        
        # Find next quarter end
        quarter_ends = [
            datetime(now.year, 3, 31),
            datetime(now.year, 6, 30),
            datetime(now.year, 9, 30),
            datetime(now.year, 12, 31)
        ]
        
        for quarter_end in quarter_ends:
            if quarter_end > now:
                return quarter_end
        
        # If past all quarter ends, use next year's first quarter
        return datetime(now.year + 1, 3, 31)
    
    def _get_contract_month(self, symbol: str) -> str:
        """Extract contract month from symbol"""
        # Simplified contract month extraction
        now = datetime.now()
        return f"{now.year}-{now.month:02d}"
    
    async def _get_supply_demand_data(self, subcategory: CommoditySubcategory) -> Dict[str, float]:
        """Get fundamental supply and demand data"""
        # Simplified supply/demand data - would use actual APIs in production
        supply_demand_data = {
            CommoditySubcategory.CRUDE_OIL: {
                'production': 100000000,  # barrels/day global
                'consumption': 99000000,  # barrels/day global
                'inventory': 1500000000   # barrels in storage
            },
            CommoditySubcategory.NATURAL_GAS: {
                'production': 4000,  # bcf/day
                'consumption': 3950, # bcf/day
                'inventory': 3500    # bcf in storage
            },
            CommoditySubcategory.GOLD: {
                'production': 3300,  # tonnes per year
                'consumption': 4200, # tonnes per year
                'inventory': 190000  # tonnes above ground
            },
            CommoditySubcategory.SILVER: {
                'production': 25000,  # tonnes per year
                'consumption': 32000, # tonnes per year
                'inventory': 2000000  # tonnes above ground
            },
            CommoditySubcategory.COPPER: {
                'production': 21000000,  # tonnes per year
                'consumption': 20500000, # tonnes per year
                'inventory': 500000      # tonnes in warehouses
            },
            CommoditySubcategory.CORN: {
                'production': 1100000000,  # bushels per year
                'consumption': 1050000000, # bushels per year
                'inventory': 1400000000    # bushels in storage
            },
            CommoditySubcategory.WHEAT: {
                'production': 750000000,   # bushels per year
                'consumption': 740000000,  # bushels per year
                'inventory': 900000000     # bushels in storage
            },
            CommoditySubcategory.SOYBEANS: {
                'production': 350000000,   # bushels per year
                'consumption': 345000000,  # bushels per year
                'inventory': 250000000     # bushels in storage
            }
        }
        
        return supply_demand_data.get(subcategory, {
            'production': 0,
            'consumption': 0,
            'inventory': 0
        })
    
    def _estimate_dollar_correlation(self, subcategory: CommoditySubcategory) -> float:
        """Estimate correlation to US Dollar Index"""
        # Most commodities negatively correlated with USD
        correlations = {
            CommoditySubcategory.GOLD: -0.7,
            CommoditySubcategory.SILVER: -0.6,
            CommoditySubcategory.CRUDE_OIL: -0.5,
            CommoditySubcategory.NATURAL_GAS: -0.3,
            CommoditySubcategory.COPPER: -0.4,
            CommoditySubcategory.CORN: -0.2,
            CommoditySubcategory.WHEAT: -0.2,
            CommoditySubcategory.SOYBEANS: -0.2
        }
        
        return correlations.get(subcategory, -0.3)
    
    def _estimate_equity_correlation(self, subcategory: CommoditySubcategory) -> float:
        """Estimate correlation to equity markets"""
        correlations = {
            CommoditySubcategory.GOLD: 0.1,     # Low correlation, safe haven
            CommoditySubcategory.SILVER: 0.3,   # Moderate correlation
            CommoditySubcategory.CRUDE_OIL: 0.6, # High correlation with economy
            CommoditySubcategory.NATURAL_GAS: 0.4,
            CommoditySubcategory.COPPER: 0.7,   # Very high correlation with economy
            CommoditySubcategory.CORN: 0.2,
            CommoditySubcategory.WHEAT: 0.2,
            CommoditySubcategory.SOYBEANS: 0.3
        }
        
        return correlations.get(subcategory, 0.3)
    
    def _estimate_commodity_beta(self, returns: pd.Series) -> float:
        """Estimate beta to broad commodity index"""
        if len(returns) < 30:
            return 1.0
        
        volatility = returns.std() * np.sqrt(252)
        
        # Higher volatility commodities tend to have higher beta
        if volatility > 0.4:
            return 1.3
        elif volatility > 0.25:
            return 1.0
        else:
            return 0.8
    
    def _assess_geopolitical_risk(self, subcategory: CommoditySubcategory) -> float:
        """Assess geopolitical risk on 0-10 scale"""
        risk_scores = {
            CommoditySubcategory.CRUDE_OIL: 8.0,     # Very high
            CommoditySubcategory.NATURAL_GAS: 7.0,   # High
            CommoditySubcategory.GOLD: 3.0,          # Low-medium
            CommoditySubcategory.SILVER: 2.0,        # Low
            CommoditySubcategory.COPPER: 4.0,        # Medium
            CommoditySubcategory.PLATINUM: 5.0,      # Medium-high
            CommoditySubcategory.PALLADIUM: 6.0,     # High
            CommoditySubcategory.CORN: 3.0,          # Low-medium
            CommoditySubcategory.WHEAT: 5.0,         # Medium-high
            CommoditySubcategory.SOYBEANS: 3.0       # Low-medium
        }
        
        return risk_scores.get(subcategory, 3.0)
    
    async def collect_spot_prices(self, commodities: List[str] = None) -> pd.DataFrame:
        """
        Collect spot prices for physical commodities using ETF proxies.
        
        Args:
            commodities: List of commodity names. If None, uses full universe.
            
        Returns:
            DataFrame with spot price data
        """
        if commodities is None:
            commodities = list(self.spot_universe.keys())
        
        self.logger.info(f"Collecting spot prices for {len(commodities)} commodities")
        
        spot_data = []
        
        for commodity in commodities:
            try:
                etf_symbol = self.spot_universe.get(commodity)
                if not etf_symbol:
                    continue
                
                ticker = yf.Ticker(etf_symbol)
                hist = ticker.history(period="1y")
                info = ticker.info
                
                if len(hist) > 30:
                    returns = hist['Close'].pct_change().dropna()
                    
                    spot_record = {
                        'commodity': commodity,
                        'etf_symbol': etf_symbol,
                        'spot_price': hist['Close'][-1],
                        'daily_volume': hist['Volume'][-10:].mean(),
                        'annual_return': self._calculate_annualized_return(hist['Close']),
                        'annual_volatility': returns.std() * np.sqrt(252),
                        'max_drawdown': self._calculate_max_drawdown(hist['Close']),
                        'current_yield': info.get('yield', 0.0) or 0.0,
                        'expense_ratio': info.get('expenseRatio', 0.0) or 0.0,
                        'last_updated': datetime.now()
                    }
                    
                    spot_data.append(spot_record)
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect spot data for {commodity}: {e}")
        
        return pd.DataFrame(spot_data)
    
    def _calculate_annualized_return(self, prices: pd.Series) -> float:
        """Calculate annualized return"""
        if len(prices) < 2:
            return 0.0
        
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = len(prices) / 252
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def calculate_convenience_yield(self, commodity: str) -> float:
        """
        Calculate convenience yield from futures curve analysis.
        
        Args:
            commodity: Commodity identifier
            
        Returns:
            Estimated convenience yield
        """
        # This would analyze the full futures curve in production
        # For now, return estimated values based on commodity characteristics
        
        convenience_yields = {
            'Crude_Oil': 0.03,      # 3% convenience yield
            'Natural_Gas': 0.05,    # 5% convenience yield (high storage costs)
            'Gold': 0.0,            # No convenience yield
            'Silver': 0.0,          # No convenience yield
            'Copper': 0.02,         # 2% convenience yield
            'Corn': 0.04,           # 4% convenience yield
            'Wheat': 0.03,          # 3% convenience yield
            'Soybeans': 0.04        # 4% convenience yield
        }
        
        return convenience_yields.get(commodity, 0.02)
    
    def get_storage_costs(self, commodity: str) -> float:
        """
        Get storage costs for physical commodity holdings.
        
        Args:
            commodity: Commodity identifier
            
        Returns:
            Annual storage cost as percentage of value
        """
        # Map commodity name to subcategory
        commodity_map = {
            'Gold': CommoditySubcategory.GOLD,
            'Silver': CommoditySubcategory.SILVER,
            'Crude_Oil': CommoditySubcategory.CRUDE_OIL,
            'Natural_Gas': CommoditySubcategory.NATURAL_GAS,
            'Copper': CommoditySubcategory.COPPER,
            'Corn': CommoditySubcategory.CORN,
            'Wheat': CommoditySubcategory.WHEAT,
            'Soybeans': CommoditySubcategory.SOYBEANS
        }
        
        subcategory = commodity_map.get(commodity)
        return self.storage_costs.get(subcategory, 0.05)  # Default 5%
    
    async def collect_supply_demand_data(self, commodity: str) -> CommoditySupplyDemandData:
        """
        Collect comprehensive supply and demand data for commodity analysis.
        
        Args:
            commodity: Commodity identifier
            
        Returns:
            CommoditySupplyDemandData object with fundamental data
        """
        # Map commodity to subcategory
        commodity_mapping = {
            'Gold': CommoditySubcategory.GOLD,
            'Silver': CommoditySubcategory.SILVER,
            'Crude_Oil': CommoditySubcategory.CRUDE_OIL,
            'Natural_Gas': CommoditySubcategory.NATURAL_GAS,
            'Copper': CommoditySubcategory.COPPER,
            'Corn': CommoditySubcategory.CORN,
            'Wheat': CommoditySubcategory.WHEAT,
            'Soybeans': CommoditySubcategory.SOYBEANS
        }
        
        subcategory = commodity_mapping.get(commodity, CommoditySubcategory.CORN)
        
        # Get fundamental data (simplified - would use actual APIs)
        fundamental_data = await self._get_supply_demand_data(subcategory)
        
        # Create comprehensive supply/demand object
        supply_demand = CommoditySupplyDemandData(
            commodity=subcategory,
            
            global_production=fundamental_data.get('production', 0),
            major_producers=self._get_major_producers(subcategory),
            production_capacity=fundamental_data.get('production', 0) * 1.1,  # 10% excess capacity
            capacity_utilization=0.91,  # 91% utilization
            
            global_consumption=fundamental_data.get('consumption', 0),
            major_consumers=self._get_major_consumers(subcategory),
            demand_growth_rate=0.025,  # 2.5% annual growth
            
            global_inventory=fundamental_data.get('inventory', 0),
            strategic_reserves=fundamental_data.get('inventory', 0) * 0.15,  # 15% strategic
            commercial_inventory=fundamental_data.get('inventory', 0) * 0.85,  # 85% commercial
            inventory_to_consumption_ratio=fundamental_data.get('inventory', 0) / max(fundamental_data.get('consumption', 1), 1),
            
            market_concentration_hhi=self._calculate_hhi(subcategory),
            cartel_influence=self._has_cartel_influence(subcategory),
            trade_restrictions=self._get_trade_restrictions(subcategory)
        )
        
        return supply_demand
    
    def _get_major_producers(self, subcategory: CommoditySubcategory) -> Dict[str, float]:
        """Get major producing countries/regions"""
        producers = {
            CommoditySubcategory.CRUDE_OIL: {
                'United States': 20.0, 'Saudi Arabia': 12.0, 'Russia': 11.0, 
                'Canada': 6.0, 'Iraq': 5.0, 'China': 4.0, 'UAE': 4.0, 'Iran': 4.0
            },
            CommoditySubcategory.GOLD: {
                'China': 11.0, 'Russia': 9.0, 'Australia': 9.0, 'United States': 6.0,
                'Canada': 5.0, 'Peru': 4.0, 'Ghana': 4.0, 'South Africa': 3.0
            },
            CommoditySubcategory.COPPER: {
                'Chile': 28.0, 'Peru': 12.0, 'China': 8.0, 'United States': 6.0,
                'Australia': 5.0, 'Zambia': 4.0, 'Mexico': 3.0, 'Russia': 3.0
            },
            CommoditySubcategory.CORN: {
                'United States': 32.0, 'China': 22.0, 'Brazil': 10.0, 'Argentina': 5.0,
                'Ukraine': 4.0, 'India': 3.0, 'Mexico': 3.0, 'Romania': 2.0
            }
        }
        
        return producers.get(subcategory, {'Global': 100.0})
    
    def _get_major_consumers(self, subcategory: CommoditySubcategory) -> Dict[str, float]:
        """Get major consuming countries/regions"""
        consumers = {
            CommoditySubcategory.CRUDE_OIL: {
                'United States': 20.0, 'China': 15.0, 'India': 5.0, 'Japan': 4.0,
                'Russia': 3.0, 'Saudi Arabia': 3.0, 'Brazil': 3.0, 'Germany': 2.5
            },
            CommoditySubcategory.GOLD: {
                'China': 30.0, 'India': 20.0, 'United States': 8.0, 'Germany': 5.0,
                'Turkey': 4.0, 'Russia': 3.0, 'Iran': 3.0, 'Thailand': 2.0
            },
            CommoditySubcategory.COPPER: {
                'China': 50.0, 'United States': 8.0, 'Germany': 4.0, 'Japan': 4.0,
                'South Korea': 3.0, 'Italy': 2.5, 'Turkey': 2.0, 'India': 2.0
            },
            CommoditySubcategory.CORN: {
                'United States': 28.0, 'China': 25.0, 'EU': 8.0, 'Brazil': 6.0,
                'Mexico': 4.0, 'Japan': 3.0, 'Egypt': 2.0, 'South Korea': 2.0
            }
        }
        
        return consumers.get(subcategory, {'Global': 100.0})
    
    def _calculate_hhi(self, subcategory: CommoditySubcategory) -> float:
        """Calculate Herfindahl-Hirschman Index for market concentration"""
        # HHI ranges from 0 (perfect competition) to 10,000 (monopoly)
        hhi_values = {
            CommoditySubcategory.CRUDE_OIL: 1200,    # Moderately concentrated
            CommoditySubcategory.GOLD: 800,          # Low concentration
            CommoditySubcategory.COPPER: 1800,       # Concentrated (Chile dominance)
            CommoditySubcategory.CORN: 1500,         # Moderately concentrated
            CommoditySubcategory.NATURAL_GAS: 1000,  # Low-moderate concentration
            CommoditySubcategory.SILVER: 600,        # Low concentration
            CommoditySubcategory.WHEAT: 900,         # Low concentration
            CommoditySubcategory.SOYBEANS: 1400      # Moderately concentrated
        }
        
        return hhi_values.get(subcategory, 1000)
    
    def _has_cartel_influence(self, subcategory: CommoditySubcategory) -> bool:
        """Check if commodity has significant cartel influence"""
        cartel_commodities = {
            CommoditySubcategory.CRUDE_OIL: True,    # OPEC
            CommoditySubcategory.NATURAL_GAS: False, # No major cartel
            CommoditySubcategory.GOLD: False,        # No cartel
            CommoditySubcategory.COPPER: False,      # No cartel
            CommoditySubcategory.CORN: False,        # No cartel
        }
        
        return cartel_commodities.get(subcategory, False)
    
    def _get_trade_restrictions(self, subcategory: CommoditySubcategory) -> List[str]:
        """Get current trade restrictions affecting commodity"""
        restrictions = {
            CommoditySubcategory.CRUDE_OIL: ['Iran sanctions', 'Russia sanctions'],
            CommoditySubcategory.NATURAL_GAS: ['Russia sanctions', 'Export licensing'],
            CommoditySubcategory.GOLD: ['Central bank regulations'],
            CommoditySubcategory.COPPER: ['Export quotas (some countries)'],
            CommoditySubcategory.CORN: ['Export restrictions during shortages'],
            CommoditySubcategory.WHEAT: ['Export bans during crises'],
        }
        
        return restrictions.get(subcategory, [])


# Example usage and testing
if __name__ == "__main__":
    async def test_commodity_collector():
        collector = CommodityDataCollector()
        
        # Test futures data collection
        test_futures = ['CL=F', 'GC=F', 'SI=F', 'NG=F']
        futures_data = await collector.collect_futures_data(test_futures)
        
        print(f"Collected data for {len(futures_data)} futures contracts")
        for future in futures_data[:2]:
            print(f"\n{future.symbol} - {future.commodity_name}")
            print(f"Exchange: {future.exchange}")
            print(f"Futures Price: ${future.futures_price:.2f}")
            print(f"Spot Price: ${future.spot_price:.2f}")
            print(f"Basis: ${future.basis:.2f}")
            print(f"Storage Cost: {future.storage_cost:.2%}")
            print(f"Convenience Yield: {future.convenience_yield:.2%}")
        
        # Test spot prices
        spot_data = await collector.collect_spot_prices(['Gold', 'Oil', 'Natural_Gas'])
        print(f"\nSpot Price Data:")
        print(spot_data[['commodity', 'spot_price', 'annual_volatility']].head())
        
        # Test supply/demand data
        supply_demand = await collector.collect_supply_demand_data('Crude_Oil')
        print(f"\nCrude Oil Supply/Demand:")
        print(f"Production: {supply_demand.global_production:,.0f}")
        print(f"Consumption: {supply_demand.global_consumption:,.0f}")
        print(f"Balance Ratio: {supply_demand.calculate_supply_demand_balance():.3f}")
    
    # Run test
    asyncio.run(test_commodity_collector())
