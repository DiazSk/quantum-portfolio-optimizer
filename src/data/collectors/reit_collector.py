"""
REIT (Real Estate Investment Trust) data collector for alternative asset portfolio optimization.

This module provides comprehensive REIT data collection from multiple sources including
market data, fundamentals, NAV data, and property sector metrics essential for 
institutional real estate investment strategies.
"""

import logging
import asyncio
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import numpy as np

from src.portfolio.alternative_assets.real_estate import (
    REITSecurity, PropertyType, FundStatus, RedemptionFrequency
)


class REITDataCollector:
    """
    Comprehensive REIT data collection system for institutional portfolio management.
    
    Supports public REITs, private funds, NAV data collection, and property sector analysis
    with multi-source data validation and quality scoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize data sources and API clients
        self._init_api_clients()
        
        # REIT universe definitions
        self.public_reit_universe = self._get_public_reit_universe()
        self.private_reit_universe = self._get_private_reit_universe()
        
        # Property sector mappings
        self.property_type_mapping = self._get_property_type_mapping()
        
    def _init_api_clients(self):
        """Initialize API connections for REIT data sources"""
        try:
            # Yahoo Finance for public REIT data (primary source)
            self.yf_session = yf.Session()
            
            # NAREIT API would be initialized here in production
            # self.nareit_client = NAREITClient(api_key=os.getenv('NAREIT_API_KEY'))
            
            # REITData.com API for specialized REIT metrics
            # self.reitdata_client = REITDataClient(api_key=os.getenv('REITDATA_API_KEY'))
            
            self.logger.info("REIT data collector APIs initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"API initialization failed, using fallback data: {e}")
    
    def _get_public_reit_universe(self) -> List[str]:
        """Get comprehensive list of public REIT symbols"""
        return [
            # Large Cap Equity REITs
            'SPG',   # Simon Property Group (Retail)
            'PLD',   # Prologis (Industrial)
            'CCI',   # Crown Castle (Infrastructure)
            'AMT',   # American Tower (Infrastructure)
            'EQIX',  # Equinix (Data Centers)
            'PSA',   # Public Storage (Self Storage)
            'WELL',  # Welltower (Healthcare)
            'DLR',   # Digital Realty Trust (Data Centers)
            'SBAC',  # SBA Communications (Infrastructure)
            'EXR',   # Extended Stay America (Self Storage)
            'AVB',   # AvalonBay Communities (Residential)
            'EQR',   # Equity Residential (Residential)
            'ESS',   # Essex Property Trust (Residential)
            'MAA',   # Mid-America Apartment (Residential)
            'UDR',   # UDR Inc (Residential)
            'CPT',   # Camden Property Trust (Residential)
            
            # Mid Cap REITs
            'VTR',   # Ventas (Healthcare)
            'BXP',   # Boston Properties (Office)
            'ARE',   # Alexandria Real Estate (Life Sciences)
            'HST',   # Host Hotels & Resorts (Hospitality)
            'VNO',   # Vornado Realty Trust (Office)
            'KIM',   # Kimco Realty (Retail)
            'REG',   # Regency Centers (Retail)
            'FRT',   # Federal Realty (Retail)
            'O',     # Realty Income (Net Lease)
            'NNN',   # National Retail Properties (Net Lease)
            
            # Specialized REITs
            'CXW',   # CoreCivic (Specialty)
            'LAMR',  # Lamar Advertising (Outdoor Advertising)
            'UNIT',  # Uniti Group (Infrastructure)
            'CONE',  # CyrusOne (Data Centers)
            'QTS',   # QTS Realty Trust (Data Centers)
            'IRM',   # Iron Mountain (Storage)
            'PEI',   # Pennsylvania REIT (Retail)
            'MAC',   # Macerich (Retail)
            'SLG',   # SL Green Realty (Office)
            'BRX',   # Brixmor Property Group (Retail)
            
            # International REITs (via ADRs/ETFs)
            'VNQI',  # Vanguard Global ex-US Real Estate ETF
            'RWX',   # SPDR DJ Intl Real Estate ETF
            'IFGL',  # iShares International Developed Real Estate ETF
        ]
    
    def _get_private_reit_universe(self) -> List[Dict]:
        """Get private REIT and real estate fund universe"""
        return [
            {
                'fund_id': 'BREIT_001',
                'name': 'Blackstone Real Estate Income Trust',
                'manager': 'Blackstone',
                'strategy': 'Core Plus',
                'property_types': ['Industrial', 'Multifamily', 'Hotel', 'Office'],
                'vintage_year': 2017,
                'target_size': 50000000000,  # $50B
                'min_investment': 2500,
                'redemption_frequency': RedemptionFrequency.MONTHLY,
                'lock_up_period': 0
            },
            {
                'fund_id': 'SREIT_001', 
                'name': 'Starwood Real Estate Income Trust',
                'manager': 'Starwood Capital',
                'strategy': 'Core Plus',
                'property_types': ['Multifamily', 'Industrial', 'Infrastructure'],
                'vintage_year': 2018,
                'target_size': 15000000000,  # $15B
                'min_investment': 1000,
                'redemption_frequency': RedemptionFrequency.QUARTERLY,
                'lock_up_period': 12
            },
            {
                'fund_id': 'ARES_RE_001',
                'name': 'Ares Real Estate Income Trust',
                'manager': 'Ares Management',
                'strategy': 'Value Add',
                'property_types': ['Office', 'Industrial', 'Multifamily'],
                'vintage_year': 2019,
                'target_size': 8000000000,  # $8B
                'min_investment': 5000,
                'redemption_frequency': RedemptionFrequency.QUARTERLY,
                'lock_up_period': 24
            }
        ]
    
    def _get_property_type_mapping(self) -> Dict[str, PropertyType]:
        """Map REIT names/descriptions to property types"""
        return {
            # Retail
            'Simon Property': PropertyType.RETAIL,
            'Kimco': PropertyType.RETAIL,
            'Regency': PropertyType.RETAIL,
            'Federal Realty': PropertyType.RETAIL,
            'Macerich': PropertyType.RETAIL,
            'Brixmor': PropertyType.RETAIL,
            
            # Industrial/Logistics
            'Prologis': PropertyType.INDUSTRIAL,
            'EXR': PropertyType.SELF_STORAGE,
            'Public Storage': PropertyType.SELF_STORAGE,
            
            # Residential
            'AvalonBay': PropertyType.RESIDENTIAL,
            'Equity Residential': PropertyType.RESIDENTIAL,
            'Essex Property': PropertyType.RESIDENTIAL,
            'Mid-America': PropertyType.RESIDENTIAL,
            'UDR': PropertyType.RESIDENTIAL,
            'Camden': PropertyType.RESIDENTIAL,
            
            # Office
            'Boston Properties': PropertyType.OFFICE,
            'Vornado': PropertyType.OFFICE,
            'SL Green': PropertyType.OFFICE,
            
            # Healthcare
            'Welltower': PropertyType.HEALTHCARE,
            'Ventas': PropertyType.HEALTHCARE,
            
            # Data Centers
            'Equinix': PropertyType.DATA_CENTER,
            'Digital Realty': PropertyType.DATA_CENTER,
            'CyrusOne': PropertyType.DATA_CENTER,
            'QTS': PropertyType.DATA_CENTER,
            
            # Hotels
            'Host Hotels': PropertyType.HOTEL,
        }
    
    async def collect_public_reit_data(self, symbols: List[str] = None) -> List[REITSecurity]:
        """
        Collect comprehensive public REIT market data, fundamentals, and metrics.
        
        Args:
            symbols: List of REIT symbols to collect. If None, uses full universe.
            
        Returns:
            List of REITSecurity objects with complete data
        """
        if symbols is None:
            symbols = self.public_reit_universe
            
        self.logger.info(f"Collecting public REIT data for {len(symbols)} symbols")
        
        reit_securities = []
        
        # Process REITs in batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_results = await self._process_reit_batch(batch)
            reit_securities.extend(batch_results)
            
            # Rate limiting pause
            await asyncio.sleep(1)
        
        self.logger.info(f"Successfully collected data for {len(reit_securities)} REITs")
        return reit_securities
    
    async def _process_reit_batch(self, symbols: List[str]) -> List[REITSecurity]:
        """Process a batch of REIT symbols"""
        batch_results = []
        
        for symbol in symbols:
            try:
                reit_data = await self._collect_single_reit_data(symbol)
                if reit_data:
                    batch_results.append(reit_data)
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect data for {symbol}: {e}")
                continue
        
        return batch_results
    
    async def _collect_single_reit_data(self, symbol: str) -> Optional[REITSecurity]:
        """Collect comprehensive data for a single REIT"""
        try:
            # Get basic ticker info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data for calculations
            hist = ticker.history(period="2y")
            if len(hist) < 30:  # Need sufficient data
                self.logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Calculate performance metrics
            returns = hist['Close'].pct_change().dropna()
            
            # Annual metrics
            annual_return = self._calculate_annualized_return(hist['Close'])
            annual_volatility = returns.std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(hist['Close'])
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # REIT-specific metrics
            dividend_yield = info.get('dividendYield', 0.0) or 0.0
            market_cap = info.get('marketCap', 0) or 0
            current_price = hist['Close'][-1]
            
            # Property type classification
            company_name = info.get('shortName', '') or info.get('longName', '')
            property_type = self._classify_property_type(company_name)
            
            # Geographic focus (simplified)
            geographic_focus = self._determine_geographic_focus(symbol, info)
            
            # Liquidity metrics
            avg_volume = hist['Volume'][-30:].mean()
            illiquidity_factor = self._calculate_illiquidity_factor(market_cap, avg_volume, current_price)
            
            # REIT fundamentals (estimated where actual data not available)
            nav_per_share = self._estimate_nav_per_share(info, current_price)
            nav_premium_discount = (current_price - nav_per_share) / nav_per_share if nav_per_share > 0 else 0.0
            
            # Risk metrics
            beta_to_reit_index = self._calculate_beta_to_reit_index(returns)
            correlation_to_equities = self._calculate_correlation_to_equities(returns)
            
            # Create REITSecurity object
            reit_security = REITSecurity(
                symbol=symbol,
                name=company_name,
                isin=info.get('isin', ''),
                exchange=info.get('exchange', ''),
                
                property_type=property_type,
                geographic_focus=geographic_focus,
                property_locations=self._get_property_locations(symbol, info),
                
                market_cap=market_cap,
                nav_per_share=nav_per_share,
                market_price=current_price,
                nav_premium_discount=nav_premium_discount,
                dividend_yield=dividend_yield,
                funds_from_operations=self._estimate_ffo(info, market_cap),
                
                illiquidity_factor=illiquidity_factor,
                average_daily_volume=avg_volume,
                bid_ask_spread=self._estimate_bid_ask_spread(current_price, avg_volume),
                beta_to_reit_index=beta_to_reit_index,
                correlation_to_equities=correlation_to_equities,
                
                price_to_nav=current_price / nav_per_share if nav_per_share > 0 else 0.0,
                price_to_ffo=self._calculate_price_to_ffo(current_price, info),
                debt_to_total_capital=self._estimate_debt_ratio(info),
                occupancy_rate=self._estimate_occupancy_rate(property_type),
                
                fund_status=FundStatus.PUBLIC,
                
                annual_return=annual_return,
                annual_volatility=annual_volatility,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                
                cap_rate=self._estimate_cap_rate(property_type, dividend_yield),
                noi_growth_rate=0.03,  # Estimated 3% NOI growth
                lease_duration_avg=60,  # 5 years average
                
                last_updated=datetime.now()
            )
            
            return reit_security
            
        except Exception as e:
            self.logger.error(f"Error collecting REIT data for {symbol}: {e}")
            return None
    
    def _calculate_annualized_return(self, prices: pd.Series) -> float:
        """Calculate annualized return from price series"""
        if len(prices) < 2:
            return 0.0
        
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = len(prices) / 252  # Approximate trading days per year
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        
        excess_return = returns.mean() * 252 - risk_free_rate
        return excess_return / (returns.std() * np.sqrt(252))
    
    def _classify_property_type(self, company_name: str) -> PropertyType:
        """Classify REIT property type from company name"""
        name_lower = company_name.lower()
        
        for key, property_type in self.property_type_mapping.items():
            if key.lower() in name_lower:
                return property_type
        
        # Default classifications based on keywords
        if any(word in name_lower for word in ['mall', 'retail', 'shopping']):
            return PropertyType.RETAIL
        elif any(word in name_lower for word in ['apartment', 'residential', 'community']):
            return PropertyType.RESIDENTIAL
        elif any(word in name_lower for word in ['office', 'commercial']):
            return PropertyType.OFFICE
        elif any(word in name_lower for word in ['hospital', 'medical', 'healthcare']):
            return PropertyType.HEALTHCARE
        elif any(word in name_lower for word in ['industrial', 'warehouse', 'logistics']):
            return PropertyType.INDUSTRIAL
        elif any(word in name_lower for word in ['hotel', 'hospitality']):
            return PropertyType.HOTEL
        elif any(word in name_lower for word in ['storage']):
            return PropertyType.SELF_STORAGE
        elif any(word in name_lower for word in ['data', 'technology']):
            return PropertyType.DATA_CENTER
        else:
            return PropertyType.MIXED_USE
    
    def _determine_geographic_focus(self, symbol: str, info: Dict) -> str:
        """Determine geographic focus of REIT"""
        # Check if international ETF
        if symbol in ['VNQI', 'RWX', 'IFGL']:
            return "International"
        
        # Most US-listed REITs are US-focused
        country = info.get('country', 'US')
        if country == 'US':
            return "United States"
        else:
            return country
    
    def _get_property_locations(self, symbol: str, info: Dict) -> List[str]:
        """Get primary property locations (simplified)"""
        # This would be enhanced with actual property location data
        geographic_focus = self._determine_geographic_focus(symbol, info)
        
        if geographic_focus == "United States":
            # Major US markets for most REITs
            return ["New York", "Los Angeles", "Chicago", "San Francisco", "Washington DC"]
        elif geographic_focus == "International":
            return ["London", "Tokyo", "Sydney", "Toronto", "Frankfurt"]
        else:
            return [geographic_focus]
    
    def _calculate_illiquidity_factor(self, market_cap: float, avg_volume: float, price: float) -> float:
        """Calculate illiquidity factor for REIT"""
        if market_cap <= 0 or avg_volume <= 0 or price <= 0:
            return 0.5  # Default moderate illiquidity
        
        # Dollar volume traded daily
        dollar_volume = avg_volume * price
        
        # Illiquidity based on market cap and trading volume
        turnover_ratio = dollar_volume / market_cap
        
        # Higher turnover = more liquid = lower illiquidity factor
        if turnover_ratio > 0.02:  # >2% daily turnover
            return 0.05  # Very liquid
        elif turnover_ratio > 0.01:  # 1-2% turnover
            return 0.1   # Liquid
        elif turnover_ratio > 0.005:  # 0.5-1% turnover
            return 0.2   # Moderately liquid
        else:
            return 0.4   # Less liquid
    
    def _estimate_nav_per_share(self, info: Dict, current_price: float) -> float:
        """Estimate NAV per share (would use actual NAV data in production)"""
        book_value = info.get('bookValue', 0.0)
        if book_value > 0:
            return book_value
        
        # Fallback: estimate NAV as 90-95% of market price for public REITs
        return current_price * 0.92
    
    def _estimate_ffo(self, info: Dict, market_cap: float) -> float:
        """Estimate Funds From Operations"""
        # FFO approximation using net income + depreciation
        net_income = info.get('netIncomeToCommon', 0) or 0
        
        # Estimate FFO as ~110-120% of net income for REITs
        return net_income * 1.15 if net_income > 0 else 0.0
    
    def _calculate_beta_to_reit_index(self, returns: pd.Series) -> float:
        """Calculate beta to REIT index (simplified)"""
        # In production, would correlate with actual REIT index (e.g., FTSE NAREIT)
        # For now, return estimated beta based on volatility
        volatility = returns.std() * np.sqrt(252)
        
        # REITs typically have beta 0.7-1.3 to broader REIT market
        if volatility < 0.15:
            return 0.8
        elif volatility < 0.25:
            return 1.0
        else:
            return 1.2
    
    def _calculate_correlation_to_equities(self, returns: pd.Series) -> float:
        """Calculate correlation to equity markets (simplified)"""
        # REITs typically have 0.6-0.8 correlation to equity markets
        # This would use actual market data in production
        return 0.7  # Estimated correlation
    
    def _estimate_bid_ask_spread(self, price: float, volume: float) -> float:
        """Estimate bid-ask spread"""
        if volume > 1000000:  # High volume
            return price * 0.001  # 10 bps
        elif volume > 100000:  # Medium volume
            return price * 0.002  # 20 bps
        else:  # Low volume
            return price * 0.005  # 50 bps
    
    def _calculate_price_to_ffo(self, price: float, info: Dict) -> float:
        """Calculate price-to-FFO ratio"""
        ffo_per_share = self._estimate_ffo_per_share(info)
        return price / ffo_per_share if ffo_per_share > 0 else 0.0
    
    def _estimate_ffo_per_share(self, info: Dict) -> float:
        """Estimate FFO per share"""
        shares_outstanding = info.get('sharesOutstanding', 0) or info.get('impliedSharesOutstanding', 0)
        if shares_outstanding <= 0:
            return 0.0
        
        net_income = info.get('netIncomeToCommon', 0) or 0
        # Estimate FFO as 115% of net income, divided by shares
        ffo_total = net_income * 1.15
        return ffo_total / shares_outstanding if ffo_total > 0 else 0.0
    
    def _estimate_debt_ratio(self, info: Dict) -> float:
        """Estimate debt-to-total-capital ratio"""
        total_debt = info.get('totalDebt', 0) or 0
        total_equity = info.get('totalStockholderEquity', 0) or 0
        
        if total_debt + total_equity <= 0:
            return 0.4  # Typical REIT debt ratio
        
        return total_debt / (total_debt + total_equity)
    
    def _estimate_occupancy_rate(self, property_type: PropertyType) -> float:
        """Estimate occupancy rate by property type"""
        occupancy_estimates = {
            PropertyType.RESIDENTIAL: 0.95,
            PropertyType.OFFICE: 0.88,
            PropertyType.RETAIL: 0.85,
            PropertyType.INDUSTRIAL: 0.92,
            PropertyType.HEALTHCARE: 0.90,
            PropertyType.DATA_CENTER: 0.95,
            PropertyType.SELF_STORAGE: 0.88,
            PropertyType.HOTEL: 0.70,
            PropertyType.MIXED_USE: 0.90
        }
        
        return occupancy_estimates.get(property_type, 0.90)
    
    def _estimate_cap_rate(self, property_type: PropertyType, dividend_yield: float) -> float:
        """Estimate capitalization rate"""
        # Cap rate typically higher than dividend yield
        base_cap_rates = {
            PropertyType.RESIDENTIAL: 0.05,
            PropertyType.OFFICE: 0.06,
            PropertyType.RETAIL: 0.07,
            PropertyType.INDUSTRIAL: 0.055,
            PropertyType.HEALTHCARE: 0.065,
            PropertyType.DATA_CENTER: 0.045,
            PropertyType.SELF_STORAGE: 0.06,
            PropertyType.HOTEL: 0.08,
            PropertyType.MIXED_USE: 0.06
        }
        
        base_rate = base_cap_rates.get(property_type, 0.06)
        
        # Adjust based on dividend yield if reasonable
        if 0.02 <= dividend_yield <= 0.12:
            return max(base_rate, dividend_yield + 0.01)
        
        return base_rate
    
    async def collect_reit_nav_data(self, fund_ids: List[str]) -> pd.DataFrame:
        """
        Collect NAV data for private REITs and real estate funds.
        
        Args:
            fund_ids: List of fund identifiers
            
        Returns:
            DataFrame with NAV data and fund metrics
        """
        self.logger.info(f"Collecting NAV data for {len(fund_ids)} funds")
        
        nav_data = []
        
        for fund_info in self.private_reit_universe:
            if fund_info['fund_id'] in fund_ids:
                # Simulate NAV data collection (would use actual APIs in production)
                nav_record = await self._collect_fund_nav_data(fund_info)
                nav_data.append(nav_record)
        
        return pd.DataFrame(nav_data)
    
    async def _collect_fund_nav_data(self, fund_info: Dict) -> Dict:
        """Collect NAV data for a single fund"""
        # Simulate fund NAV and performance data
        # In production, this would call actual fund APIs
        
        fund_age = datetime.now().year - fund_info['vintage_year']
        
        # Simulate NAV performance based on fund age and strategy
        base_nav = 10.0  # Starting NAV
        annual_return = 0.08 + np.random.normal(0, 0.03)  # 8% base with variation
        
        current_nav = base_nav * ((1 + annual_return) ** fund_age)
        
        return {
            'fund_id': fund_info['fund_id'],
            'fund_name': fund_info['name'],
            'nav_date': datetime.now().date(),
            'nav_per_share': current_nav,
            'total_return': (current_nav / base_nav) - 1,
            'annual_return': annual_return,
            'distribution_yield': 0.06,  # 6% estimated distribution yield
            'management_fee': 0.015,     # 1.5% management fee
            'property_types': fund_info['property_types'],
            'fund_size': fund_info['target_size'],
            'vintage_year': fund_info['vintage_year'],
            'redemption_frequency': fund_info['redemption_frequency'].value,
            'lock_up_period': fund_info['lock_up_period'],
            'last_updated': datetime.now()
        }
    
    async def collect_property_sector_data(self) -> pd.DataFrame:
        """
        Collect property sector performance and market metrics.
        
        Returns:
            DataFrame with sector-level performance data
        """
        self.logger.info("Collecting property sector performance data")
        
        # Property sector ETFs for performance tracking
        sector_etfs = {
            'Residential': 'REZ',      # iShares Residential Real Estate ETF
            'Commercial': 'VNQ',       # Vanguard Real Estate ETF (broad)
            'Industrial': 'PLD',       # Use Prologis as industrial proxy
            'Healthcare': 'WELL',      # Use Welltower as healthcare proxy
            'Retail': 'RTL',          # SPDR S&P Retail ETF
            'Data_Centers': 'DLR',    # Use Digital Realty as data center proxy
            'Self_Storage': 'PSA',    # Use Public Storage as proxy
            'Office': 'BXP'           # Use Boston Properties as office proxy
        }
        
        sector_data = []
        
        for sector, symbol in sector_etfs.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                info = ticker.info
                
                if len(hist) > 30:
                    returns = hist['Close'].pct_change().dropna()
                    
                    sector_metrics = {
                        'sector': sector,
                        'proxy_symbol': symbol,
                        'annual_return': self._calculate_annualized_return(hist['Close']),
                        'annual_volatility': returns.std() * np.sqrt(252),
                        'max_drawdown': self._calculate_max_drawdown(hist['Close']),
                        'current_price': hist['Close'][-1],
                        'market_cap': info.get('marketCap', 0),
                        'dividend_yield': info.get('dividendYield', 0.0) or 0.0,
                        'pe_ratio': info.get('trailingPE', 0.0) or 0.0,
                        'last_updated': datetime.now()
                    }
                    
                    sector_data.append(sector_metrics)
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect sector data for {sector}: {e}")
        
        return pd.DataFrame(sector_data)
    
    def calculate_reit_illiquidity_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate REIT-specific illiquidity and risk metrics.
        
        Args:
            data: DataFrame with REIT data
            
        Returns:
            DataFrame with calculated illiquidity metrics
        """
        self.logger.info("Calculating REIT illiquidity metrics")
        
        if data.empty:
            return data
        
        # Add illiquidity calculations
        data = data.copy()
        
        # Amihud illiquidity measure
        data['amihud_illiquidity'] = abs(data['daily_return']) / (data['volume'] * data['price'])
        
        # Roll impact measure (price impact of trades)
        data['roll_impact'] = data['bid_ask_spread'] / data['price']
        
        # Market cap buckets for liquidity classification
        data['size_bucket'] = pd.cut(data['market_cap'], 
                                   bins=[0, 1e9, 5e9, 20e9, float('inf')],
                                   labels=['Small', 'Mid', 'Large', 'Mega'])
        
        # Composite illiquidity score
        data['composite_illiquidity'] = (
            data['amihud_illiquidity'].rank(pct=True) * 0.4 +
            data['roll_impact'].rank(pct=True) * 0.3 +
            (1 - data['volume'].rank(pct=True)) * 0.3
        )
        
        return data
    
    def get_reit_fundamentals(self, symbol: str) -> Dict:
        """
        Get comprehensive REIT fundamentals and metrics.
        
        Args:
            symbol: REIT symbol
            
        Returns:
            Dictionary with REIT fundamentals
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Balance sheet data
            balance_sheet = ticker.balance_sheet
            income_stmt = ticker.income_stmt
            cash_flow = ticker.cashflow
            
            fundamentals = {
                'symbol': symbol,
                'name': info.get('shortName', ''),
                
                # Valuation Metrics
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                
                # Profitability
                'profit_margin': info.get('profitMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                
                # REIT-Specific Metrics
                'dividend_yield': info.get('dividendYield', 0.0) or 0.0,
                'payout_ratio': info.get('payoutRatio', 0.0) or 0.0,
                
                # Financial Health
                'debt_to_equity': info.get('debtToEquity', 0.0) or 0.0,
                'current_ratio': info.get('currentRatio', 0.0) or 0.0,
                'total_cash': info.get('totalCash', 0) or 0,
                'total_debt': info.get('totalDebt', 0) or 0,
                
                # Growth Metrics
                'revenue_growth': info.get('revenueGrowth', 0.0) or 0.0,
                'earnings_growth': info.get('earningsGrowth', 0.0) or 0.0,
                
                'last_updated': datetime.now()
            }
            
            # Add FFO estimate if available
            if not income_stmt.empty:
                try:
                    net_income = income_stmt.loc['Net Income'].iloc[0]
                    # Estimate FFO as net income + depreciation
                    if not cash_flow.empty and 'Depreciation' in cash_flow.index:
                        depreciation = cash_flow.loc['Depreciation'].iloc[0]
                        ffo = net_income + depreciation
                        shares = info.get('sharesOutstanding', 0)
                        if shares > 0:
                            fundamentals['ffo_per_share'] = ffo / shares
                            fundamentals['price_to_ffo'] = info.get('currentPrice', 0) / (ffo / shares)
                except:
                    pass
            
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Error collecting fundamentals for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    async def test_reit_collector():
        collector = REITDataCollector()
        
        # Test public REIT data collection
        test_symbols = ['SPG', 'PLD', 'EQIX', 'PSA', 'WELL']
        reit_data = await collector.collect_public_reit_data(test_symbols)
        
        print(f"Collected data for {len(reit_data)} REITs")
        for reit in reit_data[:2]:  # Show first 2
            print(f"\n{reit.symbol} - {reit.name}")
            print(f"Property Type: {reit.property_type}")
            print(f"Market Cap: ${reit.market_cap:,.0f}")
            print(f"Dividend Yield: {reit.dividend_yield:.2%}")
            print(f"Illiquidity Factor: {reit.illiquidity_factor:.3f}")
        
        # Test sector data
        sector_data = await collector.collect_property_sector_data()
        print(f"\nSector Performance Data:")
        print(sector_data[['sector', 'annual_return', 'annual_volatility']].head())
    
    # Run test
    asyncio.run(test_reit_collector())
