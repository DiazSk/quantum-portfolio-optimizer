"""
International Market Data Feeds Integration
Task 2.5: GlobalPortfolioOptimizer Integration
Aggregates market data from multiple international exchanges
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Supported international exchanges"""
    NASDAQ = "NASDAQ"  # USA
    NYSE = "NYSE"      # USA
    LSE = "LSE"        # London Stock Exchange
    EURONEXT = "EURONEXT"  # European markets
    DAX = "DAX"        # German market
    TSE = "TSE"        # Tokyo Stock Exchange
    HKEX = "HKEX"      # Hong Kong Exchange
    ASX = "ASX"        # Australian Securities Exchange
    TSX = "TSX"        # Toronto Stock Exchange
    BSE = "BSE"        # Bombay Stock Exchange


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    quality_level: DataQuality
    missing_data_percentage: float
    outliers_detected: int
    last_update: datetime
    error_messages: List[str]
    warnings: List[str]


@dataclass
class MarketHours:
    """Trading hours for an exchange"""
    exchange: Exchange
    timezone: str
    trading_open: str  # HH:MM format
    trading_close: str  # HH:MM format
    lunch_break_start: Optional[str] = None
    lunch_break_end: Optional[str] = None
    pre_market_open: Optional[str] = None
    after_hours_close: Optional[str] = None


class InternationalDataFeed:
    """
    International market data aggregation and normalization
    Handles multi-exchange, multi-currency, multi-timezone data feeds
    """
    
    def __init__(self, base_currency: str = "USD"):
        self.base_currency = base_currency
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_timeout = timedelta(minutes=5)
        self._exchange_configs = self._initialize_exchange_configs()
        self._fx_rates_cache = {}
        
        self.logger.info(f"InternationalDataFeed initialized with base currency: {base_currency}")
    
    def aggregate_market_data(self, exchanges: List[str]) -> pd.DataFrame:
        """
        Aggregate market data from multiple international exchanges
        
        Args:
            exchanges: List of exchange names to aggregate data from
            
        Returns:
            DataFrame with aggregated market data across exchanges
        """
        try:
            self.logger.info(f"Aggregating market data from {len(exchanges)} exchanges: {exchanges}")
            
            # Validate exchanges
            valid_exchanges = self._validate_exchanges(exchanges)
            if not valid_exchanges:
                raise ValueError("No valid exchanges provided")
            
            # Collect data from each exchange
            exchange_data = {}
            
            with ThreadPoolExecutor(max_workers=min(len(valid_exchanges), 8)) as executor:
                # Submit data collection tasks
                futures = {
                    executor.submit(self._fetch_exchange_data, exchange): exchange 
                    for exchange in valid_exchanges
                }
                
                # Collect results
                for future in futures:
                    exchange = futures[future]
                    try:
                        data = future.result(timeout=30)  # 30 second timeout per exchange
                        if not data.empty:
                            exchange_data[exchange] = data
                            self.logger.info(f"Successfully fetched data from {exchange}: {len(data)} securities")
                        else:
                            self.logger.warning(f"No data returned from {exchange}")
                    except Exception as e:
                        self.logger.error(f"Failed to fetch data from {exchange}: {e}")
            
            if not exchange_data:
                raise ValueError("No market data could be retrieved from any exchange")
            
            # Combine and standardize data
            combined_data = self._combine_exchange_data(exchange_data)
            
            # Add metadata
            combined_data['data_timestamp'] = datetime.now()
            combined_data['base_currency'] = self.base_currency
            
            self.logger.info(f"Market data aggregation completed: {len(combined_data)} total securities")
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Market data aggregation failed: {e}")
            raise
    
    def normalize_currency_data(self, data: pd.DataFrame, base_currency: str) -> pd.DataFrame:
        """
        Normalize all currency-denominated values to base currency
        
        Args:
            data: DataFrame with multi-currency data
            base_currency: Target currency for normalization
            
        Returns:
            DataFrame with all values normalized to base currency
        """
        try:
            self.logger.info(f"Normalizing currency data to {base_currency}")
            
            if data.empty:
                self.logger.warning("Empty DataFrame provided for currency normalization")
                return data
            
            # Make a copy to avoid modifying original data
            normalized_data = data.copy()
            
            # Get current FX rates
            unique_currencies = data['currency'].unique() if 'currency' in data.columns else [base_currency]
            fx_rates = self._get_fx_rates(list(unique_currencies), base_currency)
            
            # Apply currency conversion to price-related columns
            price_columns = ['price', 'market_cap', 'volume', 'high', 'low', 'open', 'close']
            
            for column in price_columns:
                if column in normalized_data.columns:
                    # Apply FX conversion row by row
                    for idx, row in normalized_data.iterrows():
                        source_currency = row.get('currency', base_currency)
                        if source_currency != base_currency:
                            fx_rate = fx_rates.get(f"{source_currency}/{base_currency}", 1.0)
                            # Ensure proper type conversion to avoid pandas warnings
                            original_value = row[column]
                            if pd.notna(original_value):
                                normalized_data.at[idx, column] = float(original_value * fx_rate)
            
            # Update currency column
            if 'currency' in normalized_data.columns:
                normalized_data['currency'] = base_currency
            
            # Add conversion metadata
            normalized_data['fx_conversion_timestamp'] = datetime.now()
            normalized_data['original_currency'] = data.get('currency', base_currency) if 'currency' in data.columns else base_currency
            
            self.logger.info(f"Currency normalization completed for {len(normalized_data)} securities")
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Currency normalization failed: {e}")
            raise
    
    def handle_market_hours_timezone(self, exchange: str) -> Dict[str, Any]:
        """
        Get comprehensive market hours and timezone information for exchange
        
        Args:
            exchange: Exchange name
            
        Returns:
            Dictionary with market hours, timezone, and trading status information
        """
        try:
            exchange_enum = Exchange(exchange.upper())
            market_hours = self._exchange_configs.get(exchange_enum)
            
            if not market_hours:
                raise ValueError(f"Market hours configuration not found for {exchange}")
            
            # Get current time in exchange timezone
            exchange_tz = timezone.utc  # Simplified - in production would use proper timezone
            current_time = datetime.now(exchange_tz)
            
            # Convert to naive datetime for comparison
            current_time_naive = current_time.replace(tzinfo=None)
            
            # Determine if market is currently open
            is_open = self._is_market_open(market_hours, current_time_naive)
            
            # Calculate next market open/close
            next_open, next_close = self._calculate_next_trading_times(market_hours, current_time_naive)
            
            return {
                'exchange': exchange,
                'timezone': market_hours.timezone,
                'trading_hours': {
                    'open': market_hours.trading_open,
                    'close': market_hours.trading_close,
                    'lunch_break': {
                        'start': market_hours.lunch_break_start,
                        'end': market_hours.lunch_break_end
                    } if market_hours.lunch_break_start else None,
                    'extended_hours': {
                        'pre_market_open': market_hours.pre_market_open,
                        'after_hours_close': market_hours.after_hours_close
                    }
                },
                'current_status': {
                    'is_open': is_open,
                    'current_time': current_time.isoformat(),
                    'next_open': next_open.isoformat() if next_open else None,
                    'next_close': next_close.isoformat() if next_close else None
                },
                'trading_calendar': {
                    'is_trading_day': self._is_trading_day(current_time_naive),
                    'market_phase': self._get_market_phase(market_hours, current_time_naive)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error handling market hours for {exchange}: {e}")
            raise
    
    def validate_data_quality(self, data: pd.DataFrame) -> ValidationResult:
        """
        Comprehensive data quality validation
        
        Args:
            data: DataFrame to validate
            
        Returns:
            ValidationResult with quality metrics and issues
        """
        try:
            self.logger.info(f"Validating data quality for {len(data)} records")
            
            if data.empty:
                return ValidationResult(
                    is_valid=False,
                    quality_score=0.0,
                    quality_level=DataQuality.UNAVAILABLE,
                    missing_data_percentage=100.0,
                    outliers_detected=0,
                    last_update=datetime.now(),
                    error_messages=["DataFrame is empty"],
                    warnings=[]
                )
            
            errors = []
            warnings = []
            
            # Check for missing data
            missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            
            if missing_percentage > 50:
                errors.append(f"Excessive missing data: {missing_percentage:.1f}%")
            elif missing_percentage > 20:
                warnings.append(f"High missing data: {missing_percentage:.1f}%")
            
            # Check for required columns
            required_columns = ['symbol', 'price', 'exchange', 'currency']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check for outliers in price data
            outliers_detected = 0
            if 'price' in data.columns:
                Q1 = data['price'].quantile(0.25)
                Q3 = data['price'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data['price'] < lower_bound) | (data['price'] > upper_bound)]
                outliers_detected = len(outliers)
                
                if outliers_detected > len(data) * 0.1:  # More than 10% outliers
                    warnings.append(f"High number of price outliers detected: {outliers_detected}")
            
            # Check for duplicate symbols
            if 'symbol' in data.columns:
                duplicates = data['symbol'].duplicated().sum()
                if duplicates > 0:
                    warnings.append(f"Duplicate symbols found: {duplicates}")
            
            # Check data freshness
            if 'timestamp' in data.columns:
                latest_data = data['timestamp'].max()
                if isinstance(latest_data, str):
                    latest_data = pd.to_datetime(latest_data)
                
                if (datetime.now() - latest_data).total_seconds() > 3600:  # More than 1 hour old
                    warnings.append("Data appears stale (>1 hour old)")
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(missing_percentage, outliers_detected, len(errors), len(warnings))
            
            # Determine quality level
            if quality_score >= 0.9:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality_level = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality_level = DataQuality.FAIR
            elif quality_score >= 0.3:
                quality_level = DataQuality.POOR
            else:
                quality_level = DataQuality.UNAVAILABLE
            
            is_valid = len(errors) == 0 and quality_score >= 0.5
            
            result = ValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                quality_level=quality_level,
                missing_data_percentage=missing_percentage,
                outliers_detected=outliers_detected,
                last_update=datetime.now(),
                error_messages=errors,
                warnings=warnings
            )
            
            self.logger.info(f"Data validation completed - Quality: {quality_level.value} ({quality_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                quality_level=DataQuality.UNAVAILABLE,
                missing_data_percentage=100.0,
                outliers_detected=0,
                last_update=datetime.now(),
                error_messages=[f"Validation error: {str(e)}"],
                warnings=[]
            )
    
    def _initialize_exchange_configs(self) -> Dict[Exchange, MarketHours]:
        """Initialize trading hours configuration for all exchanges"""
        return {
            Exchange.NASDAQ: MarketHours(
                exchange=Exchange.NASDAQ,
                timezone="America/New_York",
                trading_open="09:30",
                trading_close="16:00",
                pre_market_open="04:00",
                after_hours_close="20:00"
            ),
            Exchange.NYSE: MarketHours(
                exchange=Exchange.NYSE,
                timezone="America/New_York",
                trading_open="09:30",
                trading_close="16:00",
                pre_market_open="04:00",
                after_hours_close="20:00"
            ),
            Exchange.LSE: MarketHours(
                exchange=Exchange.LSE,
                timezone="Europe/London",
                trading_open="08:00",
                trading_close="16:30"
            ),
            Exchange.EURONEXT: MarketHours(
                exchange=Exchange.EURONEXT,
                timezone="Europe/Paris",
                trading_open="09:00",
                trading_close="17:30"
            ),
            Exchange.DAX: MarketHours(
                exchange=Exchange.DAX,
                timezone="Europe/Berlin",
                trading_open="09:00",
                trading_close="17:30"
            ),
            Exchange.TSE: MarketHours(
                exchange=Exchange.TSE,
                timezone="Asia/Tokyo",
                trading_open="09:00",
                trading_close="15:00",
                lunch_break_start="11:30",
                lunch_break_end="12:30"
            ),
            Exchange.HKEX: MarketHours(
                exchange=Exchange.HKEX,
                timezone="Asia/Hong_Kong",
                trading_open="09:30",
                trading_close="16:00",
                lunch_break_start="12:00",
                lunch_break_end="13:00"
            ),
            Exchange.ASX: MarketHours(
                exchange=Exchange.ASX,
                timezone="Australia/Sydney",
                trading_open="10:00",
                trading_close="16:00"
            ),
            Exchange.TSX: MarketHours(
                exchange=Exchange.TSX,
                timezone="America/Toronto",
                trading_open="09:30",
                trading_close="16:00"
            ),
            Exchange.BSE: MarketHours(
                exchange=Exchange.BSE,
                timezone="Asia/Kolkata",
                trading_open="09:15",
                trading_close="15:30"
            )
        }
    
    def _validate_exchanges(self, exchanges: List[str]) -> List[Exchange]:
        """Validate and convert exchange names to enum values"""
        valid_exchanges = []
        
        for exchange_name in exchanges:
            try:
                exchange_enum = Exchange(exchange_name.upper())
                valid_exchanges.append(exchange_enum)
            except ValueError:
                self.logger.warning(f"Invalid exchange name: {exchange_name}")
        
        return valid_exchanges
    
    def _fetch_exchange_data(self, exchange: Exchange) -> pd.DataFrame:
        """Fetch market data for a specific exchange"""
        # Simulate realistic market data for each exchange
        try:
            # Generate sample securities based on exchange
            n_securities = self._get_exchange_securities_count(exchange)
            
            securities_data = []
            
            for i in range(n_securities):
                symbol = self._generate_symbol(exchange, i)
                
                security_data = {
                    'symbol': symbol,
                    'exchange': exchange.value,
                    'price': np.random.uniform(10, 500),  # Random price between $10-500
                    'currency': self._get_exchange_currency(exchange),
                    'volume': np.random.randint(100000, 10000000),
                    'market_cap': np.random.uniform(1e9, 1e12),  # $1B to $1T
                    'sector': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Energy', 'Materials']),
                    'country': self._get_exchange_country(exchange),
                    'timestamp': datetime.now(),
                    'high': None,  # Would be populated with real data
                    'low': None,
                    'open': None,
                    'close': None
                }
                
                securities_data.append(security_data)
            
            df = pd.DataFrame(securities_data)
            
            # Add some realistic variation
            df['high'] = df['price'] * np.random.uniform(1.0, 1.05, len(df))
            df['low'] = df['price'] * np.random.uniform(0.95, 1.0, len(df))
            df['open'] = df['price'] * np.random.uniform(0.98, 1.02, len(df))
            df['close'] = df['price']  # Close price same as current price
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {exchange.value}: {e}")
            return pd.DataFrame()
    
    def _combine_exchange_data(self, exchange_data: Dict[Exchange, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple exchanges into single DataFrame"""
        combined_dataframes = []
        
        for exchange, data in exchange_data.items():
            # Add exchange metadata
            data['source_exchange'] = exchange.value
            data['data_source'] = f"{exchange.value}_feed"
            combined_dataframes.append(data)
        
        if not combined_dataframes:
            return pd.DataFrame()
        
        # Concatenate all exchange data
        combined_df = pd.concat(combined_dataframes, ignore_index=True)
        
        # Remove duplicates (same symbol on multiple exchanges)
        combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='first')
        
        return combined_df
    
    def _get_fx_rates(self, currencies: List[str], base_currency: str) -> Dict[str, float]:
        """Get current FX rates for currency conversion"""
        fx_rates = {}
        
        for currency in currencies:
            if currency == base_currency:
                fx_rates[f"{currency}/{base_currency}"] = 1.0
            else:
                # Simulate realistic FX rates
                rate = np.random.uniform(0.5, 2.0)  # Random rate between 0.5-2.0
                fx_rates[f"{currency}/{base_currency}"] = rate
        
        return fx_rates
    
    def _is_market_open(self, market_hours: MarketHours, current_time: datetime) -> bool:
        """Check if market is currently open"""
        # Simplified implementation - would use proper timezone handling in production
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_minutes = current_hour * 60 + current_minute
        
        # Parse trading hours
        open_hour, open_minute = map(int, market_hours.trading_open.split(':'))
        close_hour, close_minute = map(int, market_hours.trading_close.split(':'))
        
        open_time_minutes = open_hour * 60 + open_minute
        close_time_minutes = close_hour * 60 + close_minute
        
        return open_time_minutes <= current_time_minutes <= close_time_minutes
    
    def _calculate_next_trading_times(self, market_hours: MarketHours, current_time: datetime) -> tuple:
        """Calculate next market open and close times"""
        # Simplified implementation
        today = current_time.date()
        
        open_time = datetime.combine(today, datetime.strptime(market_hours.trading_open, "%H:%M").time())
        close_time = datetime.combine(today, datetime.strptime(market_hours.trading_close, "%H:%M").time())
        
        if current_time > close_time:
            # Market closed for today, next open is tomorrow
            next_open = open_time + timedelta(days=1)
            next_close = close_time + timedelta(days=1)
        elif current_time < open_time:
            # Market not yet open today
            next_open = open_time
            next_close = close_time
        else:
            # Market currently open
            next_open = None
            next_close = close_time
        
        return next_open, next_close
    
    def _is_trading_day(self, current_time: datetime) -> bool:
        """Check if current day is a trading day (simplified)"""
        # Exclude weekends (Saturday=5, Sunday=6)
        return current_time.weekday() < 5
    
    def _get_market_phase(self, market_hours: MarketHours, current_time: datetime) -> str:
        """Get current market phase (pre-market, trading, after-hours, closed)"""
        if not self._is_trading_day(current_time):
            return "closed_weekend"
        
        if self._is_market_open(market_hours, current_time):
            return "trading"
        
        # Simplified phase detection
        current_hour = current_time.hour
        open_hour = int(market_hours.trading_open.split(':')[0])
        close_hour = int(market_hours.trading_close.split(':')[0])
        
        if current_hour < open_hour:
            return "pre_market"
        elif current_hour > close_hour:
            return "after_hours"
        else:
            return "closed"
    
    def _calculate_quality_score(self, missing_percentage: float, outliers: int, errors: int, warnings: int) -> float:
        """Calculate overall data quality score"""
        base_score = 1.0
        
        # Penalize missing data
        base_score -= (missing_percentage / 100) * 0.5
        
        # Penalize errors heavily
        base_score -= errors * 0.2
        
        # Penalize warnings lightly
        base_score -= warnings * 0.05
        
        # Penalize outliers
        base_score -= min(outliers * 0.01, 0.3)  # Cap outlier penalty at 0.3
        
        return max(0.0, base_score)
    
    def _get_exchange_securities_count(self, exchange: Exchange) -> int:
        """Get typical number of securities per exchange"""
        counts = {
            Exchange.NASDAQ: 15,
            Exchange.NYSE: 12,
            Exchange.LSE: 10,
            Exchange.EURONEXT: 8,
            Exchange.DAX: 6,
            Exchange.TSE: 8,
            Exchange.HKEX: 6,
            Exchange.ASX: 5,
            Exchange.TSX: 5,
            Exchange.BSE: 4
        }
        return counts.get(exchange, 5)
    
    def _generate_symbol(self, exchange: Exchange, index: int) -> str:
        """Generate realistic symbols for each exchange"""
        prefixes = {
            Exchange.NASDAQ: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM", "INTC", "AMD", "ORCL", "CSCO", "IBM"],
            Exchange.NYSE: ["JPM", "JNJ", "PG", "UNH", "HD", "V", "MA", "DIS", "WMT", "PFE", "BAC", "KO"],
            Exchange.LSE: ["SHEL", "AZN", "BP", "VOD", "HSBA", "LLOY", "BT-A", "RIO", "BAT", "GSK"],
            Exchange.EURONEXT: ["ASML", "SAP", "OR", "SAN", "MC", "AI", "SU", "BNP"],
            Exchange.DAX: ["SAP", "SIE", "ALV", "DTE", "MUV2", "BMW"],
            Exchange.TSE: ["7203", "9984", "6758", "9432", "8035", "4063", "6861", "6367"],
            Exchange.HKEX: ["0700", "0005", "1299", "0941", "2318", "3690"],
            Exchange.ASX: ["BHP", "CBA", "CSL", "WBC", "ANZ"],
            Exchange.TSX: ["SHOP", "RY", "TD", "CNR", "ABX"],
            Exchange.BSE: ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
        }
        
        symbols = prefixes.get(exchange, [f"SYM{i}" for i in range(20)])
        return symbols[index % len(symbols)]
    
    def _get_exchange_currency(self, exchange: Exchange) -> str:
        """Get primary currency for exchange"""
        currencies = {
            Exchange.NASDAQ: "USD",
            Exchange.NYSE: "USD",
            Exchange.LSE: "GBP",
            Exchange.EURONEXT: "EUR",
            Exchange.DAX: "EUR",
            Exchange.TSE: "JPY",
            Exchange.HKEX: "HKD",
            Exchange.ASX: "AUD",
            Exchange.TSX: "CAD",
            Exchange.BSE: "INR"
        }
        return currencies.get(exchange, "USD")
    
    def _get_exchange_country(self, exchange: Exchange) -> str:
        """Get primary country for exchange"""
        countries = {
            Exchange.NASDAQ: "USA",
            Exchange.NYSE: "USA",
            Exchange.LSE: "UK",
            Exchange.EURONEXT: "Netherlands",
            Exchange.DAX: "Germany",
            Exchange.TSE: "Japan",
            Exchange.HKEX: "Hong Kong",
            Exchange.ASX: "Australia",
            Exchange.TSX: "Canada",
            Exchange.BSE: "India"
        }
        return countries.get(exchange, "Unknown")


# Convenience functions for external use
def aggregate_market_data(exchanges: List[str]) -> pd.DataFrame:
    """Convenience function for market data aggregation"""
    feed = InternationalDataFeed()
    return feed.aggregate_market_data(exchanges)


def normalize_currency_data(data: pd.DataFrame, base_currency: str) -> pd.DataFrame:
    """Convenience function for currency normalization"""
    feed = InternationalDataFeed(base_currency)
    return feed.normalize_currency_data(data, base_currency)


def handle_market_hours_timezone(exchange: str) -> Dict[str, Any]:
    """Convenience function for market hours handling"""
    feed = InternationalDataFeed()
    return feed.handle_market_hours_timezone(exchange)


def validate_data_quality(data: pd.DataFrame) -> ValidationResult:
    """Convenience function for data quality validation"""
    feed = InternationalDataFeed()
    return feed.validate_data_quality(data)


if __name__ == "__main__":
    # Example usage
    feed = InternationalDataFeed()
    
    # Test market data aggregation
    exchanges = ["NASDAQ", "LSE", "EURONEXT", "TSE", "HKEX", "ASX", "TSX", "BSE"]
    market_data = feed.aggregate_market_data(exchanges)
    print(f"Aggregated data from {len(exchanges)} exchanges: {len(market_data)} securities")
    
    # Test currency normalization
    normalized_data = feed.normalize_currency_data(market_data, "USD")
    print(f"Currency normalized to USD: {len(normalized_data)} securities")
    
    # Test market hours
    for exchange in exchanges[:3]:  # Test first 3 exchanges
        hours_info = feed.handle_market_hours_timezone(exchange)
        print(f"{exchange} market status: {hours_info['current_status']['is_open']}")
    
    # Test data quality
    validation_result = feed.validate_data_quality(market_data)
    print(f"Data quality: {validation_result.quality_level.value} (score: {validation_result.quality_score:.2f})")
