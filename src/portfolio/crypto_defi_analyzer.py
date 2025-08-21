"""
Cryptocurrency and DeFi Integration Module

Comprehensive cryptocurrency and decentralized finance integration including:
- Major cryptocurrency markets (Bitcoin, Ethereum, top altcoins)
- DeFi protocol analysis and yield farming opportunities
- Crypto correlation analysis with traditional assets
- Cryptocurrency risk modeling with volatility clustering
- Staking rewards and yield optimization

Business Value:
- Access to $2.5T+ cryptocurrency market
- DeFi yield farming strategies
- Digital asset portfolio optimization
- Modern portfolio diversification capabilities
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import time

from src.portfolio.alternative_assets.cryptocurrency import (
    CryptocurrencyAsset, Blockchain, ConsensusMechanism, AssetCategory, LiquidityTier
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeFiProtocol:
    """DeFi protocol data model for yield farming and liquidity analysis"""
    
    protocol_name: str
    protocol_type: str  # 'dex', 'lending', 'staking', 'yield_farming'
    blockchain: Blockchain
    
    # Financial Metrics
    total_value_locked: float  # USD value
    token_symbol: str
    token_price: float
    
    # Yield Metrics
    apy_rewards: float  # Annual percentage yield
    apy_fees: float  # Fee-based APY
    total_apy: float  # Combined APY
    
    # Risk Metrics
    smart_contract_risk: float  # 0-1 scale
    impermanent_loss_risk: float  # For liquidity pools
    token_volatility: float  # Historical volatility
    
    # Protocol Specifics
    audit_status: str  # 'audited', 'unaudited', 'partially_audited'
    governance_token: Optional[str] = None
    launch_date: Optional[datetime] = None


class CryptocurrencyDataCollector:
    """
    Advanced cryptocurrency data collection and DeFi analysis system
    
    Integrates major cryptocurrency exchanges and DeFi protocols for
    comprehensive digital asset portfolio management.
    """
    
    def __init__(self):
        """Initialize cryptocurrency data collector"""
        self.supported_exchanges = [
            'coinbase', 'binance', 'kraken', 'gemini', 'ftx'
        ]
        self.major_cryptocurrencies = self._get_major_crypto_universe()
        self.defi_protocols = self._get_defi_protocol_universe()
        self.stablecoin_universe = self._get_stablecoin_universe()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("Cryptocurrency data collector initialized")
    
    def _get_major_crypto_universe(self) -> List[str]:
        """Get comprehensive cryptocurrency universe by market cap"""
        return [
            # Layer 1 Blockchains
            'BTC',   # Bitcoin
            'ETH',   # Ethereum
            'BNB',   # Binance Coin
            'SOL',   # Solana
            'ADA',   # Cardano
            'AVAX',  # Avalanche
            'DOT',   # Polkadot
            'ATOM',  # Cosmos
            'NEAR',  # Near Protocol
            'ALGO',  # Algorand
            
            # Layer 2 Solutions
            'MATIC', # Polygon
            'OP',    # Optimism
            'ARB',   # Arbitrum
            
            # DeFi Tokens
            'UNI',   # Uniswap
            'AAVE',  # Aave
            'COMP',  # Compound
            'MKR',   # MakerDAO
            'SNX',   # Synthetix
            'CRV',   # Curve
            'SUSHI', # SushiSwap
            'YFI',   # Yearn Finance
            
            # Meme/Community
            'DOGE',  # Dogecoin
            'SHIB',  # Shiba Inu
            
            # Privacy Coins
            'XMR',   # Monero
            'ZEC',   # Zcash
            
            # Enterprise/Utility
            'XRP',   # Ripple
            'XLM',   # Stellar
            'VET',   # VeChain
            'LINK',  # Chainlink
        ]
    
    def _get_defi_protocol_universe(self) -> List[str]:
        """Get major DeFi protocol identifiers"""
        return [
            'uniswap-v3',
            'aave',
            'compound',
            'makerdao',
            'curve-dao-token',
            'pancakeswap-token',
            'yearn-finance',
            'synthetix',
            'balancer',
            'sushiswap',
            'convex-finance',
            'lido-dao',
            'frax',
            'olympus'
        ]
    
    def _get_stablecoin_universe(self) -> List[str]:
        """Get major stablecoin universe"""
        return [
            'USDT',  # Tether
            'USDC',  # USD Coin
            'BUSD',  # Binance USD
            'DAI',   # MakerDAO
            'FRAX',  # Frax
            'LUSD',  # Liquity
            'TUSD',  # TrueUSD
            'GUSD',  # Gemini Dollar
        ]
    
    async def collect_cryptocurrency_data(self, 
                                        symbols: Optional[List[str]] = None) -> List[CryptocurrencyAsset]:
        """
        Collect comprehensive cryptocurrency data
        
        Args:
            symbols: List of crypto symbols to collect, defaults to major universe
            
        Returns:
            List of CryptocurrencyAsset objects with complete data
        """
        if symbols is None:
            symbols = self.major_cryptocurrencies
        
        crypto_assets = []
        
        for symbol in symbols:
            try:
                await self._rate_limit()
                
                crypto_data = await self._fetch_crypto_fundamentals(symbol)
                if crypto_data:
                    crypto_assets.append(crypto_data)
                
            except Exception as e:
                logger.warning(f"Failed to collect data for {symbol}: {e}")
        
        logger.info(f"Collected data for {len(crypto_assets)} cryptocurrencies")
        return crypto_assets
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _fetch_crypto_fundamentals(self, symbol: str) -> Optional[CryptocurrencyAsset]:
        """
        Fetch comprehensive cryptocurrency fundamental data
        
        Note: In production, this would integrate with APIs like:
        - CoinGecko API for market data
        - CoinMarketCap API for comprehensive metrics
        - DefiPulse API for DeFi protocol data
        - Messari API for on-chain metrics
        """
        try:
            # Mock data generation for demo (replace with real API calls)
            base_data = self._generate_mock_crypto_data(symbol)
            
            # Determine blockchain and category
            blockchain = self._determine_blockchain(symbol)
            category = self._determine_category(symbol)
            liquidity_tier = self._determine_liquidity_tier(symbol)
            
            # Create cryptocurrency asset
            crypto_asset = CryptocurrencyAsset(
                symbol=symbol,
                name=base_data['name'],
                blockchain=blockchain,
                
                # Market Data
                market_cap=base_data['market_cap'],
                circulating_supply=base_data['circulating_supply'],
                total_supply=base_data['total_supply'],
                
                # Trading Metrics
                price_usd=base_data['price_usd'],
                trading_volume_24h=base_data['volume_24h'],
                volume_to_market_cap=base_data['volume_24h'] / base_data['market_cap'],
                number_of_exchanges=base_data['exchange_count'],
                
                # Volatility and Risk
                volatility_30d=base_data['volatility_30d'],
                volatility_90d=base_data['volatility_90d'],
                max_drawdown_1y=base_data['max_drawdown'],
                correlation_btc=base_data['correlation_btc'],
                correlation_sp500=base_data['correlation_sp500'],
                beta_to_crypto_market=base_data['crypto_beta'],
                
                # Crypto-specific
                contract_address=base_data.get('contract_address'),
                max_supply=base_data.get('max_supply'),
                network_hash_rate=base_data.get('hash_rate'),
                active_addresses=base_data['active_addresses'],
                transaction_count_24h=base_data['transactions_24h'],
                network_fees_24h=base_data['fees_24h'],
                staking_yield=base_data['staking_yield'],
                defi_protocols=self._get_associated_defi_protocols(symbol),
                blockchain_category=category,
                liquidity_tier=liquidity_tier,
                
                # Additional metrics
                developer_activity_score=base_data['dev_activity'],
                social_sentiment_score=base_data['social_sentiment'],
                institutional_adoption=base_data['institutional_adoption'],
                regulatory_risk_score=base_data['regulatory_risk'],
                
                # Performance
                annual_return=base_data['annual_return'],
                annual_volatility=base_data['annual_volatility'],
                max_drawdown=base_data['max_drawdown'],
                sharpe_ratio=base_data['sharpe_ratio']
            )
            
            return crypto_asset
            
        except Exception as e:
            logger.error(f"Error fetching crypto fundamentals for {symbol}: {e}")
            return None
    
    def _generate_mock_crypto_data(self, symbol: str) -> Dict:
        """Generate realistic mock cryptocurrency data"""
        # Set random seed based on symbol for consistency
        np.random.seed(hash(symbol) % 2**32)
        
        # Base market caps by tier
        market_caps = {
            'BTC': 500_000_000_000,  # $500B
            'ETH': 250_000_000_000,  # $250B
            'BNB': 50_000_000_000,   # $50B
            'SOL': 20_000_000_000,   # $20B
            'ADA': 15_000_000_000,   # $15B
        }
        
        base_market_cap = market_caps.get(symbol, np.random.uniform(1e9, 10e9))
        price = np.random.uniform(0.1, 50000) if symbol != 'BTC' else np.random.uniform(30000, 70000)
        
        return {
            'name': f"{symbol} Token",
            'market_cap': base_market_cap,
            'price_usd': price,
            'circulating_supply': base_market_cap / price,
            'total_supply': base_market_cap / price * np.random.uniform(1.0, 2.0),
            'max_supply': base_market_cap / price * np.random.uniform(1.5, 10.0),
            'volume_24h': base_market_cap * np.random.uniform(0.01, 0.3),
            'exchange_count': np.random.randint(5, 50),
            'volatility_30d': np.random.uniform(0.3, 2.0),  # 30-200% annualized
            'volatility_90d': np.random.uniform(0.25, 1.5),
            'max_drawdown': np.random.uniform(0.2, 0.8),
            'correlation_btc': np.random.uniform(0.3, 0.9) if symbol != 'BTC' else 1.0,
            'correlation_sp500': np.random.uniform(-0.1, 0.5),
            'crypto_beta': np.random.uniform(0.5, 2.0),
            'active_addresses': np.random.randint(10000, 1000000),
            'transactions_24h': np.random.randint(50000, 5000000),
            'fees_24h': np.random.uniform(10000, 1000000),
            'staking_yield': np.random.uniform(0.0, 0.15),
            'dev_activity': np.random.uniform(0.1, 1.0),
            'social_sentiment': np.random.uniform(0.2, 0.8),
            'institutional_adoption': np.random.uniform(0.0, 0.9),
            'regulatory_risk': np.random.uniform(0.1, 0.7),
            'annual_return': np.random.uniform(-0.5, 2.0),
            'annual_volatility': np.random.uniform(0.4, 2.5),
            'sharpe_ratio': np.random.uniform(-0.5, 2.0),
            'hash_rate': np.random.uniform(1e18, 1e20) if symbol in ['BTC'] else None,
            'contract_address': f"0x{''.join([f'{np.random.randint(0,16):x}' for _ in range(40)])}" if symbol != 'BTC' else None
        }
    
    def _determine_blockchain(self, symbol: str) -> Blockchain:
        """Determine blockchain for cryptocurrency"""
        blockchain_mapping = {
            'BTC': Blockchain.BITCOIN,
            'ETH': Blockchain.ETHEREUM,
            'BNB': Blockchain.BINANCE_SMART_CHAIN,
            'SOL': Blockchain.SOLANA,
            'ADA': Blockchain.CARDANO,
            'AVAX': Blockchain.AVALANCHE,
            'DOT': Blockchain.POLKADOT,
            'MATIC': Blockchain.POLYGON,
            'ATOM': Blockchain.COSMOS,
        }
        return blockchain_mapping.get(symbol, Blockchain.ETHEREUM)  # Default to Ethereum
    
    def _determine_category(self, symbol: str) -> AssetCategory:
        """Determine asset category for cryptocurrency"""
        category_mapping = {
            'BTC': AssetCategory.LAYER_1,
            'ETH': AssetCategory.LAYER_1,
            'SOL': AssetCategory.LAYER_1,
            'ADA': AssetCategory.LAYER_1,
            'MATIC': AssetCategory.LAYER_2,
            'OP': AssetCategory.LAYER_2,
            'UNI': AssetCategory.DEFI,
            'AAVE': AssetCategory.DEFI,
            'LINK': AssetCategory.ORACLE,
            'USDT': AssetCategory.STABLECOIN,
            'USDC': AssetCategory.STABLECOIN,
            'DOGE': AssetCategory.MEME,
            'SHIB': AssetCategory.MEME,
        }
        return category_mapping.get(symbol, AssetCategory.UTILITY)
    
    def _determine_liquidity_tier(self, symbol: str) -> LiquidityTier:
        """Determine liquidity tier for cryptocurrency"""
        tier_1 = ['BTC', 'ETH', 'BNB', 'USDT', 'USDC']
        tier_2 = ['SOL', 'ADA', 'AVAX', 'DOT', 'MATIC', 'UNI', 'LINK']
        
        if symbol in tier_1:
            return LiquidityTier.TIER_1
        elif symbol in tier_2:
            return LiquidityTier.TIER_2
        else:
            return LiquidityTier.TIER_3
    
    def _get_associated_defi_protocols(self, symbol: str) -> List[str]:
        """Get DeFi protocols associated with cryptocurrency"""
        defi_mapping = {
            'ETH': ['uniswap', 'aave', 'compound', 'makerdao'],
            'UNI': ['uniswap'],
            'AAVE': ['aave'],
            'COMP': ['compound'],
            'MKR': ['makerdao'],
            'CRV': ['curve'],
            'SUSHI': ['sushiswap'],
            'YFI': ['yearn'],
        }
        return defi_mapping.get(symbol, [])
    
    async def collect_defi_protocol_data(self) -> List[DeFiProtocol]:
        """
        Collect DeFi protocol data for yield farming analysis
        
        Returns:
            List of DeFiProtocol objects with yield and risk data
        """
        defi_protocols = []
        
        for protocol_id in self.defi_protocols:
            try:
                await self._rate_limit()
                
                protocol_data = await self._fetch_defi_protocol_data(protocol_id)
                if protocol_data:
                    defi_protocols.append(protocol_data)
                    
            except Exception as e:
                logger.warning(f"Failed to collect DeFi data for {protocol_id}: {e}")
        
        logger.info(f"Collected data for {len(defi_protocols)} DeFi protocols")
        return defi_protocols
    
    async def _fetch_defi_protocol_data(self, protocol_id: str) -> Optional[DeFiProtocol]:
        """Fetch DeFi protocol data (mock implementation)"""
        try:
            # Mock DeFi protocol data
            np.random.seed(hash(protocol_id) % 2**32)
            
            protocol_data = {
                'uniswap-v3': {
                    'name': 'Uniswap V3',
                    'type': 'dex',
                    'blockchain': Blockchain.ETHEREUM,
                    'tvl': 3_500_000_000,  # $3.5B
                    'token': 'UNI',
                    'apy_fees': 0.25,  # 25% from fees
                },
                'aave': {
                    'name': 'Aave',
                    'type': 'lending',
                    'blockchain': Blockchain.ETHEREUM,
                    'tvl': 7_000_000_000,  # $7B
                    'token': 'AAVE',
                    'apy_fees': 0.05,  # 5% lending rates
                },
                'compound': {
                    'name': 'Compound',
                    'type': 'lending',
                    'blockchain': Blockchain.ETHEREUM,
                    'tvl': 2_000_000_000,  # $2B
                    'token': 'COMP',
                    'apy_fees': 0.04,  # 4% lending rates
                }
            }
            
            base_data = protocol_data.get(protocol_id, {
                'name': protocol_id.title(),
                'type': 'defi',
                'blockchain': Blockchain.ETHEREUM,
                'tvl': np.random.uniform(100e6, 1e9),
                'token': protocol_id.upper()[:4],
                'apy_fees': np.random.uniform(0.02, 0.3)
            })
            
            # Generate protocol metrics
            token_price = np.random.uniform(1, 500)
            apy_rewards = np.random.uniform(0.0, 0.2)  # 0-20% rewards APY
            total_apy = base_data['apy_fees'] + apy_rewards
            
            defi_protocol = DeFiProtocol(
                protocol_name=base_data['name'],
                protocol_type=base_data['type'],
                blockchain=base_data['blockchain'],
                
                total_value_locked=base_data['tvl'],
                token_symbol=base_data['token'],
                token_price=token_price,
                
                apy_rewards=apy_rewards,
                apy_fees=base_data['apy_fees'],
                total_apy=total_apy,
                
                smart_contract_risk=np.random.uniform(0.1, 0.4),
                impermanent_loss_risk=np.random.uniform(0.0, 0.3),
                token_volatility=np.random.uniform(0.5, 2.0),
                
                audit_status='audited' if np.random.random() > 0.3 else 'partially_audited',
                governance_token=base_data['token'],
                launch_date=datetime.now() - timedelta(days=np.random.randint(100, 1000))
            )
            
            return defi_protocol
            
        except Exception as e:
            logger.error(f"Error fetching DeFi protocol data for {protocol_id}: {e}")
            return None
    
    def analyze_crypto_traditional_correlation(self, 
                                             crypto_returns: pd.Series,
                                             sp500_returns: pd.Series,
                                             bond_returns: pd.Series,
                                             rolling_window: int = 90) -> Dict[str, float]:
        """
        Analyze correlation between cryptocurrency and traditional assets
        
        Returns comprehensive correlation analysis for portfolio diversification
        """
        if len(crypto_returns) < rolling_window:
            return {'sp500_correlation': 0.0, 'bond_correlation': 0.0}
        
        # Align all data
        aligned_data = pd.concat([crypto_returns, sp500_returns, bond_returns], axis=1).dropna()
        if len(aligned_data) < rolling_window:
            return {'sp500_correlation': 0.0, 'bond_correlation': 0.0}
        
        crypto_col = aligned_data.columns[0]
        sp500_col = aligned_data.columns[1]
        bond_col = aligned_data.columns[2]
        
        # Calculate correlations
        sp500_corr = aligned_data[crypto_col].corr(aligned_data[sp500_col])
        bond_corr = aligned_data[crypto_col].corr(aligned_data[bond_col])
        
        # Rolling correlations for stability analysis
        rolling_sp500 = aligned_data[crypto_col].rolling(rolling_window).corr(
            aligned_data[sp500_col]
        ).dropna()
        
        rolling_bond = aligned_data[crypto_col].rolling(rolling_window).corr(
            aligned_data[bond_col]
        ).dropna()
        
        # Market regime analysis
        high_vol_periods = aligned_data[aligned_data[sp500_col].rolling(30).std() > 
                                      aligned_data[sp500_col].rolling(30).std().quantile(0.75)]
        
        high_vol_sp500_corr = (high_vol_periods[crypto_col].corr(high_vol_periods[sp500_col])
                              if len(high_vol_periods) > 30 else np.nan)
        
        return {
            'sp500_correlation': float(sp500_corr) if not np.isnan(sp500_corr) else 0.0,
            'bond_correlation': float(bond_corr) if not np.isnan(bond_corr) else 0.0,
            'rolling_sp500_mean': float(rolling_sp500.mean()) if len(rolling_sp500) > 0 else 0.0,
            'rolling_bond_mean': float(rolling_bond.mean()) if len(rolling_bond) > 0 else 0.0,
            'correlation_stability_sp500': float(1.0 - rolling_sp500.std()) if len(rolling_sp500) > 0 else 0.0,
            'correlation_stability_bond': float(1.0 - rolling_bond.std()) if len(rolling_bond) > 0 else 0.0,
            'high_volatility_sp500_correlation': float(high_vol_sp500_corr) if not np.isnan(high_vol_sp500_corr) else None,
            'diversification_benefit': float(1.0 - abs(sp500_corr)) if not np.isnan(sp500_corr) else 1.0
        }
    
    def generate_crypto_market_summary(self) -> Dict[str, any]:
        """Generate cryptocurrency market analysis summary"""
        return {
            'collector_status': 'active',
            'supported_cryptocurrencies': len(self.major_cryptocurrencies),
            'supported_defi_protocols': len(self.defi_protocols),
            'supported_stablecoins': len(self.stablecoin_universe),
            'supported_exchanges': self.supported_exchanges,
            'analysis_capabilities': [
                'fundamental_analysis',
                'defi_yield_analysis',
                'correlation_analysis',
                'liquidity_assessment',
                'risk_modeling'
            ],
            'last_updated': datetime.now().isoformat()
        }


# Demo usage function
async def demo_crypto_analysis():
    """Demonstrate cryptocurrency analysis capabilities"""
    collector = CryptocurrencyDataCollector()
    
    print("Cryptocurrency Data Collector Demo")
    print("=" * 50)
    
    # Collect sample cryptocurrency data
    sample_cryptos = ['BTC', 'ETH', 'UNI', 'AAVE']
    crypto_data = await collector.collect_cryptocurrency_data(sample_cryptos)
    
    print(f"\nCollected data for {len(crypto_data)} cryptocurrencies:")
    for crypto in crypto_data:
        print(f"- {crypto.symbol}: ${crypto.price_usd:,.2f} "
              f"(Market Cap: ${crypto.market_cap/1e9:.1f}B)")
    
    # Collect DeFi protocol data
    defi_data = await collector.collect_defi_protocol_data()
    
    print(f"\nCollected data for {len(defi_data)} DeFi protocols:")
    for protocol in defi_data[:3]:  # Show first 3
        print(f"- {protocol.protocol_name}: {protocol.total_apy:.1%} APY "
              f"(TVL: ${protocol.total_value_locked/1e9:.1f}B)")
    
    # Generate summary
    summary = collector.generate_crypto_market_summary()
    print(f"\nCryptocurrency Market Coverage:")
    print(f"- Cryptocurrencies: {summary['supported_cryptocurrencies']}")
    print(f"- DeFi Protocols: {summary['supported_defi_protocols']}")
    print(f"- Analysis Capabilities: {len(summary['analysis_capabilities'])}")


if __name__ == "__main__":
    asyncio.run(demo_crypto_analysis())
