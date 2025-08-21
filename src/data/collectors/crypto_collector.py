"""
Cryptocurrency data collector for institutional alternative asset portfolios.

Provides comprehensive data collection for cryptocurrency assets including
blockchain metrics, DeFi protocols, staking rewards, and institutional liquidity.
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import asdict

from src.portfolio.alternative_assets.cryptocurrency import (
    CryptocurrencyAsset, Blockchain, ConsensusMechanism, AssetCategory,
    LiquidityTier, DeFiProtocol, CryptoPortfolioMetrics
)


logger = logging.getLogger(__name__)


class CryptocurrencyDataCollector:
    """
    Comprehensive cryptocurrency data collector for institutional portfolios.
    
    Features:
    - Major cryptocurrency data collection
    - DeFi protocol integration
    - Staking and yield opportunities
    - Institutional liquidity analysis
    - Risk metrics and correlations
    """
    
    def __init__(self):
        """Initialize cryptocurrency data collector with universe and mappings"""
        self.crypto_universe = self._get_crypto_universe()
        self.defi_protocols = self._get_defi_protocols()
        self.blockchain_mapping = self._get_blockchain_mapping()
        self.staking_data = self._get_staking_data()
        self.exchange_tiers = self._get_exchange_tiers()
        
    def _get_crypto_universe(self) -> Dict[str, Dict]:
        """Define comprehensive cryptocurrency universe for institutional portfolios"""
        return {
            # Layer 1 Blockchains
            'BTC-USD': {
                'name': 'Bitcoin',
                'blockchain': Blockchain.BITCOIN,
                'category': AssetCategory.LAYER_1,
                'consensus': ConsensusMechanism.PROOF_OF_WORK,
                'max_supply': 21000000,
                'tier': LiquidityTier.TIER_1
            },
            'ETH-USD': {
                'name': 'Ethereum',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.LAYER_1,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': None,
                'tier': LiquidityTier.TIER_1
            },
            'ADA-USD': {
                'name': 'Cardano',
                'blockchain': Blockchain.CARDANO,
                'category': AssetCategory.LAYER_1,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 45000000000,
                'tier': LiquidityTier.TIER_2
            },
            'SOL-USD': {
                'name': 'Solana',
                'blockchain': Blockchain.SOLANA,
                'category': AssetCategory.LAYER_1,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': None,
                'tier': LiquidityTier.TIER_2
            },
            'AVAX-USD': {
                'name': 'Avalanche',
                'blockchain': Blockchain.AVALANCHE,
                'category': AssetCategory.LAYER_1,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 720000000,
                'tier': LiquidityTier.TIER_2
            },
            'DOT-USD': {
                'name': 'Polkadot',
                'blockchain': Blockchain.POLKADOT,
                'category': AssetCategory.LAYER_1,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': None,
                'tier': LiquidityTier.TIER_2
            },
            
            # DeFi Tokens
            'UNI-USD': {
                'name': 'Uniswap',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.DEFI,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 1000000000,
                'tier': LiquidityTier.TIER_2
            },
            'AAVE-USD': {
                'name': 'Aave',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.DEFI,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 16000000,
                'tier': LiquidityTier.TIER_2
            },
            'COMP-USD': {
                'name': 'Compound',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.DEFI,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 10000000,
                'tier': LiquidityTier.TIER_3
            },
            'MKR-USD': {
                'name': 'Maker',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.DEFI,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 1005577,
                'tier': LiquidityTier.TIER_3
            },
            
            # Stablecoins
            'USDC-USD': {
                'name': 'USD Coin',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.STABLECOIN,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': None,
                'tier': LiquidityTier.TIER_1
            },
            'USDT-USD': {
                'name': 'Tether',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.STABLECOIN,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': None,
                'tier': LiquidityTier.TIER_1
            },
            'DAI-USD': {
                'name': 'Dai',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.STABLECOIN,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': None,
                'tier': LiquidityTier.TIER_2
            },
            
            # Layer 2 Solutions
            'MATIC-USD': {
                'name': 'Polygon',
                'blockchain': Blockchain.POLYGON,
                'category': AssetCategory.LAYER_2,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 10000000000,
                'tier': LiquidityTier.TIER_2
            },
            'OP-USD': {
                'name': 'Optimism',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.LAYER_2,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 4294967296,
                'tier': LiquidityTier.TIER_3
            },
            
            # Institutional Infrastructure
            'LINK-USD': {
                'name': 'Chainlink',
                'blockchain': Blockchain.ETHEREUM,
                'category': AssetCategory.ORACLE,
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'max_supply': 1000000000,
                'tier': LiquidityTier.TIER_2
            }
        }
    
    def _get_defi_protocols(self) -> Dict[str, Dict]:
        """Define DeFi protocol yield opportunities"""
        return {
            'AAVE': {
                'protocol_type': 'LENDING',
                'typical_apy': 0.05,
                'risk_level': 'MEDIUM',
                'tvl_threshold': 5000000000,
                'smart_contract_risk': 0.3
            },
            'COMPOUND': {
                'protocol_type': 'LENDING',
                'typical_apy': 0.04,
                'risk_level': 'MEDIUM',
                'tvl_threshold': 3000000000,
                'smart_contract_risk': 0.2
            },
            'UNISWAP_V3': {
                'protocol_type': 'AMM',
                'typical_apy': 0.08,
                'risk_level': 'HIGH',
                'tvl_threshold': 4000000000,
                'smart_contract_risk': 0.4
            },
            'CURVE': {
                'protocol_type': 'STABLECOIN_AMM',
                'typical_apy': 0.06,
                'risk_level': 'MEDIUM',
                'tvl_threshold': 8000000000,
                'smart_contract_risk': 0.25
            },
            'LIDO': {
                'protocol_type': 'STAKING',
                'typical_apy': 0.045,
                'risk_level': 'LOW',
                'tvl_threshold': 20000000000,
                'smart_contract_risk': 0.15
            }
        }
    
    def _get_blockchain_mapping(self) -> Dict[str, Dict]:
        """Map blockchain characteristics for institutional analysis"""
        return {
            Blockchain.BITCOIN: {
                'consensus': ConsensusMechanism.PROOF_OF_WORK,
                'energy_efficiency': 0.1,
                'transaction_throughput': 7,
                'finality_time': 600,  # seconds
                'institutional_adoption': 0.9
            },
            Blockchain.ETHEREUM: {
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'energy_efficiency': 0.8,
                'transaction_throughput': 15,
                'finality_time': 384,  # seconds
                'institutional_adoption': 0.85
            },
            Blockchain.SOLANA: {
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'energy_efficiency': 0.9,
                'transaction_throughput': 2000,
                'finality_time': 13,  # seconds
                'institutional_adoption': 0.6
            },
            Blockchain.CARDANO: {
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'energy_efficiency': 0.95,
                'transaction_throughput': 250,
                'finality_time': 300,  # seconds
                'institutional_adoption': 0.5
            },
            Blockchain.AVALANCHE: {
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'energy_efficiency': 0.9,
                'transaction_throughput': 4500,
                'finality_time': 2,  # seconds
                'institutional_adoption': 0.4
            },
            Blockchain.POLKADOT: {
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'energy_efficiency': 0.85,
                'transaction_throughput': 1000,
                'finality_time': 60,  # seconds
                'institutional_adoption': 0.3
            },
            Blockchain.POLYGON: {
                'consensus': ConsensusMechanism.PROOF_OF_STAKE,
                'energy_efficiency': 0.95,
                'transaction_throughput': 7000,
                'finality_time': 2,  # seconds
                'institutional_adoption': 0.7
            }
        }
    
    def _get_staking_data(self) -> Dict[str, Dict]:
        """Define staking opportunities and requirements"""
        return {
            'ETH-USD': {
                'staking_rewards': 0.045,
                'minimum_stake': 32.0,
                'lock_period': 0,  # No lock after Shanghai upgrade
                'slashing_risk': 0.01,
                'validator_requirements': True
            },
            'ADA-USD': {
                'staking_rewards': 0.055,
                'minimum_stake': 1.0,
                'lock_period': 0,
                'slashing_risk': 0.0,
                'validator_requirements': False
            },
            'SOL-USD': {
                'staking_rewards': 0.065,
                'minimum_stake': 0.01,
                'lock_period': 2.5,  # days
                'slashing_risk': 0.005,
                'validator_requirements': False
            },
            'DOT-USD': {
                'staking_rewards': 0.12,
                'minimum_stake': 120.0,
                'lock_period': 28,  # days
                'slashing_risk': 0.02,
                'validator_requirements': False
            },
            'AVAX-USD': {
                'staking_rewards': 0.09,
                'minimum_stake': 25.0,
                'lock_period': 14,  # days
                'slashing_risk': 0.0,
                'validator_requirements': False
            },
            'MATIC-USD': {
                'staking_rewards': 0.08,
                'minimum_stake': 1.0,
                'lock_period': 80,  # days (Ethereum checkpoint delays)
                'slashing_risk': 0.01,
                'validator_requirements': False
            }
        }
    
    def _get_exchange_tiers(self) -> Dict[str, List[str]]:
        """Define exchange tiers for institutional liquidity analysis"""
        return {
            'TIER_1': [
                'Coinbase Pro', 'Binance', 'Kraken Pro', 'Bitstamp', 
                'Gemini', 'Huobi Global', 'OKX', 'Bitfinex'
            ],
            'TIER_2': [
                'FTX', 'KuCoin', 'Gate.io', 'Crypto.com Exchange',
                'Bybit', 'MEXC', 'Bitget', 'Phemex'
            ],
            'INSTITUTIONAL': [
                'Coinbase Custody', 'BitGo', 'Fidelity Digital Assets',
                'Bakkt', 'XAPO Bank', 'Anchorage Digital'
            ]
        }
    
    async def collect_crypto_data(self, symbols: Optional[List[str]] = None) -> List[CryptocurrencyAsset]:
        """
        Collect comprehensive cryptocurrency data for institutional portfolios.
        
        Args:
            symbols: List of crypto symbols to collect. If None, uses full universe.
            
        Returns:
            List of CryptocurrencyAsset objects with institutional metrics
        """
        if symbols is None:
            symbols = list(self.crypto_universe.keys())
        
        crypto_assets = []
        
        # Process symbols in parallel batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_tasks = [self._collect_single_crypto(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, CryptocurrencyAsset):
                    crypto_assets.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error collecting crypto data: {result}")
            
            # Rate limiting delay
            await asyncio.sleep(1)
        
        logger.info(f"Collected data for {len(crypto_assets)} cryptocurrency assets")
        return crypto_assets
    
    async def _collect_single_crypto(self, symbol: str) -> CryptocurrencyAsset:
        """Collect data for a single cryptocurrency asset"""
        try:
            # Get asset configuration
            asset_config = self.crypto_universe.get(symbol)
            if not asset_config:
                raise ValueError(f"Unknown crypto asset: {symbol}")
            
            # Fetch market data from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data for volatility calculations
            hist_data = ticker.history(period="90d")
            if hist_data.empty:
                raise ValueError(f"No historical data for {symbol}")
            
            # Calculate market metrics
            current_price = info.get('regularMarketPrice', hist_data['Close'].iloc[-1])
            market_cap = info.get('marketCap', 0)
            volume_24h = info.get('volume24Hr', hist_data['Volume'].iloc[-1] * current_price)
            
            # Calculate volatility metrics
            returns = hist_data['Close'].pct_change().dropna()
            volatility_30d = returns.tail(30).std() * np.sqrt(365) if len(returns) >= 30 else 0
            volatility_90d = returns.std() * np.sqrt(365) if len(returns) >= 60 else 0
            
            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown_1y = abs(drawdown.min()) if not drawdown.empty else 0
            
            # Get supply data
            circulating_supply = info.get('circulatingSupply', 0)
            total_supply = info.get('totalSupply', circulating_supply)
            max_supply = asset_config.get('max_supply')
            
            # Calculate correlations (simplified for demo)
            correlation_btc = await self._calculate_btc_correlation(symbol, hist_data)
            correlation_sp500 = await self._calculate_sp500_correlation(symbol, hist_data)
            
            # Get blockchain-specific data
            blockchain_data = self.blockchain_mapping.get(asset_config['blockchain'], {})
            staking_data = self.staking_data.get(symbol, {})
            
            # Calculate institutional metrics
            liquidity_score = self._calculate_crypto_liquidity_score(
                market_cap, volume_24h, asset_config['tier']
            )
            
            # Get DeFi data if applicable
            defi_yields = self._get_defi_yields(symbol, asset_config)
            
            # Determine exchange availability
            exchange_availability = self._get_exchange_availability(asset_config['tier'])
            custody_solutions = self._get_custody_solutions(asset_config['tier'])
            
            # Create cryptocurrency asset
            crypto_asset = CryptocurrencyAsset(
                symbol=symbol,
                name=asset_config['name'],
                blockchain=asset_config['blockchain'],
                market_cap=market_cap,
                circulating_supply=circulating_supply,
                total_supply=total_supply,
                max_supply=max_supply,
                price_usd=current_price,
                trading_volume_24h=volume_24h,
                volume_to_market_cap=volume_24h / market_cap if market_cap > 0 else 0,
                number_of_exchanges=len(exchange_availability),
                volatility_30d=volatility_30d,
                volatility_90d=volatility_90d,
                max_drawdown_1y=max_drawdown_1y,
                correlation_btc=correlation_btc,
                correlation_sp500=correlation_sp500,
                beta_to_crypto_market=self._calculate_crypto_beta(symbol, returns),
                network_hash_rate=blockchain_data.get('institutional_adoption', 0) * 1000000,
                active_addresses=market_cap // 1000,  # Proxy calculation
                consensus_mechanism=blockchain_data.get('consensus', asset_config['consensus']),
                asset_category=asset_config['category'],
                liquidity_tier=asset_config['tier'],
                staking_rewards=staking_data.get('staking_rewards', 0),
                yield_farming_apy=defi_yields.get('yield_farming', 0),
                total_value_locked=defi_yields.get('tvl', 0),
                governance_token=asset_config['category'] == AssetCategory.DEFI,
                regulatory_risk_score=self._calculate_regulatory_risk(asset_config),
                exchange_availability=exchange_availability,
                custody_solutions=custody_solutions,
                energy_efficiency=blockchain_data.get('energy_efficiency', 0.5),
                transaction_throughput=blockchain_data.get('transaction_throughput', 100),
                developer_activity_score=int(blockchain_data.get('institutional_adoption', 0.5) * 100)
            )
            
            return crypto_asset
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            raise
    
    async def _calculate_btc_correlation(self, symbol: str, hist_data: pd.DataFrame) -> float:
        """Calculate correlation with Bitcoin"""
        if symbol == 'BTC-USD':
            return 1.0
        
        try:
            # Get Bitcoin data for correlation
            btc_ticker = yf.Ticker('BTC-USD')
            btc_data = btc_ticker.history(period="90d")
            
            if btc_data.empty or len(hist_data) < 30:
                return 0.5  # Default moderate correlation
            
            # Align dates and calculate correlation
            common_dates = hist_data.index.intersection(btc_data.index)
            if len(common_dates) < 30:
                return 0.5
            
            asset_returns = hist_data.loc[common_dates, 'Close'].pct_change().dropna()
            btc_returns = btc_data.loc[common_dates, 'Close'].pct_change().dropna()
            
            if len(asset_returns) < 20 or len(btc_returns) < 20:
                return 0.5
            
            correlation = asset_returns.corr(btc_returns)
            return max(-1.0, min(1.0, correlation)) if not pd.isna(correlation) else 0.5
            
        except Exception:
            return 0.5  # Default on error
    
    async def _calculate_sp500_correlation(self, symbol: str, hist_data: pd.DataFrame) -> float:
        """Calculate correlation with S&P 500"""
        try:
            # Get S&P 500 data for correlation
            spy_ticker = yf.Ticker('SPY')
            spy_data = spy_ticker.history(period="90d")
            
            if spy_data.empty or len(hist_data) < 30:
                return 0.3  # Default low correlation
            
            # Align dates and calculate correlation
            common_dates = hist_data.index.intersection(spy_data.index)
            if len(common_dates) < 30:
                return 0.3
            
            asset_returns = hist_data.loc[common_dates, 'Close'].pct_change().dropna()
            spy_returns = spy_data.loc[common_dates, 'Close'].pct_change().dropna()
            
            if len(asset_returns) < 20 or len(spy_returns) < 20:
                return 0.3
            
            correlation = asset_returns.corr(spy_returns)
            return max(-1.0, min(1.0, correlation)) if not pd.isna(correlation) else 0.3
            
        except Exception:
            return 0.3  # Default on error
    
    def _calculate_crypto_liquidity_score(self, market_cap: float, volume_24h: float, tier: LiquidityTier) -> float:
        """Calculate liquidity score for institutional assessment"""
        # Base score from volume-to-market-cap ratio
        volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
        
        # Tier adjustments
        tier_multipliers = {
            LiquidityTier.TIER_1: 1.0,
            LiquidityTier.TIER_2: 0.8,
            LiquidityTier.TIER_3: 0.6,
            LiquidityTier.TIER_4: 0.4
        }
        
        base_score = min(1.0, volume_ratio * 50)  # Scale volume ratio
        tier_adjustment = tier_multipliers.get(tier, 0.5)
        
        return base_score * tier_adjustment
    
    def _calculate_crypto_beta(self, symbol: str, returns: pd.Series) -> float:
        """Calculate beta to crypto market (simplified)"""
        if len(returns) < 30:
            return 1.0
        
        # Use volatility as proxy for beta (actual implementation would use market returns)
        volatility = returns.std() * np.sqrt(365)
        
        # Scale to reasonable beta range
        if volatility < 0.5:
            return 0.5 + volatility
        elif volatility > 2.0:
            return min(3.0, 1.0 + volatility / 2)
        else:
            return volatility
    
    def _get_defi_yields(self, symbol: str, asset_config: Dict) -> Dict[str, float]:
        """Get DeFi yield opportunities for the asset"""
        yields = {}
        
        # Staking yields
        staking_data = self.staking_data.get(symbol, {})
        if staking_data:
            yields['staking'] = staking_data.get('staking_rewards', 0)
        
        # DeFi protocol yields
        if asset_config['category'] == AssetCategory.DEFI:
            protocol_name = asset_config['name'].upper()
            if protocol_name in self.defi_protocols:
                protocol = self.defi_protocols[protocol_name]
                yields['yield_farming'] = protocol.get('typical_apy', 0)
                yields['tvl'] = protocol.get('tvl_threshold', 0)
        
        # Liquidity providing (general estimate)
        if asset_config['tier'] in [LiquidityTier.TIER_1, LiquidityTier.TIER_2]:
            yields['liquidity_providing'] = 0.03  # 3% estimate
        
        return yields
    
    def _get_exchange_availability(self, tier: LiquidityTier) -> List[str]:
        """Get exchange availability based on liquidity tier"""
        if tier == LiquidityTier.TIER_1:
            return self.exchange_tiers['TIER_1'] + self.exchange_tiers['INSTITUTIONAL']
        elif tier == LiquidityTier.TIER_2:
            return self.exchange_tiers['TIER_1'] + self.exchange_tiers['TIER_2']
        elif tier == LiquidityTier.TIER_3:
            return self.exchange_tiers['TIER_2']
        else:
            return ['Small Exchange']
    
    def _get_custody_solutions(self, tier: LiquidityTier) -> List[str]:
        """Get institutional custody solutions"""
        if tier in [LiquidityTier.TIER_1, LiquidityTier.TIER_2]:
            return self.exchange_tiers['INSTITUTIONAL']
        else:
            return ['Self-Custody']
    
    def _calculate_regulatory_risk(self, asset_config: Dict) -> float:
        """Calculate regulatory risk score"""
        base_risk = {
            AssetCategory.LAYER_1: 3.0,
            AssetCategory.LAYER_2: 4.0,
            AssetCategory.DEFI: 6.0,
            AssetCategory.STABLECOIN: 2.0,
            AssetCategory.UTILITY: 5.0,
            AssetCategory.GOVERNANCE: 7.0,
            AssetCategory.MEME: 9.0,
            AssetCategory.PRIVACY: 8.0,
            AssetCategory.ORACLE: 4.0,
            AssetCategory.NFT: 6.0
        }.get(asset_config['category'], 5.0)
        
        # Adjust for liquidity tier (higher tier = lower regulatory risk)
        tier_adjustment = {
            LiquidityTier.TIER_1: -1.0,
            LiquidityTier.TIER_2: -0.5,
            LiquidityTier.TIER_3: 0.0,
            LiquidityTier.TIER_4: 1.0
        }.get(asset_config['tier'], 0.0)
        
        return max(1.0, min(10.0, base_risk + tier_adjustment))
    
    async def get_defi_protocol_data(self, protocol_symbols: List[str]) -> List[DeFiProtocol]:
        """Collect DeFi protocol specific data"""
        protocols = []
        
        for symbol in protocol_symbols:
            if symbol in self.crypto_universe:
                asset_config = self.crypto_universe[symbol]
                if asset_config['category'] == AssetCategory.DEFI:
                    protocol_name = asset_config['name'].upper()
                    if protocol_name in self.defi_protocols:
                        protocol_data = self.defi_protocols[protocol_name]
                        
                        protocol = DeFiProtocol(
                            protocol_name=protocol_name,
                            protocol_type=protocol_data['protocol_type'],
                            blockchain=asset_config['blockchain'],
                            total_value_locked=protocol_data.get('tvl_threshold', 0),
                            yield_opportunities={
                                'lending': protocol_data.get('typical_apy', 0),
                                'staking': 0.05,  # Default estimate
                                'liquidity_providing': 0.08  # Default estimate
                            },
                            smart_contract_risk=protocol_data.get('smart_contract_risk', 0.3),
                            governance_token_symbol=symbol,
                            audit_status="AUDITED",  # Assumption for major protocols
                            insurance_coverage=protocol_data.get('tvl_threshold', 0) > 1000000000
                        )
                        protocols.append(protocol)
        
        return protocols
    
    async def calculate_portfolio_metrics(self, crypto_assets: List[CryptocurrencyAsset], 
                                        weights: Optional[List[float]] = None) -> CryptoPortfolioMetrics:
        """Calculate portfolio-level metrics for crypto assets"""
        if not crypto_assets:
            raise ValueError("No crypto assets provided")
        
        if weights is None:
            weights = [1.0 / len(crypto_assets)] * len(crypto_assets)
        
        if len(weights) != len(crypto_assets):
            raise ValueError("Weights length must match assets length")
        
        # Portfolio calculations
        total_market_cap = sum(asset.market_cap * weight for asset, weight in zip(crypto_assets, weights))
        weighted_volatility = sum(asset.volatility_30d * weight for asset, weight in zip(crypto_assets, weights))
        portfolio_max_drawdown = max(asset.max_drawdown_1y for asset in crypto_assets)
        
        # Correlation analysis
        btc_correlations = [asset.correlation_btc for asset in crypto_assets]
        sp500_correlations = [asset.correlation_sp500 for asset in crypto_assets]
        
        avg_btc_correlation = sum(corr * weight for corr, weight in zip(btc_correlations, weights))
        avg_sp500_correlation = sum(corr * weight for corr, weight in zip(sp500_correlations, weights))
        
        # Asset allocation analysis
        asset_allocation = {}
        for asset, weight in zip(crypto_assets, weights):
            category = asset.asset_category.value
            asset_allocation[category] = asset_allocation.get(category, 0) + weight
        
        # Yield opportunities
        total_staking_yield = sum(asset.staking_rewards * weight for asset, weight in zip(crypto_assets, weights))
        total_defi_yield = sum(asset.yield_farming_apy * weight for asset, weight in zip(crypto_assets, weights))
        
        # Risk metrics
        regulatory_risk = sum(asset.regulatory_risk_score * weight for asset, weight in zip(crypto_assets, weights))
        
        return CryptoPortfolioMetrics(
            total_market_cap=total_market_cap,
            portfolio_volatility=weighted_volatility,
            portfolio_max_drawdown=portfolio_max_drawdown,
            correlation_btc=avg_btc_correlation,
            correlation_traditional_assets=avg_sp500_correlation,
            asset_allocation=asset_allocation,
            staking_yield_potential=total_staking_yield,
            defi_yield_potential=total_defi_yield,
            regulatory_risk_score=regulatory_risk,
            liquidity_assessment={
                'tier_1_allocation': sum(weight for asset, weight in zip(crypto_assets, weights) 
                                       if asset.liquidity_tier == LiquidityTier.TIER_1),
                'tier_2_allocation': sum(weight for asset, weight in zip(crypto_assets, weights) 
                                       if asset.liquidity_tier == LiquidityTier.TIER_2),
                'tier_3_allocation': sum(weight for asset, weight in zip(crypto_assets, weights) 
                                       if asset.liquidity_tier == LiquidityTier.TIER_3),
                'tier_4_allocation': sum(weight for asset, weight in zip(crypto_assets, weights) 
                                       if asset.liquidity_tier == LiquidityTier.TIER_4)
            }
        )


# Example usage and testing
async def main():
    """Example usage of cryptocurrency data collector"""
    collector = CryptocurrencyDataCollector()
    
    # Collect data for major cryptocurrencies
    major_cryptos = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'USDC-USD']
    crypto_data = await collector.collect_crypto_data(major_cryptos)
    
    print(f"Collected data for {len(crypto_data)} cryptocurrencies:")
    for crypto in crypto_data:
        print(f"- {crypto.symbol}: {crypto.name} (${crypto.price_usd:.2f})")
        print(f"  Market Cap: ${crypto.market_cap / 1e9:.1f}B")
        print(f"  30d Volatility: {crypto.volatility_30d:.1%}")
        print(f"  Liquidity Tier: {crypto.liquidity_tier.value}")
        print(f"  Staking Rewards: {crypto.staking_rewards:.1%}")
        print()
    
    # Calculate portfolio metrics
    if crypto_data:
        portfolio_metrics = await collector.calculate_portfolio_metrics(crypto_data)
        print("Portfolio Metrics:")
        print(f"Total Market Cap: ${portfolio_metrics.total_market_cap / 1e9:.1f}B")
        print(f"Portfolio Volatility: {portfolio_metrics.portfolio_volatility:.1%}")
        print(f"BTC Correlation: {portfolio_metrics.correlation_btc:.2f}")
        print(f"Staking Yield Potential: {portfolio_metrics.staking_yield_potential:.1%}")
        print(f"Regulatory Risk Score: {portfolio_metrics.regulatory_risk_score:.1f}/10")


if __name__ == "__main__":
    asyncio.run(main())
