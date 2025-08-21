"""
Cryptocurrency and digital asset data models for alternative asset portfolio optimization.

This module provides comprehensive cryptocurrency-specific data structures for digital asset
portfolio construction, DeFi protocol analysis, and blockchain-based investment strategies.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
import math


class Blockchain(Enum):
    """Major blockchain networks"""
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "binance_smart_chain"
    CARDANO = "cardano"
    SOLANA = "solana"
    AVALANCHE = "avalanche"
    POLKADOT = "polkadot"
    POLYGON = "polygon"
    FANTOM = "fantom"
    COSMOS = "cosmos"


class ConsensusMechanism(Enum):
    """Blockchain consensus mechanisms"""
    PROOF_OF_WORK = "proof_of_work"
    PROOF_OF_STAKE = "proof_of_stake"
    DELEGATED_PROOF_OF_STAKE = "delegated_proof_of_stake"
    PROOF_OF_HISTORY = "proof_of_history"
    NOMINATED_PROOF_OF_STAKE = "nominated_proof_of_stake"
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = "pbft"


class AssetCategory(Enum):
    """Cryptocurrency asset categories"""
    LAYER_1 = "layer_1"  # Base blockchain protocols
    LAYER_2 = "layer_2"  # Scaling solutions
    DEFI = "defi"        # Decentralized finance
    NFT = "nft"          # Non-fungible tokens
    GAMING = "gaming"    # Gaming and metaverse
    STABLECOIN = "stablecoin"  # Price-stable assets
    GOVERNANCE = "governance"  # Governance tokens
    UTILITY = "utility"  # Utility tokens
    MEME = "meme"       # Meme coins
    ORACLE = "oracle"   # Oracle and data feeds
    PRIVACY = "privacy" # Privacy-focused coins


class LiquidityTier(Enum):
    """Cryptocurrency liquidity classifications"""
    TIER_1 = "tier_1"  # Highest liquidity (BTC, ETH, major exchanges)
    TIER_2 = "tier_2"  # High liquidity (top 50 coins)
    TIER_3 = "tier_3"  # Moderate liquidity (top 500 coins)
    TIER_4 = "tier_4"  # Low liquidity (smaller altcoins)


@dataclass
class CryptocurrencyAsset:
    """
    Comprehensive cryptocurrency asset data model for institutional digital asset management.
    
    Includes crypto-specific metrics like on-chain data, DeFi protocols, staking yields,
    and blockchain fundamentals essential for digital asset portfolio construction.
    """
    
    # Basic Identification - Required fields
    symbol: str
    name: str
    blockchain: Blockchain
    
    # Market Data - Required fields
    market_cap: float
    circulating_supply: float
    total_supply: float
    
    # Trading Metrics - Required fields
    price_usd: float
    trading_volume_24h: float
    volume_to_market_cap: float
    number_of_exchanges: int
    
    # Volatility and Risk - Required fields
    volatility_30d: float
    volatility_90d: float
    max_drawdown_1y: float
    correlation_btc: float
    correlation_sp500: float
    beta_to_crypto_market: float
    
    # Optional fields with defaults
    contract_address: Optional[str] = None  # For tokens
    max_supply: Optional[float] = None
    
    # Fundamental Metrics - Optional
    network_hash_rate: Optional[float] = None  # For PoW coins
    active_addresses: Optional[int] = None
    transaction_volume: Optional[float] = None
    developer_activity_score: Optional[float] = None
    
    # DeFi Specific (if applicable)
    total_value_locked: Optional[float] = None  # TVL for DeFi protocols
    yield_farming_apy: Optional[float] = None
    staking_rewards: Optional[float] = None
    governance_token: bool = False
    
    # Regulatory and Risk Factors
    regulatory_risk_score: float = 0.0  # 0-10 scale
    exchange_availability: Optional[List[str]] = None
    custody_solutions: Optional[List[str]] = None
    liquidity_tier: LiquidityTier = LiquidityTier.TIER_4
    
    # Blockchain and Technical
    consensus_mechanism: Optional[ConsensusMechanism] = None
    asset_category: Optional[AssetCategory] = None
    
    # Additional Metrics
    social_sentiment_score: Optional[float] = None
    github_activity: Optional[int] = None
    institutional_adoption: Optional[float] = None
    energy_efficiency: Optional[float] = None
    transaction_throughput: Optional[int] = None
    
    # Data Quality
    last_updated: Optional[datetime] = None
    data_sources: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.exchange_availability is None:
            self.exchange_availability = []
        if self.custody_solutions is None:
            self.custody_solutions = []
        if self.data_sources is None:
            self.data_sources = []
    
    def calculate_market_cap_rank(self, all_crypto_market_caps: List[float]) -> int:
        """Calculate market cap ranking among cryptocurrencies"""
        sorted_caps = sorted(all_crypto_market_caps, reverse=True)
        try:
            return sorted_caps.index(self.market_cap) + 1
        except ValueError:
            return len(sorted_caps) + 1
    
    def calculate_volatility_score(self) -> str:
        """Classify volatility level"""
        if self.volatility_30d < 0.5:
            return "LOW"
        elif self.volatility_30d < 1.0:
            return "MEDIUM"
        elif self.volatility_30d < 2.0:
            return "HIGH"
        else:
            return "EXTREME"
    
    def calculate_liquidity_score(self) -> float:
        """Calculate liquidity score based on volume and exchanges"""
        if self.market_cap <= 0:
            return 0.0
        
        # Volume-based component (0-0.5)
        volume_ratio = self.trading_volume_24h / self.market_cap
        volume_score = min(volume_ratio * 10, 0.5)  # Cap at 0.5
        
        # Exchange availability component (0-0.3)
        exchange_score = min(len(self.exchange_availability) * 0.05, 0.3)
        
        # Liquidity tier component (0-0.2)
        tier_scores = {
            LiquidityTier.TIER_1: 0.2,
            LiquidityTier.TIER_2: 0.15,
            LiquidityTier.TIER_3: 0.1,
            LiquidityTier.TIER_4: 0.05
        }
        tier_score = tier_scores.get(self.liquidity_tier, 0.0)
        
        return min(volume_score + exchange_score + tier_score, 1.0)
    
    def get_defi_yield_opportunities(self) -> Dict[str, float]:
        """Get available DeFi yield opportunities"""
        opportunities = {}
        
        if self.staking_rewards and self.staking_rewards > 0:
            opportunities["staking"] = self.staking_rewards
        
        if self.yield_farming_apy and self.yield_farming_apy > 0:
            opportunities["yield_farming"] = self.yield_farming_apy
        
        # Add protocol-specific yields based on asset category
        if self.asset_category == AssetCategory.DEFI:
            if self.total_value_locked and self.total_value_locked > 1000000:  # $1M+ TVL
                opportunities["liquidity_providing"] = 0.08  # Estimated 8% APY
        
        return opportunities
    
    def calculate_network_security_score(self) -> float:
        """Calculate network security score"""
        security_score = 0.0
        
        # Hash rate component (for PoW)
        if self.network_hash_rate and self.consensus_mechanism == ConsensusMechanism.PROOF_OF_WORK:
            # Normalize hash rate (simplified)
            if self.network_hash_rate > 100000000:  # 100 EH/s for Bitcoin scale
                security_score += 0.4
            elif self.network_hash_rate > 1000000:  # 1 EH/s
                security_score += 0.2
            else:
                security_score += 0.1
        
        # Staking participation (for PoS)
        elif self.consensus_mechanism in [ConsensusMechanism.PROOF_OF_STAKE, 
                                        ConsensusMechanism.DELEGATED_PROOF_OF_STAKE]:
            if self.staking_rewards and self.staking_rewards > 0.05:  # 5%+ staking rewards
                security_score += 0.3
        
        # Market cap component
        if self.market_cap > 10000000000:  # $10B+
            security_score += 0.3
        elif self.market_cap > 1000000000:  # $1B+
            security_score += 0.2
        else:
            security_score += 0.1
        
        # Active addresses component
        if self.active_addresses:
            if self.active_addresses > 100000:
                security_score += 0.2
            elif self.active_addresses > 10000:
                security_score += 0.1
        
        # Developer activity component
        if self.developer_activity_score:
            security_score += min(self.developer_activity_score / 100, 0.1)
        
        return min(security_score, 1.0)
    
    def validate_crypto_data(self) -> List[str]:
        """Validate cryptocurrency data quality"""
        issues = []
        
        if not self.symbol or len(self.symbol) < 2:
            issues.append("Invalid symbol")
        
        if self.market_cap <= 0:
            issues.append("Invalid market cap")
        
        if self.price_usd <= 0:
            issues.append("Invalid price")
        
        if self.circulating_supply <= 0:
            issues.append("Invalid circulating supply")
        
        if self.max_supply and self.circulating_supply > self.max_supply:
            issues.append("Circulating supply exceeds max supply")
        
        if self.volatility_30d < 0 or self.volatility_30d > 10:
            issues.append("Volatility outside reasonable range")
        
        if self.correlation_btc < -1 or self.correlation_btc > 1:
            issues.append("BTC correlation outside valid range")
        
        if self.regulatory_risk_score < 0 or self.regulatory_risk_score > 10:
            issues.append("Regulatory risk score outside valid range")
        
        return issues


@dataclass
class DeFiProtocol:
    """
    Decentralized Finance protocol data model for DeFi investment analysis.
    """
    
    protocol_name: str
    blockchain: Blockchain
    category: str  # lending, dex, derivatives, insurance, etc.
    
    # TVL and Usage Metrics
    total_value_locked: float
    tvl_change_24h: float
    active_users: int
    transaction_volume_24h: float
    
    # Token Information
    governance_token: Optional[str] = None
    token_distribution: Optional[Dict[str, float]] = None
    
    # Yield and Rewards
    base_apy: float = 0.0
    reward_apy: float = 0.0
    total_apy: float = 0.0
    
    # Risk Metrics
    smart_contract_risk: float = 0.0  # 0-10 scale
    impermanent_loss_risk: float = 0.0  # For liquidity pools
    audit_score: float = 0.0  # Based on security audits
    
    # Financial Metrics
    revenue_24h: Optional[float] = None
    fees_24h: Optional[float] = None
    protocol_revenue: Optional[float] = None
    
    def calculate_risk_adjusted_yield(self) -> float:
        """Calculate risk-adjusted yield for DeFi protocol"""
        risk_discount = (self.smart_contract_risk + self.impermanent_loss_risk) / 20.0
        return max(0, self.total_apy - risk_discount)
    
    def get_protocol_health_score(self) -> float:
        """Calculate overall protocol health score"""
        # TVL component (0-0.3)
        tvl_score = min(self.total_value_locked / 1000000000, 0.3)  # $1B max
        
        # User activity component (0-0.3)
        user_score = min(self.active_users / 100000, 0.3)  # 100K users max
        
        # Audit component (0-0.2)
        audit_component = self.audit_score / 50.0  # Normalize to 0-0.2
        
        # Revenue component (0-0.2)
        revenue_component = 0.0
        if self.revenue_24h:
            revenue_component = min(self.revenue_24h / 1000000, 0.2)  # $1M daily revenue max
        
        return min(tvl_score + user_score + audit_component + revenue_component, 1.0)


@dataclass
class CryptoPortfolioMetrics:
    """
    Portfolio-level metrics for cryptocurrency allocation analysis.
    """
    
    total_crypto_allocation: float  # Percentage of portfolio
    asset_category_diversification: Dict[AssetCategory, float]
    blockchain_diversification: Dict[Blockchain, float]
    
    # Risk Metrics
    crypto_portfolio_volatility: float
    correlation_to_traditional_assets: float
    regulatory_risk_exposure: float
    
    # DeFi and Staking
    staking_yield_portfolio: float
    defi_exposure: float
    liquid_vs_staked_allocation: Dict[str, float]
    
    # Technical Metrics
    average_network_security: float
    custody_risk_score: float
    
    def calculate_crypto_diversification_score(self) -> float:
        """Calculate cryptocurrency diversification score"""
        category_entropy = self._calculate_entropy(self.asset_category_diversification)
        blockchain_entropy = self._calculate_entropy(self.blockchain_diversification)
        
        # Weighted combination
        return 0.6 * category_entropy + 0.4 * blockchain_entropy
    
    def _calculate_entropy(self, allocation_dict: Dict) -> float:
        """Calculate Shannon entropy for diversification"""
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
        
        max_entropy = math.log2(len(allocation_dict))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def assess_regulatory_risk(self) -> str:
        """Assess overall regulatory risk level"""
        if self.regulatory_risk_exposure < 3:
            return "LOW"
        elif self.regulatory_risk_exposure < 6:
            return "MODERATE"
        elif self.regulatory_risk_exposure < 8:
            return "HIGH"
        else:
            return "EXTREME"
    
    def calculate_yield_sustainability_score(self) -> float:
        """Calculate sustainability of crypto yields"""
        if self.staking_yield_portfolio <= 0:
            return 0.0
        
        # Factor in DeFi vs staking mix (staking more sustainable)
        staking_weight = 1.0 - self.defi_exposure
        defi_weight = self.defi_exposure
        
        # Staking yields more sustainable than DeFi yields
        sustainability = (staking_weight * 0.8) + (defi_weight * 0.4)
        
        # Adjust for yield level (very high yields less sustainable)
        if self.staking_yield_portfolio > 0.2:  # 20%+ yields
            sustainability *= 0.7
        elif self.staking_yield_portfolio > 0.1:  # 10%+ yields
            sustainability *= 0.85
        
        return min(sustainability, 1.0)


@dataclass
class NFTAsset:
    """
    Non-Fungible Token asset data model for NFT investment analysis.
    """
    
    # Basic Information
    collection_name: str
    token_id: str
    blockchain: Blockchain
    contract_address: str
    
    # Market Data
    current_floor_price: float
    last_sale_price: float
    estimated_value: float
    
    # Collection Metrics
    collection_volume_24h: float
    collection_floor_price: float
    total_supply: int
    unique_holders: int
    
    # Rarity and Attributes
    rarity_rank: Optional[int] = None
    rarity_score: Optional[float] = None
    attributes: Optional[Dict[str, str]] = None
    
    # Utility and Rights
    utility_score: float = 0.0  # 0-10 scale
    commercial_rights: bool = False
    governance_rights: bool = False
    
    def calculate_collection_liquidity(self) -> float:
        """Calculate NFT collection liquidity score"""
        if self.collection_floor_price <= 0:
            return 0.0
        
        # Volume to floor price ratio
        volume_ratio = self.collection_volume_24h / self.collection_floor_price
        
        # Holder distribution
        if self.total_supply > 0:
            holder_ratio = self.unique_holders / self.total_supply
        else:
            holder_ratio = 0.0
        
        # Combined liquidity score
        liquidity_score = (volume_ratio * 0.7) + (holder_ratio * 0.3)
        return min(liquidity_score, 1.0)
