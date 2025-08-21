"""
Alternative assets API endpoints for institutional portfolio management.

Provides REST API endpoints for CRUD operations on alternative asset data including
REITs, commodities, cryptocurrencies, and private market investments.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import logging

from src.portfolio.alternative_assets.real_estate import REITSecurity, PropertyType
from src.portfolio.alternative_assets.commodities import CommodityFuture, CommodityType
from src.portfolio.alternative_assets.cryptocurrency import CryptocurrencyAsset, LiquidityTier
from src.portfolio.alternative_assets.private_markets import PrivateMarketInvestment, PrivateMarketType

from src.data.collectors.reit_collector import REITDataCollector
from src.data.collectors.commodity_collector import CommodityDataCollector
from src.data.collectors.crypto_collector import CryptocurrencyDataCollector
from src.data.collectors.private_market_collector import PrivateMarketDataCollector


logger = logging.getLogger(__name__)

# Create router for alternative assets endpoints
router = APIRouter(prefix="/api/v1/alternative-assets", tags=["Alternative Assets"])

# Initialize data collectors
reit_collector = REITDataCollector()
commodity_collector = CommodityDataCollector()
crypto_collector = CryptocurrencyDataCollector()
private_market_collector = PrivateMarketDataCollector()


@router.get("/health")
async def health_check():
    """Health check endpoint for alternative assets service"""
    return {
        "status": "healthy",
        "service": "alternative_assets_api",
        "timestamp": datetime.now().isoformat(),
        "collectors": {
            "reit": "initialized",
            "commodity": "initialized", 
            "crypto": "initialized",
            "private_markets": "initialized"
        }
    }


# REIT Endpoints
@router.get("/reits", response_model=List[Dict[str, Any]])
async def get_reits(
    symbols: Optional[List[str]] = Query(None, description="List of REIT symbols to fetch"),
    property_type: Optional[PropertyType] = Query(None, description="Filter by property type"),
    min_market_cap: Optional[float] = Query(None, description="Minimum market cap filter"),
    max_illiquidity_factor: Optional[float] = Query(0.5, description="Maximum illiquidity factor")
):
    """
    Get REIT data with optional filtering.
    
    Returns comprehensive REIT data including NAV premiums, FFO ratios,
    and property-specific metrics for institutional analysis.
    """
    try:
        # Collect REIT data
        if symbols:
            reit_data = await reit_collector.collect_public_reit_data(symbols)
        else:
            # Get sample of major REITs
            major_reits = ['VNQ', 'SPG', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'EXR']
            reit_data = await reit_collector.collect_public_reit_data(major_reits)
        
        # Apply filters
        filtered_data = []
        for reit in reit_data:
            # Property type filter
            if property_type and reit.property_type != property_type:
                continue
                
            # Market cap filter
            if min_market_cap and reit.market_cap < min_market_cap:
                continue
                
            # Illiquidity filter
            if max_illiquidity_factor and reit.illiquidity_factor > max_illiquidity_factor:
                continue
            
            # Convert to dict for JSON response
            reit_dict = {
                "symbol": reit.symbol,
                "name": reit.name,
                "property_type": reit.property_type.value,
                "market_cap": reit.market_cap,
                "nav_per_share": reit.nav_per_share,
                "market_price": reit.market_price,
                "nav_premium_discount": reit.nav_premium_discount,
                "dividend_yield": reit.dividend_yield,
                "funds_from_operations": reit.funds_from_operations,
                "price_to_ffo": reit.price_to_ffo,
                "occupancy_rate": reit.occupancy_rate,
                "debt_to_total_capital": reit.debt_to_total_capital,
                "liquidity_tier": reit.get_liquidity_tier(),
                "illiquidity_factor": reit.illiquidity_factor,
                "geographic_focus": reit.geographic_focus,
                "property_locations": reit.property_locations
            }
            filtered_data.append(reit_dict)
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error fetching REIT data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch REIT data: {str(e)}")


@router.get("/reits/{symbol}")
async def get_reit_detail(symbol: str):
    """Get detailed data for a specific REIT"""
    try:
        reit_data = await reit_collector.collect_public_reit_data([symbol])
        
        if not reit_data:
            raise HTTPException(status_code=404, detail=f"REIT {symbol} not found")
        
        reit = reit_data[0]
        
        # Calculate additional metrics
        nav_premium_discount = reit.calculate_nav_premium_discount()
        price_to_ffo = reit.calculate_price_to_ffo()
        data_quality = reit.validate_data_quality()
        
        return {
            "basic_info": {
                "symbol": reit.symbol,
                "name": reit.name,
                "exchange": reit.exchange,
                "isin": reit.isin
            },
            "property_details": {
                "property_type": reit.property_type.value,
                "geographic_focus": reit.geographic_focus,
                "property_locations": reit.property_locations,
                "occupancy_rate": reit.occupancy_rate
            },
            "financial_metrics": {
                "market_cap": reit.market_cap,
                "nav_per_share": reit.nav_per_share,
                "market_price": reit.market_price,
                "nav_premium_discount": nav_premium_discount,
                "dividend_yield": reit.dividend_yield,
                "funds_from_operations": reit.funds_from_operations,
                "price_to_ffo": price_to_ffo,
                "debt_to_total_capital": reit.debt_to_total_capital
            },
            "liquidity_metrics": {
                "average_daily_volume": reit.average_daily_volume,
                "bid_ask_spread": reit.bid_ask_spread,
                "illiquidity_factor": reit.illiquidity_factor,
                "liquidity_tier": reit.get_liquidity_tier()
            },
            "risk_metrics": {
                "beta_to_reit_index": reit.beta_to_reit_index,
                "correlation_to_equities": reit.correlation_to_equities
            },
            "data_quality": {
                "issues": data_quality,
                "quality_score": max(0, 10 - len(data_quality))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching REIT detail for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch REIT detail: {str(e)}")


# Commodity Endpoints
@router.get("/commodities", response_model=List[Dict[str, Any]])
async def get_commodities(
    symbols: Optional[List[str]] = Query(None, description="List of commodity symbols"),
    commodity_type: Optional[CommodityType] = Query(None, description="Filter by commodity type"),
    min_open_interest: Optional[int] = Query(None, description="Minimum open interest filter")
):
    """Get commodity futures data with optional filtering"""
    try:
        # Collect commodity data
        if symbols:
            commodity_data = await commodity_collector.collect_futures_data(symbols)
        else:
            # Get sample of major commodities
            major_commodities = ['CL=F', 'GC=F', 'SI=F', 'NG=F', 'ZC=F', 'ZS=F']
            commodity_data = await commodity_collector.collect_futures_data(major_commodities)
        
        # Apply filters
        filtered_data = []
        for commodity in commodity_data:
            # Commodity type filter
            if commodity_type and commodity.commodity_type != commodity_type:
                continue
                
            # Open interest filter
            if min_open_interest and commodity.open_interest < min_open_interest:
                continue
            
            # Convert to dict
            commodity_dict = {
                "symbol": commodity.symbol,
                "commodity_name": commodity.commodity_name,
                "commodity_type": commodity.commodity_type.value,
                "subcategory": commodity.subcategory.value,
                "exchange": commodity.exchange.value,
                "contract_month": commodity.contract_month,
                "expiration_date": commodity.expiration_date.isoformat(),
                "spot_price": commodity.spot_price,
                "futures_price": commodity.futures_price,
                "basis": commodity.basis,
                "curve_structure": commodity.determine_curve_structure(),
                "contract_size": commodity.contract_size,
                "open_interest": commodity.open_interest,
                "volume": commodity.volume,
                "volatility_30d": commodity.volatility_30d,
                "storage_cost": commodity.storage_cost,
                "convenience_yield": commodity.convenience_yield,
                "days_to_expiration": commodity.calculate_days_to_expiration(),
                "geopolitical_risk_score": commodity.geopolitical_risk_score
            }
            filtered_data.append(commodity_dict)
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error fetching commodity data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch commodity data: {str(e)}")


# Cryptocurrency Endpoints  
@router.get("/crypto", response_model=List[Dict[str, Any]])
async def get_cryptocurrencies(
    symbols: Optional[List[str]] = Query(None, description="List of crypto symbols"),
    liquidity_tier: Optional[LiquidityTier] = Query(None, description="Filter by liquidity tier"),
    min_market_cap: Optional[float] = Query(None, description="Minimum market cap filter"),
    max_volatility: Optional[float] = Query(None, description="Maximum 30d volatility filter")
):
    """Get cryptocurrency data with optional filtering"""
    try:
        # Collect crypto data
        if symbols:
            crypto_data = await crypto_collector.collect_crypto_data(symbols)
        else:
            # Get sample of major cryptocurrencies
            major_cryptos = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'USDC-USD']
            crypto_data = await crypto_collector.collect_crypto_data(major_cryptos)
        
        # Apply filters
        filtered_data = []
        for crypto in crypto_data:
            # Liquidity tier filter
            if liquidity_tier and crypto.liquidity_tier != liquidity_tier:
                continue
                
            # Market cap filter
            if min_market_cap and crypto.market_cap < min_market_cap:
                continue
                
            # Volatility filter
            if max_volatility and crypto.volatility_30d > max_volatility:
                continue
            
            # Convert to dict
            crypto_dict = {
                "symbol": crypto.symbol,
                "name": crypto.name,
                "blockchain": crypto.blockchain.value,
                "market_cap": crypto.market_cap,
                "price_usd": crypto.price_usd,
                "circulating_supply": crypto.circulating_supply,
                "max_supply": crypto.max_supply,
                "trading_volume_24h": crypto.trading_volume_24h,
                "volatility_30d": crypto.volatility_30d,
                "volatility_90d": crypto.volatility_90d,
                "max_drawdown_1y": crypto.max_drawdown_1y,
                "correlation_btc": crypto.correlation_btc,
                "correlation_sp500": crypto.correlation_sp500,
                "liquidity_tier": crypto.liquidity_tier.value,
                "asset_category": crypto.asset_category.value if crypto.asset_category else None,
                "staking_rewards": crypto.staking_rewards,
                "yield_farming_apy": crypto.yield_farming_apy,
                "regulatory_risk_score": crypto.regulatory_risk_score,
                "volatility_score": crypto.calculate_volatility_score(),
                "liquidity_score": crypto.calculate_liquidity_score(),
                "network_security_score": crypto.calculate_network_security_score()
            }
            filtered_data.append(crypto_dict)
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch crypto data: {str(e)}")


# Private Market Endpoints
@router.get("/private-markets", response_model=List[Dict[str, Any]])
async def get_private_markets(
    fund_ids: Optional[List[str]] = Query(None, description="List of fund IDs"),
    strategy: Optional[PrivateMarketType] = Query(None, description="Filter by strategy"),
    min_fund_size: Optional[float] = Query(None, description="Minimum fund size filter"),
    vintage_year: Optional[int] = Query(None, description="Filter by vintage year")
):
    """Get private market fund data with optional filtering"""
    try:
        # Collect private market data
        if fund_ids:
            pm_data = await private_market_collector.collect_private_market_data(fund_ids)
        else:
            # Get sample funds
            sample_funds = ['KKR_XIII', 'SEQUOIA_XV', 'BRIDGEWATER', 'BROOKFIELD_IV']
            pm_data = await private_market_collector.collect_private_market_data(sample_funds)
        
        # Apply filters
        filtered_data = []
        for fund in pm_data:
            # Strategy filter
            if strategy and fund.strategy != strategy:
                continue
                
            # Fund size filter
            if min_fund_size and fund.fund_size < min_fund_size:
                continue
                
            # Vintage year filter
            if vintage_year and fund.vintage_year != vintage_year:
                continue
            
            # Convert to dict
            fund_dict = {
                "fund_id": fund.fund_id,
                "fund_name": fund.fund_name,
                "fund_manager": fund.fund_manager,
                "strategy": fund.strategy.value,
                "vintage_year": fund.vintage_year,
                "fund_size": fund.fund_size,
                "committed_capital": fund.committed_capital,
                "called_capital": fund.called_capital,
                "distributed_capital": fund.distributed_capital,
                "nav": fund.nav,
                "irr": fund.irr,
                "tvpi": fund.calculate_tvpi(),
                "dpi": fund.calculate_dpi(),
                "rvpi": fund.calculate_rvpi(),
                "fund_status": fund.fund_status.value if fund.fund_status else None,
                "fund_maturity_stage": fund.get_fund_maturity_stage(),
                "illiquidity_factor": fund.illiquidity_factor,
                "management_fee": fund.management_fee,
                "performance_fee": fund.performance_fee,
                "sector_focus": fund.sector_focus,
                "geographic_focus": fund.geographic_focus.value if fund.geographic_focus else None,
                "j_curve_impact": fund.estimate_j_curve_impact(datetime.now().year - fund.vintage_year),
                "illiquidity_premium": fund.calculate_illiquidity_premium()
            }
            filtered_data.append(fund_dict)
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error fetching private market data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch private market data: {str(e)}")


# Portfolio Analytics Endpoints
@router.post("/portfolio/analysis")
async def analyze_alternative_portfolio(
    allocation: Dict[str, Dict[str, float]],
    benchmark_symbols: Optional[List[str]] = None
):
    """
    Analyze alternative asset portfolio allocation.
    
    Expected allocation format:
    {
        "reits": {"VNQ": 0.3, "SPG": 0.2},
        "commodities": {"GLD": 0.15, "CL=F": 0.1},
        "crypto": {"BTC-USD": 0.15, "ETH-USD": 0.1},
        "private_markets": {"weights_by_strategy": {"private_equity": 0.5, "venture_capital": 0.3, "hedge_fund": 0.2}}
    }
    """
    try:
        portfolio_analysis = {}
        
        # Analyze REIT allocation
        if "reits" in allocation and allocation["reits"]:
            reit_symbols = list(allocation["reits"].keys())
            reit_weights = list(allocation["reits"].values())
            reit_data = await reit_collector.collect_public_reit_data(reit_symbols)
            
            # Calculate portfolio-level REIT metrics
            portfolio_analysis["reits"] = {
                "allocation_percentage": sum(reit_weights) * 100,
                "weighted_dividend_yield": sum(
                    reit.dividend_yield * weight for reit, weight in zip(reit_data, reit_weights)
                ),
                "weighted_nav_premium": sum(
                    reit.nav_premium_discount * weight for reit, weight in zip(reit_data, reit_weights)
                ),
                "property_type_diversification": _calculate_property_diversification(reit_data, reit_weights),
                "liquidity_assessment": _assess_reit_liquidity(reit_data, reit_weights)
            }
        
        # Analyze Commodity allocation
        if "commodities" in allocation and allocation["commodities"]:
            commodity_symbols = list(allocation["commodities"].keys())
            commodity_weights = list(allocation["commodities"].values())
            commodity_data = await commodity_collector.collect_futures_data(commodity_symbols)
            
            portfolio_analysis["commodities"] = {
                "allocation_percentage": sum(commodity_weights) * 100,
                "commodity_type_diversification": _calculate_commodity_diversification(commodity_data, commodity_weights),
                "weighted_volatility": sum(
                    commodity.volatility_30d * weight for commodity, weight in zip(commodity_data, commodity_weights)
                ),
                "storage_cost_impact": sum(
                    commodity.storage_cost * weight for commodity, weight in zip(commodity_data, commodity_weights)
                ),
                "geopolitical_risk": sum(
                    commodity.geopolitical_risk_score * weight for commodity, weight in zip(commodity_data, commodity_weights)
                )
            }
        
        # Analyze Crypto allocation
        if "crypto" in allocation and allocation["crypto"]:
            crypto_symbols = list(allocation["crypto"].keys())
            crypto_weights = list(allocation["crypto"].values())
            crypto_data = await crypto_collector.collect_crypto_data(crypto_symbols)
            
            if crypto_data:
                crypto_metrics = await crypto_collector.calculate_portfolio_metrics(crypto_data, crypto_weights)
                portfolio_analysis["crypto"] = {
                    "allocation_percentage": sum(crypto_weights) * 100,
                    "portfolio_volatility": crypto_metrics.portfolio_volatility,
                    "btc_correlation": crypto_metrics.correlation_btc,
                    "traditional_asset_correlation": crypto_metrics.correlation_traditional_assets,
                    "staking_yield_potential": crypto_metrics.staking_yield_potential,
                    "defi_yield_potential": crypto_metrics.defi_yield_potential,
                    "regulatory_risk_score": crypto_metrics.regulatory_risk_score,
                    "liquidity_assessment": crypto_metrics.liquidity_assessment
                }
        
        # Analyze Private Markets allocation
        if "private_markets" in allocation and allocation["private_markets"]:
            # Use sample funds for strategy allocation analysis
            all_funds = await private_market_collector.collect_private_market_data()
            
            if all_funds:
                pm_metrics = await private_market_collector.calculate_portfolio_metrics(all_funds)
                portfolio_analysis["private_markets"] = {
                    "allocation_percentage": sum(allocation["private_markets"].get("weights_by_strategy", {}).values()) * 100,
                    "portfolio_irr": pm_metrics.portfolio_irr,
                    "portfolio_tvpi": pm_metrics.portfolio_tvpi,
                    "strategy_diversification": pm_metrics.strategy_allocation,
                    "vintage_diversification": pm_metrics.vintage_diversification,
                    "illiquidity_factor": pm_metrics.illiquidity_factor,
                    "j_curve_impact": pm_metrics.j_curve_impact
                }
        
        # Overall portfolio assessment
        total_alt_allocation = sum(
            sum(asset_allocation.values()) for asset_allocation in allocation.values()
            if isinstance(asset_allocation, dict) and "weights_by_strategy" not in asset_allocation
        )
        
        if "private_markets" in allocation:
            total_alt_allocation += sum(allocation["private_markets"].get("weights_by_strategy", {}).values())
        
        portfolio_analysis["summary"] = {
            "total_alternative_allocation": total_alt_allocation * 100,
            "diversification_score": _calculate_alt_diversification_score(portfolio_analysis),
            "liquidity_profile": _assess_portfolio_liquidity(portfolio_analysis),
            "risk_assessment": _assess_portfolio_risk(portfolio_analysis),
            "yield_opportunities": _calculate_yield_opportunities(portfolio_analysis)
        }
        
        return portfolio_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing alternative portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze portfolio: {str(e)}")


# Helper functions for portfolio analysis
def _calculate_property_diversification(reit_data: List[REITSecurity], weights: List[float]) -> Dict[str, float]:
    """Calculate property type diversification for REIT portfolio"""
    property_allocation = {}
    for reit, weight in zip(reit_data, weights):
        prop_type = reit.property_type.value
        property_allocation[prop_type] = property_allocation.get(prop_type, 0) + weight
    return property_allocation


def _assess_reit_liquidity(reit_data: List[REITSecurity], weights: List[float]) -> Dict[str, float]:
    """Assess liquidity profile of REIT portfolio"""
    weighted_illiquidity = sum(reit.illiquidity_factor * weight for reit, weight in zip(reit_data, weights))
    
    liquidity_tiers = {"HIGHLY_LIQUID": 0, "LIQUID": 0, "MODERATELY_LIQUID": 0, "ILLIQUID": 0}
    for reit, weight in zip(reit_data, weights):
        tier = reit.get_liquidity_tier()
        liquidity_tiers[tier] += weight
    
    return {
        "weighted_illiquidity_factor": weighted_illiquidity,
        "liquidity_tier_allocation": liquidity_tiers
    }


def _calculate_commodity_diversification(commodity_data: List[CommodityFuture], weights: List[float]) -> Dict[str, float]:
    """Calculate commodity type diversification"""
    commodity_allocation = {}
    for commodity, weight in zip(commodity_data, weights):
        comm_type = commodity.commodity_type.value
        commodity_allocation[comm_type] = commodity_allocation.get(comm_type, 0) + weight
    return commodity_allocation


def _calculate_alt_diversification_score(portfolio_analysis: Dict) -> float:
    """Calculate overall alternative asset diversification score"""
    asset_class_count = len([k for k in portfolio_analysis.keys() if k != "summary"])
    
    # Penalize concentration in single asset class
    allocations = []
    for key, data in portfolio_analysis.items():
        if key != "summary" and isinstance(data, dict) and "allocation_percentage" in data:
            allocations.append(data["allocation_percentage"])
    
    if not allocations:
        return 0.0
    
    # Calculate Herfindahl-Hirschman Index (lower = more diversified)
    hhi = sum((alloc / 100) ** 2 for alloc in allocations)
    diversification_score = max(0, (1 - hhi) * 10)  # Scale to 0-10
    
    return diversification_score


def _assess_portfolio_liquidity(portfolio_analysis: Dict) -> str:
    """Assess overall portfolio liquidity"""
    illiquid_allocation = 0
    
    # Check private markets (highly illiquid)
    if "private_markets" in portfolio_analysis:
        illiquid_allocation += portfolio_analysis["private_markets"].get("allocation_percentage", 0)
    
    # Check REIT illiquidity
    if "reits" in portfolio_analysis:
        reit_liquidity = portfolio_analysis["reits"].get("liquidity_assessment", {})
        illiquid_reits = reit_liquidity.get("liquidity_tier_allocation", {}).get("ILLIQUID", 0)
        illiquid_allocation += illiquid_reits * portfolio_analysis["reits"].get("allocation_percentage", 0) / 100
    
    if illiquid_allocation > 30:
        return "LOW_LIQUIDITY"
    elif illiquid_allocation > 15:
        return "MODERATE_LIQUIDITY"
    else:
        return "HIGH_LIQUIDITY"


def _assess_portfolio_risk(portfolio_analysis: Dict) -> Dict[str, float]:
    """Assess portfolio-level risk metrics"""
    risk_metrics = {}
    
    # Aggregate volatility
    volatilities = []
    if "commodities" in portfolio_analysis:
        volatilities.append(portfolio_analysis["commodities"].get("weighted_volatility", 0))
    if "crypto" in portfolio_analysis:
        volatilities.append(portfolio_analysis["crypto"].get("portfolio_volatility", 0))
    
    risk_metrics["weighted_volatility"] = sum(volatilities) / len(volatilities) if volatilities else 0
    
    # Regulatory risk
    reg_risks = []
    if "crypto" in portfolio_analysis:
        reg_risks.append(portfolio_analysis["crypto"].get("regulatory_risk_score", 0))
    
    risk_metrics["regulatory_risk"] = sum(reg_risks) / len(reg_risks) if reg_risks else 0
    
    # Geopolitical risk
    if "commodities" in portfolio_analysis:
        risk_metrics["geopolitical_risk"] = portfolio_analysis["commodities"].get("geopolitical_risk", 0)
    
    return risk_metrics


def _calculate_yield_opportunities(portfolio_analysis: Dict) -> Dict[str, float]:
    """Calculate portfolio yield opportunities"""
    yield_opportunities = {}
    
    if "reits" in portfolio_analysis:
        yield_opportunities["dividend_yield"] = portfolio_analysis["reits"].get("weighted_dividend_yield", 0)
    
    if "crypto" in portfolio_analysis:
        yield_opportunities["staking_yield"] = portfolio_analysis["crypto"].get("staking_yield_potential", 0)
        yield_opportunities["defi_yield"] = portfolio_analysis["crypto"].get("defi_yield_potential", 0)
    
    if "private_markets" in portfolio_analysis:
        yield_opportunities["private_market_irr"] = portfolio_analysis["private_markets"].get("portfolio_irr", 0)
    
    return yield_opportunities


# Data refresh endpoints
@router.post("/refresh/reits")
async def refresh_reit_data():
    """Refresh REIT data universe"""
    try:
        # Reinitialize collector to refresh universe
        global reit_collector
        reit_collector = REITDataCollector()
        
        return {
            "status": "success",
            "message": "REIT data universe refreshed",
            "timestamp": datetime.now().isoformat(),
            "universe_size": len(reit_collector.public_reit_universe)
        }
    except Exception as e:
        logger.error(f"Error refreshing REIT data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh REIT data: {str(e)}")


@router.post("/refresh/commodities")
async def refresh_commodity_data():
    """Refresh commodity data universe"""
    try:
        global commodity_collector
        commodity_collector = CommodityDataCollector()
        
        return {
            "status": "success",
            "message": "Commodity data universe refreshed",
            "timestamp": datetime.now().isoformat(),
            "futures_universe_size": len(commodity_collector.futures_universe),
            "spot_universe_size": len(commodity_collector.spot_universe)
        }
    except Exception as e:
        logger.error(f"Error refreshing commodity data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh commodity data: {str(e)}")


@router.post("/refresh/crypto")
async def refresh_crypto_data():
    """Refresh cryptocurrency data universe"""
    try:
        global crypto_collector
        crypto_collector = CryptocurrencyDataCollector()
        
        return {
            "status": "success",
            "message": "Cryptocurrency data universe refreshed", 
            "timestamp": datetime.now().isoformat(),
            "universe_size": len(crypto_collector.crypto_universe)
        }
    except Exception as e:
        logger.error(f"Error refreshing crypto data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh crypto data: {str(e)}")


@router.post("/refresh/private-markets")
async def refresh_private_market_data():
    """Refresh private market data universe"""
    try:
        global private_market_collector
        private_market_collector = PrivateMarketDataCollector()
        
        return {
            "status": "success",
            "message": "Private market data universe refreshed",
            "timestamp": datetime.now().isoformat(),
            "pe_universe_size": len(private_market_collector.pe_universe),
            "vc_universe_size": len(private_market_collector.vc_universe),
            "hedge_fund_universe_size": len(private_market_collector.hedge_fund_universe),
            "infrastructure_universe_size": len(private_market_collector.infrastructure_universe)
        }
    except Exception as e:
        logger.error(f"Error refreshing private market data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh private market data: {str(e)}")


# Export router for main application
__all__ = ["router"]
