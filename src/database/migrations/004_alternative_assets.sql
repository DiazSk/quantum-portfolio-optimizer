-- Alternative Assets Database Schema Migration
-- Version: 004_alternative_assets.sql
-- Description: Comprehensive alternative asset data storage for institutional portfolio management
-- Date: 2025-08-20

-- ============================================================================
-- Alternative Assets Core Tables
-- ============================================================================

-- Main alternative assets table with flexible JSONB storage
CREATE TABLE IF NOT EXISTS alternative_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    asset_class VARCHAR(50) NOT NULL, -- reit, commodity, crypto, private_market
    symbol VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    isin VARCHAR(12),
    exchange VARCHAR(50),
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Market Data
    market_cap DECIMAL(20,2),
    current_price DECIMAL(15,4),
    daily_volume DECIMAL(20,2),
    
    -- Risk and Liquidity
    liquidity_tier VARCHAR(20), -- LIQUID, MODERATE, ILLIQUID, HIGHLY_ILLIQUID
    illiquidity_score DECIMAL(6,4) CHECK (illiquidity_score >= 0 AND illiquidity_score <= 1),
    volatility_regime VARCHAR(20), -- Low, Medium, High, Extreme
    
    -- Geographic and Sector
    sector VARCHAR(100),
    geographic_region VARCHAR(50),
    
    -- Asset-specific data stored as JSONB for flexibility
    data_model JSONB NOT NULL,
    
    -- Data Quality and Timestamps
    data_quality_score DECIMAL(4,2),
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(tenant_id, symbol, asset_class),
    CHECK (market_cap >= 0),
    CHECK (current_price >= 0),
    CHECK (data_quality_score >= 0 AND data_quality_score <= 10)
);

-- Index for efficient querying
CREATE INDEX IF NOT EXISTS idx_alternative_assets_tenant_class ON alternative_assets(tenant_id, asset_class);
CREATE INDEX IF NOT EXISTS idx_alternative_assets_symbol ON alternative_assets(symbol);
CREATE INDEX IF NOT EXISTS idx_alternative_assets_sector ON alternative_assets(sector);
CREATE INDEX IF NOT EXISTS idx_alternative_assets_updated ON alternative_assets(last_updated);

-- ============================================================================
-- Price History Tables
-- ============================================================================

-- Alternative asset price history with multiple source support
CREATE TABLE IF NOT EXISTS alternative_asset_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES alternative_assets(id) ON DELETE CASCADE,
    price_date DATE NOT NULL,
    open_price DECIMAL(15,4),
    high_price DECIMAL(15,4),
    low_price DECIMAL(15,4),
    close_price DECIMAL(15,4) NOT NULL,
    volume DECIMAL(20,2),
    adjusted_close DECIMAL(15,4),
    
    -- Data source tracking
    source VARCHAR(100) NOT NULL,
    source_quality_score DECIMAL(4,2),
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(asset_id, price_date, source),
    CHECK (open_price >= 0),
    CHECK (high_price >= 0),
    CHECK (low_price >= 0),
    CHECK (close_price >= 0),
    CHECK (volume >= 0),
    CHECK (source_quality_score >= 0 AND source_quality_score <= 10)
);

CREATE INDEX IF NOT EXISTS idx_alt_prices_asset_date ON alternative_asset_prices(asset_id, price_date DESC);
CREATE INDEX IF NOT EXISTS idx_alt_prices_date ON alternative_asset_prices(price_date);

-- ============================================================================
-- Risk Metrics Tables
-- ============================================================================

-- Alternative asset risk metrics with time series support
CREATE TABLE IF NOT EXISTS alternative_asset_risk_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES alternative_assets(id) ON DELETE CASCADE,
    calculation_date DATE NOT NULL,
    
    -- Core Risk Metrics
    volatility_30d DECIMAL(8,6),
    volatility_90d DECIMAL(8,6),
    volatility_1y DECIMAL(8,6),
    max_drawdown DECIMAL(8,6),
    
    -- Correlation Metrics
    correlation_to_market DECIMAL(8,6) CHECK (correlation_to_market >= -1 AND correlation_to_market <= 1),
    correlation_to_bonds DECIMAL(8,6) CHECK (correlation_to_bonds >= -1 AND correlation_to_bonds <= 1),
    correlation_to_commodities DECIMAL(8,6) CHECK (correlation_to_commodities >= -1 AND correlation_to_commodities <= 1),
    beta_to_market DECIMAL(8,6),
    
    -- Alternative Asset Specific
    illiquidity_factor DECIMAL(6,4) CHECK (illiquidity_factor >= 0 AND illiquidity_factor <= 1),
    risk_score DECIMAL(4,2) CHECK (risk_score >= 0 AND risk_score <= 10),
    
    -- Additional metrics stored as JSONB
    metrics JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(asset_id, calculation_date)
);

CREATE INDEX IF NOT EXISTS idx_alt_risk_asset_date ON alternative_asset_risk_metrics(asset_id, calculation_date DESC);

-- ============================================================================
-- REIT-Specific Tables
-- ============================================================================

-- Real Estate Investment Trust specific data
CREATE TABLE IF NOT EXISTS reit_fundamentals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES alternative_assets(id) ON DELETE CASCADE,
    reporting_date DATE NOT NULL,
    
    -- REIT-Specific Financials
    funds_from_operations DECIMAL(15,2), -- FFO
    net_asset_value DECIMAL(15,4), -- NAV per share
    nav_premium_discount DECIMAL(8,6), -- (price - nav) / nav
    dividend_yield DECIMAL(8,6),
    
    -- Property Metrics
    occupancy_rate DECIMAL(6,4) CHECK (occupancy_rate >= 0 AND occupancy_rate <= 1),
    cap_rate DECIMAL(6,4), -- Capitalization rate
    debt_to_total_capital DECIMAL(6,4),
    
    -- Property Details
    property_type VARCHAR(50), -- residential, commercial, industrial, etc.
    property_locations JSONB, -- Array of primary markets
    
    -- Performance Ratios
    price_to_ffo DECIMAL(8,4),
    price_to_nav DECIMAL(8,4),
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(asset_id, reporting_date),
    CHECK (occupancy_rate >= 0 AND occupancy_rate <= 1),
    CHECK (debt_to_total_capital >= 0 AND debt_to_total_capital <= 1)
);

CREATE INDEX IF NOT EXISTS idx_reit_fundamentals_asset ON reit_fundamentals(asset_id);
CREATE INDEX IF NOT EXISTS idx_reit_fundamentals_date ON reit_fundamentals(reporting_date);

-- ============================================================================
-- Commodity-Specific Tables  
-- ============================================================================

-- Commodity futures and physical asset data
CREATE TABLE IF NOT EXISTS commodity_contracts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES alternative_assets(id) ON DELETE CASCADE,
    
    -- Contract Specifications
    contract_month VARCHAR(10) NOT NULL,
    expiration_date DATE NOT NULL,
    contract_size DECIMAL(15,4) NOT NULL,
    price_unit VARCHAR(20), -- $/barrel, $/ounce, etc.
    minimum_tick DECIMAL(10,6),
    tick_value DECIMAL(10,2),
    
    -- Market Data
    spot_price DECIMAL(15,4),
    futures_price DECIMAL(15,4),
    basis DECIMAL(15,4), -- futures - spot
    open_interest INTEGER,
    
    -- Commodity-Specific Factors
    storage_cost DECIMAL(8,6), -- Annual storage cost as % of value
    convenience_yield DECIMAL(8,6),
    seasonal_factor DECIMAL(8,6),
    
    -- Supply/Demand Fundamentals
    global_production DECIMAL(20,2),
    global_consumption DECIMAL(20,2),
    inventory_levels DECIMAL(20,2),
    geopolitical_risk_score DECIMAL(4,2) CHECK (geopolitical_risk_score >= 0 AND geopolitical_risk_score <= 10),
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(asset_id, contract_month),
    CHECK (contract_size > 0),
    CHECK (futures_price >= 0),
    CHECK (spot_price >= 0)
);

CREATE INDEX IF NOT EXISTS idx_commodity_contracts_asset ON commodity_contracts(asset_id);
CREATE INDEX IF NOT EXISTS idx_commodity_contracts_expiration ON commodity_contracts(expiration_date);

-- ============================================================================
-- Cryptocurrency-Specific Tables
-- ============================================================================

-- Cryptocurrency and digital asset data
CREATE TABLE IF NOT EXISTS cryptocurrency_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES alternative_assets(id) ON DELETE CASCADE,
    measurement_date DATE NOT NULL,
    
    -- Supply Metrics
    circulating_supply DECIMAL(25,8),
    total_supply DECIMAL(25,8),
    max_supply DECIMAL(25,8),
    
    -- Trading Metrics
    trading_volume_24h DECIMAL(20,2),
    volume_to_market_cap DECIMAL(8,6),
    number_of_exchanges INTEGER,
    
    -- Blockchain Metrics
    blockchain VARCHAR(50),
    consensus_mechanism VARCHAR(50),
    network_hash_rate DECIMAL(25,8),
    active_addresses INTEGER,
    transaction_volume DECIMAL(25,8),
    
    -- DeFi Specific
    total_value_locked DECIMAL(20,2), -- TVL for DeFi protocols
    yield_farming_apy DECIMAL(8,6),
    staking_rewards DECIMAL(8,6),
    governance_token BOOLEAN DEFAULT FALSE,
    
    -- Risk Assessment
    regulatory_risk_score DECIMAL(4,2) CHECK (regulatory_risk_score >= 0 AND regulatory_risk_score <= 10),
    liquidity_tier VARCHAR(20), -- TIER_1, TIER_2, TIER_3, TIER_4
    
    -- Additional crypto metrics as JSONB
    additional_metrics JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(asset_id, measurement_date),
    CHECK (circulating_supply >= 0),
    CHECK (trading_volume_24h >= 0),
    CHECK (volume_to_market_cap >= 0)
);

CREATE INDEX IF NOT EXISTS idx_crypto_metrics_asset ON cryptocurrency_metrics(asset_id);
CREATE INDEX IF NOT EXISTS idx_crypto_metrics_date ON cryptocurrency_metrics(measurement_date);

-- ============================================================================
-- Private Market-Specific Tables
-- ============================================================================

-- Private market investments (PE, VC, Hedge Funds)
CREATE TABLE IF NOT EXISTS private_market_funds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES alternative_assets(id) ON DELETE CASCADE,
    
    -- Fund Details
    fund_manager VARCHAR(255) NOT NULL,
    vintage_year INTEGER NOT NULL,
    fund_life INTEGER DEFAULT 10, -- years
    investment_period INTEGER DEFAULT 5, -- years
    
    -- Capital Structure
    fund_size DECIMAL(20,2) NOT NULL,
    committed_capital DECIMAL(20,2),
    called_capital DECIMAL(20,2),
    distributed_capital DECIMAL(20,2),
    nav DECIMAL(20,2),
    
    -- Performance Metrics
    irr DECIMAL(8,6), -- Internal Rate of Return
    tvpi DECIMAL(8,6), -- Total Value to Paid-In
    dpi DECIMAL(8,6), -- Distributed to Paid-In
    rvpi DECIMAL(8,6), -- Residual Value to Paid-In
    
    -- Strategy Details
    investment_strategy VARCHAR(100), -- buyout, growth, venture, etc.
    sector_focus JSONB, -- Array of target sectors
    geographic_focus VARCHAR(50),
    
    -- Fee Structure
    management_fee DECIMAL(6,4), -- As decimal (e.g., 0.02 for 2%)
    performance_fee DECIMAL(6,4), -- Carried interest
    high_water_mark BOOLEAN DEFAULT FALSE,
    
    -- Liquidity Terms
    lock_up_period INTEGER, -- months
    redemption_frequency VARCHAR(20), -- monthly, quarterly, annual
    
    -- Risk and Valuation
    illiquidity_factor DECIMAL(6,4) CHECK (illiquidity_factor >= 0 AND illiquidity_factor <= 1),
    valuation_method VARCHAR(50), -- market, income, cost, nav
    last_valuation_date DATE,
    independent_valuation BOOLEAN DEFAULT FALSE,
    
    -- Fund Status
    fund_status VARCHAR(20) DEFAULT 'investing', -- fundraising, investing, harvesting, liquidating
    
    -- J-curve modeling parameters
    j_curve_parameters JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(asset_id),
    CHECK (fund_size > 0),
    CHECK (vintage_year >= 1990 AND vintage_year <= EXTRACT(YEAR FROM NOW())),
    CHECK (committed_capital >= 0),
    CHECK (called_capital >= 0),
    CHECK (distributed_capital >= 0),
    CHECK (management_fee >= 0 AND management_fee <= 0.1), -- Max 10%
    CHECK (performance_fee >= 0 AND performance_fee <= 0.5) -- Max 50%
);

CREATE INDEX IF NOT EXISTS idx_private_funds_asset ON private_market_funds(asset_id);
CREATE INDEX IF NOT EXISTS idx_private_funds_vintage ON private_market_funds(vintage_year);
CREATE INDEX IF NOT EXISTS idx_private_funds_strategy ON private_market_funds(investment_strategy);

-- ============================================================================
-- Portfolio Allocation Tables
-- ============================================================================

-- Alternative asset allocations within portfolios
CREATE TABLE IF NOT EXISTS portfolio_alternative_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL,
    asset_id UUID NOT NULL REFERENCES alternative_assets(id),
    tenant_id UUID NOT NULL,
    
    -- Allocation Details
    target_weight DECIMAL(8,6) NOT NULL CHECK (target_weight >= 0 AND target_weight <= 1),
    current_weight DECIMAL(8,6) NOT NULL CHECK (current_weight >= 0 AND current_weight <= 1),
    position_value DECIMAL(20,2) NOT NULL CHECK (position_value >= 0),
    shares_or_units DECIMAL(20,8),
    
    -- Cost Basis
    cost_basis DECIMAL(20,2),
    unrealized_pnl DECIMAL(20,2),
    
    -- Liquidity Constraints
    liquidity_constraint DECIMAL(8,6), -- Percentage that can be liquidated in 30 days
    redemption_notice_period INTEGER, -- Days notice required
    
    -- Performance Attribution
    contribution_to_return DECIMAL(10,8),
    contribution_to_risk DECIMAL(10,8),
    
    -- Rebalancing
    last_rebalance_date DATE,
    next_review_date DATE,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(portfolio_id, asset_id),
    CHECK (target_weight >= 0 AND target_weight <= 1),
    CHECK (current_weight >= 0 AND current_weight <= 1)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_alt_alloc_portfolio ON portfolio_alternative_allocations(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_alt_alloc_asset ON portfolio_alternative_allocations(asset_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_alt_alloc_tenant ON portfolio_alternative_allocations(tenant_id);

-- ============================================================================
-- Alternative Asset Universe Views
-- ============================================================================

-- View for REIT universe with fundamentals
CREATE OR REPLACE VIEW reit_universe_view AS
SELECT 
    aa.id,
    aa.symbol,
    aa.name,
    aa.market_cap,
    aa.current_price,
    aa.sector,
    aa.geographic_region,
    rf.dividend_yield,
    rf.occupancy_rate,
    rf.nav_premium_discount,
    rf.property_type,
    aarm.volatility_30d,
    aarm.correlation_to_market,
    aa.liquidity_tier,
    aa.illiquidity_score
FROM alternative_assets aa
LEFT JOIN reit_fundamentals rf ON aa.id = rf.asset_id 
    AND rf.reporting_date = (
        SELECT MAX(reporting_date) 
        FROM reit_fundamentals rf2 
        WHERE rf2.asset_id = aa.id
    )
LEFT JOIN alternative_asset_risk_metrics aarm ON aa.id = aarm.asset_id
    AND aarm.calculation_date = (
        SELECT MAX(calculation_date)
        FROM alternative_asset_risk_metrics aarm2
        WHERE aarm2.asset_id = aa.id
    )
WHERE aa.asset_class = 'reit';

-- View for cryptocurrency universe
CREATE OR REPLACE VIEW crypto_universe_view AS
SELECT 
    aa.id,
    aa.symbol,
    aa.name,
    aa.market_cap,
    aa.current_price,
    cm.blockchain,
    cm.consensus_mechanism,
    cm.trading_volume_24h,
    cm.staking_rewards,
    cm.total_value_locked,
    cm.regulatory_risk_score,
    aarm.volatility_30d,
    aarm.correlation_to_market,
    aa.liquidity_tier
FROM alternative_assets aa
LEFT JOIN cryptocurrency_metrics cm ON aa.id = cm.asset_id
    AND cm.measurement_date = (
        SELECT MAX(measurement_date)
        FROM cryptocurrency_metrics cm2
        WHERE cm2.asset_id = aa.id
    )
LEFT JOIN alternative_asset_risk_metrics aarm ON aa.id = aarm.asset_id
    AND aarm.calculation_date = (
        SELECT MAX(calculation_date)
        FROM alternative_asset_risk_metrics aarm2
        WHERE aarm2.asset_id = aa.id
    )
WHERE aa.asset_class = 'crypto';

-- ============================================================================
-- Data Quality and Maintenance
-- ============================================================================

-- Function to update data quality scores
CREATE OR REPLACE FUNCTION update_alternative_asset_data_quality()
RETURNS TRIGGER AS $$
BEGIN
    -- Simple data quality scoring based on field completeness
    NEW.data_quality_score = (
        CASE WHEN NEW.market_cap IS NOT NULL AND NEW.market_cap > 0 THEN 2 ELSE 0 END +
        CASE WHEN NEW.current_price IS NOT NULL AND NEW.current_price > 0 THEN 2 ELSE 0 END +
        CASE WHEN NEW.sector IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN NEW.geographic_region IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN NEW.liquidity_tier IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN NEW.illiquidity_score IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN NEW.data_model IS NOT NULL AND jsonb_array_length(jsonb_object_keys(NEW.data_model)) > 5 THEN 2 ELSE 0 END
    );
    
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for data quality updates
CREATE TRIGGER alternative_assets_data_quality_trigger
    BEFORE INSERT OR UPDATE ON alternative_assets
    FOR EACH ROW
    EXECUTE FUNCTION update_alternative_asset_data_quality();

-- ============================================================================
-- Indexes for Performance Optimization
-- ============================================================================

-- Additional indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_alt_assets_market_cap ON alternative_assets(market_cap DESC);
CREATE INDEX IF NOT EXISTS idx_alt_assets_liquidity_tier ON alternative_assets(liquidity_tier);
CREATE INDEX IF NOT EXISTS idx_alt_assets_class_sector ON alternative_assets(asset_class, sector);

-- Partial indexes for active assets only
CREATE INDEX IF NOT EXISTS idx_alt_assets_active_liquid 
    ON alternative_assets(tenant_id, asset_class, market_cap DESC) 
    WHERE market_cap > 100000000 AND liquidity_tier IN ('LIQUID', 'MODERATE');

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON TABLE alternative_assets IS 'Core alternative assets universe with flexible JSONB storage for asset-specific data';
COMMENT ON TABLE alternative_asset_prices IS 'Historical price data for alternative assets from multiple sources';
COMMENT ON TABLE alternative_asset_risk_metrics IS 'Time series risk calculations and correlation metrics';
COMMENT ON TABLE reit_fundamentals IS 'REIT-specific financial and property metrics';
COMMENT ON TABLE commodity_contracts IS 'Commodity futures contracts and physical asset specifications';
COMMENT ON TABLE cryptocurrency_metrics IS 'Cryptocurrency and digital asset blockchain metrics';
COMMENT ON TABLE private_market_funds IS 'Private equity, VC, and hedge fund investment data';
COMMENT ON TABLE portfolio_alternative_allocations IS 'Alternative asset positions within institutional portfolios';

-- ============================================================================
-- Migration Completion
-- ============================================================================

-- Record migration completion
INSERT INTO schema_migrations (version, description, executed_at) 
VALUES (4, 'Alternative Assets Schema', NOW())
ON CONFLICT (version) DO NOTHING;
