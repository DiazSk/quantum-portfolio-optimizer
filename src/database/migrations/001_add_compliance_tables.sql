-- Compliance Tables Migration
-- Add compliance rules and violations tracking to the database
-- Migration: 001_add_compliance_tables.sql

-- Create compliance rules table
CREATE TABLE IF NOT EXISTS compliance_rules (
    id SERIAL PRIMARY KEY,
    rule_type VARCHAR(50) NOT NULL CHECK (rule_type IN ('position_limit', 'mandate', 'concentration')),
    rule_name VARCHAR(100) NOT NULL,
    rule_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create compliance violations table
CREATE TABLE IF NOT EXISTS compliance_violations (
    id SERIAL PRIMARY KEY,
    portfolio_id UUID,
    rule_id INTEGER NOT NULL REFERENCES compliance_rules(id) ON DELETE CASCADE,
    violation_details JSONB NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    detected_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP,
    is_resolved BOOLEAN DEFAULT false
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_compliance_rules_type_active ON compliance_rules(rule_type, is_active);
CREATE INDEX IF NOT EXISTS idx_compliance_rules_name ON compliance_rules(rule_name);
CREATE INDEX IF NOT EXISTS idx_compliance_violations_portfolio ON compliance_violations(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_compliance_violations_rule ON compliance_violations(rule_id);
CREATE INDEX IF NOT EXISTS idx_compliance_violations_severity ON compliance_violations(severity);
CREATE INDEX IF NOT EXISTS idx_compliance_violations_detected ON compliance_violations(detected_at);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_compliance_rules_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
CREATE TRIGGER compliance_rules_updated_at_trigger
    BEFORE UPDATE ON compliance_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_compliance_rules_updated_at();

-- Insert default compliance rules for common scenarios
INSERT INTO compliance_rules (rule_type, rule_name, rule_config, is_active) VALUES
(
    'position_limit',
    'Maximum Single Position Limit',
    '{
        "max_single_position": 0.15,
        "description": "No single asset can exceed 15% of portfolio",
        "enforcement_level": "strict"
    }',
    true
),
(
    'concentration',
    'Sector Concentration Limit',
    '{
        "max_sector_concentration": 0.25,
        "description": "No sector can exceed 25% of portfolio",
        "enforcement_level": "warning"
    }',
    true
),
(
    'mandate',
    'ESG Minimum Requirements',
    '{
        "min_esg_score": 7.0,
        "description": "All assets must have ESG score >= 7.0",
        "enforcement_level": "strict",
        "exclude_unrated": true
    }',
    true
);

-- Add comments for documentation
COMMENT ON TABLE compliance_rules IS 'Configuration table for portfolio compliance rules';
COMMENT ON TABLE compliance_violations IS 'Historical tracking of compliance violations';
COMMENT ON COLUMN compliance_rules.rule_type IS 'Type of compliance rule: position_limit, mandate, or concentration';
COMMENT ON COLUMN compliance_rules.rule_config IS 'JSON configuration for rule parameters and thresholds';
COMMENT ON COLUMN compliance_violations.violation_details IS 'JSON details about the specific violation occurrence';
COMMENT ON COLUMN compliance_violations.severity IS 'Violation severity level: low, medium, high, or critical';
