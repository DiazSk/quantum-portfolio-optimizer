-- Story 1.3: Institutional Audit Trail & Reporting Database Schema
-- Migration: Add audit trail and regulatory reporting tables
-- Author: Development Agent (James)
-- Date: 2025-08-20

-- Enable UUID extension for PostgreSQL
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Immutable audit trail with blockchain-style verification
CREATE TABLE audit_trail_entries (
    id SERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    previous_hash VARCHAR(64),  -- Hash of previous audit entry for chain integrity
    current_hash VARCHAR(64) NOT NULL,   -- SHA-256 hash of current entry
    event_type VARCHAR(50) NOT NULL, -- 'portfolio_decision', 'trade_execution', 'risk_override', 'ml_prediction'
    event_data JSONB NOT NULL, -- Structured event information
    user_id INTEGER,
    portfolio_id INTEGER,
    model_version VARCHAR(50),
    data_sources JSONB, -- Track alternative data sources used
    digital_signature VARCHAR(512), -- RSA digital signature for verification
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_verified BOOLEAN DEFAULT false
);

-- Indexes for audit trail performance
CREATE INDEX idx_audit_trail_event_type ON audit_trail_entries(event_type);
CREATE INDEX idx_audit_trail_created_at ON audit_trail_entries(created_at);
CREATE INDEX idx_audit_trail_user_id ON audit_trail_entries(user_id);
CREATE INDEX idx_audit_trail_portfolio_id ON audit_trail_entries(portfolio_id);
CREATE INDEX idx_audit_trail_event_id ON audit_trail_entries(event_id);
CREATE INDEX idx_audit_trail_hash_chain ON audit_trail_entries(previous_hash);

-- GIN index for JSONB event data querying
CREATE INDEX idx_audit_trail_event_data ON audit_trail_entries USING GIN(event_data);
CREATE INDEX idx_audit_trail_data_sources ON audit_trail_entries USING GIN(data_sources);

-- Regulatory report templates and generation tracking
CREATE TABLE regulatory_reports (
    id SERIAL PRIMARY KEY,
    report_type VARCHAR(50) NOT NULL, -- 'form_pf', 'aifmd', 'solvency_ii'
    report_period_start DATE NOT NULL,
    report_period_end DATE NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    report_data JSONB NOT NULL,
    file_path VARCHAR(500),
    file_size_bytes INTEGER,
    checksum_sha256 VARCHAR(64),
    submission_status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'submitted', 'approved', 'rejected'
    submitted_at TIMESTAMP WITH TIME ZONE,
    approved_by INTEGER,
    approved_at TIMESTAMP WITH TIME ZONE,
    submission_reference VARCHAR(100), -- External filing reference number
    compliance_notes TEXT
);

-- Indexes for regulatory reports
CREATE INDEX idx_regulatory_reports_type ON regulatory_reports(report_type);
CREATE INDEX idx_regulatory_reports_period ON regulatory_reports(report_period_start, report_period_end);
CREATE INDEX idx_regulatory_reports_status ON regulatory_reports(submission_status);
CREATE INDEX idx_regulatory_reports_generated ON regulatory_reports(generated_at);

-- Client reporting configurations and delivery tracking
CREATE TABLE client_reports (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL,
    report_type VARCHAR(50) NOT NULL, -- 'performance', 'risk', 'attribution', 'compliance'
    report_name VARCHAR(255) NOT NULL,
    report_config JSONB NOT NULL, -- Template settings, metrics, formatting
    delivery_schedule VARCHAR(20), -- 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
    delivery_method VARCHAR(20) DEFAULT 'email', -- 'email', 'portal', 'api'
    recipients JSONB, -- Email addresses or delivery endpoints
    last_generated TIMESTAMP WITH TIME ZONE,
    next_due_date DATE,
    is_active BOOLEAN DEFAULT true,
    created_by INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for client reports
CREATE INDEX idx_client_reports_client_id ON client_reports(client_id);
CREATE INDEX idx_client_reports_type ON client_reports(report_type);
CREATE INDEX idx_client_reports_schedule ON client_reports(delivery_schedule);
CREATE INDEX idx_client_reports_due_date ON client_reports(next_due_date);
CREATE INDEX idx_client_reports_active ON client_reports(is_active);

-- Client report delivery history
CREATE TABLE client_report_deliveries (
    id SERIAL PRIMARY KEY,
    client_report_id INTEGER REFERENCES client_reports(id),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    delivered_at TIMESTAMP WITH TIME ZONE,
    delivery_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'sent', 'delivered', 'failed'
    delivery_error TEXT,
    file_path VARCHAR(500),
    file_size_bytes INTEGER,
    recipient_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    last_downloaded TIMESTAMP WITH TIME ZONE
);

-- Indexes for client report deliveries
CREATE INDEX idx_client_deliveries_report_id ON client_report_deliveries(client_report_id);
CREATE INDEX idx_client_deliveries_status ON client_report_deliveries(delivery_status);
CREATE INDEX idx_client_deliveries_generated ON client_report_deliveries(generated_at);

-- ML model lineage and data provenance tracking
CREATE TABLE ml_lineage_tracking (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_name VARCHAR(255),
    training_data_hash VARCHAR(64),
    feature_sources JSONB, -- Track which alternative data sources used
    training_dataset_info JSONB, -- Dataset metadata and statistics
    training_timestamp TIMESTAMP WITH TIME ZONE,
    training_duration_minutes INTEGER,
    hyperparameters JSONB,
    performance_metrics JSONB, -- Accuracy, precision, recall, etc.
    prediction_confidence_stats JSONB, -- Confidence score distributions
    data_lineage JSONB, -- Complete data provenance chain
    validation_results JSONB,
    deployment_status VARCHAR(20) DEFAULT 'training', -- 'training', 'validation', 'production', 'retired'
    deployed_at TIMESTAMP WITH TIME ZONE,
    retired_at TIMESTAMP WITH TIME ZONE,
    created_by INTEGER,
    notes TEXT
);

-- Indexes for ML lineage tracking
CREATE INDEX idx_ml_lineage_model_id ON ml_lineage_tracking(model_id);
CREATE INDEX idx_ml_lineage_version ON ml_lineage_tracking(model_version);
CREATE INDEX idx_ml_lineage_timestamp ON ml_lineage_tracking(training_timestamp);
CREATE INDEX idx_ml_lineage_status ON ml_lineage_tracking(deployment_status);

-- GIN indexes for JSONB columns in ML lineage
CREATE INDEX idx_ml_lineage_features ON ml_lineage_tracking USING GIN(feature_sources);
CREATE INDEX idx_ml_lineage_data_lineage ON ml_lineage_tracking USING GIN(data_lineage);
CREATE INDEX idx_ml_lineage_performance ON ml_lineage_tracking USING GIN(performance_metrics);

-- ML prediction audit log for individual predictions
CREATE TABLE ml_prediction_audit (
    id SERIAL PRIMARY KEY,
    prediction_id UUID DEFAULT uuid_generate_v4(),
    model_id VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    input_data_hash VARCHAR(64),
    input_features JSONB,
    prediction_output JSONB,
    confidence_score FLOAT,
    feature_importance JSONB,
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    execution_time_ms INTEGER,
    portfolio_id INTEGER,
    trade_id VARCHAR(100),
    business_context JSONB, -- How prediction was used in business logic
    data_sources_used JSONB, -- Alternative data sources for this prediction
    model_drift_score FLOAT,
    validation_status VARCHAR(20) DEFAULT 'pending' -- 'pending', 'validated', 'flagged'
);

-- Indexes for ML prediction audit
CREATE INDEX idx_ml_prediction_model ON ml_prediction_audit(model_id, model_version);
CREATE INDEX idx_ml_prediction_timestamp ON ml_prediction_audit(prediction_timestamp);
CREATE INDEX idx_ml_prediction_portfolio ON ml_prediction_audit(portfolio_id);
CREATE INDEX idx_ml_prediction_confidence ON ml_prediction_audit(confidence_score);

-- GIN indexes for JSONB in prediction audit
CREATE INDEX idx_ml_prediction_features ON ml_prediction_audit USING GIN(input_features);
CREATE INDEX idx_ml_prediction_output ON ml_prediction_audit USING GIN(prediction_output);

-- Compliance filing calendar and deadline tracking
CREATE TABLE compliance_filings (
    id SERIAL PRIMARY KEY,
    filing_type VARCHAR(50) NOT NULL, -- 'form_pf', 'aifmd', 'solvency_ii', 'client_report'
    filing_name VARCHAR(255) NOT NULL,
    regulatory_authority VARCHAR(100), -- 'SEC', 'ESMA', 'FCA', etc.
    filing_frequency VARCHAR(20), -- 'quarterly', 'annual', 'monthly'
    due_date DATE NOT NULL,
    filing_period_start DATE,
    filing_period_end DATE,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'in_progress', 'submitted', 'approved'
    assigned_to INTEGER,
    priority VARCHAR(10) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    estimated_hours INTEGER,
    actual_hours INTEGER,
    submission_method VARCHAR(50), -- 'online_portal', 'email', 'physical_mail'
    submission_reference VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for compliance filings
CREATE INDEX idx_compliance_filings_type ON compliance_filings(filing_type);
CREATE INDEX idx_compliance_filings_due_date ON compliance_filings(due_date);
CREATE INDEX idx_compliance_filings_status ON compliance_filings(status);
CREATE INDEX idx_compliance_filings_authority ON compliance_filings(regulatory_authority);
CREATE INDEX idx_compliance_filings_assigned ON compliance_filings(assigned_to);

-- Compliance workflow and approval tracking
CREATE TABLE compliance_workflows (
    id SERIAL PRIMARY KEY,
    workflow_type VARCHAR(50) NOT NULL, -- 'report_review', 'filing_approval', 'risk_override'
    entity_type VARCHAR(50), -- 'regulatory_report', 'client_report', 'audit_event'
    entity_id INTEGER,
    workflow_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'in_review', 'approved', 'rejected'
    initiated_by INTEGER,
    assigned_to INTEGER,
    reviewer_id INTEGER,
    approval_level VARCHAR(20), -- 'first_level', 'second_level', 'final'
    priority VARCHAR(10) DEFAULT 'medium',
    deadline DATE,
    comments TEXT,
    approval_criteria JSONB,
    review_checklist JSONB,
    initiated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for compliance workflows
CREATE INDEX idx_compliance_workflows_type ON compliance_workflows(workflow_type);
CREATE INDEX idx_compliance_workflows_status ON compliance_workflows(workflow_status);
CREATE INDEX idx_compliance_workflows_assigned ON compliance_workflows(assigned_to);
CREATE INDEX idx_compliance_workflows_deadline ON compliance_workflows(deadline);

-- Compliance rule changes and impact tracking
CREATE TABLE regulatory_change_tracking (
    id SERIAL PRIMARY KEY,
    change_type VARCHAR(50) NOT NULL, -- 'rule_change', 'new_requirement', 'deadline_change'
    regulatory_authority VARCHAR(100),
    regulation_name VARCHAR(255),
    change_description TEXT NOT NULL,
    effective_date DATE,
    impact_assessment JSONB, -- Business impact analysis
    implementation_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'in_progress', 'completed'
    implementation_deadline DATE,
    assigned_to INTEGER,
    estimated_effort_hours INTEGER,
    actual_effort_hours INTEGER,
    compliance_risk_level VARCHAR(10), -- 'low', 'medium', 'high', 'critical'
    mitigation_actions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for regulatory change tracking
CREATE INDEX idx_regulatory_changes_type ON regulatory_change_tracking(change_type);
CREATE INDEX idx_regulatory_changes_effective ON regulatory_change_tracking(effective_date);
CREATE INDEX idx_regulatory_changes_status ON regulatory_change_tracking(implementation_status);
CREATE INDEX idx_regulatory_changes_deadline ON regulatory_change_tracking(implementation_deadline);
CREATE INDEX idx_regulatory_changes_risk ON regulatory_change_tracking(compliance_risk_level);

-- Create views for common audit trail queries
CREATE VIEW audit_trail_summary AS
SELECT 
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT portfolio_id) as unique_portfolios,
    MIN(created_at) as first_event,
    MAX(created_at) as last_event,
    COUNT(CASE WHEN is_verified = true THEN 1 END) as verified_events,
    COUNT(CASE WHEN digital_signature IS NOT NULL THEN 1 END) as signed_events
FROM audit_trail_entries
GROUP BY event_type;

-- Create view for compliance dashboard
CREATE VIEW compliance_dashboard_summary AS
SELECT 
    (SELECT COUNT(*) FROM compliance_filings WHERE due_date <= CURRENT_DATE + INTERVAL '30 days' AND status != 'completed') as upcoming_filings,
    (SELECT COUNT(*) FROM compliance_filings WHERE due_date < CURRENT_DATE AND status != 'completed') as overdue_filings,
    (SELECT COUNT(*) FROM compliance_workflows WHERE workflow_status = 'pending') as pending_approvals,
    (SELECT COUNT(*) FROM regulatory_change_tracking WHERE implementation_status != 'completed' AND compliance_risk_level IN ('high', 'critical')) as high_risk_changes,
    (SELECT COUNT(*) FROM audit_trail_entries WHERE created_at >= CURRENT_DATE - INTERVAL '24 hours') as audit_events_today,
    (SELECT COUNT(*) FROM ml_prediction_audit WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL '24 hours') as ml_predictions_today;

-- Add table comments for documentation
COMMENT ON TABLE audit_trail_entries IS 'Immutable audit trail with blockchain-style hash chain verification for regulatory compliance';
COMMENT ON TABLE regulatory_reports IS 'Automated regulatory report generation and submission tracking';
COMMENT ON TABLE client_reports IS 'Client reporting configurations and delivery management';
COMMENT ON TABLE ml_lineage_tracking IS 'Complete ML model lineage and data provenance tracking';
COMMENT ON TABLE compliance_filings IS 'Regulatory filing calendar and deadline management';
COMMENT ON TABLE compliance_workflows IS 'Compliance workflow and approval process tracking';

-- Add column comments for key fields
COMMENT ON COLUMN audit_trail_entries.current_hash IS 'SHA-256 hash of entry content for tamper detection';
COMMENT ON COLUMN audit_trail_entries.previous_hash IS 'Hash of previous entry to create immutable chain';
COMMENT ON COLUMN audit_trail_entries.digital_signature IS 'RSA digital signature for cryptographic verification';
COMMENT ON COLUMN audit_trail_entries.event_data IS 'Structured JSON containing complete event details';

-- Grant permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT ON audit_trail_entries TO audit_system_role;
-- GRANT SELECT, INSERT, UPDATE ON regulatory_reports TO compliance_officer_role;
-- GRANT SELECT ON audit_trail_summary TO audit_reader_role;

-- Insert initial test data for development (remove in production)
-- This helps verify the schema works correctly
INSERT INTO audit_trail_entries (event_type, event_data, user_id, is_verified) VALUES 
('system_initialization', '{"action": "schema_migration", "version": "1.3.0", "description": "Initial audit trail schema creation"}', 1, true);

-- Migration completion
INSERT INTO audit_trail_entries (event_type, event_data, is_verified) VALUES 
('schema_migration', '{"migration": "add_audit_reporting_tables", "story": "1.3", "completed_at": "' || NOW() || '"}', true);
