-- Migration: Add Risk Alert Tables
-- Description: Create tables for risk alert thresholds, alerts, and notifications
-- Author: GitHub Copilot Dev Agent (James)
-- Date: 2025-08-20

-- Drop tables if they exist (for development)
DROP TABLE IF EXISTS alert_notifications CASCADE;
DROP TABLE IF EXISTS escalation_events CASCADE;
DROP TABLE IF EXISTS risk_alerts CASCADE;
DROP TABLE IF EXISTS risk_alert_thresholds CASCADE;

-- Alert threshold configurations
CREATE TABLE risk_alert_thresholds (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    portfolio_id VARCHAR(100),  -- NULL for global thresholds
    metric_type VARCHAR(50) NOT NULL, -- 'var_95', 'cvar_95', 'max_drawdown', 'concentration', 'leverage', 'volatility', 'sharpe_ratio'
    threshold_value DECIMAL(12,8) NOT NULL,
    comparison_operator VARCHAR(10) NOT NULL DEFAULT '>', -- '>', '<', '>=', '<='
    severity_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Alert events and history
CREATE TABLE risk_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(100) UNIQUE NOT NULL,  -- Custom alert ID
    threshold_id INTEGER REFERENCES risk_alert_thresholds(id) ON DELETE CASCADE,
    portfolio_id VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    current_value DECIMAL(12,8) NOT NULL,
    threshold_value DECIMAL(12,8) NOT NULL,
    severity_level VARCHAR(20) NOT NULL,
    alert_status VARCHAR(20) NOT NULL DEFAULT 'triggered', -- 'triggered', 'acknowledged', 'resolved', 'escalated', 'suppressed'
    description TEXT,
    triggered_at TIMESTAMP DEFAULT NOW(),
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    acknowledgment_user VARCHAR(100),
    resolution_note TEXT
);

-- Escalation events tracking
CREATE TABLE escalation_events (
    id SERIAL PRIMARY KEY,
    escalation_id VARCHAR(100) UNIQUE NOT NULL,  -- Custom escalation ID
    alert_id VARCHAR(100) NOT NULL,  -- References risk_alerts.alert_id
    escalation_rule_id VARCHAR(100),
    from_level VARCHAR(20) NOT NULL, -- 'level_1', 'level_2', 'level_3', 'level_4'
    to_level VARCHAR(20) NOT NULL,
    escalated_at TIMESTAMP DEFAULT NOW(),
    escalated_by VARCHAR(100), -- User ID or 'system'
    reason TEXT,
    notification_sent BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP
);

-- Notification delivery tracking
CREATE TABLE alert_notifications (
    id SERIAL PRIMARY KEY,
    notification_id VARCHAR(100) UNIQUE NOT NULL,  -- Custom notification ID
    alert_id VARCHAR(100) NOT NULL,  -- References risk_alerts.alert_id
    escalation_id VARCHAR(100),  -- Optional reference to escalation_events.escalation_id
    notification_type VARCHAR(20) NOT NULL, -- 'email', 'sms', 'webhook', 'dashboard'
    recipient VARCHAR(255) NOT NULL,
    delivery_status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'sent', 'delivered', 'failed', 'retrying'
    attempts INTEGER DEFAULT 0,
    template_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    sent_at TIMESTAMP,
    delivered_at TIMESTAMP,
    failed_at TIMESTAMP,
    error_message TEXT
);

-- User notification preferences
CREATE TABLE user_notification_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    webhook_url TEXT,
    enabled_types JSONB DEFAULT '["email", "dashboard"]'::jsonb, -- Array of enabled notification types
    severity_filter JSONB DEFAULT '["medium", "high", "critical"]'::jsonb, -- Array of severity levels
    quiet_hours_start TIME, -- HH:MM format
    quiet_hours_end TIME,
    timezone VARCHAR(50) DEFAULT 'UTC',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Escalation rules configuration
CREATE TABLE escalation_rules (
    id SERIAL PRIMARY KEY,
    rule_id VARCHAR(100) UNIQUE NOT NULL,  -- Custom rule ID
    rule_name VARCHAR(200) NOT NULL,
    severity_level VARCHAR(20) NOT NULL,
    escalation_path JSONB NOT NULL, -- JSON array of escalation steps
    auto_escalate BOOLEAN DEFAULT true,
    max_escalation_level VARCHAR(20) DEFAULT 'level_4',
    require_acknowledgment BOOLEAN DEFAULT true,
    escalation_conditions JSONB DEFAULT '["unacknowledged", "time_threshold"]'::jsonb,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User escalation contacts
CREATE TABLE user_escalation_contacts (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) UNIQUE NOT NULL,
    user_role VARCHAR(50) NOT NULL, -- 'analyst', 'portfolio_manager', 'risk_manager', etc.
    email VARCHAR(255),
    phone VARCHAR(50),
    backup_users JSONB, -- Array of backup user IDs
    is_available BOOLEAN DEFAULT true,
    working_hours_start TIME DEFAULT '09:00',
    working_hours_end TIME DEFAULT '17:00',
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_risk_alert_thresholds_user_portfolio ON risk_alert_thresholds(user_id, portfolio_id);
CREATE INDEX idx_risk_alert_thresholds_metric_active ON risk_alert_thresholds(metric_type, is_active);
CREATE INDEX idx_risk_alerts_portfolio_status ON risk_alerts(portfolio_id, alert_status);
CREATE INDEX idx_risk_alerts_triggered_at ON risk_alerts(triggered_at DESC);
CREATE INDEX idx_escalation_events_alert_id ON escalation_events(alert_id);
CREATE INDEX idx_escalation_events_escalated_at ON escalation_events(escalated_at DESC);
CREATE INDEX idx_alert_notifications_alert_id ON alert_notifications(alert_id);
CREATE INDEX idx_alert_notifications_status ON alert_notifications(delivery_status);
CREATE INDEX idx_alert_notifications_created_at ON alert_notifications(created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for auto-updating updated_at
CREATE TRIGGER update_risk_alert_thresholds_updated_at BEFORE UPDATE ON risk_alert_thresholds FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_user_notification_preferences_updated_at BEFORE UPDATE ON user_notification_preferences FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_escalation_rules_updated_at BEFORE UPDATE ON escalation_rules FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_user_escalation_contacts_updated_at BEFORE UPDATE ON user_escalation_contacts FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

-- Insert default escalation rules
INSERT INTO escalation_rules (rule_id, rule_name, severity_level, escalation_path, auto_escalate, require_acknowledgment) VALUES
('critical_escalation', 'Critical Alert Escalation', 'critical', 
 '[
    {"level": "level_1", "roles": ["analyst", "portfolio_manager"], "delay_minutes": 0},
    {"level": "level_2", "roles": ["risk_manager"], "delay_minutes": 5},
    {"level": "level_3", "roles": ["head_of_risk", "cio"], "delay_minutes": 15},
    {"level": "level_4", "roles": ["ceo"], "delay_minutes": 30}
  ]'::jsonb, 
 true, true),

('high_escalation', 'High Alert Escalation', 'high',
 '[
    {"level": "level_1", "roles": ["analyst", "portfolio_manager"], "delay_minutes": 0},
    {"level": "level_2", "roles": ["risk_manager"], "delay_minutes": 15},
    {"level": "level_3", "roles": ["head_of_risk"], "delay_minutes": 60}
  ]'::jsonb,
 true, true),

('medium_escalation', 'Medium Alert Escalation', 'medium',
 '[
    {"level": "level_1", "roles": ["analyst"], "delay_minutes": 0},
    {"level": "level_2", "roles": ["senior_analyst"], "delay_minutes": 30}
  ]'::jsonb,
 false, false);

-- Insert sample alert thresholds
INSERT INTO risk_alert_thresholds (user_id, metric_type, threshold_value, comparison_operator, severity_level, description) VALUES
('system', 'var_95', -0.05, '<', 'high', 'Daily VaR (95%) exceeds -5%'),
('system', 'max_drawdown', -0.20, '<', 'critical', 'Maximum drawdown exceeds -20%'),
('system', 'concentration', 0.30, '>', 'medium', 'Single position concentration exceeds 30%'),
('system', 'leverage', 1.5, '>', 'high', 'Portfolio leverage exceeds 150%'),
('system', 'volatility', 0.25, '>', 'medium', 'Annual volatility exceeds 25%'),
('system', 'cvar_95', -0.08, '<', 'critical', 'CVaR (95%) exceeds -8%'),
('system', 'sharpe_ratio', 0.5, '<', 'low', 'Sharpe ratio below 0.5');

-- Insert sample user escalation contacts
INSERT INTO user_escalation_contacts (user_id, user_role, email, phone, is_available) VALUES
('analyst_1', 'analyst', 'analyst1@company.com', '+1234567890', true),
('pm_1', 'portfolio_manager', 'pm1@company.com', '+1234567891', true),
('risk_mgr', 'risk_manager', 'risk@company.com', '+1234567892', true),
('head_risk', 'head_of_risk', 'head.risk@company.com', '+1234567893', true),
('cio', 'cio', 'cio@company.com', '+1234567894', true),
('ceo', 'ceo', 'ceo@company.com', '+1234567895', true);

-- Insert sample notification preferences
INSERT INTO user_notification_preferences (user_id, email, phone, enabled_types, severity_filter) VALUES
('analyst_1', 'analyst1@company.com', '+1234567890', '["email", "dashboard"]'::jsonb, '["medium", "high", "critical"]'::jsonb),
('pm_1', 'pm1@company.com', '+1234567891', '["email", "sms", "dashboard"]'::jsonb, '["high", "critical"]'::jsonb),
('risk_mgr', 'risk@company.com', '+1234567892', '["email", "sms", "dashboard"]'::jsonb, '["medium", "high", "critical"]'::jsonb),
('head_risk', 'head.risk@company.com', '+1234567893', '["email", "sms"]'::jsonb, '["high", "critical"]'::jsonb),
('cio', 'cio@company.com', '+1234567894', '["email", "sms"]'::jsonb, '["critical"]'::jsonb);

-- Views for easier querying
CREATE VIEW active_alerts AS
SELECT 
    ra.alert_id,
    ra.portfolio_id,
    ra.metric_type,
    ra.current_value,
    ra.threshold_value,
    ra.severity_level,
    ra.alert_status,
    ra.description,
    ra.triggered_at,
    ra.acknowledged_at,
    ra.acknowledgment_user,
    rat.user_id as threshold_owner
FROM risk_alerts ra
JOIN risk_alert_thresholds rat ON ra.threshold_id = rat.id
WHERE ra.alert_status IN ('triggered', 'acknowledged', 'escalated');

CREATE VIEW alert_statistics AS
SELECT 
    DATE_TRUNC('day', triggered_at) as alert_date,
    severity_level,
    alert_status,
    COUNT(*) as alert_count
FROM risk_alerts 
WHERE triggered_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', triggered_at), severity_level, alert_status
ORDER BY alert_date DESC, severity_level, alert_status;

-- Comments
COMMENT ON TABLE risk_alert_thresholds IS 'Configuration table for risk alert thresholds';
COMMENT ON TABLE risk_alerts IS 'Risk alert events and history tracking';
COMMENT ON TABLE escalation_events IS 'Alert escalation events and workflow tracking';
COMMENT ON TABLE alert_notifications IS 'Notification delivery tracking and status';
COMMENT ON TABLE user_notification_preferences IS 'User preferences for alert notifications';
COMMENT ON TABLE escalation_rules IS 'Escalation workflow rules and configuration';
COMMENT ON TABLE user_escalation_contacts IS 'User contact information for escalation workflows';

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO risk_monitoring_role;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO risk_monitoring_role;
