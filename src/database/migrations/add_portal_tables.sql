-- Client Portal & Dashboard Enhancement Database Schema
-- Story 3.2: Client Portal & Dashboard Enhancement
-- Migration: Add portal tables for multi-tenant client dashboard

-- Client portal customization
CREATE TABLE IF NOT EXISTS client_portal_config (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    portal_theme JSONB DEFAULT '{}', -- Colors, logos, branding
    dashboard_layout JSONB DEFAULT '{}', -- Widget positions and sizes
    notification_preferences JSONB DEFAULT '{}',
    custom_features JSONB DEFAULT '{}', -- Client-specific feature flags
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User preferences and personalization
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    dashboard_widgets JSONB DEFAULT '{}', -- Personal widget configuration
    notification_settings JSONB DEFAULT '{}',
    ui_preferences JSONB DEFAULT '{}', -- Theme, language, timezone
    favorite_portfolios JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Notification history and tracking
CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    tenant_id INTEGER NOT NULL,
    notification_type VARCHAR(50) NOT NULL DEFAULT 'info', -- 'alert', 'system', 'portfolio', 'compliance'
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    data JSONB DEFAULT '{}', -- Additional notification data
    priority INTEGER DEFAULT 1, -- 1=low, 2=medium, 3=high, 4=critical
    read_at TIMESTAMP NULL,
    acknowledged_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Portfolio dashboard widgets configuration
CREATE TABLE IF NOT EXISTS dashboard_widgets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    tenant_id INTEGER NOT NULL,
    widget_type VARCHAR(100) NOT NULL, -- 'portfolio_summary', 'risk_metrics', 'performance_chart'
    widget_config JSONB DEFAULT '{}',
    position_x INTEGER DEFAULT 0,
    position_y INTEGER DEFAULT 0,
    width INTEGER DEFAULT 4,
    height INTEGER DEFAULT 3,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Client alert rules and thresholds
CREATE TABLE IF NOT EXISTS client_alert_rules (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    user_id INTEGER NULL, -- NULL for tenant-wide rules
    rule_name VARCHAR(200) NOT NULL,
    rule_type VARCHAR(100) NOT NULL, -- 'portfolio_performance', 'risk_threshold', 'compliance'
    conditions JSONB NOT NULL, -- Rule conditions and thresholds
    notification_channels JSONB DEFAULT '[]', -- email, sms, in-app
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Dashboard session tracking
CREATE TABLE IF NOT EXISTS dashboard_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    tenant_id INTEGER NOT NULL,
    session_id VARCHAR(255) NOT NULL UNIQUE,
    last_activity TIMESTAMP DEFAULT NOW(),
    session_data JSONB DEFAULT '{}',
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_client_portal_config_tenant_id ON client_portal_config(tenant_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_tenant_id ON notifications(tenant_id);
CREATE INDEX IF NOT EXISTS idx_notifications_read_at ON notifications(read_at);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dashboard_widgets_user_tenant ON dashboard_widgets(user_id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_client_alert_rules_tenant_id ON client_alert_rules(tenant_id);
CREATE INDEX IF NOT EXISTS idx_client_alert_rules_user_id ON client_alert_rules(user_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_sessions_user_id ON dashboard_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_sessions_session_id ON dashboard_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_sessions_expires_at ON dashboard_sessions(expires_at);

-- Insert default portal configuration for demo tenant
INSERT INTO client_portal_config (tenant_id, portal_theme, dashboard_layout, notification_preferences, custom_features)
VALUES (
    1, -- demo tenant
    '{
        "name": "Demo Professional",
        "primary_color": "#1f77b4",
        "secondary_color": "#ff7f0e",
        "accent_color": "#2ca02c",
        "background_color": "#ffffff",
        "text_color": "#212529",
        "logo_url": null,
        "company_name": "Demo Asset Management"
    }',
    '{
        "default_layout": "grid",
        "sidebar_collapsed": false,
        "theme": "professional",
        "show_performance_summary": true,
        "show_risk_metrics": true,
        "show_portfolio_allocation": true
    }',
    '{
        "email_enabled": true,
        "browser_notifications": true,
        "alert_frequency": "immediate",
        "daily_summary": true,
        "risk_threshold_alerts": true
    }',
    '{
        "advanced_analytics": true,
        "custom_reports": true,
        "api_access": false,
        "white_label": true,
        "export_capabilities": ["pdf", "excel", "csv"]
    }'
)
ON CONFLICT DO NOTHING;

-- Insert default user preferences for demo users
INSERT INTO user_preferences (user_id, dashboard_widgets, notification_settings, ui_preferences, favorite_portfolios)
VALUES 
(1, -- demo viewer
    '{
        "portfolio_summary": {"enabled": true, "position": {"x": 0, "y": 0}},
        "performance_chart": {"enabled": true, "position": {"x": 1, "y": 0}},
        "holdings_table": {"enabled": true, "position": {"x": 0, "y": 1}}
    }',
    '{
        "email": true,
        "push": false,
        "frequency": "daily"
    }',
    '{
        "theme": "light",
        "language": "en",
        "timezone": "UTC",
        "currency": "USD",
        "number_format": "US"
    }',
    '[1, 2]'
),
(2, -- demo analyst
    '{
        "portfolio_summary": {"enabled": true, "position": {"x": 0, "y": 0}},
        "performance_chart": {"enabled": true, "position": {"x": 1, "y": 0}},
        "risk_metrics": {"enabled": true, "position": {"x": 2, "y": 0}},
        "factor_exposure": {"enabled": true, "position": {"x": 0, "y": 1}},
        "holdings_table": {"enabled": true, "position": {"x": 1, "y": 1}}
    }',
    '{
        "email": true,
        "push": true,
        "frequency": "immediate"
    }',
    '{
        "theme": "light",
        "language": "en",
        "timezone": "UTC",
        "currency": "USD",
        "number_format": "US"
    }',
    '[1, 2, 3]'
)
ON CONFLICT DO NOTHING;

-- Insert sample notification for demo
INSERT INTO notifications (user_id, tenant_id, notification_type, title, message, data, priority)
VALUES (
    1, 1, 'system', 
    'Welcome to Dashboard',
    'Your professional client portal is now ready. Explore portfolio analytics, risk metrics, and reporting features.',
    '{"category": "welcome", "action_required": false}',
    1
)
ON CONFLICT DO NOTHING;

COMMENT ON TABLE client_portal_config IS 'Tenant-specific portal configuration and branding';
COMMENT ON TABLE user_preferences IS 'Individual user preferences and dashboard customization';
COMMENT ON TABLE notifications IS 'Notification history and delivery tracking';
COMMENT ON TABLE dashboard_widgets IS 'Dashboard widget configuration and positioning';
COMMENT ON TABLE client_alert_rules IS 'Client-specific alert rules and thresholds';
COMMENT ON TABLE dashboard_sessions IS 'Active dashboard sessions for state management';
