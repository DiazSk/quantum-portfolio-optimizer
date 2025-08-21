"""
Multi-Tenant Authentication Database Schema
Story 3.1: Database tables for enterprise authentication and RBAC
"""

-- Multi-Tenant Authentication Schema Extension
-- Extends existing portfolio optimization database with enterprise auth

-- =============================================================================
-- TENANT MANAGEMENT
-- =============================================================================

-- Core tenant information and configuration
CREATE TABLE IF NOT EXISTS tenants (
    id SERIAL PRIMARY KEY,
    tenant_code VARCHAR(50) UNIQUE NOT NULL,
    company_name VARCHAR(200) NOT NULL,
    domain VARCHAR(100) UNIQUE,
    sso_config JSONB DEFAULT '{}',
    encryption_key_id VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tenant-specific configuration and settings
CREATE TABLE IF NOT EXISTS tenant_settings (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    setting_key VARCHAR(100) NOT NULL,
    setting_value JSONB,
    is_encrypted BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, setting_key)
);

-- =============================================================================
-- USER MANAGEMENT
-- =============================================================================

-- Enhanced user table with tenant isolation
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    external_user_id VARCHAR(100), -- From SSO provider (AD, Okta, etc.)
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(200) NOT NULL,
    password_hash VARCHAR(255), -- For local accounts only
    phone VARCHAR(20),
    department VARCHAR(100),
    job_title VARCHAR(100),
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    account_locked_until TIMESTAMP,
    password_reset_token VARCHAR(255),
    password_reset_expires TIMESTAMP,
    email_verified BOOLEAN DEFAULT false,
    email_verification_token VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Ensure email uniqueness within tenant for multi-tenant support
    CONSTRAINT unique_tenant_email UNIQUE(tenant_id, email)
);

-- User profile additional information
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    avatar_url VARCHAR(500),
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    notification_preferences JSONB DEFAULT '{}',
    ui_preferences JSONB DEFAULT '{}',
    api_access_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- ROLE-BASED ACCESS CONTROL (RBAC)
-- =============================================================================

-- Hierarchical role system with tenant isolation
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    role_name VARCHAR(100) NOT NULL,
    role_description TEXT,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_system_role BOOLEAN DEFAULT false, -- System vs custom roles
    parent_role_id INTEGER REFERENCES roles(id), -- Role hierarchy
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_tenant_role UNIQUE(tenant_id, role_name)
);

-- User role assignments with temporal control
CREATE TABLE IF NOT EXISTS user_roles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    assigned_by INTEGER REFERENCES users(id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP, -- Optional role expiration
    is_active BOOLEAN DEFAULT true,
    notes TEXT,
    
    CONSTRAINT unique_active_user_role UNIQUE(user_id, role_id)
);

-- Resource-level permissions for fine-grained access control
CREATE TABLE IF NOT EXISTS resource_permissions (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL, -- 'portfolio', 'report', 'user', etc.
    resource_id VARCHAR(100), -- Specific resource ID (optional)
    permission_type VARCHAR(20) NOT NULL, -- 'read', 'write', 'delete', 'admin'
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- SESSION MANAGEMENT
-- =============================================================================

-- User session tracking for security and audit
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    device_fingerprint VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- OAuth/SSO token storage for enterprise integration
CREATE TABLE IF NOT EXISTS oauth_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL, -- 'azure_ad', 'okta', 'auth0', etc.
    provider_user_id VARCHAR(200),
    access_token TEXT,
    refresh_token TEXT,
    token_type VARCHAR(20) DEFAULT 'Bearer',
    expires_at TIMESTAMP,
    scope TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- AUDIT AND SECURITY
-- =============================================================================

-- Comprehensive audit trail for authentication events
CREATE TABLE IF NOT EXISTS auth_audit_log (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    user_id INTEGER REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL, -- 'login', 'logout', 'role_change', etc.
    event_status VARCHAR(20) NOT NULL, -- 'success', 'failure', 'blocked'
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Security event tracking for threat detection
CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    user_id INTEGER REFERENCES users(id),
    event_category VARCHAR(50) NOT NULL, -- 'suspicious_login', 'privilege_escalation', etc.
    event_severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    event_description TEXT NOT NULL,
    source_ip INET,
    additional_data JSONB,
    is_resolved BOOLEAN DEFAULT false,
    resolved_by INTEGER REFERENCES users(id),
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Password history for security compliance
CREATE TABLE IF NOT EXISTS password_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- API ACCESS AND RATE LIMITING
-- =============================================================================

-- API key management for programmatic access
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    api_key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions JSONB DEFAULT '[]',
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Rate limiting tracking
CREATE TABLE IF NOT EXISTS api_rate_limits (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    api_key_id INTEGER REFERENCES api_keys(id) ON DELETE CASCADE,
    endpoint VARCHAR(200) NOT NULL,
    request_count INTEGER DEFAULT 0,
    window_start TIMESTAMP DEFAULT NOW(),
    window_end TIMESTAMP,
    is_blocked BOOLEAN DEFAULT false
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Authentication performance indexes
CREATE INDEX IF NOT EXISTS idx_users_tenant_email ON users(tenant_id, email);
CREATE INDEX IF NOT EXISTS idx_users_external_id ON users(external_user_id);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active, tenant_id);

-- Role and permission indexes
CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_user_roles_expires ON user_roles(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_roles_tenant ON roles(tenant_id, is_active);

-- Session management indexes
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON user_sessions(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);

-- Audit and security indexes
CREATE INDEX IF NOT EXISTS idx_auth_audit_tenant_time ON auth_audit_log(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_auth_audit_user_time ON auth_audit_log(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_security_events_tenant ON security_events(tenant_id, is_resolved);

-- API access indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_user_active ON api_keys(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_api_rate_limits_window ON api_rate_limits(tenant_id, window_start, window_end);

-- =============================================================================
-- DEFAULT DATA AND SYSTEM ROLES
-- =============================================================================

-- Create system tenant for platform administration
INSERT INTO tenants (tenant_code, company_name, domain, encryption_key_id) 
VALUES ('SYSTEM', 'System Administration', 'system.local', 'system_key_001')
ON CONFLICT (tenant_code) DO NOTHING;

-- Default system roles (will be created per tenant)
-- This is handled by the application during tenant creation

-- =============================================================================
-- SECURITY FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Function to update 'updated_at' timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to relevant tables
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_roles_updated_at BEFORE UPDATE ON roles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically create user profile on user creation
CREATE OR REPLACE FUNCTION create_user_profile()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO user_profiles (user_id) VALUES (NEW.id);
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER create_user_profile_trigger AFTER INSERT ON users
    FOR EACH ROW EXECUTE FUNCTION create_user_profile();

-- Function to log authentication events
CREATE OR REPLACE FUNCTION log_auth_event(
    p_tenant_id INTEGER,
    p_user_id INTEGER,
    p_event_type VARCHAR,
    p_event_status VARCHAR,
    p_ip_address INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_details JSONB DEFAULT '{}'
) RETURNS VOID AS $$
BEGIN
    INSERT INTO auth_audit_log (
        tenant_id, user_id, event_type, event_status, 
        ip_address, user_agent, details
    ) VALUES (
        p_tenant_id, p_user_id, p_event_type, p_event_status,
        p_ip_address, p_user_agent, p_details
    );
END;
$$ language 'plpgsql';

-- =============================================================================
-- MULTI-TENANT DATA ISOLATION VIEWS
-- =============================================================================

-- Tenant-specific user view for data isolation
CREATE OR REPLACE VIEW tenant_users AS
SELECT 
    u.id,
    u.tenant_id,
    u.email,
    u.full_name,
    u.department,
    u.job_title,
    u.last_login,
    u.is_active,
    u.created_at,
    t.tenant_code,
    t.company_name
FROM users u
JOIN tenants t ON u.tenant_id = t.id
WHERE t.is_active = true AND u.is_active = true;

-- User roles view with tenant context
CREATE OR REPLACE VIEW user_roles_view AS
SELECT 
    ur.user_id,
    ur.role_id,
    r.role_name,
    r.role_description,
    r.permissions,
    ur.assigned_at,
    ur.expires_at,
    ur.is_active,
    u.tenant_id,
    t.tenant_code
FROM user_roles ur
JOIN roles r ON ur.role_id = r.id
JOIN users u ON ur.user_id = u.id
JOIN tenants t ON u.tenant_id = t.id
WHERE ur.is_active = true 
  AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
  AND r.is_active = true
  AND u.is_active = true
  AND t.is_active = true;

-- =============================================================================
-- PERFORMANCE AND MAINTENANCE
-- =============================================================================

-- Cleanup expired sessions (run periodically)
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Cleanup old audit logs (keep 2 years)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM auth_audit_log 
    WHERE created_at < NOW() - INTERVAL '2 years';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

COMMENT ON SCHEMA public IS 'Multi-Tenant Authentication Schema - Story 3.1 Enterprise Security';
