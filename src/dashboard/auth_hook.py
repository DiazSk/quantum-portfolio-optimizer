"""
Lightweight auth hook for Streamlit client portal.
This provides a small interface to integrate with `src/auth/multi_tenant_auth.py`.
If the real module isn't present, the hook fails gracefully and returns informative errors.

Enhanced with role-based access control for Task 2.2
"""
from typing import Optional, Dict, List


def validate_token(token: str) -> Dict[str, Optional[str]]:
    """Validate JWT token via MultiTenantAuthManager if available.

    Returns a dict with keys: authenticated (bool), user (dict|None), error (str|None)
    """
    try:
        from src.auth.multi_tenant_auth import MultiTenantAuthManager
    except Exception as e:
        return {
            'authenticated': False,
            'user': None,
            'error': 'Auth manager not available: ' + str(e)
        }

    try:
        manager = MultiTenantAuthManager()
        user = manager.validate_jwt(token)
        return {'authenticated': True, 'user': user, 'error': None}
    except Exception as e:
        return {'authenticated': False, 'user': None, 'error': str(e)}


def get_user_roles(user_info: Dict) -> List[str]:
    """Get user roles from user info"""
    if not user_info:
        return []
    
    # Primary role
    primary_role = user_info.get('role', 'viewer')
    roles = [primary_role]
    
    # Additional roles (if supported)
    additional_roles = user_info.get('additional_roles', [])
    if isinstance(additional_roles, list):
        roles.extend(additional_roles)
    
    return roles


def check_permission(user_info: Dict, required_permission: str) -> bool:
    """Check if user has a specific permission"""
    try:
        from src.dashboard.components.navigation import get_navigation_manager
        nav_manager = get_navigation_manager()
        user_permissions = nav_manager.get_user_permissions(user_info)
        return required_permission in user_permissions
    except Exception:
        # Fallback: basic role-based check
        user_role = user_info.get('role', 'viewer').lower()
        
        # Basic permission mapping
        basic_permissions = {
            'viewer': ['basic_access', 'view_portfolio', 'generate_reports'],
            'analyst': ['basic_access', 'view_portfolio', 'view_analytics', 'view_risk', 'view_alerts', 'generate_reports'],
            'portfolio_manager': ['basic_access', 'view_portfolio', 'view_analytics', 'view_risk', 'view_compliance', 'view_alerts', 'generate_reports', 'modify_portfolio'],
            'admin': ['basic_access', 'view_portfolio', 'view_analytics', 'view_risk', 'view_compliance', 'view_alerts', 'generate_reports', 'modify_portfolio', 'admin_access', 'user_management']
        }
        
        user_permissions = basic_permissions.get(user_role, basic_permissions['viewer'])
        return required_permission in user_permissions


def get_tenant_context(user_info: Dict) -> Dict:
    """Get tenant context for multi-tenant operations"""
    tenant_id = user_info.get('tenant_id', 'default')
    tenant_name = user_info.get('tenant_name', 'Demo Company')
    
    return {
        'tenant_id': tenant_id,
        'tenant_name': tenant_name,
        'company_name': tenant_name,
        'tenant_settings': user_info.get('tenant_settings', {}),
        'branding': user_info.get('tenant_branding', {})
    }


def create_demo_user_session(email: str, role: str = 'viewer', tenant_code: str = 'demo') -> Dict:
    """Create a demo user session for testing (when real auth unavailable)"""
    from datetime import datetime, timedelta
    
    demo_users = {
        'viewer@demo.com': {
            'id': 'demo_viewer_1',
            'full_name': 'Demo Viewer',
            'role': 'viewer',
            'email': 'viewer@demo.com',
            'tenant_id': 'demo_tenant',
            'tenant_name': 'Demo Company'
        },
        'analyst@demo.com': {
            'id': 'demo_analyst_1', 
            'full_name': 'Demo Analyst',
            'role': 'analyst',
            'email': 'analyst@demo.com',
            'tenant_id': 'demo_tenant',
            'tenant_name': 'Demo Company'
        },
        'manager@demo.com': {
            'id': 'demo_manager_1',
            'full_name': 'Demo Portfolio Manager', 
            'role': 'portfolio_manager',
            'email': 'manager@demo.com',
            'tenant_id': 'demo_tenant',
            'tenant_name': 'Demo Company'
        },
        'admin@demo.com': {
            'id': 'demo_admin_1',
            'full_name': 'Demo Administrator',
            'role': 'admin', 
            'email': 'admin@demo.com',
            'tenant_id': 'demo_tenant',
            'tenant_name': 'Demo Company'
        }
    }
    
    user_data = demo_users.get(email.lower())
    if not user_data:
        # Create generic demo user
        user_data = {
            'id': f'demo_{role}_generic',
            'full_name': f'Demo {role.title()}',
            'role': role,
            'email': email,
            'tenant_id': tenant_code,
            'tenant_name': tenant_code.title() + ' Company'
        }
    
    # Add session metadata
    user_data.update({
        'session_start': datetime.now(),
        'session_expires': datetime.now() + timedelta(hours=8),
        'last_activity': datetime.now(),
        'demo_session': True
    })
    
    return {
        'authenticated': True,
        'user': user_data,
        'error': None
    }
