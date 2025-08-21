"""
Role-based navigation system for Streamlit dashboard
Task 2.2: Role-Based Menu System
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NavigationManager:
    """Manages role-based navigation and access control"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.menu_items = self._define_menu_structure()
        self.role_permissions = self._define_role_permissions()
    
    def _define_menu_structure(self) -> Dict[str, Dict]:
        """Define the complete menu structure"""
        return {
            "dashboard": {
                "title": "📊 Portfolio Dashboard",
                "description": "Real-time portfolio monitoring and analytics",
                "required_permissions": ["view_portfolio"],
                "page_function": "render_enhanced_dashboard",
                "module": "src.dashboard.pages.dashboard"
            },
            "analytics": {
                "title": "📈 Performance Analytics", 
                "description": "Advanced portfolio analysis and attribution",
                "required_permissions": ["view_analytics"],
                "page_function": "render_analytics_page",
                "module": "src.dashboard.pages.analytics"
            },
            "risk": {
                "title": "⚠️ Risk Monitoring",
                "description": "Risk metrics and stress testing", 
                "required_permissions": ["view_risk"],
                "page_function": "render_risk_monitoring",
                "module": "src.dashboard.pages.risk"
            },
            "compliance": {
                "title": "📋 Compliance",
                "description": "Regulatory compliance and reporting",
                "required_permissions": ["view_compliance"],
                "page_function": "render_compliance_page",
                "module": "src.dashboard.pages.compliance"
            },
            "alerts": {
                "title": "🔔 Alerts & Notifications",
                "description": "Real-time alerts and notification management",
                "required_permissions": ["view_alerts"],
                "page_function": "render_alerts",
                "module": "src.dashboard.client_portal"
            },
            "reports": {
                "title": "📄 Reports",
                "description": "Generate and download portfolio reports",
                "required_permissions": ["generate_reports"],
                "page_function": "render_reports",
                "module": "src.dashboard.client_portal"
            },
            "admin": {
                "title": "⚙️ Administration",
                "description": "User management and system settings",
                "required_permissions": ["admin_access"],
                "page_function": "render_admin_page", 
                "module": "src.dashboard.pages.admin"
            },
            "settings": {
                "title": "🔧 Settings",
                "description": "Personal preferences and configuration",
                "required_permissions": ["basic_access"],
                "page_function": "render_settings",
                "module": "src.dashboard.client_portal"
            }
        }
    
    def _define_role_permissions(self) -> Dict[str, List[str]]:
        """Define permissions for each role"""
        return {
            "viewer": [
                "basic_access",
                "view_portfolio",
                "generate_reports"
            ],
            "analyst": [
                "basic_access", 
                "view_portfolio",
                "view_analytics",
                "view_risk",
                "view_alerts",
                "generate_reports"
            ],
            "portfolio_manager": [
                "basic_access",
                "view_portfolio", 
                "view_analytics",
                "view_risk",
                "view_compliance",
                "view_alerts",
                "generate_reports",
                "modify_portfolio"
            ],
            "admin": [
                "basic_access",
                "view_portfolio",
                "view_analytics", 
                "view_risk",
                "view_compliance",
                "view_alerts",
                "generate_reports",
                "modify_portfolio",
                "admin_access",
                "user_management"
            ],
            "compliance_officer": [
                "basic_access",
                "view_portfolio",
                "view_compliance",
                "view_risk",
                "generate_reports"
            ]
        }
    
    def get_user_permissions(self, user_info: Dict) -> List[str]:
        """Get permissions for a user based on their role"""
        user_role = user_info.get('role', 'viewer').lower()
        return self.role_permissions.get(user_role, self.role_permissions['viewer'])
    
    def get_accessible_menu_items(self, user_info: Dict) -> Dict[str, Dict]:
        """Get menu items accessible to the user"""
        user_permissions = self.get_user_permissions(user_info)
        accessible_items = {}
        
        for menu_key, menu_config in self.menu_items.items():
            required_perms = menu_config['required_permissions']
            
            # Check if user has all required permissions
            if all(perm in user_permissions for perm in required_perms):
                accessible_items[menu_key] = menu_config
        
        return accessible_items
    
    def render_navigation_sidebar(self) -> Optional[str]:
        """Render the navigation sidebar and return selected page"""
        user_info = st.session_state.get('user_info', {})
        tenant_info = st.session_state.get('tenant_info', {})
        
        if not user_info:
            return None
        
        with st.sidebar:
            self._render_tenant_branding(tenant_info)
            self._render_user_info(user_info)
            
            st.divider()
            
            # Get accessible menu items
            accessible_items = self.get_accessible_menu_items(user_info)
            
            if not accessible_items:
                st.error("No accessible menu items. Please contact your administrator.")
                return None
            
            # Navigation menu
            st.subheader("Navigation")
            
            menu_options = list(accessible_items.keys())
            menu_labels = [accessible_items[key]['title'] for key in menu_options]
            
            # Get current selection from session state or default to first item
            if 'current_page' not in st.session_state:
                st.session_state.current_page = menu_options[0]
            
            # Ensure current page is accessible (handle role changes)
            if st.session_state.current_page not in menu_options:
                st.session_state.current_page = menu_options[0]
            
            current_index = menu_options.index(st.session_state.current_page)
            
            selected_index = st.selectbox(
                "Select Page",
                range(len(menu_labels)),
                index=current_index,
                format_func=lambda x: menu_labels[x],
                key="navigation_select"
            )
            
            selected_page = menu_options[selected_index]
            st.session_state.current_page = selected_page
            
            # Show page description
            page_config = accessible_items[selected_page]
            st.caption(page_config['description'])
            
            st.divider()
            
            self._render_quick_actions(user_info)
            self._render_user_controls()
            
            return selected_page
    
    def _render_tenant_branding(self, tenant_info: Dict):
        """Render tenant-specific branding"""
        company_name = tenant_info.get('company_name', 'Demo Company')
        tenant_logo = tenant_info.get('logo_url', None)
        
        if tenant_logo:
            st.image(tenant_logo, width=150)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; border-radius: 8px; margin-bottom: 1rem;">
            <h3>{company_name}</h3>
            <p>Portfolio Management</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_user_info(self, user_info: Dict):
        """Render user information"""
        full_name = user_info.get('full_name', 'User')
        role = user_info.get('role', 'viewer').title()
        email = user_info.get('email', '')
        
        st.markdown(f"👤 **{full_name}**")
        st.markdown(f"🏢 Role: **{role}**")
        if email:
            st.markdown(f"📧 {email}")
        
        # Show user permissions (for debugging/transparency)
        with st.expander("My Permissions"):
            permissions = self.get_user_permissions(user_info)
            for perm in permissions:
                st.text(f"✓ {perm.replace('_', ' ').title()}")
    
    def _render_quick_actions(self, user_info: Dict):
        """Render quick action buttons"""
        st.subheader("Quick Actions")
        
        user_permissions = self.get_user_permissions(user_info)
        
        # Refresh data action (always available)
        if st.button("🔄 Refresh Data", use_container_width=True):
            self._handle_refresh_data()
        
        # Generate report action
        if "generate_reports" in user_permissions:
            if st.button("📊 Generate Report", use_container_width=True):
                self._handle_generate_report()
        
        # Create alert action  
        if "view_alerts" in user_permissions:
            if st.button("🚨 Create Alert", use_container_width=True):
                self._handle_create_alert()
        
        # Admin actions
        if "admin_access" in user_permissions:
            if st.button("⚙️ Admin Panel", use_container_width=True):
                self._handle_admin_panel()
    
    def _render_user_controls(self):
        """Render user control buttons"""
        st.divider()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh and 'last_auto_refresh' not in st.session_state:
            st.session_state.last_auto_refresh = True
            st.rerun()
        
        # Theme selector
        theme = st.selectbox(
            "Theme",
            ["Professional", "Dark", "Light"],
            index=0
        )
        if theme != st.session_state.get('theme', 'Professional'):
            st.session_state.theme = theme
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            self._handle_logout()
    
    def _handle_refresh_data(self):
        """Handle refresh data action"""
        try:
            from src.dashboard.services.portfolio_service import portfolio_service
            portfolio_service.clear_cache()
            st.session_state.last_refresh = datetime.now()
            st.success("Data refreshed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to refresh data: {e}")
    
    def _handle_generate_report(self):
        """Handle generate report action"""
        st.session_state.show_report_modal = True
        st.info("Report generation modal would open here")
    
    def _handle_create_alert(self):
        """Handle create alert action"""
        st.session_state.show_alert_modal = True
        st.info("Alert creation modal would open here")
    
    def _handle_admin_panel(self):
        """Handle admin panel action"""
        st.session_state.current_page = 'admin'
        st.rerun()
    
    def _handle_logout(self):
        """Handle user logout"""
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Logged out successfully!")
        st.rerun()
    
    def check_page_permission(self, page_key: str, user_info: Dict) -> bool:
        """Check if user has permission to access a specific page"""
        if page_key not in self.menu_items:
            return False
        
        user_permissions = self.get_user_permissions(user_info)
        required_perms = self.menu_items[page_key]['required_permissions']
        
        return all(perm in user_permissions for perm in required_perms)
    
    def render_access_denied(self, page_key: str):
        """Render access denied message"""
        st.error("🚫 Access Denied")
        st.markdown(f"""
        You don't have permission to access the **{self.menu_items.get(page_key, {}).get('title', page_key)}** page.
        
        Please contact your administrator if you believe you should have access to this feature.
        """)
        
        user_info = st.session_state.get('user_info', {})
        user_role = user_info.get('role', 'viewer')
        st.info(f"Your current role: **{user_role.title()}**")


# Global navigation manager instance
nav_manager = NavigationManager()


def get_navigation_manager() -> NavigationManager:
    """Get the global navigation manager instance"""
    return nav_manager
