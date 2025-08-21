"""
Multi-tenant theming and branding support
Task 2.2: Role-Based Menu System - Tenant customization
"""

import streamlit as st
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class TenantThemeManager:
    """Manages tenant-specific themes and branding"""
    
    def __init__(self):
        self.default_theme = self._get_default_theme()
        self.tenant_themes = self._load_tenant_themes()
    
    def _get_default_theme(self) -> Dict:
        """Get the default professional theme"""
        return {
            'name': 'Professional',
            'primary_color': '#2a5298',
            'secondary_color': '#1e3c72', 
            'accent_color': '#4CAF50',
            'background_color': '#ffffff',
            'sidebar_color': '#f8f9fa',
            'text_color': '#333333',
            'success_color': '#28a745',
            'warning_color': '#ffc107',
            'error_color': '#dc3545',
            'info_color': '#17a2b8',
            'font_family': 'Inter, sans-serif',
            'header_gradient': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)'
        }
    
    def _load_tenant_themes(self) -> Dict[str, Dict]:
        """Load tenant-specific theme configurations"""
        return {
            'demo_tenant': {
                'name': 'Demo Professional',
                'primary_color': '#2a5298',
                'secondary_color': '#1e3c72',
                'accent_color': '#4CAF50',
                'logo_url': None,
                'company_colors': ['#2a5298', '#1e3c72', '#4CAF50']
            },
            'enterprise_tenant': {
                'name': 'Enterprise Blue',
                'primary_color': '#003f7f',
                'secondary_color': '#001a33', 
                'accent_color': '#0066cc',
                'logo_url': '/assets/enterprise_logo.png',
                'company_colors': ['#003f7f', '#0066cc', '#004080']
            },
            'fintech_tenant': {
                'name': 'FinTech Green',
                'primary_color': '#00b894',
                'secondary_color': '#00a085',
                'accent_color': '#55efc4',
                'logo_url': '/assets/fintech_logo.png', 
                'company_colors': ['#00b894', '#00a085', '#55efc4']
            }
        }
    
    def get_tenant_theme(self, tenant_id: str) -> Dict:
        """Get theme configuration for a specific tenant"""
        tenant_theme = self.tenant_themes.get(tenant_id, {})
        
        # Merge with default theme
        theme = self.default_theme.copy()
        theme.update(tenant_theme)
        
        return theme
    
    def apply_tenant_theme(self, tenant_info: Dict):
        """Apply tenant-specific theme to Streamlit"""
        tenant_id = tenant_info.get('tenant_id', 'demo_tenant')
        theme = self.get_tenant_theme(tenant_id)
        
        # Apply custom CSS
        custom_css = self._generate_custom_css(theme)
        st.markdown(custom_css, unsafe_allow_html=True)
    
    def _generate_custom_css(self, theme: Dict) -> str:
        """Generate custom CSS from theme configuration"""
        return f"""
        <style>
        /* Tenant-specific theme: {theme['name']} */
        
        /* Main container styling */
        .main {{
            padding-top: 1rem;
            font-family: {theme['font_family']};
            color: {theme['text_color']};
        }}
        
        /* Enterprise header styling */
        .enterprise-header {{
            background: {theme['header_gradient']};
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        /* Professional metric cards */
        .metric-card {{
            background: {theme['background_color']};
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid {theme['primary_color']};
            margin-bottom: 1rem;
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: {theme['primary_color']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: background-color 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {theme['secondary_color']};
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {theme['sidebar_color']};
        }}
        
        /* Success/Error styling */
        .stSuccess {{
            background-color: {theme['success_color']};
        }}
        
        .stError {{
            background-color: {theme['error_color']};
        }}
        
        .stWarning {{
            background-color: {theme['warning_color']};
        }}
        
        .stInfo {{
            background-color: {theme['info_color']};
        }}
        
        /* Selectbox styling */
        .stSelectbox > div > div {{
            border-color: {theme['primary_color']};
        }}
        
        /* Metric styling */
        [data-testid="metric-container"] {{
            border: 1px solid {theme['primary_color']};
            border-radius: 8px;
            padding: 1rem;
            background: {theme['background_color']};
        }}
        
        /* Navigation menu styling */
        .nav-menu {{
            background-color: {theme['secondary_color']};
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }}
        
        /* Professional table styling */
        .dataframe {{
            border: 1px solid {theme['primary_color']};
            border-radius: 4px;
        }}
        
        /* Custom accent elements */
        .accent-border {{
            border-left: 4px solid {theme['accent_color']};
        }}
        
        /* Hide Streamlit default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {theme['primary_color']};
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {theme['secondary_color']};
        }}
        </style>
        """
    
    def get_tenant_config(self, tenant_info: Dict) -> Dict:
        """Get tenant-specific configuration"""
        tenant_id = tenant_info.get('tenant_id', 'default')
        
        base_config = {
            'tenant_id': tenant_id,
            'company_name': tenant_info.get('company_name', 'Demo Company'),
            'features_enabled': ['dashboard', 'analytics', 'reports'],
            'branding': self.get_tenant_theme(tenant_id),
            'permissions': {
                'export_data': True,
                'custom_reports': True,
                'api_access': False
            }
        }
        
        # Add tenant-specific overrides
        if tenant_id == 'demo_tenant':
            base_config.update({
                'features_enabled': ['dashboard', 'analytics', 'reports', 'demo_features'],
                'permissions': {
                    'export_data': True,
                    'custom_reports': True,
                    'api_access': True,
                    'admin_features': True
                }
            })
        
        return base_config

    def get_chart_colors(self, tenant_info: Dict) -> List[str]:
        """Get color palette for charts based on tenant theme"""
        tenant_id = tenant_info.get('tenant_id', 'demo_tenant')
        theme = self.get_tenant_theme(tenant_id)
        
        # Default color palette based on theme
        base_colors = [
            theme['primary_color'],
            theme['secondary_color'],
            theme['accent_color'],
            theme['success_color'],
            theme['info_color']
        ]
        
        # Extend with company-specific colors if available
        company_colors = theme.get('company_colors', [])
        if company_colors:
            base_colors.extend(company_colors)
        
        return base_colors
    
    def render_tenant_logo(self, tenant_info: Dict):
        """Render tenant logo if available"""
        tenant_id = tenant_info.get('tenant_id', 'demo_tenant')
        theme = self.get_tenant_theme(tenant_id)
        
        logo_url = theme.get('logo_url')
        company_name = tenant_info.get('company_name', 'Demo Company')
        
        if logo_url:
            try:
                st.image(logo_url, width=150, caption=company_name)
            except Exception as e:
                logger.warning(f"Failed to load tenant logo: {e}")
                self._render_text_logo(company_name, theme)
        else:
            self._render_text_logo(company_name, theme)
    
    def _render_text_logo(self, company_name: str, theme: Dict):
        """Render text-based logo when image is not available"""
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: {theme['header_gradient']}; 
                    color: white; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin: 0; font-family: {theme['font_family']};">{company_name}</h3>
            <p style="margin: 0; opacity: 0.8;">Portfolio Management</p>
        </div>
        """, unsafe_allow_html=True)
    
    def get_theme_options(self) -> list:
        """Get available theme options for user selection"""
        themes = ['Professional', 'Dark', 'Light', 'High Contrast']
        return themes
    
    def apply_user_theme_preference(self, theme_name: str):
        """Apply user's theme preference"""
        if theme_name == 'Dark':
            self._apply_dark_theme()
        elif theme_name == 'Light':
            self._apply_light_theme()
        elif theme_name == 'High Contrast':
            self._apply_high_contrast_theme()
        # Professional is default, no additional styling needed
    
    def _apply_dark_theme(self):
        """Apply dark theme modifications"""
        st.markdown("""
        <style>
        .main {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        .metric-card {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        
        .stDataFrame {
            background-color: #2d2d2d;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_light_theme(self):
        """Apply light theme modifications"""
        st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
            color: #333333;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_high_contrast_theme(self):
        """Apply high contrast theme for accessibility"""
        st.markdown("""
        <style>
        .main {
            background-color: #000000;
            color: #ffffff;
        }
        
        .stButton > button {
            background-color: #ffffff;
            color: #000000;
            border: 2px solid #ffffff;
        }
        
        .metric-card {
            background-color: #000000;
            color: #ffffff;
            border: 2px solid #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)


# Global theme manager instance
theme_manager = TenantThemeManager()


def get_theme_manager() -> TenantThemeManager:
    """Get the global theme manager instance"""
    return theme_manager
