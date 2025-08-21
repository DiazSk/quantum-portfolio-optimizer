"""
Enhanced UX Dashboard Component with Accessibility and Professional Formatting
===========================================================================

This module provides enhanced user experience components for the Streamlit dashboard
with improved accessibility, professional styling, and export functionality integration.

Dependencies:
- streamlit: Dashboard framework
- plotly: Interactive charts
- pandas: Data manipulation
- streamlit-aggrid: Enhanced data tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import base64
import io

# Import our export services
try:
    from .services.pdf_export import generate_portfolio_report, create_risk_analysis_pdf
    from .services.excel_export import create_portfolio_workbook, export_holdings_data
    EXPORT_SERVICES_AVAILABLE = True
except ImportError:
    EXPORT_SERVICES_AVAILABLE = False

# Enhanced data table support
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    from st_aggrid.shared import GridUpdateMode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False


class EnhancedUXDashboard:
    """
    Enhanced UX dashboard with professional styling, accessibility features,
    and integrated export functionality.
    
    Features:
    - WCAG 2.1 AA accessibility compliance
    - Professional color schemes and typography
    - Interactive export buttons with progress indicators
    - Enhanced data tables with sorting and filtering
    - Responsive design with mobile-friendly layouts
    - Real-time data refresh with loading states
    """
    
    def __init__(self):
        self.setup_page_config()
        self.load_custom_css()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page with professional settings."""
        st.set_page_config(
            page_title="Quantum Portfolio Optimizer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/quantum-portfolio-optimizer/help',
                'Report a Bug': 'https://github.com/quantum-portfolio-optimizer/issues',
                'About': "Quantum Portfolio Optimizer - Professional Investment Management Platform"
            }
        )
    
    def load_custom_css(self):
        """Load custom CSS for professional styling and accessibility."""
        custom_css = """
        <style>
        /* Professional color scheme */
        :root {
            --primary-color: #1f77b4;
            --secondary-color: #ff7f0e;
            --success-color: #2ca02c;
            --warning-color: #d62728;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
        }
        
        /* Enhanced accessibility */
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 0.5rem;
        }
        
        .section-header {
            font-size: 1.5rem;
            font-weight: 500;
            color: var(--dark-color);
            margin: 1.5rem 0 1rem 0;
            padding-left: 0.5rem;
            border-left: 4px solid var(--secondary-color);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .export-button {
            background: linear-gradient(135deg, var(--primary-color) 0%, #1565c0 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            margin: 0.25rem;
            transition: all 0.3s ease;
            min-width: 120px;
        }
        
        .export-button:hover {
            background: linear-gradient(135deg, #1565c0 0%, var(--primary-color) 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
        }
        
        .export-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-success { background-color: var(--success-color); }
        .status-warning { background-color: var(--warning-color); }
        .status-info { background-color: var(--info-color); }
        
        .data-table {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Accessibility improvements */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0,0,0,0);
            white-space: nowrap;
            border: 0;
        }
        
        /* Focus indicators */
        button:focus, input:focus, select:focus {
            outline: 2px solid var(--primary-color);
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .metric-card {
                border: 2px solid var(--dark-color);
            }
            
            .export-button {
                border: 2px solid white;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            .metric-card, .export-button {
                transition: none;
            }
            
            .metric-card:hover, .export-button:hover {
                transform: none;
            }
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .section-header {
                font-size: 1.25rem;
            }
            
            .metric-card {
                padding: 1rem;
            }
            
            .export-button {
                width: 100%;
                margin: 0.25rem 0;
            }
        }
        
        /* Loading spinner */
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Progress bar */
        .progress-container {
            width: 100%;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .progress-bar {
            height: 8px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'export_in_progress' not in st.session_state:
            st.session_state.export_in_progress = False
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'selected_portfolio' not in st.session_state:
            st.session_state.selected_portfolio = None
        if 'theme_preference' not in st.session_state:
            st.session_state.theme_preference = 'professional'
    
    def render_header(self, title: str, subtitle: Optional[str] = None):
        """Render professional page header with accessibility."""
        st.markdown(f'<h1 class="main-header" role="banner">{title}</h1>', 
                   unsafe_allow_html=True)
        
        if subtitle:
            st.markdown(f'<p class="lead" style="font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem;">{subtitle}</p>', 
                       unsafe_allow_html=True)
        
        # Add refresh indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            refresh_time = st.session_state.last_refresh.strftime("%H:%M:%S")
            st.caption(f"📊 Last updated: {refresh_time}")
    
    def render_export_controls(self, portfolio_id: str, tenant_id: str):
        """Render enhanced export controls with progress indicators."""
        st.markdown('<div class="section-header">📋 Export Reports</div>', 
                   unsafe_allow_html=True)
        
        if not EXPORT_SERVICES_AVAILABLE:
            st.warning("⚠️ Export services not available. Please install required dependencies.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 PDF Report", key="pdf_export", 
                        disabled=st.session_state.export_in_progress,
                        help="Generate comprehensive PDF portfolio report"):
                self._export_pdf_report(portfolio_id, tenant_id)
        
        with col2:
            if st.button("📊 Excel Workbook", key="excel_export",
                        disabled=st.session_state.export_in_progress,
                        help="Create detailed Excel workbook with multiple sheets"):
                self._export_excel_workbook(portfolio_id, tenant_id)
        
        with col3:
            if st.button("⚠️ Risk Analysis", key="risk_export",
                        disabled=st.session_state.export_in_progress,
                        help="Generate comprehensive risk analysis report"):
                self._export_risk_analysis(portfolio_id)
        
        with col4:
            if st.button("📈 Performance Report", key="perf_export",
                        disabled=st.session_state.export_in_progress,
                        help="Export performance attribution analysis"):
                self._export_performance_report(portfolio_id)
    
    def render_enhanced_metrics_grid(self, metrics_data: Dict[str, Any]):
        """Render enhanced metrics grid with professional styling."""
        st.markdown('<div class="section-header">📊 Portfolio Metrics</div>', 
                   unsafe_allow_html=True)
        
        # Create responsive grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_metric_card(
                "Total Value",
                f"${metrics_data.get('total_value', 0):,.0f}",
                "💰",
                self._get_value_status(metrics_data.get('total_value', 0))
            )
        
        with col2:
            self._render_metric_card(
                "Total Return",
                f"{metrics_data.get('total_return', 0):.2%}",
                "📈",
                self._get_return_status(metrics_data.get('total_return', 0))
            )
        
        with col3:
            self._render_metric_card(
                "Sharpe Ratio",
                f"{metrics_data.get('sharpe_ratio', 0):.2f}",
                "⚡",
                self._get_sharpe_status(metrics_data.get('sharpe_ratio', 0))
            )
        
        with col4:
            self._render_metric_card(
                "VaR (95%)",
                f"{metrics_data.get('var_95', 0):.2%}",
                "⚠️",
                self._get_risk_status(metrics_data.get('var_95', 0))
            )
    
    def render_enhanced_data_table(self, data: pd.DataFrame, title: str, 
                                  key: str, height: int = 400):
        """Render enhanced data table with sorting and filtering."""
        st.markdown(f'<div class="section-header">{title}</div>', 
                   unsafe_allow_html=True)
        
        if AGGRID_AVAILABLE:
            self._render_aggrid_table(data, key, height)
        else:
            self._render_basic_table(data)
    
    def render_interactive_charts(self, chart_data: Dict[str, Any]):
        """Render interactive charts with professional styling."""
        st.markdown('<div class="section-header">📈 Performance Analytics</div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Performance", "Allocation", "Risk"])
        
        with tab1:
            self._render_performance_chart(chart_data.get('performance', {}))
        
        with tab2:
            self._render_allocation_chart(chart_data.get('allocation', {}))
        
        with tab3:
            self._render_risk_chart(chart_data.get('risk', {}))
    
    def render_accessibility_controls(self):
        """Render accessibility and preference controls."""
        with st.sidebar:
            st.markdown("### ♿ Accessibility")
            
            # High contrast toggle
            high_contrast = st.checkbox("High Contrast Mode", 
                                       help="Enable high contrast colors for better visibility")
            
            # Font size adjustment
            font_size = st.select_slider("Font Size", 
                                        options=["Small", "Medium", "Large", "Extra Large"],
                                        value="Medium",
                                        help="Adjust text size for better readability")
            
            # Reduced motion
            reduced_motion = st.checkbox("Reduce Motion", 
                                       help="Disable animations and transitions")
            
            # Apply accessibility settings
            if high_contrast or font_size != "Medium" or reduced_motion:
                self._apply_accessibility_settings(high_contrast, font_size, reduced_motion)
    
    def _render_metric_card(self, title: str, value: str, icon: str, status: str):
        """Render individual metric card with status indicator."""
        status_class = f"status-{status}"
        
        card_html = f"""
        <div class="metric-card" role="region" aria-labelledby="metric-{title.lower().replace(' ', '-')}">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span class="status-indicator {status_class}" aria-hidden="true"></span>
                <span style="font-size: 1.5rem; margin-right: 0.5rem;" aria-hidden="true">{icon}</span>
                <h3 id="metric-{title.lower().replace(' ', '-')}" style="margin: 0; font-size: 0.9rem; color: #6c757d;">{title}</h3>
            </div>
            <div style="font-size: 1.8rem; font-weight: 600; color: #2c3e50;" aria-describedby="metric-{title.lower().replace(' ', '-')}">{value}</div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    def _render_aggrid_table(self, data: pd.DataFrame, key: str, height: int):
        """Render enhanced AgGrid table with advanced features."""
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination(paginationPageSize=20)
        gb.configure_side_bar()
        gb.configure_selection("single")
        gb.configure_default_column(
            resizable=True,
            sortable=True,
            filter=True,
            groupable=True
        )
        
        # Add number formatting for numeric columns
        for col in data.select_dtypes(include=[np.number]).columns:
            if 'weight' in col.lower() or 'return' in col.lower() or 'ratio' in col.lower():
                gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], 
                                   valueFormatter="value.toLocaleString('en-US', {style: 'percent', minimumFractionDigits: 2})")
            elif 'value' in col.lower() or 'price' in col.lower():
                gb.configure_column(col, type=["numericColumn", "numberColumnFilter"],
                                   valueFormatter="value.toLocaleString('en-US', {style: 'currency', currency: 'USD'})")
        
        gridOptions = gb.build()
        
        grid_response = AgGrid(
            data,
            gridOptions=gridOptions,
            data_return_mode="AS_INPUT",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=True,
            theme="streamlit",
            height=height,
            key=key
        )
        
        # Handle row selection
        if grid_response["selected_rows"] is not None and len(grid_response["selected_rows"]) > 0:
            selected_row = grid_response["selected_rows"][0]
            st.info(f"Selected: {selected_row}")
    
    def _render_basic_table(self, data: pd.DataFrame):
        """Render basic Streamlit table as fallback."""
        st.dataframe(
            data,
            use_container_width=True,
            height=400
        )
    
    def _render_performance_chart(self, performance_data: Dict):
        """Render performance chart with Plotly."""
        if not performance_data:
            # Generate sample data
            dates = pd.date_range('2024-01-01', periods=252, freq='D')
            portfolio_returns = np.random.normal(0.0008, 0.015, 252).cumsum()
            benchmark_returns = np.random.normal(0.0006, 0.012, 252).cumsum()
            
            performance_data = {
                'dates': dates,
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_data['dates'],
            y=performance_data['portfolio'],
            name='Portfolio',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['dates'],
            y=performance_data['benchmark'],
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Portfolio vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_allocation_chart(self, allocation_data: Dict):
        """Render allocation pie chart."""
        if not allocation_data:
            # Generate sample data
            allocation_data = {
                'sectors': ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy'],
                'weights': [0.35, 0.25, 0.20, 0.15, 0.05]
            }
        
        fig = px.pie(
            values=allocation_data['weights'],
            names=allocation_data['sectors'],
            title="Sector Allocation"
        )
        
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.1%}<extra></extra>'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_chart(self, risk_data: Dict):
        """Render risk metrics chart."""
        if not risk_data:
            # Generate sample data
            risk_data = {
                'metrics': ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'Volatility'],
                'values': [-0.023, -0.041, -0.031, 0.155]
            }
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_data['metrics'],
                y=risk_data['values'],
                marker_color=['#d62728' if v < 0 else '#2ca02c' for v in risk_data['values']],
                hovertemplate='<b>%{x}</b><br>Value: %{y:.2%}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Risk Metrics",
            xaxis_title="Risk Measure",
            yaxis_title="Value",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _export_pdf_report(self, portfolio_id: str, tenant_id: str):
        """Handle PDF report export with progress indication."""
        try:
            st.session_state.export_in_progress = True
            
            with st.spinner("🔄 Generating PDF report..."):
                pdf_bytes = generate_portfolio_report(portfolio_id, tenant_id)
                
                if pdf_bytes:
                    # Create download button
                    b64_pdf = base64.b64encode(pdf_bytes).decode()
                    href = f'data:application/pdf;base64,{b64_pdf}'
                    filename = f"portfolio_report_{portfolio_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    
                    st.success("✅ PDF report generated successfully!")
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        key="download_pdf"
                    )
                else:
                    st.error("❌ Failed to generate PDF report")
                    
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
        finally:
            st.session_state.export_in_progress = False
    
    def _export_excel_workbook(self, portfolio_id: str, tenant_id: str):
        """Handle Excel workbook export with progress indication."""
        try:
            st.session_state.export_in_progress = True
            
            with st.spinner("🔄 Creating Excel workbook..."):
                excel_bytes = create_portfolio_workbook(portfolio_id, tenant_id)
                
                if excel_bytes:
                    filename = f"portfolio_workbook_{portfolio_id}_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    
                    st.success("✅ Excel workbook created successfully!")
                    st.download_button(
                        label="📥 Download Excel Workbook",
                        data=excel_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel"
                    )
                else:
                    st.error("❌ Failed to create Excel workbook")
                    
        except Exception as e:
            st.error(f"❌ Error creating Excel workbook: {str(e)}")
        finally:
            st.session_state.export_in_progress = False
    
    def _export_risk_analysis(self, portfolio_id: str):
        """Handle risk analysis export."""
        try:
            st.session_state.export_in_progress = True
            
            with st.spinner("🔄 Generating risk analysis..."):
                pdf_bytes = create_risk_analysis_pdf(portfolio_id)
                
                if pdf_bytes:
                    filename = f"risk_analysis_{portfolio_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    
                    st.success("✅ Risk analysis generated successfully!")
                    st.download_button(
                        label="📥 Download Risk Analysis",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        key="download_risk"
                    )
                else:
                    st.error("❌ Failed to generate risk analysis")
                    
        except Exception as e:
            st.error(f"❌ Error generating risk analysis: {str(e)}")
        finally:
            st.session_state.export_in_progress = False
    
    def _export_performance_report(self, portfolio_id: str):
        """Handle performance report export."""
        try:
            st.session_state.export_in_progress = True
            
            with st.spinner("🔄 Creating performance report..."):
                excel_bytes = export_holdings_data(portfolio_id, include_transactions=True)
                
                if excel_bytes:
                    filename = f"performance_report_{portfolio_id}_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    
                    st.success("✅ Performance report created successfully!")
                    st.download_button(
                        label="📥 Download Performance Report",
                        data=excel_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_performance"
                    )
                else:
                    st.error("❌ Failed to create performance report")
                    
        except Exception as e:
            st.error(f"❌ Error creating performance report: {str(e)}")
        finally:
            st.session_state.export_in_progress = False
    
    def _get_value_status(self, value: float) -> str:
        """Determine status based on portfolio value."""
        if value >= 10000000:
            return "success"
        elif value >= 5000000:
            return "info"
        else:
            return "warning"
    
    def _get_return_status(self, return_value: float) -> str:
        """Determine status based on return value."""
        if return_value >= 0.1:
            return "success"
        elif return_value >= 0.05:
            return "info"
        else:
            return "warning"
    
    def _get_sharpe_status(self, sharpe: float) -> str:
        """Determine status based on Sharpe ratio."""
        if sharpe >= 1.0:
            return "success"
        elif sharpe >= 0.5:
            return "info"
        else:
            return "warning"
    
    def _get_risk_status(self, var: float) -> str:
        """Determine status based on VaR."""
        if var >= -0.02:
            return "success"
        elif var >= -0.05:
            return "info"
        else:
            return "warning"
    
    def _apply_accessibility_settings(self, high_contrast: bool, font_size: str, reduced_motion: bool):
        """Apply accessibility settings via CSS."""
        accessibility_css = ""
        
        if high_contrast:
            accessibility_css += """
            :root {
                --primary-color: #000000;
                --secondary-color: #ffffff;
                --success-color: #008000;
                --warning-color: #ff0000;
            }
            .metric-card {
                border: 2px solid #000000;
                background: #ffffff;
            }
            """
        
        if font_size != "Medium":
            size_multiplier = {"Small": 0.8, "Large": 1.2, "Extra Large": 1.4}.get(font_size, 1.0)
            accessibility_css += f"""
            html, body, .main-header, .section-header, .metric-card {{
                font-size: calc(1rem * {size_multiplier});
            }}
            """
        
        if reduced_motion:
            accessibility_css += """
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
            """
        
        if accessibility_css:
            st.markdown(f"<style>{accessibility_css}</style>", unsafe_allow_html=True)


def create_enhanced_dashboard():
    """Factory function to create enhanced dashboard instance."""
    return EnhancedUXDashboard()


# Demo usage
if __name__ == "__main__":
    # Create dashboard instance
    dashboard = EnhancedUXDashboard()
    
    # Render demo page
    dashboard.render_header(
        "Quantum Portfolio Optimizer", 
        "Professional Investment Management Platform with Enhanced UX"
    )
    
    # Sample portfolio data
    sample_metrics = {
        'total_value': 12500000.0,
        'total_return': 0.087,
        'sharpe_ratio': 0.65,
        'var_95': -0.023
    }
    
    # Render components
    dashboard.render_enhanced_metrics_grid(sample_metrics)
    dashboard.render_export_controls("demo_portfolio_001", "demo_tenant_001")
    
    # Sample data table
    sample_holdings = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 'Tesla Inc.'],
        'Weight': [0.20, 0.18, 0.15, 0.12, 0.10],
        'Value': [2500000, 2250000, 1875000, 1500000, 1250000],
        'Return': [0.095, 0.082, 0.076, 0.091, 0.145]
    })
    
    dashboard.render_enhanced_data_table(sample_holdings, "📊 Portfolio Holdings", "holdings_table")
    
    # Sample chart data
    sample_charts = {
        'performance': {},  # Will use generated sample data
        'allocation': {},   # Will use generated sample data
        'risk': {}         # Will use generated sample data
    }
    
    dashboard.render_interactive_charts(sample_charts)
    dashboard.render_accessibility_controls()
    
    st.sidebar.success("✅ Enhanced UX Dashboard loaded successfully!")
