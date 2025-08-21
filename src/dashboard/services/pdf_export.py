"""
PDF Export Engine for Professional Portfolio Reports
==================================================

This module provides comprehensive PDF generation capabilities for portfolio reports,
risk analysis, and performance attribution with tenant-specific branding.

Dependencies:
- reportlab: Professional PDF generation
- plotly: Chart conversion to PDF graphics
- pandas: Data manipulation
- Pillow: Image processing for logos/branding
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import os

# PDF Generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import Color, blue, black, red, green, white
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.platypus.flowables import KeepTogether
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("Warning: ReportLab not available. PDF export functionality will be limited.")
    REPORTLAB_AVAILABLE = False

# Plotly integration
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available. Chart export functionality will be limited.")
    PLOTLY_AVAILABLE = False

# Image processing
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: Pillow not available. Logo/branding functionality will be limited.")
    PIL_AVAILABLE = False


@dataclass
class TenantBranding:
    """Tenant-specific branding configuration for PDF exports."""
    
    tenant_id: str
    company_name: str
    logo_path: Optional[str] = None
    primary_color: str = "#1f77b4"  # Default blue
    secondary_color: str = "#ff7f0e"  # Default orange
    accent_color: str = "#2ca02c"    # Default green
    font_family: str = "Helvetica"
    watermark_text: Optional[str] = None
    footer_text: Optional[str] = None
    confidentiality_notice: str = "CONFIDENTIAL - For Internal Use Only"
    
    def to_reportlab_color(self, hex_color: str) -> Color:
        """Convert hex color to ReportLab Color object."""
        if not REPORTLAB_AVAILABLE:
            return None
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return Color(rgb[0], rgb[1], rgb[2])


@dataclass
class PortfolioReportData:
    """Data structure for portfolio report generation."""
    
    portfolio_id: str
    portfolio_name: str
    tenant_id: str
    report_date: date
    
    # Portfolio Summary
    total_value: float
    total_return: float
    benchmark_return: float
    active_return: float
    
    # Risk Metrics
    volatility: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    beta: float
    
    # Holdings Data
    holdings: List[Dict[str, Any]]
    
    # Performance Data
    performance_history: pd.DataFrame
    
    # Attribution Data
    sector_attribution: Optional[pd.DataFrame] = None
    factor_attribution: Optional[pd.DataFrame] = None


class PDFExportEngine:
    """
    Professional PDF export engine for portfolio reports with tenant branding.
    
    Features:
    - Comprehensive portfolio reports with charts and analytics
    - Risk analysis with VaR, stress tests, correlation matrices
    - Performance attribution with factor breakdown
    - Tenant-specific branding and customization
    - Professional formatting and layout
    """
    
    def __init__(self):
        self.styles = self._initialize_styles()
        
    def _initialize_styles(self) -> Dict[str, Any]:
        """Initialize ReportLab styles for consistent formatting."""
        if not REPORTLAB_AVAILABLE:
            return {}
            
        styles = getSampleStyleSheet()
        
        # Custom styles
        custom_styles = {
            'CustomTitle': ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER
            ),
            'SectionHeader': ParagraphStyle(
                'SectionHeader',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                textColor=blue
            ),
            'SubHeader': ParagraphStyle(
                'SubHeader',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=8
            ),
            'TableHeader': ParagraphStyle(
                'TableHeader',
                parent=styles['Normal'],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=blue
            ),
            'Confidential': ParagraphStyle(
                'Confidential',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=red
            )
        }
        
        styles.add(custom_styles['CustomTitle'])
        styles.add(custom_styles['SectionHeader'])
        styles.add(custom_styles['SubHeader'])
        styles.add(custom_styles['TableHeader'])
        styles.add(custom_styles['Confidential'])
        
        return styles
    
    def generate_portfolio_report(self, portfolio_id: str, tenant_id: str, 
                                 branding: Optional[TenantBranding] = None) -> bytes:
        """
        Generate comprehensive PDF portfolio report with charts and analytics.
        
        Args:
            portfolio_id: Unique portfolio identifier
            tenant_id: Tenant identifier for data isolation
            branding: Optional tenant branding configuration
            
        Returns:
            PDF file as bytes
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_fallback_report(portfolio_id, tenant_id)
            
        try:
            # Get portfolio data (would integrate with portfolio service)
            report_data = self._get_portfolio_data(portfolio_id, tenant_id)
            
            # Set up PDF document
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build report content
            story = []
            
            # Header with branding
            story.extend(self._build_header(report_data, branding))
            
            # Executive Summary
            story.extend(self._build_executive_summary(report_data))
            
            # Portfolio Holdings
            story.extend(self._build_holdings_section(report_data))
            
            # Performance Analysis
            story.extend(self._build_performance_section(report_data))
            
            # Risk Analysis
            story.extend(self._build_risk_section(report_data))
            
            # Footer
            story.extend(self._build_footer(branding))
            
            # Generate PDF
            doc.build(story)
            
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            print(f"Error generating portfolio report: {e}")
            return self._generate_error_report(portfolio_id, str(e))
    
    def create_risk_analysis_pdf(self, portfolio_id: str, date_range: str = "1Y") -> bytes:
        """
        Export detailed risk analysis with VaR, stress tests, correlation matrices.
        
        Args:
            portfolio_id: Portfolio identifier
            date_range: Analysis period (1M, 3M, 6M, 1Y, 2Y)
            
        Returns:
            Risk analysis PDF as bytes
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_fallback_report(portfolio_id, "risk_analysis")
            
        try:
            # Get risk data
            risk_data = self._get_risk_analysis_data(portfolio_id, date_range)
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            
            # Title
            story.append(Paragraph("Risk Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # VaR Analysis
            story.extend(self._build_var_section(risk_data))
            
            # Stress Testing
            story.extend(self._build_stress_test_section(risk_data))
            
            # Correlation Analysis
            story.extend(self._build_correlation_section(risk_data))
            
            # Generate PDF
            doc.build(story)
            
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            print(f"Error generating risk analysis: {e}")
            return self._generate_error_report(portfolio_id, str(e))
    
    def export_performance_attribution(self, portfolio_id: str, period: str = "1Y") -> bytes:
        """
        Generate performance attribution analysis with factor breakdown.
        
        Args:
            portfolio_id: Portfolio identifier
            period: Attribution period
            
        Returns:
            Performance attribution PDF as bytes
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_fallback_report(portfolio_id, "performance_attribution")
            
        try:
            # Get attribution data
            attribution_data = self._get_attribution_data(portfolio_id, period)
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            
            # Title
            story.append(Paragraph("Performance Attribution Analysis", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Sector Attribution
            story.extend(self._build_sector_attribution(attribution_data))
            
            # Factor Attribution
            story.extend(self._build_factor_attribution(attribution_data))
            
            # Security Selection
            story.extend(self._build_security_selection(attribution_data))
            
            doc.build(story)
            
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            print(f"Error generating performance attribution: {e}")
            return self._generate_error_report(portfolio_id, str(e))
    
    def apply_tenant_branding(self, pdf_template: bytes, tenant_config: Dict) -> bytes:
        """
        Apply tenant-specific logos, colors, and branding to PDF exports.
        
        Args:
            pdf_template: Base PDF template as bytes
            tenant_config: Tenant branding configuration
            
        Returns:
            Branded PDF as bytes
        """
        # For now, return the original PDF with a note about branding
        # In a full implementation, this would overlay branding elements
        return pdf_template
    
    def _get_portfolio_data(self, portfolio_id: str, tenant_id: str) -> PortfolioReportData:
        """Get portfolio data for report generation (stub implementation)."""
        
        # Sample data for demonstration
        return PortfolioReportData(
            portfolio_id=portfolio_id,
            portfolio_name="Sample Portfolio",
            tenant_id=tenant_id,
            report_date=date.today(),
            total_value=10000000.0,
            total_return=0.087,
            benchmark_return=0.075,
            active_return=0.012,
            volatility=0.155,
            sharpe_ratio=0.52,
            var_95=-0.023,
            cvar_95=-0.031,
            beta=1.05,
            holdings=[
                {"symbol": "AAPL", "name": "Apple Inc.", "weight": 0.15, "value": 1500000},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "weight": 0.12, "value": 1200000},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "weight": 0.10, "value": 1000000},
                {"symbol": "AMZN", "name": "Amazon.com Inc.", "weight": 0.08, "value": 800000},
                {"symbol": "TSLA", "name": "Tesla Inc.", "weight": 0.05, "value": 500000},
            ],
            performance_history=pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=12, freq='ME'),
                'portfolio_return': np.random.normal(0.008, 0.02, 12),
                'benchmark_return': np.random.normal(0.006, 0.018, 12)
            })
        )
    
    def _get_risk_analysis_data(self, portfolio_id: str, date_range: str) -> Dict:
        """Get risk analysis data (stub implementation)."""
        return {
            'var_95': -0.023,
            'cvar_95': -0.031,
            'volatility': 0.155,
            'stress_scenarios': {
                '2008_crisis': -0.387,
                'covid_crash': -0.341,
                'dot_com_bubble': -0.492
            },
            'correlations': np.random.uniform(-1, 1, (10, 10))
        }
    
    def _get_attribution_data(self, portfolio_id: str, period: str) -> Dict:
        """Get performance attribution data (stub implementation)."""
        return {
            'sector_attribution': pd.DataFrame({
                'sector': ['Technology', 'Healthcare', 'Financials', 'Consumer'],
                'allocation_effect': [0.005, -0.002, 0.001, 0.003],
                'selection_effect': [0.008, 0.004, -0.001, 0.002],
                'total_effect': [0.013, 0.002, 0.0, 0.005]
            }),
            'factor_attribution': pd.DataFrame({
                'factor': ['Market', 'Size', 'Value', 'Momentum', 'Quality'],
                'exposure': [1.05, 0.23, -0.15, 0.08, 0.31],
                'return': [0.065, 0.012, -0.008, 0.025, 0.018],
                'contribution': [0.068, 0.003, 0.001, 0.002, 0.006]
            })
        }
    
    def _build_header(self, report_data: PortfolioReportData, 
                     branding: Optional[TenantBranding]) -> List:
        """Build report header with branding."""
        elements = []
        
        if branding and branding.logo_path and os.path.exists(branding.logo_path):
            # Add logo if available
            try:
                logo = Image(branding.logo_path, width=2*inch, height=1*inch)
                elements.append(logo)
            except:
                pass
        
        # Title
        title = f"{report_data.portfolio_name} - Portfolio Report"
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        
        # Date and basic info
        info_text = f"Report Date: {report_data.report_date.strftime('%B %d, %Y')}<br/>"
        info_text += f"Portfolio ID: {report_data.portfolio_id}<br/>"
        info_text += f"Total Value: ${report_data.total_value:,.0f}"
        
        elements.append(Paragraph(info_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        if branding and branding.confidentiality_notice:
            elements.append(Paragraph(branding.confidentiality_notice, self.styles['Confidential']))
            elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_executive_summary(self, report_data: PortfolioReportData) -> List:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Performance summary table
        summary_data = [
            ['Metric', 'Portfolio', 'Benchmark', 'Active'],
            ['Total Return', f"{report_data.total_return:.1%}", 
             f"{report_data.benchmark_return:.1%}", f"{report_data.active_return:.1%}"],
            ['Volatility', f"{report_data.volatility:.1%}", "15.2%", "-"],
            ['Sharpe Ratio', f"{report_data.sharpe_ratio:.2f}", "0.45", "-"],
            ['VaR (95%)', f"{report_data.var_95:.1%}", "-2.5%", "-"],
            ['Beta', f"{report_data.beta:.2f}", "1.00", f"{report_data.beta-1:.2f}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_holdings_section(self, report_data: PortfolioReportData) -> List:
        """Build portfolio holdings section."""
        elements = []
        
        elements.append(Paragraph("Portfolio Holdings", self.styles['SectionHeader']))
        
        # Holdings table
        holdings_data = [['Symbol', 'Security Name', 'Weight', 'Market Value']]
        
        for holding in report_data.holdings:
            holdings_data.append([
                holding['symbol'],
                holding['name'],
                f"{holding['weight']:.1%}",
                f"${holding['value']:,.0f}"
            ])
        
        holdings_table = Table(holdings_data, colWidths=[1*inch, 3*inch, 1*inch, 1.5*inch])
        holdings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        elements.append(holdings_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_performance_section(self, report_data: PortfolioReportData) -> List:
        """Build performance analysis section."""
        elements = []
        
        elements.append(Paragraph("Performance Analysis", self.styles['SectionHeader']))
        
        # Performance chart would go here (requires plotly integration)
        elements.append(Paragraph("Performance Chart: [Chart would be embedded here]", 
                                self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_risk_section(self, report_data: PortfolioReportData) -> List:
        """Build risk analysis section."""
        elements = []
        
        elements.append(Paragraph("Risk Analysis", self.styles['SectionHeader']))
        
        risk_text = f"""
        The portfolio exhibits a volatility of {report_data.volatility:.1%}, indicating moderate risk levels.
        Value at Risk (95% confidence) is {report_data.var_95:.1%}, suggesting potential maximum daily loss.
        The portfolio beta of {report_data.beta:.2f} indicates higher sensitivity to market movements.
        """
        
        elements.append(Paragraph(risk_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_footer(self, branding: Optional[TenantBranding]) -> List:
        """Build report footer."""
        elements = []
        
        footer_text = "This report is generated by the Quantum Portfolio Optimizer platform."
        if branding and branding.footer_text:
            footer_text = branding.footer_text
            
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(footer_text, self.styles['Normal']))
        
        return elements
    
    def _build_var_section(self, risk_data: Dict) -> List:
        """Build VaR analysis section."""
        elements = []
        
        elements.append(Paragraph("Value at Risk Analysis", self.styles['SectionHeader']))
        
        var_text = f"""
        95% Value at Risk: {risk_data['var_95']:.1%}<br/>
        95% Conditional VaR: {risk_data['cvar_95']:.1%}<br/>
        Portfolio Volatility: {risk_data['volatility']:.1%}
        """
        
        elements.append(Paragraph(var_text, self.styles['Normal']))
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_stress_test_section(self, risk_data: Dict) -> List:
        """Build stress testing section."""
        elements = []
        
        elements.append(Paragraph("Stress Test Results", self.styles['SubHeader']))
        
        stress_data = [['Scenario', 'Portfolio Impact']]
        for scenario, impact in risk_data['stress_scenarios'].items():
            stress_data.append([scenario.replace('_', ' ').title(), f"{impact:.1%}"])
        
        stress_table = Table(stress_data, colWidths=[3*inch, 2*inch])
        stress_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        elements.append(stress_table)
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_correlation_section(self, risk_data: Dict) -> List:
        """Build correlation analysis section."""
        elements = []
        
        elements.append(Paragraph("Correlation Analysis", self.styles['SubHeader']))
        elements.append(Paragraph("Correlation matrix analysis would be displayed here.", 
                                self.styles['Normal']))
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _build_sector_attribution(self, attribution_data: Dict) -> List:
        """Build sector attribution section."""
        elements = []
        
        elements.append(Paragraph("Sector Attribution", self.styles['SectionHeader']))
        
        if 'sector_attribution' in attribution_data:
            df = attribution_data['sector_attribution']
            
            # Convert DataFrame to table
            table_data = [['Sector', 'Allocation Effect', 'Selection Effect', 'Total Effect']]
            for _, row in df.iterrows():
                table_data.append([
                    row['sector'],
                    f"{row['allocation_effect']:.2%}",
                    f"{row['selection_effect']:.2%}",
                    f"{row['total_effect']:.2%}"
                ])
            
            attribution_table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            attribution_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            
            elements.append(attribution_table)
        
        elements.append(Spacer(1, 12))
        return elements
    
    def _build_factor_attribution(self, attribution_data: Dict) -> List:
        """Build factor attribution section."""
        elements = []
        
        elements.append(Paragraph("Factor Attribution", self.styles['SubHeader']))
        
        if 'factor_attribution' in attribution_data:
            df = attribution_data['factor_attribution']
            
            # Convert DataFrame to table
            table_data = [['Factor', 'Exposure', 'Return', 'Contribution']]
            for _, row in df.iterrows():
                table_data.append([
                    row['factor'],
                    f"{row['exposure']:.2f}",
                    f"{row['return']:.2%}",
                    f"{row['contribution']:.2%}"
                ])
            
            factor_table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            factor_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            
            elements.append(factor_table)
        
        elements.append(Spacer(1, 12))
        return elements
    
    def _build_security_selection(self, attribution_data: Dict) -> List:
        """Build security selection section."""
        elements = []
        
        elements.append(Paragraph("Security Selection Analysis", self.styles['SubHeader']))
        elements.append(Paragraph("Security-level attribution analysis would be displayed here.", 
                                self.styles['Normal']))
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _generate_fallback_report(self, portfolio_id: str, report_type: str) -> bytes:
        """Generate a simple text-based report when ReportLab is not available."""
        report_content = f"""
        PORTFOLIO REPORT - {report_type.upper()}
        ================================
        
        Portfolio ID: {portfolio_id}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        This is a fallback text report. For full PDF functionality,
        please install the required dependencies:
        - reportlab
        - plotly
        - Pillow
        
        Report Type: {report_type}
        Status: Generated successfully
        """
        
        return report_content.encode('utf-8')
    
    def _generate_error_report(self, portfolio_id: str, error_message: str) -> bytes:
        """Generate an error report when PDF generation fails."""
        error_content = f"""
        PORTFOLIO REPORT - ERROR
        ========================
        
        Portfolio ID: {portfolio_id}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Error occurred during report generation:
        {error_message}
        
        Please check your data and try again.
        """
        
        return error_content.encode('utf-8')


# Convenience functions for external use
def generate_portfolio_report(portfolio_id: str, tenant_id: str) -> bytes:
    """Generate a comprehensive portfolio report."""
    engine = PDFExportEngine()
    return engine.generate_portfolio_report(portfolio_id, tenant_id)

def create_risk_analysis_pdf(portfolio_id: str, date_range: str = "1Y") -> bytes:
    """Generate a risk analysis report."""
    engine = PDFExportEngine()
    return engine.create_risk_analysis_pdf(portfolio_id, date_range)

def export_performance_attribution(portfolio_id: str, period: str = "1Y") -> bytes:
    """Generate a performance attribution report."""
    engine = PDFExportEngine()
    return engine.export_performance_attribution(portfolio_id, period)

def apply_tenant_branding(pdf_template: bytes, tenant_config: Dict) -> bytes:
    """Apply tenant branding to a PDF."""
    engine = PDFExportEngine()
    return engine.apply_tenant_branding(pdf_template, tenant_config)


if __name__ == "__main__":
    # Test the PDF export engine
    print("Testing PDF Export Engine...")
    
    # Test portfolio report generation
    try:
        pdf_bytes = generate_portfolio_report("test_portfolio_001", "test_tenant_001")
        print(f"Portfolio report generated: {len(pdf_bytes)} bytes")
        
        # Save test report
        with open("test_portfolio_report.pdf", "wb") as f:
            f.write(pdf_bytes)
        print("Test report saved as test_portfolio_report.pdf")
        
    except Exception as e:
        print(f"Error testing PDF export: {e}")
    
    print("PDF Export Engine test completed.")
