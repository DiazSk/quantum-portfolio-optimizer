"""
Compliance Dashboard & Filing Management System for Quantum Portfolio Optimizer

This module provides an integrated compliance dashboard with regulatory filing management,
compliance monitoring, and automated reporting capabilities for institutional requirements.

Features:
- Real-time compliance monitoring dashboard
- Regulatory filing calendar and tracking
- Automated compliance reporting
- Risk threshold monitoring and alerts
- Audit trail visualization
- ML model compliance tracking
- Client communication management
- Regulatory deadlines and reminders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
from pathlib import Path
import calendar

from ..database.connection import DatabaseConnection
from ..models.regulatory_reporting import RegulatoryReportingEngine
from ..models.client_reporting import ClientReportingSystem
from ..models.enhanced_model_manager import MLLineageReportingSystem
from ..utils.immutable_audit_trail import ImmutableAuditTrail
from ..utils.professional_logging import get_logger

logger = get_logger(__name__)


class ComplianceDashboard:
    """
    Main compliance dashboard providing comprehensive regulatory monitoring
    and filing management capabilities.
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.regulatory_engine = RegulatoryReportingEngine(db_connection)
        self.client_reporting = ClientReportingSystem(db_connection)
        self.ml_lineage = MLLineageReportingSystem(db_connection)
        self.audit_trail = ImmutableAuditTrail(db_connection)
    
    async def render_main_dashboard(self):
        """Render the main compliance dashboard."""
        
        st.set_page_config(
            page_title="Quantum Portfolio Optimizer - Compliance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏛️ Institutional Compliance Dashboard")
        st.markdown("*Real-time regulatory compliance monitoring and filing management*")
        
        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            dashboard_section = st.selectbox(
                "Select Dashboard Section",
                [
                    "📊 Compliance Overview",
                    "📋 Regulatory Filings",
                    "👥 Client Reporting",
                    "🤖 ML Model Compliance",
                    "🔍 Audit Trail",
                    "⚠️ Risk Monitoring",
                    "📅 Compliance Calendar",
                    "⚙️ Settings"
                ]
            )
        
        # Route to selected dashboard section
        if dashboard_section == "📊 Compliance Overview":
            await self.render_compliance_overview()
        elif dashboard_section == "📋 Regulatory Filings":
            await self.render_regulatory_filings()
        elif dashboard_section == "👥 Client Reporting":
            await self.render_client_reporting()
        elif dashboard_section == "🤖 ML Model Compliance":
            await self.render_ml_compliance()
        elif dashboard_section == "🔍 Audit Trail":
            await self.render_audit_trail()
        elif dashboard_section == "⚠️ Risk Monitoring":
            await self.render_risk_monitoring()
        elif dashboard_section == "📅 Compliance Calendar":
            await self.render_compliance_calendar()
        elif dashboard_section == "⚙️ Settings":
            await self.render_settings()
    
    async def render_compliance_overview(self):
        """Render compliance overview dashboard."""
        
        st.header("📊 Compliance Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Get compliance metrics
        compliance_metrics = await self._get_compliance_metrics()
        
        with col1:
            st.metric(
                "Compliance Score",
                f"{compliance_metrics['compliance_score']:.1f}%",
                delta=f"{compliance_metrics['compliance_trend']:+.1f}%"
            )
        
        with col2:
            st.metric(
                "Pending Filings",
                compliance_metrics['pending_filings'],
                delta=f"{compliance_metrics['filing_trend']:+d}"
            )
        
        with col3:
            st.metric(
                "Active Audits",
                compliance_metrics['active_audits'],
                delta=None
            )
        
        with col4:
            st.metric(
                "Risk Alerts",
                compliance_metrics['risk_alerts'],
                delta=f"{compliance_metrics['alert_trend']:+d}"
            )
        
        # Compliance status overview
        st.subheader("📈 Compliance Status Overview")
        
        # Create compliance status chart
        compliance_data = await self._get_compliance_status_data()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = self._create_compliance_trend_chart(compliance_data)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Compliance breakdown pie chart
            fig_pie = self._create_compliance_breakdown_pie(compliance_metrics)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent activity
        st.subheader("📋 Recent Compliance Activity")
        
        recent_activity = await self._get_recent_compliance_activity()
        
        for activity in recent_activity[:5]:  # Show last 5 activities
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.write(f"🕒 {activity['timestamp'].strftime('%H:%M')}")
                
                with col2:
                    st.write(f"**{activity['type']}:** {activity['description']}")
                
                with col3:
                    status_color = {"completed": "🟢", "pending": "🟡", "failed": "🔴"}
                    st.write(status_color.get(activity['status'], "⚪") + f" {activity['status'].title()}")
                
                st.divider()
        
        # Upcoming deadlines
        st.subheader("⏰ Upcoming Regulatory Deadlines")
        
        deadlines = await self._get_upcoming_deadlines()
        
        if deadlines:
            deadline_df = pd.DataFrame(deadlines)
            deadline_df['days_until'] = (deadline_df['deadline'] - pd.Timestamp.now()).dt.days
            
            # Color code by urgency
            def style_deadline_urgency(val):
                if val <= 3:
                    return 'background-color: #ffebee'  # Red
                elif val <= 7:
                    return 'background-color: #fff3e0'  # Orange
                elif val <= 14:
                    return 'background-color: #fffde7'  # Yellow
                return ''
            
            styled_df = deadline_df.style.applymap(
                style_deadline_urgency, 
                subset=['days_until']
            )
            
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No upcoming regulatory deadlines in the next 30 days.")
    
    async def render_regulatory_filings(self):
        """Render regulatory filings management interface."""
        
        st.header("📋 Regulatory Filings Management")
        
        # Filing controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            filing_type = st.selectbox(
                "Filing Type",
                ["All", "Form PF", "AIFMD", "Solvency II", "MiFID II", "CFTC", "SEC"]
            )
        
        with col2:
            filing_status = st.selectbox(
                "Status",
                ["All", "Draft", "Pending Review", "Submitted", "Accepted", "Rejected"]
            )
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(date.today() - timedelta(days=30), date.today()),
                max_value=date.today()
            )
        
        # Filing summary cards
        filing_summary = await self._get_filing_summary(filing_type, filing_status, date_range)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Filings", filing_summary['total'])
        
        with col2:
            st.metric("Submitted", filing_summary['submitted'])
        
        with col3:
            st.metric("Pending", filing_summary['pending'])
        
        with col4:
            st.metric("Overdue", filing_summary['overdue'])
        
        # Filing actions
        st.subheader("📄 Filing Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Create New Filing", type="primary"):
                await self._show_new_filing_dialog()
        
        with col2:
            if st.button("📊 Generate Report"):
                await self._generate_filing_report()
        
        with col3:
            if st.button("📤 Bulk Submit"):
                await self._show_bulk_submit_dialog()
        
        # Filing list
        st.subheader("📋 Recent Filings")
        
        filings = await self._get_filings_list(filing_type, filing_status, date_range)
        
        if filings:
            filing_df = pd.DataFrame(filings)
            
            # Add action buttons
            for idx, filing in filing_df.iterrows():
                with st.expander(f"{filing['filing_type']} - {filing['filing_id']} ({filing['status']})"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Due Date:** {filing['due_date']}")
                        st.write(f"**Created:** {filing['created_at']}")
                        st.write(f"**Description:** {filing['description']}")
                    
                    with col2:
                        if st.button(f"👁️ View", key=f"view_{filing['filing_id']}"):
                            await self._view_filing_details(filing['filing_id'])
                    
                    with col3:
                        if filing['status'] in ['Draft', 'Pending Review']:
                            if st.button(f"✏️ Edit", key=f"edit_{filing['filing_id']}"):
                                await self._edit_filing(filing['filing_id'])
        else:
            st.info("No filings found for the selected criteria.")
    
    async def render_client_reporting(self):
        """Render client reporting management interface."""
        
        st.header("👥 Client Reporting Dashboard")
        
        # Client selection
        clients = await self._get_active_clients()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_client = st.selectbox(
                "Select Client",
                options=["All Clients"] + [client['name'] for client in clients],
                index=0
            )
        
        with col2:
            report_period = st.selectbox(
                "Report Period",
                ["Current Month", "Previous Month", "Quarter", "Year-to-Date", "Custom"]
            )
        
        # Client reporting metrics
        client_metrics = await self._get_client_reporting_metrics(selected_client, report_period)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Reports Generated", client_metrics['reports_generated'])
        
        with col2:
            st.metric("On-Time Delivery", f"{client_metrics['on_time_percentage']:.1f}%")
        
        with col3:
            st.metric("Client Satisfaction", f"{client_metrics['satisfaction_score']:.1f}/5")
        
        with col4:
            st.metric("Pending Reports", client_metrics['pending_reports'])
        
        # Report generation
        st.subheader("📊 Generate Client Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Performance Report", type="primary"):
                await self._generate_performance_report(selected_client)
        
        with col2:
            if st.button("⚠️ Risk Report"):
                await self._generate_risk_report(selected_client)
        
        with col3:
            if st.button("📋 Compliance Report"):
                await self._generate_compliance_report(selected_client)
        
        # Recent client reports
        st.subheader("📄 Recent Client Reports")
        
        recent_reports = await self._get_recent_client_reports(selected_client)
        
        if recent_reports:
            for report in recent_reports:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{report['report_type']}** - {report['client_name']}")
                        st.write(f"Period: {report['period']}")
                    
                    with col2:
                        st.write(f"📅 {report['created_at'].strftime('%Y-%m-%d')}")
                    
                    with col3:
                        status_icons = {
                            'generated': '✅',
                            'delivered': '📧',
                            'pending': '⏳',
                            'failed': '❌'
                        }
                        st.write(f"{status_icons.get(report['status'], '❓')} {report['status'].title()}")
                    
                    with col4:
                        if st.button("📥", key=f"download_{report['report_id']}"):
                            await self._download_report(report['report_id'])
                
                st.divider()
        else:
            st.info("No recent client reports found.")
    
    async def render_ml_compliance(self):
        """Render ML model compliance dashboard."""
        
        st.header("🤖 ML Model Compliance Dashboard")
        
        # ML compliance overview
        ml_metrics = await self._get_ml_compliance_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Production Models",
                ml_metrics['production_models'],
                delta=f"{ml_metrics['model_trend']:+d}"
            )
        
        with col2:
            st.metric(
                "Compliance Rate",
                f"{ml_metrics['compliance_rate']:.1f}%",
                delta=f"{ml_metrics['compliance_trend']:+.1f}%"
            )
        
        with col3:
            st.metric(
                "Model Performance",
                f"{ml_metrics['avg_performance']:.1f}%",
                delta=f"{ml_metrics['performance_trend']:+.1f}%"
            )
        
        with col4:
            st.metric(
                "Data Quality",
                f"{ml_metrics['data_quality']:.1f}%",
                delta=f"{ml_metrics['quality_trend']:+.1f}%"
            )
        
        # Model lineage visualization
        st.subheader("🔗 Model Lineage & Performance")
        
        # Model selection
        models = await self._get_active_models()
        selected_model = st.selectbox(
            "Select Model for Detailed View",
            options=[model['model_id'] for model in models]
        )
        
        if selected_model:
            # Get model lineage
            lineage_data = await self.ml_lineage.lineage_tracker.get_model_lineage(selected_model)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Model Information")
                model_info = lineage_data['model_info']
                st.write(f"**Model Name:** {model_info['model_name']}")
                st.write(f"**Version:** {model_info['model_version']}")
                st.write(f"**Status:** {model_info['deployment_status']}")
                st.write(f"**Created:** {model_info['created_at']}")
                st.write(f"**Compliance Approved:** {'✅' if model_info['compliance_approved'] else '❌'}")
            
            with col2:
                st.subheader("📈 Performance Metrics")
                if 'performance_metrics' in model_info:
                    perf_data = json.loads(model_info['performance_metrics'])
                    st.write(f"**Accuracy:** {perf_data.get('accuracy', 'N/A'):.3f}")
                    st.write(f"**Precision:** {perf_data.get('precision', 'N/A'):.3f}")
                    st.write(f"**Recall:** {perf_data.get('recall', 'N/A'):.3f}")
                    st.write(f"**F1 Score:** {perf_data.get('f1_score', 'N/A'):.3f}")
            
            # Feature importance chart
            if lineage_data['model_info'] and 'performance_metrics' in lineage_data['model_info']:
                perf_metrics = json.loads(lineage_data['model_info']['performance_metrics'])
                if 'feature_importance' in perf_metrics:
                    st.subheader("🎯 Feature Importance")
                    feature_importance = perf_metrics['feature_importance']
                    
                    fig = px.bar(
                        x=list(feature_importance.values()),
                        y=list(feature_importance.keys()),
                        orientation='h',
                        title="Feature Importance Rankings"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Model compliance issues
        st.subheader("⚠️ Model Compliance Issues")
        
        compliance_issues = await self._get_ml_compliance_issues()
        
        if compliance_issues:
            for issue in compliance_issues:
                severity_colors = {
                    'critical': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }
                
                with st.container():
                    st.write(f"{severity_colors.get(issue['severity'], '⚪')} **{issue['type'].replace('_', ' ').title()}**")
                    st.write(f"Description: {issue['description']}")
                    st.write(f"Recommendation: {issue['recommendation']}")
                    
                    if st.button(f"Resolve Issue", key=f"resolve_{issue.get('model_id', 'general')}"):
                        await self._resolve_compliance_issue(issue)
                
                st.divider()
        else:
            st.success("No ML compliance issues found!")
    
    async def render_audit_trail(self):
        """Render audit trail visualization."""
        
        st.header("🔍 Audit Trail Dashboard")
        
        # Audit trail filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_type = st.selectbox(
                "Event Type",
                ["All", "Portfolio Decision", "Trade Execution", "Risk Override", "ML Prediction", "System Access"]
            )
        
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(date.today() - timedelta(days=7), date.today()),
                max_value=date.today()
            )
        
        with col3:
            user_filter = st.text_input("User ID Filter (optional)")
        
        # Audit statistics
        audit_stats = await self._get_audit_statistics(event_type, date_range, user_filter)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", audit_stats['total_events'])
        
        with col2:
            st.metric("Unique Users", audit_stats['unique_users'])
        
        with col3:
            st.metric("Event Types", audit_stats['event_types'])
        
        with col4:
            st.metric("Integrity Score", f"{audit_stats['integrity_score']:.1f}%")
        
        # Audit trail timeline
        st.subheader("📈 Audit Event Timeline")
        
        timeline_data = await self._get_audit_timeline(event_type, date_range, user_filter)
        
        if timeline_data:
            fig = px.line(
                timeline_data,
                x='timestamp',
                y='event_count',
                title="Audit Events Over Time",
                labels={'event_count': 'Number of Events', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent audit events
        st.subheader("📋 Recent Audit Events")
        
        recent_events = await self._get_recent_audit_events(event_type, date_range, user_filter)
        
        if recent_events:
            # Display events in expandable format
            for event in recent_events[:10]:  # Show last 10 events
                with st.expander(f"{event['event_type']} - {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write(f"**Event ID:** {event['event_id']}")
                        st.write(f"**User ID:** {event['user_id']}")
                        st.write(f"**IP Address:** {event.get('ip_address', 'N/A')}")
                        st.write(f"**Session ID:** {event.get('session_id', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Hash:** {event['hash_value'][:16]}...")
                        st.write(f"**Previous Hash:** {event['previous_hash'][:16] if event['previous_hash'] else 'Genesis'}...")
                        st.write(f"**Signature Valid:** {'✅' if event.get('signature_valid', True) else '❌'}")
                    
                    # Event data
                    if event.get('event_data'):
                        st.subheader("Event Data")
                        st.json(event['event_data'])
        else:
            st.info("No audit events found for the selected criteria.")
        
        # Audit integrity verification
        st.subheader("🔐 Audit Trail Integrity Verification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Verify Integrity", type="primary"):
                with st.spinner("Verifying audit trail integrity..."):
                    integrity_result = await self._verify_audit_integrity()
                    
                    if integrity_result['valid']:
                        st.success(f"✅ Audit trail integrity verified! {integrity_result['verified_entries']} entries checked.")
                    else:
                        st.error(f"❌ Integrity violation detected! {integrity_result['invalid_entries']} invalid entries found.")
                        
                        for violation in integrity_result.get('violations', []):
                            st.error(f"Violation at entry {violation['entry_id']}: {violation['description']}")
        
        with col2:
            if st.button("📊 Generate Audit Report"):
                await self._generate_audit_report(event_type, date_range, user_filter)
    
    async def render_compliance_calendar(self):
        """Render compliance calendar with deadlines and reminders."""
        
        st.header("📅 Compliance Calendar")
        
        # Calendar view selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            view_type = st.selectbox(
                "Calendar View",
                ["Month", "Quarter", "Year"]
            )
        
        with col2:
            selected_date = st.date_input(
                "Calendar Date",
                value=date.today()
            )
        
        # Get calendar data
        calendar_data = await self._get_compliance_calendar_data(view_type, selected_date)
        
        # Calendar visualization
        if view_type == "Month":
            await self._render_monthly_calendar(selected_date, calendar_data)
        elif view_type == "Quarter":
            await self._render_quarterly_calendar(selected_date, calendar_data)
        else:
            await self._render_yearly_calendar(selected_date, calendar_data)
        
        # Upcoming deadlines summary
        st.subheader("⏰ Upcoming Deadlines")
        
        upcoming_deadlines = await self._get_upcoming_deadlines(30)  # Next 30 days
        
        if upcoming_deadlines:
            deadline_df = pd.DataFrame(upcoming_deadlines)
            deadline_df['days_remaining'] = (deadline_df['deadline'] - pd.Timestamp.now()).dt.days
            
            # Sort by urgency
            deadline_df = deadline_df.sort_values('days_remaining')
            
            for _, deadline in deadline_df.iterrows():
                urgency_color = "🔴" if deadline['days_remaining'] <= 3 else "🟡" if deadline['days_remaining'] <= 7 else "🟢"
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"{urgency_color} **{deadline['title']}**")
                        st.write(f"Type: {deadline['type']}")
                    
                    with col2:
                        st.write(f"📅 {deadline['deadline'].strftime('%Y-%m-%d')}")
                    
                    with col3:
                        st.write(f"⏱️ {deadline['days_remaining']} days")
                        
                        if st.button("✅", key=f"complete_{deadline['id']}"):
                            await self._mark_deadline_complete(deadline['id'])
                
                st.divider()
        else:
            st.info("No upcoming deadlines in the next 30 days.")
    
    # Helper methods for data retrieval and processing
    
    async def _get_compliance_metrics(self) -> Dict[str, Any]:
        """Get key compliance metrics."""
        # This would query the database for actual metrics
        return {
            'compliance_score': 94.5,
            'compliance_trend': 2.1,
            'pending_filings': 3,
            'filing_trend': -1,
            'active_audits': 2,
            'risk_alerts': 5,
            'alert_trend': 2
        }
    
    async def _get_compliance_status_data(self) -> Dict[str, Any]:
        """Get compliance status trend data."""
        # Mock data - would be replaced with actual database queries
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        scores = np.random.normal(92, 3, len(dates))
        scores = np.clip(scores, 85, 100)  # Keep within reasonable range
        
        return {
            'dates': dates,
            'compliance_scores': scores,
            'target_score': 90
        }
    
    def _create_compliance_trend_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create compliance trend chart."""
        fig = go.Figure()
        
        # Compliance score line
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['compliance_scores'],
            mode='lines+markers',
            name='Compliance Score',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Target line
        fig.add_hline(
            y=data['target_score'],
            line_dash="dash",
            line_color="red",
            annotation_text="Target (90%)"
        )
        
        fig.update_layout(
            title="Compliance Score Trend",
            xaxis_title="Date",
            yaxis_title="Compliance Score (%)",
            yaxis=dict(range=[80, 100]),
            height=400
        )
        
        return fig
    
    def _create_compliance_breakdown_pie(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create compliance breakdown pie chart."""
        compliant = metrics['compliance_score']
        non_compliant = 100 - compliant
        
        fig = go.Figure(data=[go.Pie(
            labels=['Compliant', 'Non-Compliant'],
            values=[compliant, non_compliant],
            colors=['#2E8B57', '#DC143C'],
            hole=0.3
        )])
        
        fig.update_layout(
            title="Compliance Status",
            height=400,
            showlegend=True
        )
        
        return fig
    
    async def _get_recent_compliance_activity(self) -> List[Dict[str, Any]]:
        """Get recent compliance activity."""
        # Mock data - would query audit trail
        activities = []
        for i in range(10):
            activities.append({
                'timestamp': datetime.now() - timedelta(hours=i),
                'type': f'Activity {i+1}',
                'description': f'Sample compliance activity {i+1}',
                'status': np.random.choice(['completed', 'pending', 'failed'])
            })
        
        return activities
    
    async def _get_upcoming_deadlines(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Get upcoming regulatory deadlines."""
        # Mock data - would query compliance calendar
        deadlines = []
        for i in range(5):
            deadlines.append({
                'id': f'deadline_{i}',
                'title': f'Regulatory Filing {i+1}',
                'type': np.random.choice(['Form PF', 'AIFMD', 'Solvency II']),
                'deadline': datetime.now() + timedelta(days=np.random.randint(1, days_ahead)),
                'status': 'pending'
            })
        
        return deadlines


class ComplianceCalendarWidget:
    """Widget for rendering compliance calendar views."""
    
    def __init__(self, dashboard: ComplianceDashboard):
        self.dashboard = dashboard
    
    async def render_monthly_view(self, selected_date: date, calendar_data: Dict[str, Any]):
        """Render monthly calendar view."""
        
        # Get calendar for selected month
        cal = calendar.monthcalendar(selected_date.year, selected_date.month)
        month_name = calendar.month_name[selected_date.month]
        
        st.subheader(f"📅 {month_name} {selected_date.year}")
        
        # Create calendar grid
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Header row
        cols = st.columns(7)
        for i, day in enumerate(days):
            with cols[i]:
                st.write(f"**{day}**")
        
        # Calendar rows
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                with cols[i]:
                    if day == 0:
                        st.write("")  # Empty cell for days not in month
                    else:
                        # Check for events on this day
                        day_events = self._get_day_events(
                            date(selected_date.year, selected_date.month, day),
                            calendar_data
                        )
                        
                        # Display day number
                        if day == selected_date.day:
                            st.markdown(f"**🔴 {day}**")  # Highlight today
                        else:
                            st.write(str(day))
                        
                        # Display event indicators
                        for event in day_events[:3]:  # Max 3 events per day
                            event_icon = self._get_event_icon(event['type'])
                            st.write(f"{event_icon} {event['title'][:10]}...")
    
    def _get_day_events(self, target_date: date, calendar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get events for a specific day."""
        # Filter events for the target date
        day_events = []
        for event in calendar_data.get('events', []):
            if event['date'].date() == target_date:
                day_events.append(event)
        
        return day_events
    
    def _get_event_icon(self, event_type: str) -> str:
        """Get icon for event type."""
        icons = {
            'filing': '📋',
            'audit': '🔍',
            'review': '👁️',
            'deadline': '⏰',
            'meeting': '🤝',
            'report': '📊'
        }
        return icons.get(event_type, '📅')


# Main dashboard entry point
async def main():
    """Main entry point for the compliance dashboard."""
    
    try:
        # Initialize database connection
        db = DatabaseConnection()
        await db.connect()
        
        # Create and render dashboard
        dashboard = ComplianceDashboard(db)
        await dashboard.render_main_dashboard()
        
    except Exception as e:
        st.error(f"Dashboard Error: {str(e)}")
        logger.error(f"Dashboard initialization failed: {e}")
    
    finally:
        if 'db' in locals():
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())
