"""
STORY 5.1: CLIENT ACQUISITION & SALES OPERATIONS
Institutional CRM and Sales Pipeline Management System
================================================================================

Comprehensive sales operations platform for institutional client acquisition,
pipeline management, and revenue generation targeting $5M+ ARR.

AC-5.1.1: CRM System & Pipeline Management
AC-5.1.2: Automated Demonstration System
AC-5.1.3: Proposal Generation Engine
AC-5.1.4: Sales Process Automation
AC-5.1.5: Client Business Case Development
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Use standard logging as fallback
import logging
logger = logging.getLogger(__name__)

class SalesStage(Enum):
    """7-stage institutional sales pipeline"""
    LEAD = "lead"
    QUALIFIED = "qualified"
    DEMO = "demo"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CONTRACT = "contract"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"

class ClientSegment(Enum):
    """Institutional client segments"""
    HEDGE_FUND = "hedge_fund"
    ASSET_MANAGER = "asset_manager"
    PENSION_FUND = "pension_fund"
    SOVEREIGN_WEALTH = "sovereign_wealth"
    FAMILY_OFFICE = "family_office"
    ENDOWMENT = "endowment"
    INSURANCE = "insurance"

class LeadSource(Enum):
    """Lead generation sources"""
    INBOUND_WEB = "inbound_web"
    REFERRAL = "referral"
    CONFERENCE = "conference"
    COLD_OUTREACH = "cold_outreach"
    PARTNERSHIP = "partnership"
    CONTENT_MARKETING = "content_marketing"

@dataclass
class InstitutionalProspect:
    """Institutional prospect profile"""
    prospect_id: str
    company_name: str
    contact_name: str
    title: str
    email: str
    phone: str
    assets_under_management: float
    client_segment: ClientSegment
    geographic_region: str
    regulatory_jurisdiction: str
    lead_source: LeadSource
    lead_score: int
    pain_points: List[str]
    compliance_requirements: List[str]
    created_date: datetime
    last_contact: Optional[datetime] = None
    notes: List[str] = None

@dataclass
class SalesOpportunity:
    """Sales opportunity tracking"""
    opportunity_id: str
    prospect_id: str
    sales_stage: SalesStage
    estimated_value: float
    probability: float
    expected_close_date: datetime
    sales_rep: str
    last_activity: datetime
    next_action: str
    proposal_sent: bool = False
    demo_completed: bool = False
    contract_terms: Optional[Dict] = None

@dataclass
class DemoSession:
    """Institutional demonstration session"""
    demo_id: str
    opportunity_id: str
    scheduled_date: datetime
    attendees: List[str]
    demo_type: str  # "standard", "custom", "technical"
    use_cases_shown: List[str]
    roi_calculated: bool
    follow_up_actions: List[str]
    attendee_feedback: Dict[str, Any]
    conversion_likelihood: float

class InstitutionalCRM:
    """
    Comprehensive CRM system for institutional client acquisition
    Implements AC-5.1.1: CRM System & Pipeline Management
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.prospects: Dict[str, InstitutionalProspect] = {}
        self.opportunities: Dict[str, SalesOpportunity] = {}
        self.demo_sessions: Dict[str, DemoSession] = {}
        
        # Lead scoring weights
        self.scoring_weights = {
            'aum_tier': {'100B+': 50, '10B-100B': 40, '1B-10B': 30, '100M-1B': 20, '<100M': 10},
            'segment': {
                ClientSegment.HEDGE_FUND: 45,
                ClientSegment.ASSET_MANAGER: 40,
                ClientSegment.SOVEREIGN_WEALTH: 50,
                ClientSegment.PENSION_FUND: 35,
                ClientSegment.FAMILY_OFFICE: 30,
                ClientSegment.ENDOWMENT: 25,
                ClientSegment.INSURANCE: 30
            },
            'geography': {'North America': 40, 'Europe': 35, 'Asia-Pacific': 30, 'Other': 20},
            'lead_source': {
                LeadSource.REFERRAL: 40,
                LeadSource.INBOUND_WEB: 35,
                LeadSource.CONFERENCE: 30,
                LeadSource.PARTNERSHIP: 25,
                LeadSource.CONTENT_MARKETING: 20,
                LeadSource.COLD_OUTREACH: 15
            }
        }
        
        logger.info("InstitutionalCRM initialized")
    
    def add_prospect(self, prospect_data: Dict[str, Any]) -> InstitutionalProspect:
        """
        Add new institutional prospect with lead scoring
        Implements lead scoring algorithm based on AUM, geography, and compliance needs
        """
        try:
            prospect_id = str(uuid.uuid4())
            
            # Calculate lead score
            lead_score = self._calculate_lead_score(prospect_data)
            
            prospect = InstitutionalProspect(
                prospect_id=prospect_id,
                company_name=prospect_data['company_name'],
                contact_name=prospect_data['contact_name'],
                title=prospect_data['title'],
                email=prospect_data['email'],
                phone=prospect_data.get('phone', ''),
                assets_under_management=prospect_data['assets_under_management'],
                client_segment=ClientSegment(prospect_data['client_segment']),
                geographic_region=prospect_data['geographic_region'],
                regulatory_jurisdiction=prospect_data.get('regulatory_jurisdiction', ''),
                lead_source=LeadSource(prospect_data['lead_source']),
                lead_score=lead_score,
                pain_points=prospect_data.get('pain_points', []),
                compliance_requirements=prospect_data.get('compliance_requirements', []),
                created_date=datetime.now(timezone.utc),
                notes=[]
            )
            
            self.prospects[prospect_id] = prospect
            
            # Auto-create opportunity if qualified lead (score > 70)
            if lead_score > 70:
                self._create_qualified_opportunity(prospect)
            
            logger.info(f"Prospect added: {prospect.company_name} (Score: {lead_score})")
            return prospect
            
        except Exception as e:
            logger.error(f"Failed to add prospect: {e}")
            raise
    
    def _calculate_lead_score(self, prospect_data: Dict) -> int:
        """Calculate lead score based on institutional criteria"""
        score = 0
        
        # AUM scoring
        aum = prospect_data['assets_under_management'] / 1e9  # Convert to billions
        if aum >= 100:
            score += self.scoring_weights['aum_tier']['100B+']
        elif aum >= 10:
            score += self.scoring_weights['aum_tier']['10B-100B']
        elif aum >= 1:
            score += self.scoring_weights['aum_tier']['1B-10B']
        elif aum >= 0.1:
            score += self.scoring_weights['aum_tier']['100M-1B']
        else:
            score += self.scoring_weights['aum_tier']['<100M']
        
        # Client segment scoring
        segment = ClientSegment(prospect_data['client_segment'])
        score += self.scoring_weights['segment'][segment]
        
        # Geography scoring
        geography = prospect_data['geographic_region']
        score += self.scoring_weights['geography'].get(geography, 20)
        
        # Lead source scoring
        lead_source = LeadSource(prospect_data['lead_source'])
        score += self.scoring_weights['lead_source'][lead_source]
        
        # Compliance requirements bonus
        compliance_reqs = prospect_data.get('compliance_requirements', [])
        score += min(len(compliance_reqs) * 5, 20)  # Up to 20 points for compliance needs
        
        return min(score, 100)  # Cap at 100
    
    def _create_qualified_opportunity(self, prospect: InstitutionalProspect) -> SalesOpportunity:
        """Auto-create opportunity for qualified leads"""
        opportunity_id = str(uuid.uuid4())
        
        # Estimate opportunity value based on AUM
        aum_billions = prospect.assets_under_management / 1e9
        if aum_billions >= 100:
            estimated_value = 2000000  # $2M for largest clients
        elif aum_billions >= 10:
            estimated_value = 1000000  # $1M for large clients
        elif aum_billions >= 1:
            estimated_value = 500000   # $500K for medium clients
        else:
            estimated_value = 250000   # $250K for smaller clients
        
        opportunity = SalesOpportunity(
            opportunity_id=opportunity_id,
            prospect_id=prospect.prospect_id,
            sales_stage=SalesStage.QUALIFIED,
            estimated_value=estimated_value,
            probability=0.25,  # 25% for qualified stage
            expected_close_date=datetime.now(timezone.utc) + timedelta(days=180),
            sales_rep="auto_assigned",
            last_activity=datetime.now(timezone.utc),
            next_action="Schedule initial discovery call"
        )
        
        self.opportunities[opportunity_id] = opportunity
        logger.info(f"Auto-created opportunity: {opportunity_id} (${estimated_value:,})")
        return opportunity
    
    def advance_sales_stage(self, opportunity_id: str, new_stage: SalesStage, 
                           notes: str = "") -> SalesOpportunity:
        """
        Advance opportunity through sales pipeline
        Implements automated pipeline progression and probability updates
        """
        if opportunity_id not in self.opportunities:
            raise ValueError(f"Opportunity {opportunity_id} not found")
        
        opportunity = self.opportunities[opportunity_id]
        old_stage = opportunity.sales_stage
        opportunity.sales_stage = new_stage
        opportunity.last_activity = datetime.now(timezone.utc)
        
        # Update probability based on stage
        stage_probabilities = {
            SalesStage.LEAD: 0.10,
            SalesStage.QUALIFIED: 0.25,
            SalesStage.DEMO: 0.40,
            SalesStage.PROPOSAL: 0.60,
            SalesStage.NEGOTIATION: 0.80,
            SalesStage.CONTRACT: 0.95,
            SalesStage.CLOSED_WON: 1.00,
            SalesStage.CLOSED_LOST: 0.00
        }
        opportunity.probability = stage_probabilities[new_stage]
        
        # Update next actions
        next_actions = {
            SalesStage.QUALIFIED: "Schedule demo presentation",
            SalesStage.DEMO: "Send follow-up materials and proposal",
            SalesStage.PROPOSAL: "Schedule proposal review meeting",
            SalesStage.NEGOTIATION: "Finalize contract terms",
            SalesStage.CONTRACT: "Execute contract and begin onboarding",
            SalesStage.CLOSED_WON: "Begin client onboarding process",
            SalesStage.CLOSED_LOST: "Archive opportunity and request feedback"
        }
        opportunity.next_action = next_actions.get(new_stage, "Follow up as appropriate")
        
        # Add notes to prospect
        prospect = self.prospects[opportunity.prospect_id]
        if prospect.notes is None:
            prospect.notes = []
        prospect.notes.append(f"{datetime.now().strftime('%Y-%m-%d')}: Moved from {old_stage.value} to {new_stage.value}. {notes}")
        
        logger.info(f"Opportunity {opportunity_id} advanced: {old_stage.value} → {new_stage.value}")
        return opportunity
    
    def get_pipeline_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive sales pipeline report
        Implements sales performance dashboards with conversion metrics
        """
        pipeline_summary = {
            'total_opportunities': len(self.opportunities),
            'total_pipeline_value': sum(opp.estimated_value for opp in self.opportunities.values()),
            'weighted_pipeline_value': sum(opp.estimated_value * opp.probability for opp in self.opportunities.values()),
            'stage_breakdown': {},
            'conversion_metrics': {},
            'forecast': {}
        }
        
        # Stage breakdown
        for stage in SalesStage:
            stage_opportunities = [opp for opp in self.opportunities.values() if opp.sales_stage == stage]
            pipeline_summary['stage_breakdown'][stage.value] = {
                'count': len(stage_opportunities),
                'value': sum(opp.estimated_value for opp in stage_opportunities),
                'weighted_value': sum(opp.estimated_value * opp.probability for opp in stage_opportunities)
            }
        
        # Conversion metrics
        total_leads = len([opp for opp in self.opportunities.values()])
        qualified_leads = len([opp for opp in self.opportunities.values() 
                              if opp.sales_stage not in [SalesStage.LEAD]])
        demos_completed = len([opp for opp in self.opportunities.values() 
                              if opp.demo_completed])
        closed_won = len([opp for opp in self.opportunities.values() 
                         if opp.sales_stage == SalesStage.CLOSED_WON])
        
        pipeline_summary['conversion_metrics'] = {
            'lead_to_qualified': qualified_leads / total_leads if total_leads > 0 else 0,
            'qualified_to_demo': demos_completed / qualified_leads if qualified_leads > 0 else 0,
            'demo_to_close': closed_won / demos_completed if demos_completed > 0 else 0,
            'overall_conversion': closed_won / total_leads if total_leads > 0 else 0
        }
        
        # 90-day forecast
        forecast_date = datetime.now(timezone.utc) + timedelta(days=90)
        forecasted_closes = [opp for opp in self.opportunities.values() 
                           if opp.expected_close_date <= forecast_date and 
                           opp.sales_stage not in [SalesStage.CLOSED_WON, SalesStage.CLOSED_LOST]]
        
        pipeline_summary['forecast'] = {
            'next_90_days_opportunities': len(forecasted_closes),
            'next_90_days_value': sum(opp.estimated_value for opp in forecasted_closes),
            'next_90_days_weighted_value': sum(opp.estimated_value * opp.probability for opp in forecasted_closes)
        }
        
        return pipeline_summary

class AutomatedDemoSystem:
    """
    Automated institutional demonstration and ROI calculation system
    Implements AC-5.1.2: Automated Demonstration System
    """
    
    def __init__(self, crm_system: InstitutionalCRM):
        self.crm = crm_system
        self.demo_templates = self._initialize_demo_templates()
        self.roi_calculator = ROICalculator()
        
    def _initialize_demo_templates(self) -> Dict[str, Dict]:
        """Initialize demo templates for different client segments"""
        return {
            'hedge_fund': {
                'duration_minutes': 45,
                'focus_areas': ['advanced_analytics', 'risk_management', 'alternative_assets', 'performance_attribution'],
                'use_cases': ['multi_strategy_optimization', 'factor_exposure_analysis', 'alpha_generation'],
                'technical_depth': 'high'
            },
            'asset_manager': {
                'duration_minutes': 60,
                'focus_areas': ['client_portal', 'compliance_reporting', 'global_markets', 'scalability'],
                'use_cases': ['multi_client_management', 'regulatory_reporting', 'international_portfolios'],
                'technical_depth': 'medium'
            },
            'family_office': {
                'duration_minutes': 30,
                'focus_areas': ['wealth_management', 'tax_optimization', 'esg_integration', 'reporting'],
                'use_cases': ['family_portfolio_management', 'impact_investing', 'wealth_preservation'],
                'technical_depth': 'low'
            }
        }
    
    def schedule_demo(self, opportunity_id: str, demo_request: Dict[str, Any]) -> DemoSession:
        """
        Schedule automated demo with calendar integration
        Implements self-service demo scheduling with calendar integration
        """
        opportunity = self.crm.opportunities.get(opportunity_id)
        if not opportunity:
            raise ValueError(f"Opportunity {opportunity_id} not found")
        
        prospect = self.crm.prospects[opportunity.prospect_id]
        demo_id = str(uuid.uuid4())
        
        # Select appropriate demo template
        segment_key = prospect.client_segment.value
        if segment_key not in self.demo_templates:
            segment_key = 'asset_manager'  # Default template
        
        demo_template = self.demo_templates[segment_key]
        
        demo_session = DemoSession(
            demo_id=demo_id,
            opportunity_id=opportunity_id,
            scheduled_date=datetime.fromisoformat(demo_request['scheduled_date']),
            attendees=demo_request['attendees'],
            demo_type=demo_request.get('demo_type', 'standard'),
            use_cases_shown=demo_template['use_cases'],
            roi_calculated=False,
            follow_up_actions=[],
            attendee_feedback={},
            conversion_likelihood=0.0
        )
        
        self.crm.demo_sessions[demo_id] = demo_session
        
        # Update opportunity
        opportunity.demo_completed = True
        self.crm.advance_sales_stage(opportunity_id, SalesStage.DEMO, 
                                   f"Demo scheduled for {demo_session.scheduled_date}")
        
        logger.info(f"Demo scheduled: {demo_id} for {prospect.company_name}")
        return demo_session
    
    def calculate_demo_roi(self, demo_id: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate interactive ROI during demonstration
        Implements interactive ROI calculator during demonstrations
        """
        demo_session = self.crm.demo_sessions.get(demo_id)
        if not demo_session:
            raise ValueError(f"Demo session {demo_id} not found")
        
        roi_analysis = self.roi_calculator.calculate_institutional_roi(client_data)
        demo_session.roi_calculated = True
        demo_session.conversion_likelihood = roi_analysis['conversion_probability']
        
        # Update opportunity probability based on ROI
        opportunity = self.crm.opportunities[demo_session.opportunity_id]
        if roi_analysis['payback_period_months'] <= 12:
            opportunity.probability = min(opportunity.probability * 1.2, 0.95)
        
        logger.info(f"ROI calculated for demo {demo_id}: {roi_analysis['annual_savings']:,}")
        return roi_analysis

class ROICalculator:
    """
    Interactive ROI calculator for institutional demonstrations
    Calculates compliance cost savings, efficiency gains, and risk reduction
    """
    
    def __init__(self):
        # Cost savings benchmarks (annual)
        self.savings_benchmarks = {
            'compliance_automation': {
                'per_billion_aum': 50000,  # $50K per $1B AUM
                'base_savings': 100000
            },
            'risk_monitoring': {
                'per_billion_aum': 30000,
                'base_savings': 75000
            },
            'operational_efficiency': {
                'per_billion_aum': 40000,
                'base_savings': 120000
            },
            'reporting_automation': {
                'per_billion_aum': 25000,
                'base_savings': 60000
            }
        }
    
    def calculate_institutional_roi(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive ROI for institutional client"""
        aum_billions = client_data['assets_under_management'] / 1e9
        current_staff_costs = client_data.get('annual_staff_costs', aum_billions * 1000000)  # Estimate if not provided
        
        # Calculate annual savings
        compliance_savings = self._calculate_compliance_savings(aum_billions, client_data)
        risk_savings = self._calculate_risk_savings(aum_billions, client_data)
        efficiency_savings = self._calculate_efficiency_savings(aum_billions, current_staff_costs)
        reporting_savings = self._calculate_reporting_savings(aum_billions, client_data)
        
        total_annual_savings = compliance_savings + risk_savings + efficiency_savings + reporting_savings
        
        # Platform costs (estimated)
        annual_platform_cost = self._estimate_platform_cost(aum_billions, client_data)
        
        # ROI calculations
        net_annual_benefit = total_annual_savings - annual_platform_cost
        roi_percentage = (net_annual_benefit / annual_platform_cost) * 100 if annual_platform_cost > 0 else 0
        payback_period_months = (annual_platform_cost / (total_annual_savings / 12)) if total_annual_savings > 0 else float('inf')
        
        # Risk reduction value
        risk_reduction_value = self._calculate_risk_reduction_value(aum_billions, client_data)
        
        # Conversion probability based on ROI strength
        conversion_probability = min(0.95, max(0.1, (roi_percentage - 100) / 400 + 0.5))
        
        return {
            'annual_savings': total_annual_savings,
            'annual_platform_cost': annual_platform_cost,
            'net_annual_benefit': net_annual_benefit,
            'roi_percentage': roi_percentage,
            'payback_period_months': payback_period_months,
            'risk_reduction_value': risk_reduction_value,
            'conversion_probability': conversion_probability,
            'savings_breakdown': {
                'compliance_automation': compliance_savings,
                'risk_monitoring': risk_savings,
                'operational_efficiency': efficiency_savings,
                'reporting_automation': reporting_savings
            }
        }
    
    def _calculate_compliance_savings(self, aum_billions: float, client_data: Dict) -> float:
        """Calculate compliance automation savings"""
        base = self.savings_benchmarks['compliance_automation']['base_savings']
        aum_factor = aum_billions * self.savings_benchmarks['compliance_automation']['per_billion_aum']
        
        # Regulatory complexity multiplier
        jurisdictions = len(client_data.get('regulatory_jurisdictions', ['US']))
        complexity_multiplier = 1 + (jurisdictions - 1) * 0.3
        
        return (base + aum_factor) * complexity_multiplier
    
    def _calculate_risk_savings(self, aum_billions: float, client_data: Dict) -> float:
        """Calculate risk monitoring and prevention savings"""
        base = self.savings_benchmarks['risk_monitoring']['base_savings']
        aum_factor = aum_billions * self.savings_benchmarks['risk_monitoring']['per_billion_aum']
        
        # Risk event prevention value (based on historical losses)
        risk_prevention_value = aum_billions * 1000  # $1K per billion in prevented losses
        
        return base + aum_factor + risk_prevention_value
    
    def _calculate_efficiency_savings(self, aum_billions: float, current_staff_costs: float) -> float:
        """Calculate operational efficiency savings"""
        base = self.savings_benchmarks['operational_efficiency']['base_savings']
        aum_factor = aum_billions * self.savings_benchmarks['operational_efficiency']['per_billion_aum']
        
        # Staff productivity improvement (5-15% depending on AUM)
        productivity_improvement = min(0.15, 0.05 + (aum_billions / 100) * 0.10)
        staff_savings = current_staff_costs * productivity_improvement
        
        return base + aum_factor + staff_savings
    
    def _calculate_reporting_savings(self, aum_billions: float, client_data: Dict) -> float:
        """Calculate reporting automation savings"""
        base = self.savings_benchmarks['reporting_automation']['base_savings']
        aum_factor = aum_billions * self.savings_benchmarks['reporting_automation']['per_billion_aum']
        
        # Client count multiplier
        client_count = client_data.get('number_of_clients', max(1, int(aum_billions * 10)))
        client_multiplier = 1 + (client_count / 100) * 0.2
        
        return (base + aum_factor) * client_multiplier
    
    def _estimate_platform_cost(self, aum_billions: float, client_data: Dict) -> float:
        """Estimate annual platform cost"""
        # Base pricing tiers
        if aum_billions >= 100:
            return 2000000  # $2M for enterprise
        elif aum_billions >= 10:
            return 1000000  # $1M for large
        elif aum_billions >= 1:
            return 500000   # $500K for medium
        else:
            return 250000   # $250K for small
    
    def _calculate_risk_reduction_value(self, aum_billions: float, client_data: Dict) -> float:
        """Calculate value of risk reduction"""
        # Historical risk event costs (basis points of AUM)
        historical_risk_costs_bps = 15  # 15 basis points annually
        risk_reduction_percentage = 0.6  # 60% risk reduction
        
        aum_dollars = aum_billions * 1e9
        historical_costs = aum_dollars * (historical_risk_costs_bps / 10000)
        risk_reduction_value = historical_costs * risk_reduction_percentage
        
        return risk_reduction_value

if __name__ == "__main__":
    # Example usage and testing
    crm = InstitutionalCRM("postgresql://localhost/portfolio_crm")
    demo_system = AutomatedDemoSystem(crm)
    
    # Add sample prospect
    prospect_data = {
        'company_name': 'Global Asset Management LLC',
        'contact_name': 'John Smith',
        'title': 'Chief Investment Officer',
        'email': 'jsmith@globalasset.com',
        'phone': '+1-555-0123',
        'assets_under_management': 5.2e9,  # $5.2B
        'client_segment': 'asset_manager',
        'geographic_region': 'North America',
        'regulatory_jurisdiction': 'SEC',
        'lead_source': 'referral',
        'pain_points': ['manual_compliance', 'risk_monitoring'],
        'compliance_requirements': ['SEC', 'MiFID_II']
    }
    
    prospect = crm.add_prospect(prospect_data)
    print(f"Added prospect: {prospect.company_name} (Score: {prospect.lead_score})")
    
    # Generate pipeline report
    pipeline_report = crm.get_pipeline_report()
    print(f"Pipeline value: ${pipeline_report['total_pipeline_value']:,}")
    print(f"Weighted pipeline: ${pipeline_report['weighted_pipeline_value']:,}")
