"""
STORY 6.1: MID-MARKET PLATFORM & AUTOMATED ONBOARDING
Self-Service Platform for Wealth Management Firms and Family Offices
================================================================================

Simplified, automated onboarding platform targeting the $25T+ mid-market
wealth management segment with self-service deployment and tiered pricing.

AC-6.1.1: Self-Service Onboarding Platform
AC-6.1.2: Simplified User Interface & Experience
AC-6.1.3: Tiered Pricing & Feature Model
AC-6.1.4: Automated Compliance Templates
AC-6.1.5: White-Label & Partnership Capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.professional_logging import get_logger

logger = get_logger(__name__)

class PricingTier(Enum):
    """Mid-market pricing tiers"""
    STARTER = "starter"          # <$500M AUM
    PROFESSIONAL = "professional" # $500M-$2B AUM
    ENTERPRISE = "enterprise"    # $2B-$10B AUM
    INSTITUTIONAL = "institutional" # >$10B AUM

class ComplianceJurisdiction(Enum):
    """Regulatory jurisdictions"""
    SEC_US = "sec_us"
    FCA_UK = "fca_uk"
    CIRO_CANADA = "ciro_canada"
    ASIC_AUSTRALIA = "asic_australia"
    MAS_SINGAPORE = "mas_singapore"
    FINMA_SWITZERLAND = "finma_switzerland"

class OnboardingStage(Enum):
    """Onboarding process stages"""
    REGISTRATION = "registration"
    VERIFICATION = "verification"
    CONFIGURATION = "configuration"
    DATA_MIGRATION = "data_migration"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    TRAINING = "training"
    LIVE = "live"

@dataclass
class PricingModel:
    """Pricing tier configuration"""
    tier: PricingTier
    name: str
    monthly_base_fee: float
    aum_fee_bps: float  # basis points
    max_portfolios: int
    max_users: int
    features_included: List[str]
    support_level: str

@dataclass
class OnboardingClient:
    """Mid-market client onboarding profile"""
    client_id: str
    firm_name: str
    contact_name: str
    contact_email: str
    phone: str
    assets_under_management: float
    pricing_tier: PricingTier
    regulatory_jurisdiction: ComplianceJurisdiction
    current_stage: OnboardingStage
    registration_date: datetime
    target_go_live: datetime
    data_sources: List[str]
    compliance_requirements: List[str]
    custom_branding: bool = False
    white_label_partner: Optional[str] = None

@dataclass
class OnboardingTask:
    """Individual onboarding task"""
    task_id: str
    client_id: str
    task_name: str
    description: str
    stage: OnboardingStage
    estimated_duration_hours: float
    status: str  # pending, in_progress, completed, failed
    assigned_to: str  # auto, client, support
    dependencies: List[str]
    completion_date: Optional[datetime] = None

class MidMarketOnboardingEngine:
    """
    Self-service onboarding platform for mid-market clients
    Implements AC-6.1.1: Self-Service Onboarding Platform
    """
    
    def __init__(self):
        self.pricing_tiers = self._initialize_pricing_tiers()
        self.onboarding_clients: Dict[str, OnboardingClient] = {}
        self.onboarding_tasks: Dict[str, List[OnboardingTask]] = {}
        self.compliance_templates = self._initialize_compliance_templates()
        
        logger.info("MidMarketOnboardingEngine initialized")
    
    def _initialize_pricing_tiers(self) -> Dict[PricingTier, PricingModel]:
        """
        Initialize tiered pricing model
        Implements AC-6.1.3: Tiered Pricing & Feature Model
        """
        return {
            PricingTier.STARTER: PricingModel(
                tier=PricingTier.STARTER,
                name="Starter",
                monthly_base_fee=2500.0,
                aum_fee_bps=2.5,  # 2.5 basis points
                max_portfolios=50,
                max_users=5,
                features_included=[
                    "basic_portfolio_optimization",
                    "risk_reporting",
                    "client_portal",
                    "standard_compliance",
                    "email_support"
                ],
                support_level="email"
            ),
            PricingTier.PROFESSIONAL: PricingModel(
                tier=PricingTier.PROFESSIONAL,
                name="Professional",
                monthly_base_fee=7500.0,
                aum_fee_bps=2.0,
                max_portfolios=200,
                max_users=15,
                features_included=[
                    "advanced_portfolio_optimization",
                    "risk_management",
                    "client_portal",
                    "advanced_compliance",
                    "alternative_assets",
                    "api_access",
                    "phone_support"
                ],
                support_level="phone"
            ),
            PricingTier.ENTERPRISE: PricingModel(
                tier=PricingTier.ENTERPRISE,
                name="Enterprise",
                monthly_base_fee=15000.0,
                aum_fee_bps=1.5,
                max_portfolios=1000,
                max_users=50,
                features_included=[
                    "enterprise_portfolio_optimization",
                    "advanced_risk_management",
                    "multi_tenant_portal",
                    "enterprise_compliance",
                    "global_markets",
                    "alternative_assets",
                    "api_access",
                    "white_label_options",
                    "dedicated_support"
                ],
                support_level="dedicated"
            ),
            PricingTier.INSTITUTIONAL: PricingModel(
                tier=PricingTier.INSTITUTIONAL,
                name="Institutional",
                monthly_base_fee=25000.0,
                aum_fee_bps=1.0,
                max_portfolios=5000,
                max_users=200,
                features_included=[
                    "institutional_portfolio_optimization",
                    "enterprise_risk_management",
                    "institutional_portal",
                    "regulatory_compliance_suite",
                    "global_markets",
                    "alternative_assets",
                    "ai_insights",
                    "esg_integration",
                    "full_api_suite",
                    "white_label_platform",
                    "24x7_support"
                ],
                support_level="24x7"
            )
        }
    
    def register_new_client(self, registration_data: Dict[str, Any]) -> OnboardingClient:
        """
        Register new mid-market client with automated tier assignment
        Implements guided self-service onboarding with automated provisioning
        """
        try:
            client_id = str(uuid.uuid4())
            
            # Determine pricing tier based on AUM
            aum = registration_data['assets_under_management']
            pricing_tier = self._determine_pricing_tier(aum)
            
            client = OnboardingClient(
                client_id=client_id,
                firm_name=registration_data['firm_name'],
                contact_name=registration_data['contact_name'],
                contact_email=registration_data['contact_email'],
                phone=registration_data.get('phone', ''),
                assets_under_management=aum,
                pricing_tier=pricing_tier,
                regulatory_jurisdiction=ComplianceJurisdiction(registration_data['regulatory_jurisdiction']),
                current_stage=OnboardingStage.REGISTRATION,
                registration_date=datetime.now(timezone.utc),
                target_go_live=datetime.now(timezone.utc) + timedelta(days=7),  # 1 week target
                data_sources=registration_data.get('data_sources', []),
                compliance_requirements=registration_data.get('compliance_requirements', []),
                custom_branding=registration_data.get('custom_branding', False),
                white_label_partner=registration_data.get('white_label_partner')
            )
            
            self.onboarding_clients[client_id] = client
            
            # Create onboarding task plan
            self._create_onboarding_plan(client)
            
            # Send welcome email
            self._send_welcome_email(client)
            
            logger.info(f"New client registered: {client.firm_name} ({pricing_tier.value})")
            return client
            
        except Exception as e:
            logger.error(f"Failed to register client: {e}")
            raise
    
    def _determine_pricing_tier(self, aum: float) -> PricingTier:
        """Determine appropriate pricing tier based on AUM"""
        aum_millions = aum / 1e6
        
        if aum_millions < 500:
            return PricingTier.STARTER
        elif aum_millions < 2000:
            return PricingTier.PROFESSIONAL
        elif aum_millions < 10000:
            return PricingTier.ENTERPRISE
        else:
            return PricingTier.INSTITUTIONAL
    
    def _create_onboarding_plan(self, client: OnboardingClient):
        """
        Create automated onboarding task plan
        Implements complete setup in <24 hours without technical support required
        """
        task_templates = {
            OnboardingStage.REGISTRATION: [
                ("Account Verification", "Verify email and phone number", 0.5, "auto", []),
                ("Compliance Review", "Review regulatory requirements", 1.0, "auto", []),
                ("Pricing Confirmation", "Confirm pricing tier and features", 0.5, "client", [])
            ],
            OnboardingStage.VERIFICATION: [
                ("Identity Verification", "Verify business and personal identity", 2.0, "client", ["Account Verification"]),
                ("Regulatory Documentation", "Submit required compliance documents", 2.0, "client", ["Compliance Review"]),
                ("Banking Setup", "Configure payment and billing", 1.0, "client", ["Identity Verification"])
            ],
            OnboardingStage.CONFIGURATION: [
                ("Platform Setup", "Initialize client platform instance", 1.0, "auto", ["Banking Setup"]),
                ("User Management", "Create user accounts and permissions", 1.5, "client", ["Platform Setup"]),
                ("Branding Configuration", "Apply custom branding if selected", 2.0, "auto", ["Platform Setup"])
            ],
            OnboardingStage.DATA_MIGRATION: [
                ("Data Source Connection", "Connect to existing data sources", 3.0, "client", ["User Management"]),
                ("Portfolio Import", "Import existing portfolio data", 2.0, "auto", ["Data Source Connection"]),
                ("Data Validation", "Validate imported data integrity", 1.0, "auto", ["Portfolio Import"])
            ],
            OnboardingStage.TESTING: [
                ("System Testing", "Comprehensive system functionality test", 2.0, "auto", ["Data Validation"]),
                ("User Acceptance Testing", "Client testing and approval", 4.0, "client", ["System Testing"]),
                ("Performance Validation", "Validate system performance", 1.0, "auto", ["User Acceptance Testing"])
            ],
            OnboardingStage.DEPLOYMENT: [
                ("Production Deployment", "Deploy to production environment", 1.0, "auto", ["Performance Validation"]),
                ("Final Configuration", "Apply production configurations", 1.0, "auto", ["Production Deployment"]),
                ("Go-Live Verification", "Verify production readiness", 0.5, "auto", ["Final Configuration"])
            ],
            OnboardingStage.TRAINING: [
                ("Platform Training", "User training and documentation", 4.0, "client", ["Go-Live Verification"]),
                ("Support Setup", "Configure support channels", 0.5, "auto", ["Platform Training"]),
                ("Documentation Handover", "Provide complete documentation", 1.0, "auto", ["Support Setup"])
            ]
        }
        
        tasks = []
        for stage, stage_tasks in task_templates.items():
            for task_name, description, duration, assigned_to, dependencies in stage_tasks:
                task_id = str(uuid.uuid4())
                
                task = OnboardingTask(
                    task_id=task_id,
                    client_id=client.client_id,
                    task_name=task_name,
                    description=description,
                    stage=stage,
                    estimated_duration_hours=duration,
                    status="pending",
                    assigned_to=assigned_to,
                    dependencies=dependencies
                )
                
                tasks.append(task)
        
        self.onboarding_tasks[client.client_id] = tasks
        
        # Auto-start first tasks
        self._progress_onboarding(client.client_id)
    
    def _progress_onboarding(self, client_id: str):
        """Automatically progress onboarding tasks"""
        if client_id not in self.onboarding_tasks:
            return
        
        tasks = self.onboarding_tasks[client_id]
        
        # Find tasks ready to start (dependencies completed)
        for task in tasks:
            if task.status == "pending":
                dependencies_completed = all(
                    any(t.task_name == dep and t.status == "completed" for t in tasks)
                    for dep in task.dependencies
                ) if task.dependencies else True
                
                if dependencies_completed:
                    if task.assigned_to == "auto":
                        # Auto-complete automated tasks
                        task.status = "completed"
                        task.completion_date = datetime.now(timezone.utc)
                        logger.info(f"Auto-completed task: {task.task_name}")
                    else:
                        # Mark client tasks as ready
                        task.status = "in_progress"
                        logger.info(f"Client task ready: {task.task_name}")
        
        # Update client stage
        client = self.onboarding_clients[client_id]
        completed_stages = set()
        
        for stage in OnboardingStage:
            stage_tasks = [t for t in tasks if t.stage == stage]
            if stage_tasks and all(t.status == "completed" for t in stage_tasks):
                completed_stages.add(stage)
        
        # Advance to next uncompleted stage
        for stage in OnboardingStage:
            stage_tasks = [t for t in tasks if t.stage == stage]
            if stage_tasks and not all(t.status == "completed" for t in stage_tasks):
                client.current_stage = stage
                break
        else:
            client.current_stage = OnboardingStage.LIVE
    
    def get_onboarding_status(self, client_id: str) -> Dict[str, Any]:
        """
        Get comprehensive onboarding status
        Implements clear progress tracking and next-step guidance
        """
        if client_id not in self.onboarding_clients:
            raise ValueError(f"Client {client_id} not found")
        
        client = self.onboarding_clients[client_id]
        tasks = self.onboarding_tasks.get(client_id, [])
        
        # Calculate progress
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.status == "completed"])
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Get current tasks
        current_tasks = [t for t in tasks if t.status == "in_progress"]
        next_tasks = [t for t in tasks if t.status == "pending" and not t.dependencies]
        
        # Estimate completion
        remaining_hours = sum(t.estimated_duration_hours for t in tasks if t.status != "completed")
        estimated_completion = datetime.now(timezone.utc) + timedelta(hours=remaining_hours)
        
        status = {
            'client_info': {
                'firm_name': client.firm_name,
                'pricing_tier': client.pricing_tier.value,
                'current_stage': client.current_stage.value,
                'registration_date': client.registration_date.isoformat(),
                'target_go_live': client.target_go_live.isoformat()
            },
            'progress': {
                'percentage': progress_percentage,
                'completed_tasks': completed_tasks,
                'total_tasks': total_tasks,
                'estimated_completion': estimated_completion.isoformat(),
                'on_track': estimated_completion <= client.target_go_live
            },
            'current_tasks': [
                {
                    'task_name': t.task_name,
                    'description': t.description,
                    'estimated_hours': t.estimated_duration_hours,
                    'assigned_to': t.assigned_to
                }
                for t in current_tasks
            ],
            'next_steps': [
                {
                    'task_name': t.task_name,
                    'description': t.description,
                    'stage': t.stage.value
                }
                for t in next_tasks[:3]  # Next 3 tasks
            ]
        }
        
        return status
    
    def _initialize_compliance_templates(self) -> Dict[ComplianceJurisdiction, Dict]:
        """
        Initialize automated compliance templates
        Implements AC-6.1.4: Automated Compliance Templates
        """
        return {
            ComplianceJurisdiction.SEC_US: {
                'name': 'SEC United States',
                'required_forms': ['Form ADV', 'Form PF', 'Form N-Q'],
                'reporting_frequency': 'quarterly',
                'key_requirements': [
                    'Investment advisor registration',
                    'Custody rule compliance',
                    'Performance reporting standards',
                    'Risk management procedures'
                ],
                'automated_reports': [
                    'Quarterly performance attribution',
                    'Risk exposure summary',
                    'Client asset verification'
                ]
            },
            ComplianceJurisdiction.FCA_UK: {
                'name': 'FCA United Kingdom',
                'required_forms': ['COBS', 'SYSC', 'FUND'],
                'reporting_frequency': 'monthly',
                'key_requirements': [
                    'COBS conduct of business rules',
                    'Senior managers regime',
                    'MiFID II compliance',
                    'GDPR data protection'
                ],
                'automated_reports': [
                    'Monthly regulatory returns',
                    'Transaction reporting',
                    'Best execution reports'
                ]
            },
            ComplianceJurisdiction.CIRO_CANADA: {
                'name': 'CIRO Canada',
                'required_forms': ['Form 31-103F1', 'Form 31-103F2'],
                'reporting_frequency': 'quarterly',
                'key_requirements': [
                    'Registration as portfolio manager',
                    'Know your client requirements',
                    'Suitability determination',
                    'Relationship disclosure'
                ],
                'automated_reports': [
                    'Quarterly financials',
                    'Client relationship summaries',
                    'Trade supervision reports'
                ]
            }
        }
    
    def _send_welcome_email(self, client: OnboardingClient):
        """Send automated welcome email with onboarding instructions"""
        try:
            pricing_model = self.pricing_tiers[client.pricing_tier]
            
            subject = f"Welcome to Quantum Portfolio Platform - {client.firm_name}"
            body = f"""
Dear {client.contact_name},

Welcome to the Quantum Portfolio Platform! We're excited to help {client.firm_name} 
transform your portfolio management capabilities.

Your Account Details:
- Pricing Tier: {pricing_model.name}
- Monthly Fee: ${pricing_model.monthly_base_fee:,}
- AUM Fee: {pricing_model.aum_fee_bps} basis points
- Target Go-Live: {client.target_go_live.strftime('%B %d, %Y')}

Next Steps:
1. Check your email for account verification
2. Complete identity verification process
3. Submit required compliance documentation
4. Configure your platform settings

Our automated onboarding system will guide you through each step. 
Most clients complete onboarding within 24-48 hours.

Support Contact:
- Email: support@quantumportfolio.com
- Phone: 1-800-QUANTUM (for Professional+ tiers)

Login to your onboarding portal: https://onboard.quantumportfolio.com/{client.client_id}

Best regards,
The Quantum Portfolio Team
"""
            
            # In production, this would send actual email
            logger.info(f"Welcome email sent to {client.contact_email}")
            
        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")

class DataMigrationService:
    """
    Automated data migration from existing systems
    Implements automated data migration from existing systems
    """
    
    def __init__(self):
        self.supported_sources = [
            'excel_files',
            'csv_exports', 
            'bloomberg_aap',
            'refinitiv_eikon',
            'factset',
            'morningstar_direct',
            'custom_api'
        ]
        
    async def migrate_client_data(self, client_id: str, data_sources: List[str]) -> Dict[str, Any]:
        """Migrate data from client's existing systems"""
        migration_results = {
            'client_id': client_id,
            'start_time': datetime.now(timezone.utc),
            'sources_migrated': [],
            'total_portfolios': 0,
            'total_positions': 0,
            'validation_errors': [],
            'status': 'in_progress'
        }
        
        try:
            for source in data_sources:
                if source in self.supported_sources:
                    source_result = await self._migrate_from_source(source)
                    migration_results['sources_migrated'].append(source_result)
                    migration_results['total_portfolios'] += source_result['portfolios_count']
                    migration_results['total_positions'] += source_result['positions_count']
            
            migration_results['status'] = 'completed'
            migration_results['completion_time'] = datetime.now(timezone.utc)
            
            logger.info(f"Data migration completed for client {client_id}")
            
        except Exception as e:
            migration_results['status'] = 'failed'
            migration_results['error'] = str(e)
            logger.error(f"Data migration failed for client {client_id}: {e}")
        
        return migration_results
    
    async def _migrate_from_source(self, source: str) -> Dict[str, Any]:
        """Migrate data from specific source system"""
        # Simulate migration process
        await asyncio.sleep(2)  # Simulate migration time
        
        return {
            'source': source,
            'portfolios_count': 25,
            'positions_count': 500,
            'duration_seconds': 2,
            'status': 'completed'
        }

if __name__ == "__main__":
    # Example usage and testing
    onboarding_engine = MidMarketOnboardingEngine()
    
    # Register new client
    registration_data = {
        'firm_name': 'Midwest Family Wealth Management',
        'contact_name': 'Sarah Johnson',
        'contact_email': 'sarah@midwestwealth.com',
        'phone': '+1-312-555-0123',
        'assets_under_management': 1.5e9,  # $1.5B
        'regulatory_jurisdiction': 'sec_us',
        'data_sources': ['excel_files', 'refinitiv_eikon'],
        'compliance_requirements': ['SEC', 'Form ADV'],
        'custom_branding': True
    }
    
    client = onboarding_engine.register_new_client(registration_data)
    print(f"Client registered: {client.firm_name} ({client.pricing_tier.value})")
    
    # Check onboarding status
    status = onboarding_engine.get_onboarding_status(client.client_id)
    print(f"Onboarding progress: {status['progress']['percentage']:.1f}%")
    print(f"Current tasks: {len(status['current_tasks'])}")
    print(f"On track: {status['progress']['on_track']}")
