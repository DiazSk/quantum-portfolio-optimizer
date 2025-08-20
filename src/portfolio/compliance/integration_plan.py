"""
Compliance Engine Integration Points
Planning document for integrating compliance engine with existing services
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class IntegrationPoint:
    """Represents an integration point with existing services"""
    service_name: str
    integration_type: str  # 'modify', 'extend', 'new_endpoint'
    description: str
    dependencies: List[str]
    risk_level: str  # 'low', 'medium', 'high'


class ComplianceIntegrationPlan:
    """Integration plan for compliance engine with existing services"""
    
    INTEGRATION_POINTS = [
        IntegrationPoint(
            service_name="Portfolio Optimization API",
            integration_type="extend",
            description="Add pre-optimization compliance checking to existing /api/optimize endpoint",
            dependencies=["ComplianceEngine", "PositionLimitValidator"],
            risk_level="medium"
        ),
        IntegrationPoint(
            service_name="Database Schema",
            integration_type="extend", 
            description="Add compliance_rules and compliance_violations tables to existing PostgreSQL schema",
            dependencies=["Migration Script"],
            risk_level="low"
        ),
        IntegrationPoint(
            service_name="FastAPI Router",
            integration_type="new_endpoint",
            description="Add new /api/compliance route for compliance management",
            dependencies=["ComplianceEngine", "ViolationReporter"],
            risk_level="low"
        ),
        IntegrationPoint(
            service_name="Risk Service",
            integration_type="modify",
            description="Leverage existing risk calculation infrastructure for compliance metrics",
            dependencies=["RiskService", "ComplianceEngine"],
            risk_level="medium"
        ),
        IntegrationPoint(
            service_name="Model Manager",
            integration_type="extend",
            description="Add compliance data models to existing model management",
            dependencies=["ComplianceModels"],
            risk_level="low"
        )
    ]
    
    API_MODIFICATIONS = {
        "/api/optimize": {
            "new_request_fields": [
                "skip_compliance_check: Optional[bool] = False",
                "compliance_rule_sets: Optional[List[int]] = None"
            ],
            "new_response_fields": [
                "compliance_status: str",
                "compliance_violations: Optional[List[ViolationDetails]] = None",
                "compliance_score: Optional[float] = None"
            ],
            "backward_compatibility": "100% - new fields are optional",
            "performance_impact": "<50ms additional latency"
        }
    }
    
    DATABASE_CHANGES = {
        "new_tables": ["compliance_rules", "compliance_violations"],
        "modified_tables": [],
        "new_indexes": [
            "idx_compliance_rules_type_active",
            "idx_compliance_violations_portfolio",
            "idx_compliance_violations_severity"
        ],
        "breaking_changes": False
    }
    
    SERVICE_DEPENDENCIES = {
        "ComplianceEngine": {
            "depends_on": ["Database", "RuleConfigManager", "ValidatorRegistry"],
            "consumed_by": ["OptimizationAPI", "ComplianceAPI", "Dashboard"]
        },
        "PositionLimitValidator": {
            "depends_on": ["MarketData", "PortfolioContext"],
            "consumed_by": ["ComplianceEngine"]
        },
        "MandateValidator": {
            "depends_on": ["ESGData", "CreditRatings", "LiquidityScores"],
            "consumed_by": ["ComplianceEngine"]
        },
        "ViolationReporter": {
            "depends_on": ["Database", "ComplianceViolations"],
            "consumed_by": ["Dashboard", "ComplianceAPI", "AlertSystem"]
        }
    }
    
    CONFIGURATION_REQUIREMENTS = {
        "environment_variables": [
            "COMPLIANCE_ENGINE_ENABLED=true",
            "COMPLIANCE_CHECK_TIMEOUT_MS=50",
            "COMPLIANCE_VIOLATION_ALERT_THRESHOLD=high"
        ],
        "feature_flags": [
            "enable_strict_compliance_mode",
            "enable_compliance_caching",
            "enable_compliance_metrics_collection"
        ]
    }


def get_integration_sequence() -> List[str]:
    """Return recommended sequence for implementing integration points"""
    return [
        "1. Database Migration - Add compliance tables",
        "2. Data Models - Create compliance models and interfaces", 
        "3. Core Validators - Implement position limit and mandate validators",
        "4. Compliance Engine - Implement main compliance orchestration service",
        "5. API Integration - Extend /api/optimize with compliance checking",
        "6. New API Endpoints - Add /api/compliance management endpoints",
        "7. Violation Reporting - Implement reporting and dashboard integration",
        "8. Testing Integration - Add compliance tests to existing test suite",
        "9. Performance Optimization - Add caching and performance monitoring"
    ]


def get_rollback_plan() -> Dict[str, str]:
    """Return rollback procedures for each integration point"""
    return {
        "database_migration": "Run rollback script to drop compliance tables",
        "api_modifications": "Deploy previous version - new fields are optional",
        "compliance_service": "Disable compliance checking via feature flag",
        "data_corruption": "Restore from backup - compliance data is separate",
        "performance_issues": "Bypass compliance checking via environment variable"
    }


def get_testing_strategy() -> Dict[str, List[str]]:
    """Return comprehensive testing strategy for integration"""
    return {
        "unit_tests": [
            "Test all compliance validators independently",
            "Test rule configuration validation",
            "Test violation detection logic",
            "Test API request/response models"
        ],
        "integration_tests": [
            "Test compliance engine with real portfolio data",
            "Test API endpoint with compliance checking enabled/disabled",
            "Test database operations and migrations",
            "Test performance under load"
        ],
        "regression_tests": [
            "Verify existing /api/optimize functionality unchanged",
            "Verify existing portfolio optimization algorithms unchanged", 
            "Verify existing database queries and performance",
            "Verify existing authentication and authorization"
        ],
        "performance_tests": [
            "Measure compliance checking latency (<50ms target)",
            "Test concurrent compliance validations (1000+ requests)",
            "Measure database query performance with new indexes",
            "Test memory usage and resource consumption"
        ]
    }
