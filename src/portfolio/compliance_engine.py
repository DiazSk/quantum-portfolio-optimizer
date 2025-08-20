"""
Compliance Engine Implementation
Main orchestrator for portfolio compliance validation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from ..database.production_db import get_db_manager
from ..models.compliance import ComplianceRule, ComplianceViolation
from ..models.compliance import (
    ComplianceValidationRequest,
    ComplianceValidationResult,
    ViolationDetails,
    ViolationSeverity,
    RuleType,
    ComplianceReport
)
from .compliance import (
    ComplianceEngine,
    ValidationContext,
    ValidationResult,
    validator_registry
)
from .compliance.position_validators import PositionLimitValidator
from .compliance.mandate_validators import MandateValidator

logger = logging.getLogger(__name__)


class ProductionComplianceEngine(ComplianceEngine):
    """Production implementation of compliance engine"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session or get_db_manager().SessionLocal()
        self._initialize_validators()
        
    def _initialize_validators(self):
        """Initialize and register all validators"""
        # Register validators with the global registry
        validator_registry.register(RuleType.POSITION_LIMIT, PositionLimitValidator)
        validator_registry.register(RuleType.MANDATE, MandateValidator)
        
        logger.info("Compliance engine initialized with validators: %s", 
                   [rule_type.value for rule_type in validator_registry.get_supported_types()])
    
    async def validate_portfolio(
        self, 
        context: ValidationContext,
        rule_ids: Optional[List[int]] = None
    ) -> ComplianceValidationResult:
        """
        Validate portfolio against compliance rules
        
        Args:
            context: Validation context with portfolio weights and metadata
            rule_ids: Optional list of specific rule IDs to validate against
            
        Returns:
            Comprehensive validation result
        """
        start_time = datetime.utcnow()
        
        try:
            # Get active rules to validate against
            rules = await self._get_validation_rules(rule_ids)
            
            if not rules:
                logger.warning("No compliance rules found for validation")
                return ComplianceValidationResult(
                    is_compliant=True,
                    violations=[],
                    validation_timestamp=datetime.utcnow(),
                    rule_set_used=[],
                    performance_metrics={'validation_time_ms': 0}
                )
            
            # Run validation for each rule
            all_violations = []
            failed_rules = []
            
            for rule in rules:
                try:
                    validator = validator_registry.get_validator(
                        RuleType(rule['rule_type']), 
                        rule['rule_config']
                    )
                    
                    result, violations = validator.validate(context)
                    
                    if violations:
                        all_violations.extend(violations)
                        
                        # Log violations to database
                        for violation in violations:
                            await self.log_violation(violation, context, rule['id'])
                    
                    if result == ValidationResult.FAIL:
                        failed_rules.append(rule['id'])
                        
                except Exception as e:
                    logger.error(f"Error validating rule {rule['id']}: {str(e)}")
                    # Create error violation
                    error_violation = ViolationDetails(
                        rule_name=rule.get('rule_name', f"Rule {rule['id']}"),
                        rule_type=RuleType(rule['rule_type']),
                        violation_description=f"[CRITICAL] Rule validation failed: {str(e)}",
                        recommended_action="Review rule configuration and portfolio data"
                    )
                    all_violations.append(error_violation)
                    failed_rules.append(rule['id'])
            
            # Determine overall compliance status
            critical_violations = [v for v in all_violations if '[CRITICAL]' in v.violation_description]
            high_violations = [v for v in all_violations if '[HIGH]' in v.violation_description]
            
            is_compliant = len(critical_violations) == 0 and len(high_violations) == 0
            
            # Calculate performance metrics
            end_time = datetime.utcnow()
            validation_time_ms = (end_time - start_time).total_seconds() * 1000
            
            performance_metrics = {
                'validation_time_ms': validation_time_ms,
                'rules_processed': len(rules),
                'violations_detected': len(all_violations),
                'failed_rules': len(failed_rules)
            }
            
            logger.info(f"Portfolio validation completed: compliant={is_compliant}, "
                       f"violations={len(all_violations)}, time={validation_time_ms:.2f}ms")
            
            return ComplianceValidationResult(
                is_compliant=is_compliant,
                violations=all_violations,
                validation_timestamp=end_time,
                rule_set_used=[rule['id'] for rule in rules],
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Critical error in portfolio validation: {str(e)}")
            error_violation = ViolationDetails(
                rule_name="System Error",
                rule_type=RuleType.POSITION_LIMIT,  # Default type for system errors
                violation_description=f"[CRITICAL] Compliance validation system error: {str(e)}",
                recommended_action="Contact system administrator"
            )
            
            return ComplianceValidationResult(
                is_compliant=False,
                violations=[error_violation],
                validation_timestamp=datetime.utcnow(),
                rule_set_used=[],
                performance_metrics={'validation_time_ms': 0, 'system_error': True}
            )
    
    async def _get_validation_rules(self, rule_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Get rules for validation from database"""
        try:
            query = self.db_session.query(ComplianceRule).filter(ComplianceRule.is_active == True)
            
            if rule_ids:
                query = query.filter(ComplianceRule.id.in_(rule_ids))
            
            rules = query.all()
            
            return [
                {
                    'id': rule.id,
                    'rule_type': rule.rule_type,
                    'rule_name': rule.rule_name,
                    'rule_config': dict(rule.rule_config, rule_name=rule.rule_name)
                }
                for rule in rules
            ]
            
        except Exception as e:
            logger.error(f"Error fetching validation rules: {str(e)}")
            return []
    
    async def get_active_rules(self, rule_type: Optional[RuleType] = None) -> List[Dict[str, Any]]:
        """Get all active compliance rules, optionally filtered by type"""
        try:
            query = self.db_session.query(ComplianceRule).filter(ComplianceRule.is_active == True)
            
            if rule_type:
                query = query.filter(ComplianceRule.rule_type == rule_type.value)
            
            rules = query.all()
            
            return [
                {
                    'id': rule.id,
                    'rule_type': rule.rule_type,
                    'rule_name': rule.rule_name,
                    'rule_config': rule.rule_config,
                    'created_at': rule.created_at.isoformat(),
                    'updated_at': rule.updated_at.isoformat()
                }
                for rule in rules
            ]
            
        except Exception as e:
            logger.error(f"Error fetching active rules: {str(e)}")
            return []
    
    async def create_rule(self, rule_data: Dict[str, Any]) -> int:
        """Create a new compliance rule, returns rule ID"""
        try:
            new_rule = ComplianceRule(
                rule_type=rule_data['rule_type'],
                rule_name=rule_data['rule_name'],
                rule_config=rule_data['rule_config'],
                is_active=rule_data.get('is_active', True)
            )
            
            self.db_session.add(new_rule)
            self.db_session.commit()
            
            logger.info(f"Created compliance rule: {new_rule.id} - {new_rule.rule_name}")
            return new_rule.id
            
        except Exception as e:
            logger.error(f"Error creating compliance rule: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def update_rule(self, rule_id: int, rule_data: Dict[str, Any]) -> bool:
        """Update existing compliance rule"""
        try:
            rule = self.db_session.query(ComplianceRule).filter(ComplianceRule.id == rule_id).first()
            
            if not rule:
                logger.warning(f"Compliance rule {rule_id} not found for update")
                return False
            
            # Update fields
            if 'rule_name' in rule_data:
                rule.rule_name = rule_data['rule_name']
            if 'rule_config' in rule_data:
                rule.rule_config = rule_data['rule_config']
            if 'is_active' in rule_data:
                rule.is_active = rule_data['is_active']
            
            rule.updated_at = datetime.utcnow()
            
            self.db_session.commit()
            
            logger.info(f"Updated compliance rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating compliance rule {rule_id}: {str(e)}")
            self.db_session.rollback()
            return False
    
    async def deactivate_rule(self, rule_id: int) -> bool:
        """Deactivate a compliance rule"""
        try:
            rule = self.db_session.query(ComplianceRule).filter(ComplianceRule.id == rule_id).first()
            
            if not rule:
                logger.warning(f"Compliance rule {rule_id} not found for deactivation")
                return False
            
            rule.is_active = False
            rule.updated_at = datetime.utcnow()
            
            self.db_session.commit()
            
            logger.info(f"Deactivated compliance rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating compliance rule {rule_id}: {str(e)}")
            self.db_session.rollback()
            return False
    
    async def log_violation(
        self, 
        violation: ViolationDetails, 
        context: ValidationContext,
        rule_id: int
    ) -> int:
        """Log a compliance violation, returns violation ID"""
        try:
            # Extract severity from violation description
            severity = 'medium'  # default
            if '[CRITICAL]' in violation.violation_description:
                severity = 'critical'
            elif '[HIGH]' in violation.violation_description:
                severity = 'high'
            elif '[LOW]' in violation.violation_description:
                severity = 'low'
            
            new_violation = ComplianceViolation(
                portfolio_id=context.portfolio_id,
                rule_id=rule_id,
                violation_details={
                    'rule_name': violation.rule_name,
                    'rule_type': violation.rule_type.value,
                    'description': violation.violation_description,
                    'current_value': violation.current_value,
                    'threshold_value': violation.threshold_value,
                    'affected_assets': violation.affected_assets,
                    'recommended_action': violation.recommended_action
                },
                severity=severity
            )
            
            self.db_session.add(new_violation)
            self.db_session.commit()
            
            logger.debug(f"Logged compliance violation: {new_violation.id}")
            return new_violation.id
            
        except Exception as e:
            logger.error(f"Error logging compliance violation: {str(e)}")
            self.db_session.rollback()
            raise


class ComplianceAPIIntegration:
    """Integration layer for compliance engine with API endpoints"""
    
    def __init__(self, compliance_engine: Optional[ComplianceEngine] = None):
        self.compliance_engine = compliance_engine or ProductionComplianceEngine()
    
    async def validate_portfolio_request(
        self, 
        request: ComplianceValidationRequest
    ) -> ComplianceValidationResult:
        """
        Process portfolio validation request from API
        
        Args:
            request: Compliance validation request
            
        Returns:
            Validation result for API response
        """
        # Create validation context
        context = ValidationContext(
            portfolio_weights=request.portfolio_weights,
            portfolio_id=request.portfolio_id
        )
        
        # Run validation
        result = await self.compliance_engine.validate_portfolio(
            context=context,
            rule_ids=request.rule_set_ids
        )
        
        return result
    
    async def get_portfolio_compliance_summary(
        self, 
        portfolio_weights: Dict[str, float],
        portfolio_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get compliance summary for portfolio"""
        request = ComplianceValidationRequest(
            portfolio_weights=portfolio_weights,
            portfolio_id=portfolio_id
        )
        
        result = await self.validate_portfolio_request(request)
        
        # Create summary
        violation_counts = {}
        for violation in result.violations:
            severity = 'medium'  # default
            if '[CRITICAL]' in violation.violation_description:
                severity = 'critical'
            elif '[HIGH]' in violation.violation_description:
                severity = 'high'
            elif '[LOW]' in violation.violation_description:
                severity = 'low'
                
            violation_counts[severity] = violation_counts.get(severity, 0) + 1
        
        return {
            'is_compliant': result.is_compliant,
            'total_violations': len(result.violations),
            'violation_counts': violation_counts,
            'compliance_score': self._calculate_compliance_score(result),
            'validation_time_ms': result.performance_metrics.get('validation_time_ms', 0),
            'rules_evaluated': len(result.rule_set_used)
        }
    
    def _calculate_compliance_score(self, result: ComplianceValidationResult) -> float:
        """Calculate compliance score (0-1) based on violations"""
        if not result.violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.7, 'critical': 1.0}
        total_penalty = 0.0
        
        for violation in result.violations:
            severity = 'medium'  # default
            if '[CRITICAL]' in violation.violation_description:
                severity = 'critical'
            elif '[HIGH]' in violation.violation_description:
                severity = 'high'
            elif '[LOW]' in violation.violation_description:
                severity = 'low'
                
            total_penalty += severity_weights.get(severity, 0.3)
        
        # Calculate score (max penalty of 10 violations = 0 score)
        score = max(0.0, 1.0 - (total_penalty / 10.0))
        return round(score, 3)


# Singleton instance for application use
compliance_engine = ProductionComplianceEngine()
compliance_api = ComplianceAPIIntegration(compliance_engine)
