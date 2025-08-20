"""
Compliance Validation Interface
Abstract base classes and interfaces for compliance validation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ...models.compliance import (
    ViolationDetails, 
    ViolationSeverity, 
    RuleType,
    ComplianceValidationResult
)


class ValidationResult(Enum):
    """Validation result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class ValidationContext:
    """Context information for validation"""
    portfolio_weights: Dict[str, float]
    portfolio_id: Optional[str] = None
    market_data: Optional[Dict[str, Any]] = None
    additional_context: Optional[Dict[str, Any]] = None


class ComplianceValidator(ABC):
    """Abstract base class for all compliance validators"""
    
    def __init__(self, rule_config: Dict[str, Any]):
        self.rule_config = rule_config
        self.rule_type = self._get_rule_type()
        
    @abstractmethod
    def _get_rule_type(self) -> RuleType:
        """Return the rule type this validator handles"""
        pass
    
    @abstractmethod
    def validate(self, context: ValidationContext) -> Tuple[ValidationResult, List[ViolationDetails]]:
        """
        Validate portfolio against compliance rules
        
        Args:
            context: Validation context containing portfolio data and metadata
            
        Returns:
            Tuple of (validation_result, list_of_violations)
        """
        pass
    
    @abstractmethod
    def get_rule_description(self) -> str:
        """Return human-readable description of the rule"""
        pass
    
    def _create_violation(
        self, 
        description: str, 
        severity: ViolationSeverity,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        affected_assets: Optional[List[str]] = None,
        recommended_action: str = "Review and adjust portfolio allocation"
    ) -> ViolationDetails:
        """Helper method to create violation details"""
        return ViolationDetails(
            rule_name=self.rule_config.get('rule_name', 'Unknown Rule'),
            rule_type=self.rule_type,
            violation_description=description,
            current_value=current_value,
            threshold_value=threshold_value,
            affected_assets=affected_assets or [],
            recommended_action=recommended_action
        )


class ComplianceEngine(ABC):
    """Abstract interface for compliance engines"""
    
    @abstractmethod
    async def validate_portfolio(
        self, 
        context: ValidationContext,
        rule_ids: Optional[List[int]] = None
    ) -> ComplianceValidationResult:
        """
        Validate portfolio against compliance rules
        
        Args:
            context: Validation context
            rule_ids: Optional list of specific rule IDs to validate against
            
        Returns:
            Comprehensive validation result
        """
        pass
    
    @abstractmethod
    async def get_active_rules(self, rule_type: Optional[RuleType] = None) -> List[Dict[str, Any]]:
        """Get all active compliance rules, optionally filtered by type"""
        pass
    
    @abstractmethod
    async def create_rule(self, rule_data: Dict[str, Any]) -> int:
        """Create a new compliance rule, returns rule ID"""
        pass
    
    @abstractmethod
    async def update_rule(self, rule_id: int, rule_data: Dict[str, Any]) -> bool:
        """Update existing compliance rule"""
        pass
    
    @abstractmethod
    async def deactivate_rule(self, rule_id: int) -> bool:
        """Deactivate a compliance rule"""
        pass
    
    @abstractmethod
    async def log_violation(
        self, 
        violation: ViolationDetails, 
        context: ValidationContext,
        rule_id: int
    ) -> int:
        """Log a compliance violation, returns violation ID"""
        pass


class RuleConfigManager(ABC):
    """Abstract interface for rule configuration management"""
    
    @abstractmethod
    async def load_rule_config(self, rule_id: int) -> Dict[str, Any]:
        """Load rule configuration by ID"""
        pass
    
    @abstractmethod
    async def validate_rule_config(self, rule_type: RuleType, config: Dict[str, Any]) -> bool:
        """Validate rule configuration format and parameters"""
        pass
    
    @abstractmethod
    async def get_rule_schema(self, rule_type: RuleType) -> Dict[str, Any]:
        """Get JSON schema for rule type configuration"""
        pass


class ViolationReporter(ABC):
    """Abstract interface for violation reporting"""
    
    @abstractmethod
    async def generate_compliance_report(
        self, 
        portfolio_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        pass
    
    @abstractmethod
    async def get_violation_history(
        self, 
        portfolio_id: Optional[str] = None,
        rule_id: Optional[int] = None,
        severity: Optional[ViolationSeverity] = None
    ) -> List[Dict[str, Any]]:
        """Get historical violations with optional filtering"""
        pass
    
    @abstractmethod
    async def get_compliance_metrics(self, portfolio_id: Optional[str] = None) -> Dict[str, float]:
        """Get compliance performance metrics"""
        pass


# ==================== Validator Registry ====================

class ValidatorRegistry:
    """Registry for compliance validators"""
    
    def __init__(self):
        self._validators: Dict[RuleType, type] = {}
    
    def register(self, rule_type: RuleType, validator_class: type):
        """Register a validator class for a rule type"""
        if not issubclass(validator_class, ComplianceValidator):
            raise ValueError(f"Validator class must inherit from ComplianceValidator")
        self._validators[rule_type] = validator_class
    
    def get_validator(self, rule_type: RuleType, rule_config: Dict[str, Any]) -> ComplianceValidator:
        """Get validator instance for rule type"""
        if rule_type not in self._validators:
            raise ValueError(f"No validator registered for rule type: {rule_type}")
        return self._validators[rule_type](rule_config)
    
    def get_supported_types(self) -> List[RuleType]:
        """Get list of supported rule types"""
        return list(self._validators.keys())


# Global validator registry instance
validator_registry = ValidatorRegistry()
