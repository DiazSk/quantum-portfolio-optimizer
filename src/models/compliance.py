"""
Compliance Engine Data Models
Data models for regulatory compliance checking and violation tracking
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

from ..database.production_db import Base


class RuleType(str, Enum):
    """Types of compliance rules"""
    POSITION_LIMIT = "position_limit"
    MANDATE = "mandate" 
    CONCENTRATION = "concentration"


class ViolationSeverity(str, Enum):
    """Compliance violation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceRule(Base):
    """Compliance rule definitions table"""
    __tablename__ = 'compliance_rules'
    
    id = Column(Integer, primary_key=True)
    rule_type = Column(String(50), nullable=False)  # RuleType enum
    rule_name = Column(String(100), nullable=False)
    rule_config = Column(JSONB, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    violations = relationship("ComplianceViolation", back_populates="rule")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_compliance_rules_type_active', 'rule_type', 'is_active'),
        Index('idx_compliance_rules_name', 'rule_name'),
    )


class ComplianceViolation(Base):
    """Compliance violations tracking table"""
    __tablename__ = 'compliance_violations'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(UUID(as_uuid=True), nullable=True)  # Can be None for pre-optimization checks
    rule_id = Column(Integer, ForeignKey('compliance_rules.id'), nullable=False)
    violation_details = Column(JSONB, nullable=False)
    severity = Column(String(20), nullable=False)  # ViolationSeverity enum
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    is_resolved = Column(Boolean, default=False)
    
    # Relationships
    rule = relationship("ComplianceRule", back_populates="violations")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_compliance_violations_portfolio', 'portfolio_id'),
        Index('idx_compliance_violations_rule', 'rule_id'),
        Index('idx_compliance_violations_severity', 'severity'),
        Index('idx_compliance_violations_detected', 'detected_at'),
    )


# ==================== Pydantic Models for API ====================

class RuleConfigModel(BaseModel):
    """Base rule configuration model"""
    rule_type: RuleType
    parameters: Dict[str, Any]


class PositionLimitConfig(RuleConfigModel):
    """Position limit rule configuration"""
    max_single_position: Optional[float] = Field(None, ge=0, le=1, description="Maximum single asset weight (0-1)")
    max_sector_concentration: Optional[float] = Field(None, ge=0, le=1, description="Maximum sector concentration (0-1)")
    max_geographic_exposure: Optional[Dict[str, float]] = Field(None, description="Maximum geographic exposure by region")
    excluded_assets: Optional[List[str]] = Field(None, description="List of excluded asset symbols")


class MandateConfig(RuleConfigModel):
    """Investment mandate rule configuration"""
    esg_requirements: Optional[Dict[str, Any]] = Field(None, description="ESG scoring requirements")
    min_credit_rating: Optional[str] = Field(None, description="Minimum credit rating (e.g., 'BBB')")
    min_liquidity_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum liquidity score (0-1)")
    required_sectors: Optional[List[str]] = Field(None, description="Required sector allocations")
    forbidden_sectors: Optional[List[str]] = Field(None, description="Forbidden sectors")


class ComplianceRuleCreate(BaseModel):
    """Model for creating compliance rules"""
    rule_type: RuleType
    rule_name: str = Field(..., min_length=1, max_length=100)
    rule_config: Dict[str, Any]
    is_active: bool = True


class ComplianceRuleResponse(BaseModel):
    """Model for compliance rule API responses"""
    id: int
    rule_type: RuleType
    rule_name: str
    rule_config: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ViolationDetails(BaseModel):
    """Model for violation details"""
    rule_name: str
    rule_type: RuleType
    violation_description: str
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    affected_assets: Optional[List[str]] = None
    recommended_action: str


class ComplianceViolationResponse(BaseModel):
    """Model for compliance violation API responses"""
    id: int
    portfolio_id: Optional[str]
    rule_id: int
    violation_details: ViolationDetails
    severity: ViolationSeverity
    detected_at: datetime
    resolved_at: Optional[datetime]
    is_resolved: bool
    
    class Config:
        from_attributes = True


class ComplianceValidationRequest(BaseModel):
    """Model for compliance validation requests"""
    portfolio_weights: Optional[Dict[str, float]] = Field(None, description="Asset weights {symbol: weight}")
    allocations: Optional[Dict[str, float]] = Field(None, description="Asset allocations (alternative name for portfolio_weights)")
    rule_set_ids: Optional[List[int]] = Field(None, description="Specific rule IDs to validate against")
    rule_sets: Optional[List[str]] = Field(None, description="Rule set names (alternative to rule_set_ids)")
    portfolio_id: Optional[str] = Field(None, description="Portfolio ID for tracking")
    portfolio_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional portfolio metadata")
    
    def model_post_init(self, __context) -> None:
        """Ensure either portfolio_weights or allocations is provided"""
        if not self.portfolio_weights and not self.allocations:
            raise ValueError("Either portfolio_weights or allocations must be provided")
        # If both are provided, use portfolio_weights as primary
        if self.allocations and not self.portfolio_weights:
            self.portfolio_weights = self.allocations


class ComplianceValidationResult(BaseModel):
    """Model for compliance validation results"""
    is_compliant: bool
    violations: List[ViolationDetails]
    validation_timestamp: datetime
    rule_set_used: List[int]
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class ComplianceReport(BaseModel):
    """Model for comprehensive compliance reports"""
    portfolio_id: Optional[str]
    report_timestamp: datetime
    overall_compliance_score: float = Field(ge=0, le=1)
    total_violations: int
    violations_by_severity: Dict[ViolationSeverity, int]
    recent_violations: List[ComplianceViolationResponse]
    rule_performance: Dict[str, Dict[str, Any]]
