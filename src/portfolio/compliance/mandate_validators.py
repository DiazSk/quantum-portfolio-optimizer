"""
Investment Mandate Validators
Validators for ESG restrictions, credit rating minimums, and liquidity requirements
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from . import ComplianceValidator, ValidationContext, ValidationResult, ViolationDetails, ViolationSeverity, RuleType

logger = logging.getLogger(__name__)


@dataclass
class ESGScores:
    """ESG scoring data for assets"""
    environmental_score: Optional[float] = None
    social_score: Optional[float] = None
    governance_score: Optional[float] = None
    overall_score: Optional[float] = None
    provider: Optional[str] = None


@dataclass
class CreditRating:
    """Credit rating information for assets"""
    rating: Optional[str] = None
    numeric_score: Optional[int] = None  # AAA=1, AA=2, etc.
    agency: Optional[str] = None


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for assets"""
    avg_daily_volume: Optional[float] = None
    bid_ask_spread: Optional[float] = None
    liquidity_score: Optional[float] = None  # 0-1 scale


class MandateValidator(ComplianceValidator):
    """Validator for investment mandates including ESG, credit ratings, and liquidity"""
    
    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.esg_requirements = rule_config.get('esg_requirements', {})
        self.min_credit_rating = rule_config.get('min_credit_rating')
        self.min_liquidity_score = rule_config.get('min_liquidity_score')
        self.required_sectors = set(rule_config.get('required_sectors', []))
        self.forbidden_sectors = set(rule_config.get('forbidden_sectors', []))
        
    def _get_rule_type(self) -> RuleType:
        return RuleType.MANDATE
    
    def _create_violation_with_severity(
        self, 
        description: str, 
        severity: ViolationSeverity,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        affected_assets: Optional[List[str]] = None,
        recommended_action: str = "Review and adjust portfolio allocation"
    ) -> ViolationDetails:
        """Helper method to create violation details with severity in description"""
        return ViolationDetails(
            rule_name=self.rule_config.get('rule_name', 'Unknown Rule'),
            rule_type=self.rule_type,
            violation_description=f"[{severity.value.upper()}] {description}",
            current_value=current_value,
            threshold_value=threshold_value,
            affected_assets=affected_assets or [],
            recommended_action=recommended_action
        )
    
    def validate(self, context: ValidationContext) -> Tuple[ValidationResult, List[ViolationDetails]]:
        """
        Validate portfolio against investment mandates
        
        Args:
            context: Validation context with portfolio weights and metadata
            
        Returns:
            Tuple of (validation_result, violations_list)
        """
        violations = []
        violation_severities = []
        
        try:
            # Check ESG requirements
            esg_violations, esg_severities = self._check_esg_requirements(context)
            violations.extend(esg_violations)
            violation_severities.extend(esg_severities)
            
            # Check credit rating requirements
            credit_violations, credit_severities = self._check_credit_rating_requirements(context)
            violations.extend(credit_violations)
            violation_severities.extend(credit_severities)
            
            # Check liquidity requirements
            liquidity_violations, liquidity_severities = self._check_liquidity_requirements(context)
            violations.extend(liquidity_violations)
            violation_severities.extend(liquidity_severities)
            
            # Check sector requirements
            sector_violations, sector_severities = self._check_sector_requirements(context)
            violations.extend(sector_violations)
            violation_severities.extend(sector_severities)
            
            # Determine overall result
            if any(s in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL] for s in violation_severities):
                result = ValidationResult.FAIL
            elif violations:
                result = ValidationResult.WARNING
            else:
                result = ValidationResult.PASS
                
            logger.info(f"Mandate validation completed: {result.value}, {len(violations)} violations")
            return result, violations
            
        except Exception as e:
            logger.error(f"Error in mandate validation: {str(e)}")
            error_violation = self._create_violation_with_severity(
                f"Investment mandate validation failed: {str(e)}",
                ViolationSeverity.CRITICAL,
                recommended_action="Review portfolio data and mandate configuration"
            )
            return ValidationResult.FAIL, [error_violation]
    
    def _check_esg_requirements(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check ESG scoring requirements"""
        violations = []
        severities = []
        
        if not self.esg_requirements:
            return violations, severities
            
        min_esg_score = self.esg_requirements.get('min_esg_score')
        exclude_unrated = self.esg_requirements.get('exclude_unrated', False)
        min_environmental = self.esg_requirements.get('min_environmental_score')
        min_social = self.esg_requirements.get('min_social_score')
        min_governance = self.esg_requirements.get('min_governance_score')
        
        for symbol, weight in context.portfolio_weights.items():
            if weight <= 0:
                continue
                
            esg_data = self._get_esg_data(symbol, context)
            
            # Check if asset has ESG data
            if not esg_data or esg_data.overall_score is None:
                if exclude_unrated:
                    violation = self._create_violation_with_severity(
                        f"Asset {symbol} lacks ESG rating and unrated assets are excluded",
                        ViolationSeverity.HIGH,
                        affected_assets=[symbol],
                        recommended_action=f"Remove {symbol} or obtain ESG rating"
                    )
                    violations.append(violation)
                    severities.append(ViolationSeverity.HIGH)
                continue
            
            # Check minimum overall ESG score
            if min_esg_score and esg_data.overall_score < min_esg_score:
                severity = ViolationSeverity.HIGH if esg_data.overall_score < min_esg_score * 0.8 else ViolationSeverity.MEDIUM
                violation = self._create_violation_with_severity(
                    f"Asset {symbol} ESG score {esg_data.overall_score:.1f} below minimum {min_esg_score}",
                    severity,
                    current_value=esg_data.overall_score,
                    threshold_value=min_esg_score,
                    affected_assets=[symbol],
                    recommended_action=f"Replace {symbol} with higher ESG-rated alternative"
                )
                violations.append(violation)
                severities.append(severity)
            
            # Check individual ESG component scores
            if min_environmental and esg_data.environmental_score and esg_data.environmental_score < min_environmental:
                violation = self._create_violation_with_severity(
                    f"Asset {symbol} environmental score {esg_data.environmental_score:.1f} below minimum {min_environmental}",
                    ViolationSeverity.MEDIUM,
                    current_value=esg_data.environmental_score,
                    threshold_value=min_environmental,
                    affected_assets=[symbol],
                    recommended_action=f"Replace {symbol} with environmentally better alternative"
                )
                violations.append(violation)
                severities.append(ViolationSeverity.MEDIUM)
                
        return violations, severities
    
    def _check_credit_rating_requirements(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check credit rating minimum requirements"""
        violations = []
        severities = []
        
        if not self.min_credit_rating:
            return violations, severities
            
        min_rating_numeric = self._rating_to_numeric(self.min_credit_rating)
        
        for symbol, weight in context.portfolio_weights.items():
            if weight <= 0:
                continue
                
            credit_data = self._get_credit_rating(symbol, context)
            
            # Skip if no credit rating available (equity assets)
            if not credit_data or not credit_data.rating:
                continue
                
            current_rating_numeric = self._rating_to_numeric(credit_data.rating)
            
            # Lower numeric scores are better (AAA=1, BB=10, etc.)
            if current_rating_numeric > min_rating_numeric:
                severity = ViolationSeverity.HIGH if current_rating_numeric > min_rating_numeric + 2 else ViolationSeverity.MEDIUM
                violation = self._create_violation_with_severity(
                    f"Asset {symbol} credit rating {credit_data.rating} below minimum {self.min_credit_rating}",
                    severity,
                    current_value=float(current_rating_numeric),
                    threshold_value=float(min_rating_numeric),
                    affected_assets=[symbol],
                    recommended_action=f"Replace {symbol} with higher-rated alternative"
                )
                violations.append(violation)
                severities.append(severity)
                
        return violations, severities
    
    def _check_liquidity_requirements(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check liquidity score requirements"""
        violations = []
        severities = []
        
        if not self.min_liquidity_score:
            return violations, severities
            
        for symbol, weight in context.portfolio_weights.items():
            if weight <= 0:
                continue
                
            liquidity_data = self._get_liquidity_metrics(symbol, context)
            
            if not liquidity_data or liquidity_data.liquidity_score is None:
                # Default assumption for major stocks
                if symbol in self._get_major_stocks():
                    continue
                    
                violation = self._create_violation_with_severity(
                    f"Asset {symbol} lacks liquidity data for validation",
                    ViolationSeverity.MEDIUM,
                    affected_assets=[symbol],
                    recommended_action=f"Obtain liquidity metrics for {symbol} or remove from portfolio"
                )
                violations.append(violation)
                severities.append(ViolationSeverity.MEDIUM)
                continue
            
            if liquidity_data.liquidity_score < self.min_liquidity_score:
                severity = ViolationSeverity.HIGH if liquidity_data.liquidity_score < self.min_liquidity_score * 0.7 else ViolationSeverity.MEDIUM
                violation = self._create_violation_with_severity(
                    f"Asset {symbol} liquidity score {liquidity_data.liquidity_score:.2f} below minimum {self.min_liquidity_score}",
                    severity,
                    current_value=liquidity_data.liquidity_score,
                    threshold_value=self.min_liquidity_score,
                    affected_assets=[symbol],
                    recommended_action=f"Replace {symbol} with more liquid alternative"
                )
                violations.append(violation)
                severities.append(severity)
                
        return violations, severities
    
    def _check_sector_requirements(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check required and forbidden sectors"""
        violations = []
        severities = []
        
        # Get portfolio sector allocation
        sector_allocations = self._calculate_sector_allocations(context)
        
        # Check required sectors
        for required_sector in self.required_sectors:
            if required_sector not in sector_allocations or sector_allocations[required_sector] == 0:
                violation = self._create_violation_with_severity(
                    f"Portfolio missing required sector allocation: {required_sector}",
                    ViolationSeverity.MEDIUM,
                    current_value=sector_allocations.get(required_sector, 0.0),
                    threshold_value=0.01,  # Minimum 1% allocation
                    recommended_action=f"Add allocation to {required_sector} sector"
                )
                violations.append(violation)
                severities.append(ViolationSeverity.MEDIUM)
        
        # Check forbidden sectors
        for forbidden_sector in self.forbidden_sectors:
            if forbidden_sector in sector_allocations and sector_allocations[forbidden_sector] > 0:
                affected_assets = [symbol for symbol, weight in context.portfolio_weights.items() 
                                 if self._get_asset_sector(symbol, context) == forbidden_sector and weight > 0]
                violation = self._create_violation_with_severity(
                    f"Portfolio contains forbidden sector allocation: {forbidden_sector}",
                    ViolationSeverity.HIGH,
                    current_value=sector_allocations[forbidden_sector],
                    threshold_value=0.0,
                    affected_assets=affected_assets,
                    recommended_action=f"Remove all {forbidden_sector} sector allocations"
                )
                violations.append(violation)
                severities.append(ViolationSeverity.HIGH)
                
        return violations, severities
    
    def _get_esg_data(self, symbol: str, context: ValidationContext) -> Optional[ESGScores]:
        """Get ESG data for asset"""
        # Try to get from market data first
        if context.market_data and symbol in context.market_data:
            esg_info = context.market_data[symbol].get('esg')
            if esg_info:
                return ESGScores(
                    environmental_score=esg_info.get('environmental'),
                    social_score=esg_info.get('social'),
                    governance_score=esg_info.get('governance'),
                    overall_score=esg_info.get('overall'),
                    provider=esg_info.get('provider')
                )
        
        # Fallback to mock ESG data for testing
        mock_esg_data = {
            'AAPL': ESGScores(environmental_score=8.5, social_score=8.0, governance_score=9.0, overall_score=8.5),
            'MSFT': ESGScores(environmental_score=9.0, social_score=8.5, governance_score=9.5, overall_score=9.0),
            'GOOGL': ESGScores(environmental_score=7.5, social_score=7.0, governance_score=8.0, overall_score=7.5),
            'TSLA': ESGScores(environmental_score=9.5, social_score=6.0, governance_score=5.0, overall_score=6.8),
            'XOM': ESGScores(environmental_score=3.0, social_score=4.0, governance_score=6.0, overall_score=4.3),
            'TOBACCO_STOCK': ESGScores(environmental_score=2.0, social_score=2.0, governance_score=5.0, overall_score=3.0),
        }
        return mock_esg_data.get(symbol)
    
    def _get_credit_rating(self, symbol: str, context: ValidationContext) -> Optional[CreditRating]:
        """Get credit rating for asset"""
        # Try to get from market data first
        if context.market_data and symbol in context.market_data:
            rating_info = context.market_data[symbol].get('credit_rating')
            if rating_info:
                return CreditRating(
                    rating=rating_info.get('rating'),
                    numeric_score=rating_info.get('numeric_score'),
                    agency=rating_info.get('agency')
                )
        
        # Mock credit ratings for bonds/debt instruments
        mock_ratings = {
            'GOVT_BOND': CreditRating(rating='AAA', numeric_score=1, agency='S&P'),
            'CORP_BOND_A': CreditRating(rating='A', numeric_score=6, agency='Moody\'s'),
            'CORP_BOND_BB': CreditRating(rating='BB', numeric_score=10, agency='S&P'),
            'JUNK_BOND': CreditRating(rating='B', numeric_score=12, agency='Fitch'),
        }
        return mock_ratings.get(symbol)
    
    def _get_liquidity_metrics(self, symbol: str, context: ValidationContext) -> Optional[LiquidityMetrics]:
        """Get liquidity metrics for asset"""
        # Try to get from market data first
        if context.market_data and symbol in context.market_data:
            liquidity_info = context.market_data[symbol].get('liquidity')
            if liquidity_info:
                return LiquidityMetrics(
                    avg_daily_volume=liquidity_info.get('avg_daily_volume'),
                    bid_ask_spread=liquidity_info.get('bid_ask_spread'),
                    liquidity_score=liquidity_info.get('score')
                )
        
        # Mock liquidity data
        mock_liquidity = {
            'AAPL': LiquidityMetrics(avg_daily_volume=50000000, bid_ask_spread=0.01, liquidity_score=0.95),
            'MSFT': LiquidityMetrics(avg_daily_volume=30000000, bid_ask_spread=0.02, liquidity_score=0.90),
            'GOOGL': LiquidityMetrics(avg_daily_volume=25000000, bid_ask_spread=0.03, liquidity_score=0.85),
            'SMALL_CAP': LiquidityMetrics(avg_daily_volume=100000, bid_ask_spread=0.20, liquidity_score=0.35),
            'ILLIQUID_STOCK': LiquidityMetrics(avg_daily_volume=10000, bid_ask_spread=0.50, liquidity_score=0.15),
        }
        
        # Default high liquidity for major stocks
        if symbol in self._get_major_stocks():
            return LiquidityMetrics(avg_daily_volume=20000000, bid_ask_spread=0.05, liquidity_score=0.80)
            
        return mock_liquidity.get(symbol)
    
    def _calculate_sector_allocations(self, context: ValidationContext) -> Dict[str, float]:
        """Calculate portfolio allocation by sector"""
        sector_allocations = {}
        for symbol, weight in context.portfolio_weights.items():
            sector = self._get_asset_sector(symbol, context)
            if sector:
                sector_allocations[sector] = sector_allocations.get(sector, 0.0) + weight
        return sector_allocations
    
    def _get_asset_sector(self, symbol: str, context: ValidationContext) -> Optional[str]:
        """Get sector for asset from market data or default mapping"""
        # Try to get from market data first
        if context.market_data and symbol in context.market_data:
            return context.market_data[symbol].get('sector')
            
        # Fallback to basic sector mapping
        sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology',
            'JPM': 'Financials', 'JNJ': 'Healthcare', 'PG': 'Consumer Staples', 'HD': 'Consumer Discretionary',
            'BAC': 'Financials', 'XOM': 'Energy', 'WMT': 'Consumer Staples', 'CVX': 'Energy',
            'TOBACCO_STOCK': 'Consumer Staples', 'DEFENSE_STOCK': 'Industrials'
        }
        return sector_mapping.get(symbol, 'Unknown')
    
    def _get_major_stocks(self) -> set:
        """Get set of major liquid stocks"""
        return {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 
            'PG', 'HD', 'BAC', 'XOM', 'WMT', 'CVX', 'KO', 'DIS', 'IBM', 'INTC'
        }
    
    def _rating_to_numeric(self, rating: str) -> int:
        """Convert credit rating to numeric score (lower is better)"""
        rating_map = {
            'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
            'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10,
            'BB+': 11, 'BB': 12, 'BB-': 13,
            'B+': 14, 'B': 15, 'B-': 16,
            'CCC+': 17, 'CCC': 18, 'CCC-': 19,
            'CC': 20, 'C': 21, 'D': 22
        }
        return rating_map.get(rating.upper(), 99)  # Default to worst score if unknown
    
    def get_rule_description(self) -> str:
        """Return human-readable description of mandate rules"""
        descriptions = []
        
        if self.esg_requirements:
            if 'min_esg_score' in self.esg_requirements:
                descriptions.append(f"Minimum ESG score: {self.esg_requirements['min_esg_score']}")
            if self.esg_requirements.get('exclude_unrated'):
                descriptions.append("Exclude ESG unrated assets")
                
        if self.min_credit_rating:
            descriptions.append(f"Minimum credit rating: {self.min_credit_rating}")
            
        if self.min_liquidity_score:
            descriptions.append(f"Minimum liquidity score: {self.min_liquidity_score}")
            
        if self.required_sectors:
            descriptions.append(f"Required sectors: {', '.join(self.required_sectors)}")
            
        if self.forbidden_sectors:
            descriptions.append(f"Forbidden sectors: {', '.join(self.forbidden_sectors)}")
            
        return "; ".join(descriptions) if descriptions else "No mandate requirements configured"
