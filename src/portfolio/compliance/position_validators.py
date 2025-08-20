"""
Position Limit Validators
Validators for single asset, sector concentration, and geographic exposure limits
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from . import ComplianceValidator, ValidationContext, ValidationResult, ViolationDetails, ViolationSeverity, RuleType

logger = logging.getLogger(__name__)


@dataclass
class AssetMetadata:
    """Metadata for assets used in position limit validation"""
    symbol: str
    sector: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    market_cap: Optional[float] = None
    liquidity_score: Optional[float] = None


class PositionLimitValidator(ComplianceValidator):
    """Validator for position limits including single asset, sector, and geographic exposure"""
    
    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.max_single_position = rule_config.get('max_single_position')
        self.max_sector_concentration = rule_config.get('max_sector_concentration')
        self.max_geographic_exposure = rule_config.get('max_geographic_exposure', {})
        self.excluded_assets = set(rule_config.get('excluded_assets', []))
        
    def _get_rule_type(self) -> RuleType:
        return RuleType.POSITION_LIMIT
    
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
        Validate portfolio against position limits
        
        Args:
            context: Validation context with portfolio weights and metadata
            
        Returns:
            Tuple of (validation_result, violations_list)
        """
        violations = []
        violation_severities = []
        
        try:
            # Check single asset position limits
            single_asset_violations, single_severities = self._check_single_asset_limits(context)
            violations.extend(single_asset_violations)
            violation_severities.extend(single_severities)
            
            # Check sector concentration limits  
            sector_violations, sector_severities = self._check_sector_concentration(context)
            violations.extend(sector_violations)
            violation_severities.extend(sector_severities)
            
            # Check geographic exposure limits
            geographic_violations, geo_severities = self._check_geographic_exposure(context)
            violations.extend(geographic_violations)
            violation_severities.extend(geo_severities)
            
            # Check excluded assets
            excluded_violations, excluded_severities = self._check_excluded_assets(context)
            violations.extend(excluded_violations)
            violation_severities.extend(excluded_severities)
            
            # Determine overall result
            if any(s in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL] for s in violation_severities):
                result = ValidationResult.FAIL
            elif violations:
                result = ValidationResult.WARNING
            else:
                result = ValidationResult.PASS
                
            logger.info(f"Position limit validation completed: {result.value}, {len(violations)} violations")
            return result, violations
            
        except Exception as e:
            logger.error(f"Error in position limit validation: {str(e)}")
            error_violation = self._create_violation_with_severity(
                f"Position limit validation failed: {str(e)}",
                ViolationSeverity.CRITICAL,
                recommended_action="Review portfolio data and validation configuration"
            )
            return ValidationResult.FAIL, [error_violation]
    
    def _check_single_asset_limits(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check individual asset position limits"""
        violations = []
        severities = []
        
        if self.max_single_position is None:
            return violations, severities
            
        for symbol, weight in context.portfolio_weights.items():
            if weight > self.max_single_position:
                severity = ViolationSeverity.CRITICAL if weight > self.max_single_position * 1.5 else ViolationSeverity.HIGH
                
                violation = self._create_violation_with_severity(
                    f"Asset {symbol} exceeds maximum single position limit",
                    severity,
                    current_value=weight,
                    threshold_value=self.max_single_position,
                    affected_assets=[symbol],
                    recommended_action=f"Reduce {symbol} allocation to {self.max_single_position:.2%} or less"
                )
                violations.append(violation)
                severities.append(severity)
                
        return violations, severities
    
    def _check_sector_concentration(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check sector concentration limits"""
        violations = []
        severities = []
        
        if self.max_sector_concentration is None:
            return violations, severities
            
        # Get sector mappings from market data or use default mapping
        sector_exposures = self._calculate_sector_exposures(context)
        
        for sector, exposure in sector_exposures.items():
            if exposure > self.max_sector_concentration:
                affected_assets = [symbol for symbol, weight in context.portfolio_weights.items() 
                                 if self._get_asset_sector(symbol, context) == sector]
                
                severity = ViolationSeverity.HIGH if exposure > self.max_sector_concentration * 1.2 else ViolationSeverity.MEDIUM
                
                violation = self._create_violation_with_severity(
                    f"Sector '{sector}' exceeds maximum concentration limit",
                    severity,
                    current_value=exposure,
                    threshold_value=self.max_sector_concentration,
                    affected_assets=affected_assets,
                    recommended_action=f"Reduce {sector} sector allocation to {self.max_sector_concentration:.2%} or less"
                )
                violations.append(violation)
                severities.append(severity)
                
        return violations, severities
    
    def _check_geographic_exposure(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check geographic exposure limits"""
        violations = []
        severities = []
        
        if not self.max_geographic_exposure:
            return violations, severities
            
        # Calculate geographic exposures
        geographic_exposures = self._calculate_geographic_exposures(context)
        
        for region, max_exposure in self.max_geographic_exposure.items():
            current_exposure = geographic_exposures.get(region, 0.0)
            
            if current_exposure > max_exposure:
                affected_assets = [symbol for symbol, weight in context.portfolio_weights.items() 
                                 if self._get_asset_region(symbol, context) == region]
                
                severity = ViolationSeverity.HIGH if current_exposure > max_exposure * 1.2 else ViolationSeverity.MEDIUM
                
                violation = self._create_violation_with_severity(
                    f"Geographic region '{region}' exceeds maximum exposure limit",
                    severity,
                    current_value=current_exposure,
                    threshold_value=max_exposure,
                    affected_assets=affected_assets,
                    recommended_action=f"Reduce {region} regional allocation to {max_exposure:.2%} or less"
                )
                violations.append(violation)
                severities.append(severity)
                
        return violations, severities
    
    def _check_excluded_assets(self, context: ValidationContext) -> Tuple[List[ViolationDetails], List[ViolationSeverity]]:
        """Check for excluded assets in portfolio"""
        violations = []
        severities = []
        
        if not self.excluded_assets:
            return violations, severities
            
        for symbol, weight in context.portfolio_weights.items():
            if symbol in self.excluded_assets and weight > 0:
                violation = self._create_violation_with_severity(
                    f"Portfolio contains excluded asset: {symbol}",
                    ViolationSeverity.CRITICAL,
                    current_value=weight,
                    threshold_value=0.0,
                    affected_assets=[symbol],
                    recommended_action=f"Remove {symbol} from portfolio allocation"
                )
                violations.append(violation)
                severities.append(ViolationSeverity.CRITICAL)
                
        return violations, severities
    
    def _calculate_sector_exposures(self, context: ValidationContext) -> Dict[str, float]:
        """Calculate total exposure by sector"""
        sector_exposures = {}
        
        for symbol, weight in context.portfolio_weights.items():
            sector = self._get_asset_sector(symbol, context)
            if sector:
                sector_exposures[sector] = sector_exposures.get(sector, 0.0) + weight
                
        return sector_exposures
    
    def _calculate_geographic_exposures(self, context: ValidationContext) -> Dict[str, float]:
        """Calculate total exposure by geographic region"""
        geographic_exposures = {}
        
        for symbol, weight in context.portfolio_weights.items():
            region = self._get_asset_region(symbol, context)
            if region:
                geographic_exposures[region] = geographic_exposures.get(region, 0.0) + weight
                
        return geographic_exposures
    
    def _get_asset_sector(self, symbol: str, context: ValidationContext) -> Optional[str]:
        """Get sector for asset from market data or default mapping"""
        # Try to get from market data first
        if context.market_data and symbol in context.market_data:
            return context.market_data[symbol].get('sector')
            
        # Fallback to basic sector mapping (in production, this would come from a data service)
        sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology',
            'JPM': 'Financials', 'JNJ': 'Healthcare', 'PG': 'Consumer Staples', 'HD': 'Consumer Discretionary',
            'BAC': 'Financials', 'XOM': 'Energy', 'WMT': 'Consumer Staples', 'CVX': 'Energy'
        }
        return sector_mapping.get(symbol, 'Unknown')
    
    def _get_asset_region(self, symbol: str, context: ValidationContext) -> Optional[str]:
        """Get geographic region for asset"""
        # Try to get from market data first
        if context.market_data and symbol in context.market_data:
            return context.market_data[symbol].get('region')
            
        # Fallback to basic region mapping (in production, this would come from a data service)
        # For simplicity, assume US stocks for common symbols
        us_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 
            'PG', 'HD', 'BAC', 'XOM', 'WMT', 'CVX', 'KO', 'DIS', 'IBM', 'INTC'
        }
        
        if symbol in us_symbols:
            return 'North America'
        else:
            return 'Unknown'
    
    def get_rule_description(self) -> str:
        """Return human-readable description of position limit rules"""
        descriptions = []
        
        if self.max_single_position:
            descriptions.append(f"Maximum single asset position: {self.max_single_position:.2%}")
            
        if self.max_sector_concentration:
            descriptions.append(f"Maximum sector concentration: {self.max_sector_concentration:.2%}")
            
        if self.max_geographic_exposure:
            geo_desc = ", ".join([f"{region}: {limit:.2%}" for region, limit in self.max_geographic_exposure.items()])
            descriptions.append(f"Geographic exposure limits: {geo_desc}")
            
        if self.excluded_assets:
            descriptions.append(f"Excluded assets: {', '.join(self.excluded_assets)}")
            
        return "; ".join(descriptions) if descriptions else "No position limits configured"


class ConfigurableThresholdManager:
    """Manages configurable thresholds for position limit checking"""
    
    def __init__(self):
        self.default_thresholds = {
            'max_single_position': 0.10,  # 10%
            'max_sector_concentration': 0.25,  # 25%
            'max_geographic_exposure': {
                'North America': 0.70,  # 70%
                'Europe': 0.30,  # 30%
                'Asia Pacific': 0.20,  # 20%
                'Emerging Markets': 0.15  # 15%
            },
            'excluded_assets': []
        }
        
    def get_threshold_config(self, portfolio_type: str = 'default') -> Dict[str, Any]:
        """Get threshold configuration for portfolio type"""
        # In production, this would load from database based on portfolio type
        portfolio_configs = {
            'conservative': {
                'max_single_position': 0.05,  # 5%
                'max_sector_concentration': 0.20,  # 20%
                'max_geographic_exposure': {
                    'North America': 0.60,
                    'Europe': 0.25,
                    'Asia Pacific': 0.15,
                    'Emerging Markets': 0.10
                }
            },
            'balanced': self.default_thresholds,
            'aggressive': {
                'max_single_position': 0.20,  # 20%
                'max_sector_concentration': 0.40,  # 40%
                'max_geographic_exposure': {
                    'North America': 0.80,
                    'Europe': 0.40,
                    'Asia Pacific': 0.30,
                    'Emerging Markets': 0.25
                }
            }
        }
        
        return portfolio_configs.get(portfolio_type, self.default_thresholds)
    
    def validate_threshold_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate threshold configuration parameters"""
        errors = []
        
        # Validate single position limit
        if 'max_single_position' in config:
            if not isinstance(config['max_single_position'], (int, float)) or not 0 < config['max_single_position'] <= 1:
                errors.append("max_single_position must be a number between 0 and 1")
                
        # Validate sector concentration limit
        if 'max_sector_concentration' in config:
            if not isinstance(config['max_sector_concentration'], (int, float)) or not 0 < config['max_sector_concentration'] <= 1:
                errors.append("max_sector_concentration must be a number between 0 and 1")
                
        # Validate geographic exposure limits
        if 'max_geographic_exposure' in config:
            geo_config = config['max_geographic_exposure']
            if not isinstance(geo_config, dict):
                errors.append("max_geographic_exposure must be a dictionary")
            else:
                for region, limit in geo_config.items():
                    if not isinstance(limit, (int, float)) or not 0 < limit <= 1:
                        errors.append(f"Geographic limit for {region} must be a number between 0 and 1")
                        
        # Validate excluded assets
        if 'excluded_assets' in config:
            if not isinstance(config['excluded_assets'], list):
                errors.append("excluded_assets must be a list")
            else:
                for asset in config['excluded_assets']:
                    if not isinstance(asset, str):
                        errors.append("All excluded assets must be strings")
                        
        return len(errors) == 0, errors
