"""
Unit Tests for Position Limit Validators
Tests for single asset, sector concentration, and geographic exposure validation
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.portfolio.compliance import ValidationContext, ValidationResult, ViolationSeverity
from src.portfolio.compliance.position_validators import (
    PositionLimitValidator, 
    ConfigurableThresholdManager,
    AssetMetadata
)


class TestPositionLimitValidator:
    """Test cases for PositionLimitValidator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.basic_config = {
            'rule_name': 'Test Position Limits',
            'max_single_position': 0.10,  # 10%
            'max_sector_concentration': 0.25,  # 25%
            'max_geographic_exposure': {
                'North America': 0.70,
                'Europe': 0.30
            },
            'excluded_assets': ['RISKY_STOCK']
        }
        
        self.validator = PositionLimitValidator(self.basic_config)
        
        self.sample_portfolio = {
            'AAPL': 0.08,   # Technology, North America
            'MSFT': 0.07,   # Technology, North America
            'GOOGL': 0.06,  # Technology, North America
            'JPM': 0.05,    # Financials, North America
            'JNJ': 0.04,    # Healthcare, North America
            'PG': 0.03,     # Consumer Staples, North America
            'HD': 0.02,     # Consumer Discretionary, North America
            'WMT': 0.02     # Consumer Staples, North America
        }
    
    def test_single_asset_limit_pass(self):
        """Test portfolio passing single asset limits"""
        context = ValidationContext(portfolio_weights=self.sample_portfolio)
        result, violations = self.validator.validate(context)
        
        # Check for single asset violations specifically
        single_asset_violations = [v for v in violations if 'single position limit' in v.violation_description]
        assert len(single_asset_violations) == 0
        
    def test_single_asset_limit_violation(self):
        """Test portfolio violating single asset limits"""
        portfolio_with_violation = self.sample_portfolio.copy()
        portfolio_with_violation['AAPL'] = 0.15  # Exceeds 10% limit
        
        context = ValidationContext(portfolio_weights=portfolio_with_violation)
        result, violations = self.validator.validate(context)
        
        # Should have violation for AAPL
        single_asset_violations = [v for v in violations if 'AAPL' in str(v.affected_assets) and 'single position limit' in v.violation_description]
        assert len(single_asset_violations) == 1
        assert single_asset_violations[0].current_value == 0.15
        assert single_asset_violations[0].threshold_value == 0.10
        # Check severity through violation description
        assert 'HIGH' in single_asset_violations[0].violation_description or 'CRITICAL' in single_asset_violations[0].violation_description
        
    def test_single_asset_limit_critical_violation(self):
        """Test portfolio with critical single asset violation"""
        portfolio_with_critical = self.sample_portfolio.copy()
        portfolio_with_critical['AAPL'] = 0.20  # Exceeds 15% (1.5 * 10%)
        
        context = ValidationContext(portfolio_weights=portfolio_with_critical)
        result, violations = self.validator.validate(context)
        
        single_asset_violations = [v for v in violations if 'AAPL' in str(v.affected_assets) and 'single position limit' in v.violation_description]
        assert len(single_asset_violations) == 1
        assert 'CRITICAL' in single_asset_violations[0].violation_description
        
    def test_sector_concentration_pass(self):
        """Test portfolio passing sector concentration limits"""
        # Balanced portfolio across sectors
        balanced_portfolio = {
            'AAPL': 0.08,   # Technology
            'MSFT': 0.07,   # Technology  
            'GOOGL': 0.05,  # Technology - Total Tech: 20%
            'JPM': 0.10,    # Financials
            'BAC': 0.10,    # Financials - Total Financials: 20%
            'JNJ': 0.15,    # Healthcare
            'PG': 0.10,     # Consumer Staples
            'XOM': 0.15     # Energy
        }
        
        context = ValidationContext(portfolio_weights=balanced_portfolio)
        result, violations = self.validator.validate(context)
        
        sector_violations = [v for v in violations if 'sector' in v.violation_description.lower()]
        assert len(sector_violations) == 0
        
    def test_sector_concentration_violation(self):
        """Test portfolio violating sector concentration limits"""
        # Heavy tech concentration
        tech_heavy_portfolio = {
            'AAPL': 0.10,   # Technology
            'MSFT': 0.10,   # Technology
            'GOOGL': 0.08,  # Technology - Total Tech: 28% (exceeds 25%)
            'JPM': 0.05,    # Financials
            'JNJ': 0.05     # Healthcare
        }
        
        context = ValidationContext(portfolio_weights=tech_heavy_portfolio)
        result, violations = self.validator.validate(context)
        
        sector_violations = [v for v in violations if 'Technology' in v.violation_description]
        assert len(sector_violations) == 1
        assert sector_violations[0].current_value == 0.28
        assert sector_violations[0].threshold_value == 0.25
        
    def test_geographic_exposure_pass(self):
        """Test portfolio passing geographic exposure limits"""
        # Portfolio within North America limits (70%)
        context = ValidationContext(portfolio_weights=self.sample_portfolio)
        result, violations = self.validator.validate(context)
        
        geo_violations = [v for v in violations if 'geographic' in v.violation_description.lower()]
        assert len(geo_violations) == 0
        
    def test_geographic_exposure_violation(self):
        """Test portfolio violating geographic exposure limits"""
        # Portfolio exceeding North America limit
        heavy_us_portfolio = {
            'AAPL': 0.20,   # North America
            'MSFT': 0.20,   # North America  
            'GOOGL': 0.20,  # North America
            'JPM': 0.15,    # North America - Total: 75% (exceeds 70%)
        }
        
        context = ValidationContext(portfolio_weights=heavy_us_portfolio)
        result, violations = self.validator.validate(context)
        
        geo_violations = [v for v in violations if 'North America' in v.violation_description]
        assert len(geo_violations) == 1
        assert abs(geo_violations[0].current_value - 0.75) < 0.0001  # Handle floating point precision
        assert geo_violations[0].threshold_value == 0.70
        
    def test_excluded_assets_violation(self):
        """Test portfolio containing excluded assets"""
        portfolio_with_excluded = self.sample_portfolio.copy()
        portfolio_with_excluded['RISKY_STOCK'] = 0.05
        
        context = ValidationContext(portfolio_weights=portfolio_with_excluded)
        result, violations = self.validator.validate(context)
        
        excluded_violations = [v for v in violations if 'excluded asset' in v.violation_description]
        assert len(excluded_violations) == 1
        assert 'RISKY_STOCK' in str(excluded_violations[0].affected_assets)
        assert 'CRITICAL' in excluded_violations[0].violation_description
        
    def test_no_config_no_violations(self):
        """Test validator with no limits configured"""
        empty_config = {'rule_name': 'No Limits'}
        validator = PositionLimitValidator(empty_config)
        
        # Even with extreme portfolio, should pass
        extreme_portfolio = {'SINGLE_STOCK': 1.0}
        context = ValidationContext(portfolio_weights=extreme_portfolio)
        result, violations = validator.validate(context)
        
        assert result == ValidationResult.PASS
        assert len(violations) == 0
        
    def test_market_data_integration(self):
        """Test validator using market data for sector/region info"""
        market_data = {
            'CUSTOM1': {'sector': 'Technology', 'region': 'Asia Pacific'},
            'CUSTOM2': {'sector': 'Technology', 'region': 'Asia Pacific'}
        }
        
        portfolio = {'CUSTOM1': 0.15, 'CUSTOM2': 0.15}  # 30% in Technology
        context = ValidationContext(
            portfolio_weights=portfolio,
            market_data=market_data
        )
        
        result, violations = self.validator.validate(context)
        
        # Should detect Technology sector violation (30% > 25%)
        sector_violations = [v for v in violations if 'Technology' in v.violation_description]
        assert len(sector_violations) == 1
        
    def test_validation_error_handling(self):
        """Test validator error handling"""
        # Create validator that will fail during validation
        with patch.object(PositionLimitValidator, '_check_single_asset_limits', side_effect=Exception("Test error")):
            context = ValidationContext(portfolio_weights=self.sample_portfolio)
            result, violations = self.validator.validate(context)
            
            assert result == ValidationResult.FAIL
            assert len(violations) == 1
            assert 'CRITICAL' in violations[0].violation_description
            assert "validation failed" in violations[0].violation_description
            
    def test_rule_description(self):
        """Test rule description generation"""
        description = self.validator.get_rule_description()
        
        assert "Maximum single asset position: 10.00%" in description
        assert "Maximum sector concentration: 25.00%" in description
        assert "Geographic exposure limits:" in description
        assert "North America: 70.00%" in description
        assert "Excluded assets: RISKY_STOCK" in description


class TestConfigurableThresholdManager:
    """Test cases for ConfigurableThresholdManager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.manager = ConfigurableThresholdManager()
    
    def test_default_config(self):
        """Test default threshold configuration"""
        config = self.manager.get_threshold_config()
        
        assert config['max_single_position'] == 0.10
        assert config['max_sector_concentration'] == 0.25
        assert 'North America' in config['max_geographic_exposure']
        assert config['excluded_assets'] == []
        
    def test_conservative_config(self):
        """Test conservative threshold configuration"""
        config = self.manager.get_threshold_config('conservative')
        
        assert config['max_single_position'] == 0.05  # More restrictive
        assert config['max_sector_concentration'] == 0.20  # More restrictive
        
    def test_aggressive_config(self):
        """Test aggressive threshold configuration"""
        config = self.manager.get_threshold_config('aggressive')
        
        assert config['max_single_position'] == 0.20  # Less restrictive
        assert config['max_sector_concentration'] == 0.40  # Less restrictive
        
    def test_unknown_portfolio_type(self):
        """Test unknown portfolio type defaults to default config"""
        config = self.manager.get_threshold_config('unknown_type')
        default_config = self.manager.get_threshold_config('default')
        
        assert config == default_config
        
    def test_valid_config_validation(self):
        """Test validation of valid configuration"""
        valid_config = {
            'max_single_position': 0.15,
            'max_sector_concentration': 0.30,
            'max_geographic_exposure': {
                'North America': 0.80,
                'Europe': 0.20
            },
            'excluded_assets': ['STOCK1', 'STOCK2']
        }
        
        is_valid, errors = self.manager.validate_threshold_config(valid_config)
        
        assert is_valid
        assert len(errors) == 0
        
    def test_invalid_single_position_config(self):
        """Test validation of invalid single position configuration"""
        invalid_configs = [
            {'max_single_position': -0.1},  # Negative
            {'max_single_position': 1.5},   # > 1
            {'max_single_position': 'invalid'},  # Not numeric
        ]
        
        for config in invalid_configs:
            is_valid, errors = self.manager.validate_threshold_config(config)
            assert not is_valid
            assert any('max_single_position' in error for error in errors)
            
    def test_invalid_sector_config(self):
        """Test validation of invalid sector configuration"""
        invalid_config = {'max_sector_concentration': 0}  # Should be > 0
        
        is_valid, errors = self.manager.validate_threshold_config(invalid_config)
        
        assert not is_valid
        assert any('max_sector_concentration' in error for error in errors)
        
    def test_invalid_geographic_config(self):
        """Test validation of invalid geographic configuration"""
        invalid_configs = [
            {'max_geographic_exposure': 'not_dict'},  # Not a dict
            {'max_geographic_exposure': {'Region': -0.1}},  # Negative value
            {'max_geographic_exposure': {'Region': 1.5}},   # > 1
        ]
        
        for config in invalid_configs:
            is_valid, errors = self.manager.validate_threshold_config(config)
            assert not is_valid
            
    def test_invalid_excluded_assets_config(self):
        """Test validation of invalid excluded assets configuration"""
        invalid_configs = [
            {'excluded_assets': 'not_list'},  # Not a list
            {'excluded_assets': [123, 'STOCK']},  # Contains non-string
        ]
        
        for config in invalid_configs:
            is_valid, errors = self.manager.validate_threshold_config(config)
            assert not is_valid


@pytest.fixture
def sample_position_validator():
    """Fixture providing a configured position validator"""
    config = {
        'rule_name': 'Test Position Limits',
        'max_single_position': 0.10,
        'max_sector_concentration': 0.25,
        'max_geographic_exposure': {'North America': 0.70},
        'excluded_assets': ['EXCLUDED_STOCK']
    }
    return PositionLimitValidator(config)


@pytest.fixture
def sample_portfolio():
    """Fixture providing a sample portfolio"""
    return {
        'AAPL': 0.08,
        'MSFT': 0.07,
        'GOOGL': 0.06,
        'JPM': 0.05,
        'JNJ': 0.04
    }


class TestPositionValidatorIntegration:
    """Integration tests for position validators"""
    
    def test_comprehensive_validation(self, sample_position_validator, sample_portfolio):
        """Test comprehensive validation with multiple rule types"""
        context = ValidationContext(portfolio_weights=sample_portfolio)
        result, violations = sample_position_validator.validate(context)
        
        # Should pass all validations for this balanced portfolio
        assert result in [ValidationResult.PASS, ValidationResult.WARNING]
        
        # If there are violations, they should be categorized properly
        if violations:
            for violation in violations:
                assert violation.rule_type.value == 'position_limit'
                # Check that severity is encoded in description
                assert any(severity in violation.violation_description for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
                assert violation.affected_assets is not None
                assert violation.recommended_action is not None
                
    def test_performance_benchmark(self, sample_position_validator):
        """Test validation performance with large portfolio"""
        import time
        
        # Create large portfolio (100 assets)
        large_portfolio = {f'STOCK_{i}': 0.01 for i in range(100)}
        context = ValidationContext(portfolio_weights=large_portfolio)
        
        start_time = time.time()
        result, violations = sample_position_validator.validate(context)
        end_time = time.time()
        
        # Validation should complete quickly (< 50ms target)
        validation_time = (end_time - start_time) * 1000  # Convert to ms
        assert validation_time < 50, f"Validation took {validation_time:.2f}ms, exceeding 50ms target"
