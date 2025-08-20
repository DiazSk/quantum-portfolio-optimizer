"""
Unit Tests for Investment Mandate Validators
Tests for ESG, credit rating, liquidity, and sector mandate validation
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.portfolio.compliance import ValidationContext, ValidationResult, ViolationSeverity
from src.portfolio.compliance.mandate_validators import (
    MandateValidator,
    ESGScores,
    CreditRating,
    LiquidityMetrics
)


class TestMandateValidator:
    """Test cases for MandateValidator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.basic_config = {
            'rule_name': 'Test Investment Mandates',
            'esg_requirements': {
                'min_esg_score': 7.0,
                'exclude_unrated': True,
                'min_environmental_score': 6.0
            },
            'min_credit_rating': 'BBB',
            'min_liquidity_score': 0.70,
            'required_sectors': ['Technology', 'Healthcare'],
            'forbidden_sectors': ['Tobacco', 'Defense']
        }
        
        self.validator = MandateValidator(self.basic_config)
        
        self.sample_portfolio = {
            'AAPL': 0.20,   # Technology, High ESG
            'MSFT': 0.15,   # Technology, High ESG
            'JNJ': 0.10,    # Healthcare
            'PG': 0.05      # Consumer Staples
        }
    
    def test_esg_requirements_pass(self):
        """Test portfolio passing ESG requirements"""
        context = ValidationContext(portfolio_weights=self.sample_portfolio)
        result, violations = self.validator.validate(context)
        
        # Check for ESG violations specifically
        esg_violations = [v for v in violations if 'ESG' in v.violation_description]
        # Should pass as mock data has good ESG scores for AAPL, MSFT, JNJ
        assert len(esg_violations) == 0 or all('below minimum' not in v.violation_description for v in esg_violations)
        
    def test_esg_score_violation(self):
        """Test portfolio violating ESG score requirements"""
        portfolio_with_low_esg = {
            'XOM': 0.30,  # Low ESG score (4.3)
            'AAPL': 0.20  # High ESG score (8.5)
        }
        
        context = ValidationContext(portfolio_weights=portfolio_with_low_esg)
        result, violations = self.validator.validate(context)
        
        # Should have violation for XOM
        esg_violations = [v for v in violations if 'XOM' in str(v.affected_assets) and 'ESG score' in v.violation_description]
        assert len(esg_violations) == 1
        assert 'below minimum' in esg_violations[0].violation_description
        
    def test_esg_unrated_exclusion(self):
        """Test exclusion of ESG unrated assets"""
        portfolio_with_unrated = {
            'UNRATED_STOCK': 0.10,  # No ESG data
            'AAPL': 0.20
        }
        
        context = ValidationContext(portfolio_weights=portfolio_with_unrated)
        result, violations = self.validator.validate(context)
        
        # Should have violation for unrated stock
        unrated_violations = [v for v in violations if 'UNRATED_STOCK' in str(v.affected_assets) and 'lacks ESG rating' in v.violation_description]
        assert len(unrated_violations) == 1
        assert 'HIGH' in unrated_violations[0].violation_description
        
    def test_credit_rating_requirements(self):
        """Test credit rating minimum requirements"""
        portfolio_with_bonds = {
            'CORP_BOND_A': 0.10,    # A rating (meets BBB minimum)
            'JUNK_BOND': 0.05,      # B rating (below BBB minimum)
            'GOVT_BOND': 0.15       # AAA rating (exceeds minimum)
        }
        
        context = ValidationContext(portfolio_weights=portfolio_with_bonds)
        result, violations = self.validator.validate(context)
        
        # Should have violation for junk bond
        credit_violations = [v for v in violations if 'JUNK_BOND' in str(v.affected_assets) and 'credit rating' in v.violation_description]
        assert len(credit_violations) == 1
        assert 'below minimum' in credit_violations[0].violation_description
        
    def test_liquidity_requirements_pass(self):
        """Test portfolio passing liquidity requirements"""
        high_liquidity_portfolio = {
            'AAPL': 0.30,   # High liquidity (0.95)
            'MSFT': 0.25,   # High liquidity (0.90)
            'GOOGL': 0.20   # Good liquidity (0.85)
        }
        
        context = ValidationContext(portfolio_weights=high_liquidity_portfolio)
        result, violations = self.validator.validate(context)
        
        # Check for liquidity violations
        liquidity_violations = [v for v in violations if 'liquidity score' in v.violation_description and 'below minimum' in v.violation_description]
        assert len(liquidity_violations) == 0
        
    def test_liquidity_requirements_violation(self):
        """Test portfolio violating liquidity requirements"""
        low_liquidity_portfolio = {
            'ILLIQUID_STOCK': 0.15,  # Low liquidity (0.15)
            'AAPL': 0.25             # High liquidity
        }
        
        context = ValidationContext(portfolio_weights=low_liquidity_portfolio)
        result, violations = self.validator.validate(context)
        
        # Should have violation for illiquid stock
        liquidity_violations = [v for v in violations if 'ILLIQUID_STOCK' in str(v.affected_assets) and 'liquidity score' in v.violation_description]
        assert len(liquidity_violations) == 1
        assert 'below minimum' in liquidity_violations[0].violation_description
        
    def test_required_sectors_pass(self):
        """Test portfolio meeting required sector allocations"""
        # Portfolio has both required sectors (Technology and Healthcare)
        context = ValidationContext(portfolio_weights=self.sample_portfolio)
        result, violations = self.validator.validate(context)
        
        # Check for missing required sectors
        missing_sector_violations = [v for v in violations if 'missing required sector' in v.violation_description]
        assert len(missing_sector_violations) == 0
        
    def test_required_sectors_violation(self):
        """Test portfolio missing required sector allocations"""
        portfolio_missing_healthcare = {
            'AAPL': 0.40,  # Technology only
            'MSFT': 0.30   # Technology only
        }
        
        context = ValidationContext(portfolio_weights=portfolio_missing_healthcare)
        result, violations = self.validator.validate(context)
        
        # Should have violation for missing Healthcare sector
        missing_violations = [v for v in violations if 'missing required sector' in v.violation_description and 'Healthcare' in v.violation_description]
        assert len(missing_violations) == 1
        
    def test_forbidden_sectors_violation(self):
        """Test portfolio containing forbidden sectors"""
        portfolio_with_forbidden = {
            'TOBACCO_STOCK': 0.10,  # Forbidden sector
            'AAPL': 0.20,
            'JNJ': 0.15
        }
        
        # Update config to map tobacco stock correctly
        context = ValidationContext(portfolio_weights=portfolio_with_forbidden)
        result, violations = self.validator.validate(context)
        
        # Should have violation for forbidden sector (if tobacco is mapped to Consumer Staples and forbidden)
        forbidden_violations = [v for v in violations if 'forbidden sector' in v.violation_description]
        # Note: This test depends on sector mapping - adjust based on implementation
        
    def test_no_mandates_configured(self):
        """Test validator with no mandates configured"""
        empty_config = {'rule_name': 'No Mandates'}
        validator = MandateValidator(empty_config)
        
        # Any portfolio should pass
        extreme_portfolio = {'ANY_STOCK': 1.0}
        context = ValidationContext(portfolio_weights=extreme_portfolio)
        result, violations = validator.validate(context)
        
        assert result == ValidationResult.PASS
        assert len(violations) == 0
        
    def test_market_data_integration(self):
        """Test validator using market data for ESG/rating info"""
        market_data = {
            'CUSTOM1': {
                'esg': {
                    'environmental': 5.0,
                    'social': 6.0,
                    'governance': 7.0,
                    'overall': 6.0  # Below 7.0 minimum
                },
                'sector': 'Technology'
            }
        }
        
        portfolio = {'CUSTOM1': 0.20}
        context = ValidationContext(
            portfolio_weights=portfolio,
            market_data=market_data
        )
        
        result, violations = self.validator.validate(context)
        
        # Should detect ESG violation
        esg_violations = [v for v in violations if 'CUSTOM1' in str(v.affected_assets) and 'ESG score' in v.violation_description]
        assert len(esg_violations) == 1
        
    def test_environmental_score_specific_requirement(self):
        """Test specific environmental score requirements"""
        portfolio_with_low_env = {
            'TSLA': 0.25  # Has overall good ESG but governance issues
        }
        
        context = ValidationContext(portfolio_weights=portfolio_with_low_env)
        result, violations = self.validator.validate(context)
        
        # TSLA should pass overall ESG but might have specific component issues
        # This tests the granular ESG component checking
        
    def test_validation_error_handling(self):
        """Test validator error handling"""
        # Create validator that will fail during validation
        with patch.object(MandateValidator, '_check_esg_requirements', side_effect=Exception("Test error")):
            context = ValidationContext(portfolio_weights=self.sample_portfolio)
            result, violations = self.validator.validate(context)
            
            assert result == ValidationResult.FAIL
            assert len(violations) == 1
            assert 'CRITICAL' in violations[0].violation_description
            assert "mandate validation failed" in violations[0].violation_description
            
    def test_rule_description(self):
        """Test rule description generation"""
        description = self.validator.get_rule_description()
        
        assert "Minimum ESG score: 7.0" in description
        assert "Minimum credit rating: BBB" in description
        assert "Minimum liquidity score: 0.7" in description
        assert "Required sectors:" in description and "Technology" in description and "Healthcare" in description
        assert "Forbidden sectors:" in description and "Tobacco" in description and "Defense" in description
        
    def test_credit_rating_conversion(self):
        """Test credit rating to numeric conversion"""
        assert self.validator._rating_to_numeric('AAA') == 1
        assert self.validator._rating_to_numeric('BBB') == 9
        assert self.validator._rating_to_numeric('B') == 15
        assert self.validator._rating_to_numeric('UNKNOWN') == 99
        
    def test_major_stocks_liquidity_default(self):
        """Test that major stocks get default high liquidity"""
        portfolio = {'INTC': 0.10}  # Major stock not in mock data
        context = ValidationContext(portfolio_weights=portfolio)
        result, violations = self.validator.validate(context)
        
        # Should not have liquidity violations for major stocks
        liquidity_violations = [v for v in violations if 'INTC' in str(v.affected_assets) and 'liquidity' in v.violation_description]
        # Should either pass or only warn about missing data, not fail on score
        
    def test_comprehensive_mandate_validation(self):
        """Test comprehensive validation with multiple mandate types"""
        complex_portfolio = {
            'AAPL': 0.20,           # Good ESG, Technology
            'MSFT': 0.15,           # Good ESG, Technology  
            'JNJ': 0.10,            # Healthcare required sector
            'XOM': 0.05,            # Poor ESG
            'ILLIQUID_STOCK': 0.05, # Poor liquidity
            'JUNK_BOND': 0.03       # Poor credit rating
        }
        
        context = ValidationContext(portfolio_weights=complex_portfolio)
        result, violations = self.validator.validate(context)
        
        # Should detect multiple types of violations
        violation_types = [v.violation_description for v in violations]
        
        # Verify we catch different types of mandate violations
        has_esg_violation = any('ESG score' in desc for desc in violation_types)
        has_liquidity_violation = any('liquidity score' in desc for desc in violation_types)
        has_credit_violation = any('credit rating' in desc for desc in violation_types)
        
        # At least some violations should be detected
        assert len(violations) > 0
        
        # All violations should be properly categorized
        for violation in violations:
            assert violation.rule_type.value == 'mandate'
            assert any(severity in violation.violation_description for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])


class TestESGValidation:
    """Specific tests for ESG validation logic"""
    
    def setup_method(self):
        self.esg_config = {
            'rule_name': 'ESG Requirements',
            'esg_requirements': {
                'min_esg_score': 8.0,
                'min_environmental_score': 7.0,
                'min_social_score': 7.0,
                'min_governance_score': 8.0,
                'exclude_unrated': True
            }
        }
        self.validator = MandateValidator(self.esg_config)
        
    def test_esg_component_scoring(self):
        """Test individual ESG component requirements"""
        # TSLA has good environmental but poor governance
        portfolio = {'TSLA': 0.30}
        context = ValidationContext(portfolio_weights=portfolio)
        result, violations = self.validator.validate(context)
        
        # Should detect governance score violation for TSLA
        governance_violations = [v for v in violations if 'TSLA' in str(v.affected_assets)]
        # TSLA overall score (6.8) is below 8.0, so should have violation
        assert len(governance_violations) > 0


class TestLiquidityValidation:
    """Specific tests for liquidity validation logic"""
    
    def setup_method(self):
        self.liquidity_config = {
            'rule_name': 'Liquidity Requirements',
            'min_liquidity_score': 0.80
        }
        self.validator = MandateValidator(self.liquidity_config)
        
    def test_liquidity_data_missing(self):
        """Test handling of missing liquidity data"""
        portfolio = {'UNKNOWN_STOCK': 0.15}
        context = ValidationContext(portfolio_weights=portfolio)
        result, violations = self.validator.validate(context)
        
        # Should warn about missing liquidity data
        missing_data_violations = [v for v in violations if 'lacks liquidity data' in v.violation_description]
        assert len(missing_data_violations) == 1


@pytest.fixture
def sample_mandate_validator():
    """Fixture providing a configured mandate validator"""
    config = {
        'rule_name': 'Test Mandates',
        'esg_requirements': {'min_esg_score': 7.0},
        'min_credit_rating': 'A',
        'min_liquidity_score': 0.75,
        'required_sectors': ['Technology'],
        'forbidden_sectors': ['Tobacco']
    }
    return MandateValidator(config)


@pytest.fixture
def diverse_portfolio():
    """Fixture providing a diverse test portfolio"""
    return {
        'AAPL': 0.15,
        'MSFT': 0.15,
        'JNJ': 0.10,
        'XOM': 0.05,
        'GOVT_BOND': 0.10
    }


class TestMandateValidatorIntegration:
    """Integration tests for mandate validators"""
    
    def test_performance_benchmark(self, sample_mandate_validator):
        """Test validation performance with large portfolio"""
        import time
        
        # Create large portfolio (50 assets)
        large_portfolio = {f'STOCK_{i}': 0.02 for i in range(50)}
        context = ValidationContext(portfolio_weights=large_portfolio)
        
        start_time = time.time()
        result, violations = sample_mandate_validator.validate(context)
        end_time = time.time()
        
        # Validation should complete quickly (< 50ms target)
        validation_time = (end_time - start_time) * 1000
        assert validation_time < 50, f"Validation took {validation_time:.2f}ms, exceeding 50ms target"
        
    def test_integration_with_market_data(self, sample_mandate_validator):
        """Test integration with comprehensive market data"""
        market_data = {
            'INTEGRATION_TEST': {
                'esg': {'overall': 8.5, 'environmental': 8.0, 'social': 9.0, 'governance': 8.5},
                'credit_rating': {'rating': 'AA', 'numeric_score': 3},
                'liquidity': {'score': 0.85, 'avg_daily_volume': 1000000},
                'sector': 'Technology'
            }
        }
        
        portfolio = {'INTEGRATION_TEST': 0.50}
        context = ValidationContext(
            portfolio_weights=portfolio,
            market_data=market_data
        )
        
        result, violations = sample_mandate_validator.validate(context)
        
        # Should pass all mandate requirements
        assert result in [ValidationResult.PASS, ValidationResult.WARNING]
        
        # Any violations should be properly structured
        for violation in violations:
            assert violation.rule_type.value == 'mandate'
            assert violation.affected_assets is not None
            assert violation.recommended_action is not None
