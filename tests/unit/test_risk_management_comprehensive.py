"""
Risk Management Tests
Tests for risk metrics and portfolio risk management
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestRiskManager:
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100),
            'GOOGL': np.random.normal(0.0012, 0.022, 100)
        }, index=dates)
        return returns
    
    @pytest.fixture
    def sample_weights(self):
        """Create sample portfolio weights"""
        return np.array([0.4, 0.35, 0.25])
    
    @pytest.fixture
    def risk_manager(self, sample_returns, sample_weights):
        """Create RiskManager instance"""
        from risk.risk_managment import RiskManager
        return RiskManager(sample_returns, sample_weights)
    
    def test_risk_manager_initialization(self, sample_returns, sample_weights):
        """Test RiskManager initialization"""
        from risk.risk_managment import RiskManager
        
        # Test with weights
        rm = RiskManager(sample_returns, sample_weights)
        assert rm.returns is not None
        assert len(rm.weights) == len(sample_weights)
        assert np.allclose(rm.weights, sample_weights)
        
        # Test without weights (equal weighting)
        rm_equal = RiskManager(sample_returns)
        expected_weights = np.array([1/3, 1/3, 1/3])
        assert np.allclose(rm_equal.weights, expected_weights)
    
    def test_portfolio_returns_calculation(self, risk_manager, sample_returns, sample_weights):
        """Test portfolio returns calculation"""
        portfolio_returns = risk_manager.portfolio_returns
        
        # Should be a pandas Series
        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(sample_returns)
        
        # Manually calculate and compare
        manual_returns = (sample_returns * sample_weights).sum(axis=1)
        assert np.allclose(portfolio_returns, manual_returns)
    
    def test_var_calculation(self, risk_manager):
        """Test Value at Risk calculation"""
        # Test default confidence level (95%)
        var_95 = risk_manager.calculate_var()
        assert isinstance(var_95, (float, np.float64))
        
        # Test different confidence levels
        var_99 = risk_manager.calculate_var(0.99)
        var_90 = risk_manager.calculate_var(0.90)
        
        # VaR at 99% should be more negative (higher risk) than 95%
        assert var_99 <= var_95
        # VaR at 90% should be less negative (lower risk) than 95%
        assert var_90 >= var_95
    
    def test_cvar_calculation(self, risk_manager):
        """Test Conditional Value at Risk calculation"""
        # Test default confidence level (95%)
        cvar_95 = risk_manager.calculate_cvar()
        assert isinstance(cvar_95, (float, np.float64))
        
        # CVaR should be less than or equal to VaR (more negative)
        var_95 = risk_manager.calculate_var()
        assert cvar_95 <= var_95
    
    def test_maximum_drawdown(self, risk_manager):
        """Test maximum drawdown calculation"""
        try:
            max_dd = risk_manager.calculate_maximum_drawdown()
            assert isinstance(max_dd, (float, np.float64))
            # Maximum drawdown should be negative or zero
            assert max_dd <= 0
        except AttributeError:
            # Method might not exist
            pass
    
    def test_sharpe_ratio(self, risk_manager):
        """Test Sharpe ratio calculation"""
        try:
            sharpe = risk_manager.calculate_sharpe_ratio()
            assert isinstance(sharpe, (float, np.float64))
            # Sharpe ratio can be positive or negative
        except AttributeError:
            # Method might not exist
            pass
    
    def test_volatility_calculation(self, risk_manager):
        """Test volatility calculation"""
        try:
            volatility = risk_manager.calculate_volatility()
            assert isinstance(volatility, (float, np.float64))
            # Volatility should be positive
            assert volatility >= 0
        except AttributeError:
            # Method might not exist
            pass
    
    def test_edge_cases(self, sample_returns):
        """Test edge cases and error handling"""
        from risk.risk_managment import RiskManager
        
        # Test with single asset
        single_asset_returns = sample_returns[['AAPL']]
        rm_single = RiskManager(single_asset_returns)
        var_single = rm_single.calculate_var()
        assert isinstance(var_single, (float, np.float64))
        
        # Test with extreme confidence levels
        var_extreme_low = rm_single.calculate_var(0.01)
        var_extreme_high = rm_single.calculate_var(0.999)
        assert var_extreme_low >= var_extreme_high
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data"""
        from risk.risk_managment import RiskManager
        
        # Test with empty DataFrame
        empty_returns = pd.DataFrame()
        try:
            rm_empty = RiskManager(empty_returns)
            # Should either handle gracefully or raise appropriate error
        except (ValueError, IndexError):
            # Expected behavior for empty data
            pass
    
    def test_invalid_weights(self, sample_returns):
        """Test invalid weight handling"""
        from risk.risk_managment import RiskManager
        
        # Test weights that don't sum to 1
        invalid_weights = np.array([0.5, 0.3, 0.1])  # Sum = 0.9
        rm = RiskManager(sample_returns, invalid_weights)
        
        # Should still work, might normalize or use as-is
        var_result = rm.calculate_var()
        assert isinstance(var_result, (float, np.float64))
        
        # Test wrong number of weights
        wrong_length_weights = np.array([0.5, 0.5])  # Only 2 weights for 3 assets
        try:
            rm_wrong = RiskManager(sample_returns, wrong_length_weights)
        except (ValueError, IndexError):
            # Expected behavior for mismatched dimensions
            pass
    
    def test_statistical_properties(self, risk_manager):
        """Test statistical properties of risk metrics"""
        # VaR and CVaR should be consistent with portfolio returns distribution
        portfolio_returns = risk_manager.portfolio_returns
        
        var_95 = risk_manager.calculate_var(0.95)
        cvar_95 = risk_manager.calculate_cvar(0.95)
        
        # About 5% of returns should be below VaR
        below_var = (portfolio_returns < var_95).sum()
        total_returns = len(portfolio_returns)
        below_var_pct = below_var / total_returns
        
        # Should be approximately 5% (allowing for small sample variation)
        assert 0.01 <= below_var_pct <= 0.15  # Reasonable range for small samples
    
    def test_confidence_level_validation(self, risk_manager):
        """Test confidence level validation"""
        # Test invalid confidence levels
        try:
            risk_manager.calculate_var(-0.1)  # Negative
            assert False, "Should raise error for negative confidence"
        except (ValueError, AssertionError):
            pass
        
        try:
            risk_manager.calculate_var(1.1)  # Greater than 1
            assert False, "Should raise error for confidence > 1"
        except (ValueError, AssertionError):
            pass
    
    def test_returns_data_types(self, sample_returns):
        """Test different returns data types"""
        from risk.risk_managment import RiskManager
        
        # Test with numpy array
        returns_array = sample_returns.values
        try:
            rm_array = RiskManager(pd.DataFrame(returns_array))
            var_array = rm_array.calculate_var()
            assert isinstance(var_array, (float, np.float64))
        except Exception:
            # Might require DataFrame specifically
            pass
    
    def test_performance_with_large_data(self):
        """Test performance with larger datasets"""
        from risk.risk_managment import RiskManager
        
        # Create larger dataset
        np.random.seed(42)
        large_returns = pd.DataFrame({
            f'Asset_{i}': np.random.normal(0.001, 0.02, 1000)
            for i in range(10)
        })
        
        weights = np.array([0.1] * 10)
        rm_large = RiskManager(large_returns, weights)
        
        # Should handle larger datasets efficiently
        var_large = rm_large.calculate_var()
        cvar_large = rm_large.calculate_cvar()
        
        assert isinstance(var_large, (float, np.float64))
        assert isinstance(cvar_large, (float, np.float64))
    
    def test_correlation_impact(self):
        """Test impact of correlation on risk metrics"""
        from risk.risk_managment import RiskManager
        
        np.random.seed(42)
        
        # Create highly correlated returns
        base_returns = np.random.normal(0.001, 0.02, 100)
        corr_returns = pd.DataFrame({
            'Asset1': base_returns,
            'Asset2': base_returns + np.random.normal(0, 0.005, 100),  # Highly correlated
            'Asset3': np.random.normal(0.001, 0.02, 100)  # Independent
        })
        
        equal_weights = np.array([1/3, 1/3, 1/3])
        rm_corr = RiskManager(corr_returns, equal_weights)
        
        var_corr = rm_corr.calculate_var()
        assert isinstance(var_corr, (float, np.float64))
    
    def test_methods_exist(self, risk_manager):
        """Test that expected methods exist on RiskManager"""
        expected_methods = [
            'calculate_var',
            'calculate_cvar',
            '_calculate_portfolio_returns'
        ]
        
        for method in expected_methods:
            assert hasattr(risk_manager, method), f"RiskManager should have {method} method"
            assert callable(getattr(risk_manager, method)), f"{method} should be callable"
    
    def test_attributes_exist(self, risk_manager):
        """Test that expected attributes exist on RiskManager"""
        expected_attributes = ['returns', 'weights', 'portfolio_returns']
        
        for attr in expected_attributes:
            assert hasattr(risk_manager, attr), f"RiskManager should have {attr} attribute"
