"""
Integration tests for the complete system
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.portfolio.portfolio_optimizer import PortfolioOptimizer

class TestIntegration:
    
    @patch('yfinance.download')
    def test_full_optimization_pipeline(self, mock_download):
        """Test complete optimization pipeline"""
        # Mock Yahoo Finance data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')
        mock_prices = pd.DataFrame({
            'AAPL': np.random.uniform(140, 160, 500),
            'MSFT': np.random.uniform(280, 320, 500),
            'GOOGL': np.random.uniform(2600, 2900, 500)
        }, index=dates)
        mock_download.return_value = mock_prices
        
        # Run optimization
        optimizer = PortfolioOptimizer(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            lookback_years=2,
            risk_free_rate=0.04,
            max_position_size=0.5
        )
        
        result = optimizer.run(method='max_sharpe')
        
        # Validate result
        assert result is not None
        assert 'weights' in result
        assert 'metrics' in result
        assert 'tickers' in result
        
        # Check weights
        assert len(result['weights']) == 3
        assert np.isclose(sum(result['weights']), 1.0)
        assert all(0 <= w <= 0.5 for w in result['weights'])
        
        # Check metrics
        assert 'return' in result['metrics']
        assert 'volatility' in result['metrics']
        assert 'sharpe' in result['metrics']
        
    def test_different_methods_produce_different_weights(self):
        """Test that different optimization methods produce different results"""
        optimizer = PortfolioOptimizer(['AAPL', 'MSFT', 'GOOGL'])
        
        # Mock data
        optimizer.returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.01 + 0.001,
            columns=['AAPL', 'MSFT', 'GOOGL']
        )
        optimizer.prices = pd.DataFrame(
            np.abs(np.random.randn(100, 3)) * 100 + 100,
            columns=['AAPL', 'MSFT', 'GOOGL']
        )
        
        results = {}
        for method in ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']:
            results[method] = optimizer.run(method=method)
        
        # Equal weight should be exactly equal
        equal_weights = results['equal_weight']['weights']
        assert all(np.isclose(w, 1/3) for w in equal_weights)
        
        # Other methods should produce different weights
        max_sharpe_weights = results['max_sharpe']['weights']
        min_var_weights = results['min_variance']['weights']
        
        # Weights should be different (allowing for small numerical differences)
        assert not np.allclose(max_sharpe_weights, min_var_weights, rtol=0.01)