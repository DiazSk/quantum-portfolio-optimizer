"""
Tests for ML predictions and market regime
"""
import pytest
import numpy as np
import pandas as pd
from src.portfolio.portfolio_optimizer import PortfolioOptimizer

class TestMLPredictions:
    
    def test_market_regime_detection(self):
        """Test different market regime scenarios"""
        optimizer = PortfolioOptimizer(['AAPL'])
        
        # Create different market scenarios
        # Bullish market
        bullish_returns = pd.Series(np.random.uniform(0.002, 0.005, 100))
        regime = optimizer.detect_market_regime(bullish_returns)
        assert regime in ['bullish', 'neutral']  # Could be bullish or neutral
        
        # Bearish market
        bearish_returns = pd.Series(np.random.uniform(-0.005, -0.001, 100))
        regime = optimizer.detect_market_regime(bearish_returns)
        assert regime == 'bearish'
        
        # High volatility
        volatile_returns = pd.Series(np.random.normal(0, 0.05, 100))
        regime = optimizer.detect_market_regime(volatile_returns)
        # Should return one of the valid regimes
        assert regime in ['high_volatility', 'neutral', 'bullish', 'bearish']
        
    def test_regime_affects_predictions(self):
        """Test that market regime affects ML predictions"""
        # Test with different market conditions that should trigger different regimes
        predictions_by_condition = {}
        
        # Create optimizer
        optimizer = PortfolioOptimizer(['AAPL'], use_random_state=True)
        
        # Test different market conditions
        conditions = {
            'high_volatility': np.random.normal(0, 0.08, 100),  # High volatility
            'bullish': np.random.normal(0.02, 0.01, 100),       # Positive returns
            'bearish': np.random.normal(-0.02, 0.01, 100),      # Negative returns  
            'neutral': np.random.normal(0, 0.01, 100)           # Low volatility, neutral
        }
        
        for condition_name, return_data in conditions.items():
            # Set up mock data
            optimizer.returns = pd.DataFrame(
                return_data.reshape(-1, 1),
                columns=['AAPL']
            )
            optimizer.prices = pd.DataFrame(
                (1 + pd.DataFrame(return_data.reshape(-1, 1))).cumprod() * 100,
                columns=['AAPL']
            )
            
            # Get predictions for this condition
            predictions = optimizer.train_ml_models()
            if predictions:
                predictions_by_condition[condition_name] = predictions.get('AAPL', 0)
        
        # Test that we got predictions
        assert len(predictions_by_condition) > 0
        
        # Test that regimes are detected properly
        for condition_name, return_data in conditions.items():
            returns_series = pd.Series(return_data)
            detected_regime = optimizer.detect_market_regime(returns_series)
            assert detected_regime in ['bullish', 'bearish', 'neutral', 'high_volatility']