"""
Enhanced Portfolio Optimizer with Advanced Ensemble ML Pipeline
Integrates the new ensemble models with the existing portfolio optimization system
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import existing components
from src.portfolio.portfolio_optimizer import PortfolioOptimizer
from src.models.advanced_ensemble_pipeline import AdvancedEnsembleManager, AdvancedFeatureEngineer, EnsembleModelValidator
from src.utils.professional_logging import get_logger

logger = get_logger(__name__)

class EnhancedPortfolioOptimizer(PortfolioOptimizer):
    """
    Enhanced portfolio optimizer that uses the advanced ensemble ML pipeline
    for return predictions and portfolio optimization.
    """
    
    def __init__(self, tickers, lookback_years=2, risk_free_rate=0.04, 
                 max_position_size=0.60, use_ensemble=True, ensemble_path=None):
        super().__init__(tickers, lookback_years, risk_free_rate, max_position_size)
        
        self.use_ensemble = use_ensemble
        self.ensemble_path = ensemble_path or "models/advanced_ensemble.pkl"
        self.ensemble_manager = None
        self.ensemble_predictions = {}
        self.ensemble_confidence = {}
        
        if self.use_ensemble:
            self._initialize_ensemble()
    
    def _initialize_ensemble(self):
        """Initialize the ensemble ML system"""
        try:
            logger.info("Initializing advanced ensemble ML system...")
            
            feature_engineer = AdvancedFeatureEngineer()
            validator = EnsembleModelValidator()
            self.ensemble_manager = AdvancedEnsembleManager(feature_engineer, validator)
            
            # Try to load existing ensemble
            if os.path.exists(self.ensemble_path):
                logger.info(f"Loading existing ensemble from {self.ensemble_path}")
                self.ensemble_manager.load_ensemble(self.ensemble_path)
            else:
                logger.info("No existing ensemble found. Will train new ensemble during optimization.")
                
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {e}")
            self.use_ensemble = False
    
    def train_ensemble_models(self):
        """Train ensemble models for all tickers"""
        if not self.use_ensemble or self.prices is None:
            return
        
        logger.info("Training ensemble models for all tickers...")
        
        for ticker in self.tickers:
            if ticker in self.prices.columns:
                try:
                    logger.info(f"Training ensemble for {ticker}...")
                    
                    # Train ensemble for this ticker
                    training_results = self.ensemble_manager.train_ensemble(
                        self.prices, self.returns, ticker
                    )
                    
                    # Save ensemble with ticker-specific name
                    ticker_ensemble_path = f"models/ensemble_{ticker}.pkl"
                    os.makedirs("models", exist_ok=True)
                    self.ensemble_manager.save_ensemble(ticker_ensemble_path)
                    
                    logger.info(f"Ensemble training complete for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Failed to train ensemble for {ticker}: {e}")
    
    def get_ensemble_predictions(self) -> Dict[str, float]:
        """Get predictions from the ensemble for all tickers"""
        if not self.use_ensemble or not self.ensemble_manager:
            return {}
        
        predictions = {}
        
        try:
            # Generate features for current data
            features = self.ensemble_manager.feature_engineer.engineer_features(
                self.prices, self.returns
            )
            
            if features.empty:
                logger.warning("No features generated for ensemble prediction")
                return {}
            
            # Get latest features for prediction
            latest_features = features.tail(1)
            
            for ticker in self.tickers:
                try:
                    # Load ticker-specific ensemble if it exists
                    ticker_ensemble_path = f"models/ensemble_{ticker}.pkl"
                    if os.path.exists(ticker_ensemble_path):
                        # Create temporary ensemble manager for this ticker
                        ticker_ensemble = AdvancedEnsembleManager()
                        ticker_ensemble.load_ensemble(ticker_ensemble_path)
                        
                        # Make prediction
                        prediction = ticker_ensemble.predict_ensemble(latest_features)
                        predictions[ticker] = float(prediction[0])
                        
                        # Get confidence score (inverse of ensemble variance)
                        model_predictions = []
                        for model_name, model in ticker_ensemble.trained_models.items():
                            if model_name in ticker_ensemble.ensemble_weights:
                                pred = model.predict(latest_features)
                                model_predictions.append(pred[0])
                        
                        if model_predictions:
                            confidence = 1.0 / (np.var(model_predictions) + 1e-6)
                            self.ensemble_confidence[ticker] = min(confidence, 10.0)  # Cap confidence
                        else:
                            self.ensemble_confidence[ticker] = 1.0
                            
                        logger.debug(f"Ensemble prediction for {ticker}: {predictions[ticker]:.6f} (confidence: {self.ensemble_confidence[ticker]:.2f})")
                        
                except Exception as e:
                    logger.warning(f"Failed to get ensemble prediction for {ticker}: {e}")
                    # Fallback to traditional prediction
                    predictions[ticker] = self.get_traditional_prediction(ticker)
                    self.ensemble_confidence[ticker] = 0.5
            
            self.ensemble_predictions = predictions
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to generate ensemble predictions: {e}")
            return {}
    
    def get_traditional_prediction(self, ticker: str) -> float:
        """Fallback to traditional prediction method"""
        if ticker in self.returns.columns and len(self.returns) > 30:
            # Simple momentum-based prediction
            recent_return = self.returns[ticker].tail(10).mean()
            return float(recent_return)
        return 0.0
    
    def calculate_ensemble_enhanced_expected_returns(self):
        """Calculate expected returns using ensemble predictions"""
        expected_returns = np.zeros(len(self.tickers))
        
        # Get ensemble predictions
        ensemble_preds = self.get_ensemble_predictions()
        
        for i, ticker in enumerate(self.tickers):
            if ticker in ensemble_preds:
                # Use ensemble prediction
                ensemble_pred = ensemble_preds[ticker]
                confidence = self.ensemble_confidence.get(ticker, 1.0)
                
                # Traditional momentum-based prediction
                traditional_pred = self.get_traditional_prediction(ticker)
                
                # Blend ensemble and traditional predictions based on confidence
                # Higher confidence = more weight on ensemble
                ensemble_weight = min(confidence / 2.0, 0.8)  # Cap at 80% ensemble weight
                traditional_weight = 1.0 - ensemble_weight
                
                expected_return = (ensemble_weight * ensemble_pred + 
                                 traditional_weight * traditional_pred)
                
                expected_returns[i] = expected_return
                
                logger.debug(f"{ticker}: ensemble={ensemble_pred:.6f}, traditional={traditional_pred:.6f}, "
                           f"final={expected_return:.6f} (conf={confidence:.2f})")
            else:
                # Fallback to traditional method
                expected_returns[i] = self.get_traditional_prediction(ticker)
        
        return expected_returns
    
    def run_enhanced_optimization(self):
        """Run portfolio optimization with ensemble-enhanced predictions"""
        logger.info("Starting enhanced portfolio optimization with ensemble ML...")
        
        # Fetch data
        if self.fetch_data() is None:
            logger.error("Failed to fetch market data")
            return None
        
        # Train ensemble models if not already trained
        if self.use_ensemble and self.ensemble_manager:
            # Check if we need to train models
            trained_models_exist = all(
                os.path.exists(f"models/ensemble_{ticker}.pkl") 
                for ticker in self.tickers
            )
            
            if not trained_models_exist:
                logger.info("Training ensemble models for the first time...")
                self.train_ensemble_models()
        
        # Calculate expected returns using ensemble
        if self.use_ensemble:
            expected_returns = self.calculate_ensemble_enhanced_expected_returns()
        else:
            # Fallback to traditional method
            expected_returns = np.array([self.get_traditional_prediction(ticker) for ticker in self.tickers])
        
        # Calculate covariance matrix
        cov_matrix = self.returns.cov().values
        
        # Portfolio optimization
        n_assets = len(self.tickers)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}
        ]
        
        # Bounds (0 to max_position_size for each asset)
        bounds = [(0.0, self.max_position_size) for _ in range(n_assets)]
        
        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return -1000  # Penalty for zero volatility
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate/252) / portfolio_vol
            return -sharpe_ratio  # Negative because we minimize
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimization
        from scipy.optimize import minimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights * expected_returns) * 252
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))) * np.sqrt(252)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Create result dictionary
            optimization_result = {
                'success': True,
                'tickers': self.tickers,
                'weights': optimal_weights.tolist(),
                'expected_returns': expected_returns.tolist(),
                'expected_annual_return': portfolio_return,
                'annual_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'ensemble_used': self.use_ensemble,
                'ensemble_predictions': self.ensemble_predictions,
                'ensemble_confidence': self.ensemble_confidence
            }
            
            logger.info("Enhanced portfolio optimization completed successfully")
            logger.info(f"Expected Annual Return: {portfolio_return:.4f}")
            logger.info(f"Annual Volatility: {portfolio_vol:.4f}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            logger.info(f"Ensemble Used: {self.use_ensemble}")
            
            return optimization_result
            
        else:
            logger.error(f"Portfolio optimization failed: {result.message}")
            return None


def run_enhanced_portfolio_demo():
    """
    Demonstration of the enhanced portfolio optimizer with ensemble ML
    """
    print("\n" + "="*80)
    print("🚀 ENHANCED PORTFOLIO OPTIMIZER WITH ENSEMBLE ML")
    print("="*80 + "\n")
    
    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
    
    try:
        # Initialize enhanced optimizer
        print("🤖 Initializing Enhanced Portfolio Optimizer...")
        optimizer = EnhancedPortfolioOptimizer(
            tickers=tickers,
            lookback_years=2,
            use_ensemble=True
        )
        
        # Run optimization
        print("\n🔥 Running Enhanced Portfolio Optimization...")
        result = optimizer.run_enhanced_optimization()
        
        if result and result['success']:
            print("\n✅ OPTIMIZATION SUCCESSFUL!")
            print("=" * 50)
            
            # Display portfolio weights
            print("\n📊 OPTIMAL PORTFOLIO WEIGHTS:")
            for ticker, weight in zip(result['tickers'], result['weights']):
                print(f"  {ticker}: {weight:.4f} ({weight*100:.2f}%)")
            
            # Display performance metrics
            print(f"\n📈 PORTFOLIO METRICS:")
            print(f"  Expected Annual Return: {result['expected_annual_return']:.4f} ({result['expected_annual_return']*100:.2f}%)")
            print(f"  Annual Volatility:      {result['annual_volatility']:.4f} ({result['annual_volatility']*100:.2f}%)")
            print(f"  Sharpe Ratio:           {result['sharpe_ratio']:.4f}")
            
            # Display ensemble information
            if result['ensemble_used']:
                print(f"\n🤖 ENSEMBLE ML INFORMATION:")
                print(f"  Ensemble Models Used: ✅ YES")
                
                if result['ensemble_predictions']:
                    print(f"  Ensemble Predictions:")
                    for ticker, pred in result['ensemble_predictions'].items():
                        conf = result['ensemble_confidence'].get(ticker, 0)
                        print(f"    {ticker}: {pred:.6f} (confidence: {conf:.2f})")
            else:
                print(f"\n🤖 ENSEMBLE ML INFORMATION:")
                print(f"  Ensemble Models Used: ❌ NO (fallback to traditional)")
            
            print("\n🎯 Enhanced Portfolio Optimization Complete!")
            print("🚀 Ready for FAANG demonstration and technical interviews!")
            
            return True
            
        else:
            print("\n❌ Portfolio optimization failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Enhanced portfolio optimization failed: {e}")
        logger.error(f"Enhanced portfolio demo error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the demo
    success = run_enhanced_portfolio_demo()
    
    if success:
        print("\n🚀 Enhanced Portfolio Optimizer ready for production!")
    else:
        print("\n⚠️ System needs debugging before deployment")
