"""
Quick test to ensure all components are working
Run this after setup to verify installation
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf
        import cvxpy as cp
        import xgboost as xgb
        import torch
        import fastapi
        print("✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_data_fetch():
    """Test market data fetching"""
    print("\nTesting data fetch...")
    
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1mo")
        if not hist.empty:
            print(f"✓ Fetched {len(hist)} days of AAPL data")
            return True
        else:
            print("✗ No data fetched")
            return False
    except Exception as e:
        print(f"✗ Data fetch failed: {e}")
        return False

def test_optimization():
    """Test basic portfolio optimization"""
    print("\nTesting portfolio optimization...")
    
    try:
        import numpy as np
        import cvxpy as cp
        
        # Simple 3-asset optimization
        returns = np.array([0.10, 0.12, 0.08])
        cov_matrix = np.array([
            [0.05, 0.01, 0.02],
            [0.01, 0.06, 0.01],
            [0.02, 0.01, 0.04]
        ])
        
        weights = cp.Variable(3)
        portfolio_return = returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        objective = cp.Maximize(portfolio_return - 2 * portfolio_risk)
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= 0.5
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            print(f"✓ Optimization successful")
            print(f"  Optimal weights: {weights.value.round(3)}")
            return True
        else:
            print("✗ Optimization failed")
            return False
            
    except Exception as e:
        print(f"✗ Optimization error: {e}")
        return False

def test_ml_model():
    """Test ML model training"""
    print("\nTesting ML model...")
    
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate dummy data
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Make prediction
        pred = model.predict(X[:5])
        
        print(f"✓ ML model trained successfully")
        print(f"  Feature importance shape: {model.feature_importances_.shape}")
        return True
        
    except Exception as e:
        print(f"✗ ML model error: {e}")
        return False

def test_api_setup():
    """Test FastAPI setup"""
    print("\nTesting API setup...")
    
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        
        app = FastAPI()
        
        class TestModel(BaseModel):
            value: float
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        print("✓ FastAPI setup successful")
        return True
        
    except Exception as e:
        print(f"✗ API setup error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("QUANTUM PORTFOLIO OPTIMIZER - SYSTEM TEST")
    print("="*50)
    
    tests = [
        test_imports,
        test_data_fetch,
        test_optimization,
        test_ml_model,
        test_api_setup
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - System ready!")
        print("\nNext steps:")
        print("1. Add your API keys to .env file")
        print("2. Run: python src/portfolio/ml_portfolio_optimizer.py")
        print("3. Start API: python src/api/main.py")
        print("4. Deploy to cloud (Heroku/AWS)")
    else:
        print("✗ Some tests failed - check requirements")
        print("Run: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)