"""
Portfolio Optimizer - DYNAMIC ML VERSION
ML predictions now vary between runs
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Portfolio optimizer with dynamic ML predictions"""
    
    def __init__(self, tickers, lookback_years=2, risk_free_rate=0.04, max_position_size=0.60, use_random_state=False):
        self.tickers = tickers
        self.lookback_years = lookback_years
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.prices = None
        self.returns = None
        self.ml_models = {}
        self.use_random_state = use_random_state  # Control randomness
        self.market_regime = None
        self.ml_predictions = {}

    def detect_market_regime(self, returns):
        """Detect current market regime for adaptive predictions"""
        recent_returns = returns.tail(20).mean()
        recent_vol = returns.tail(20).std()
        
        if recent_vol > returns.std() * 1.5:
            return "high_volatility"
        elif recent_returns < 0:
            return "bearish"
        elif recent_returns > returns.mean() * 1.5:
            return "bullish"
        else:
            return "neutral"

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def fetch_data(self):
        """Download price data - FIXED VERSION"""
        print("📊 Fetching price data from Yahoo Finance...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.lookback_years)
        
        # Method 1: Try downloading all at once
        try:
            data = yf.download(
                self.tickers,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            # Extract close prices properly
            if len(self.tickers) == 1:
                self.prices = pd.DataFrame(data['Close'])
                self.prices.columns = self.tickers
            else:
                # Multiple tickers - check data structure
                if 'Close' in str(data.columns):
                    self.prices = data['Close']
                else:
                    self.prices = data
            
            # Ensure column names are strings, not tuples
            if isinstance(self.prices.columns[0], tuple):
                self.prices.columns = [col if isinstance(col, str) else col[0] for col in self.prices.columns]
            
            self.prices = self.prices.dropna(axis=1, how='all')
            self.returns = self.prices.pct_change().dropna()
            
            print(f"✅ Downloaded {len(self.prices)} days of data for {len(self.prices.columns)} assets")
            print(f"   Assets loaded: {', '.join([str(c) for c in self.prices.columns])}")
            
            return self.prices
            
        except Exception as e:
            print(f"   Method 1 failed: {str(e)[:50]}")
            print("   Trying individual ticker download...")
        
        # Method 2: Download each ticker individually (more reliable)
        all_data = []
        successful_tickers = []
        
        for ticker in self.tickers:
            try:
                print(f"   Downloading {ticker}...", end='')
                tick = yf.Ticker(ticker)
                hist = tick.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    all_data.append(hist['Close'])
                    successful_tickers.append(ticker)
                    print(" ✅")
                else:
                    print(" ❌ (no data)")
                    
            except Exception as e:
                print(f" ❌ ({str(e)[:30]})")
        
        if all_data:
            # Combine all series into a DataFrame
            self.prices = pd.DataFrame({
                ticker: data for ticker, data in zip(successful_tickers, all_data)
            })
            
            # Align all series to same index
            self.prices = self.prices.dropna(how='all')
            self.returns = self.prices.pct_change().dropna()
            
            print(f"\n✅ Successfully loaded {len(successful_tickers)} assets:")
            print(f"   {', '.join(successful_tickers)}")
            print(f"   Data points: {len(self.prices)} days")
            
            return self.prices
        else:
            print("❌ Could not fetch any data")
            return None
    
    def train_ml_models(self):
        """Train XGBoost with DYNAMIC predictions and market regime awareness"""
        if self.returns is None or self.returns.empty:
            print("❌ No data available for ML training")
            return {}
            
        print("\n🤖 Training ML models for return prediction...")
        
        # DETECT MARKET REGIME HERE
        market_regime = self.detect_market_regime(self.returns.mean(axis=1))
        if market_regime and isinstance(market_regime, str):
            print(f"   📊 Market Regime Detected: {market_regime.upper()}")
        else:
            market_regime = "neutral"
            print(f"   📊 Market Regime Detected: NEUTRAL (default)")
        
        predictions = {}
        
        # Generate a unique seed based on current time if not using fixed seed
        import time
        if not self.use_random_state:
            base_seed = None  # True randomness
            print("   Using dynamic ML predictions (random forest)")
        else:
            base_seed = 42
            print("   Using fixed random state for reproducibility")
        
        # ADJUST HYPERPARAMETERS BASED ON MARKET REGIME
        if market_regime == "high_volatility":
            # More conservative in volatile markets
            n_estimators_range = (40, 60)  # More trees for stability
            max_depth_range = (2, 3)  # Shallower trees
            learning_rate_range = (0.03, 0.08)  # Lower learning rate
            prediction_adjustment = 0.7  # Reduce prediction magnitude
            print("   ⚠️ High volatility mode: Using conservative parameters")
        elif market_regime == "bearish":
            # Defensive in bear markets
            n_estimators_range = (35, 50)
            max_depth_range = (2, 4)
            learning_rate_range = (0.05, 0.10)
            prediction_adjustment = 0.5  # More conservative predictions
            print("   🐻 Bearish mode: Using defensive parameters")
        elif market_regime == "bullish":
            # More aggressive in bull markets
            n_estimators_range = (25, 40)
            max_depth_range = (3, 5)
            learning_rate_range = (0.10, 0.20)
            prediction_adjustment = 1.2  # Slightly amplify positive predictions
            print("   🐂 Bullish mode: Using growth-oriented parameters")
        else:  # neutral
            # Standard parameters
            n_estimators_range = (25, 50)
            max_depth_range = (2, 5)
            learning_rate_range = (0.05, 0.20)
            prediction_adjustment = 1.0
            print("   ⚖️ Neutral mode: Using balanced parameters")
        
        for i, ticker in enumerate(self.prices.columns):
            try:
                # Create features with some randomness in feature engineering
                features = pd.DataFrame(index=self.prices.index)
                
                # Basic features
                features['returns_5d'] = self.returns[ticker].rolling(5).mean()
                features['returns_20d'] = self.returns[ticker].rolling(20).mean()
                features['volatility'] = self.returns[ticker].rolling(20).std()
                features['rsi'] = self.calculate_rsi(self.prices[ticker])
                
                # ADD REGIME-SPECIFIC FEATURES
                if market_regime == "high_volatility":
                    # Add more volatility-focused features
                    features['volatility_5d'] = self.returns[ticker].rolling(5).std()
                    features['volatility_ratio'] = features['volatility_5d'] / features['volatility']
                elif market_regime == "bearish":
                    # Add downside risk features
                    features['downside_vol'] = self.returns[ticker][self.returns[ticker] < 0].rolling(20).std()
                    features['max_drawdown_20d'] = self.returns[ticker].rolling(20).min()
                elif market_regime == "bullish":
                    # Add momentum features
                    features['momentum_10d'] = self.returns[ticker].rolling(10).mean()
                    features['momentum_30d'] = self.returns[ticker].rolling(30).mean()
                
                # Add random feature windows for diversity
                if not self.use_random_state:
                    momentum_window = np.random.randint(10, 30)
                    features[f'momentum_{momentum_window}'] = self.returns[ticker].rolling(momentum_window).mean()
                    
                    # Add volume-based feature if possible
                    vol_window = np.random.randint(5, 15)
                    features[f'vol_ratio_{vol_window}'] = (
                        self.returns[ticker].rolling(vol_window).std() / 
                        self.returns[ticker].rolling(vol_window * 2).std()
                    )
                
                # Target
                target = self.returns[ticker].shift(-1)
                
                # Clean data
                valid_mask = ~(features.isna().any(axis=1) | target.isna())
                X = features[valid_mask].values
                y = target[valid_mask].values
                
                if len(X) > 50:
                    # USE REGIME-ADJUSTED HYPERPARAMETERS
                    if not self.use_random_state:
                        n_estimators = np.random.randint(*n_estimators_range)
                        max_depth = np.random.randint(*max_depth_range)
                        learning_rate = np.random.uniform(*learning_rate_range)
                        subsample = np.random.uniform(0.6, 0.9)
                        
                        model_seed = None  # This ensures different results each run
                    else:
                        n_estimators = 30
                        max_depth = 3
                        learning_rate = 0.1
                        subsample = 0.8
                        model_seed = base_seed + i if base_seed else None
                    
                    # Train model with dynamic parameters
                    model = xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        random_state=model_seed,
                        verbosity=0
                    )
                    
                    # Random train/test split
                    if not self.use_random_state:
                        split_ratio = np.random.uniform(0.7, 0.85)
                    else:
                        split_ratio = 0.8
                    
                    split = int(len(X) * split_ratio)
                    model.fit(X[:split], y[:split])
                    
                    # Make prediction with regime adjustment
                    if len(features.dropna()) > 0:
                        last_features = features.dropna().iloc[-1:].values
                        pred = model.predict(last_features)[0]
                        
                        # APPLY REGIME-BASED ADJUSTMENT
                        pred = pred * prediction_adjustment
                        
                        # Add small market noise for realism
                        if not self.use_random_state:
                            # Regime-specific noise
                            if market_regime == "high_volatility":
                                market_noise = np.random.normal(0, 0.0005)  # Higher noise
                            else:
                                market_noise = np.random.normal(0, 0.0002)  # Normal noise
                            pred = pred + market_noise
                        
                        predictions[ticker] = pred
                        print(f"  {ticker}: Predicted return = {pred:+.3%} "
                              f"(n={n_estimators}, d={max_depth}, lr={learning_rate:.3f})")
                    else:
                        predictions[ticker] = self.returns[ticker].mean() * prediction_adjustment
                else:
                    # Not enough data, use historical mean with regime adjustment
                    base_return = self.returns[ticker].mean() * prediction_adjustment
                    if not self.use_random_state:
                        noise = np.random.normal(0, 0.0001)
                        predictions[ticker] = base_return + noise
                    else:
                        predictions[ticker] = base_return
                    print(f"  {ticker}: Using historical mean (regime-adjusted)")
                    
            except Exception as e:
                print(f"  {ticker}: Error - {str(e)[:30]}")
                predictions[ticker] = 0.0
        
        # Show prediction variance with regime info
        if predictions:
            pred_values = list(predictions.values())
            print(f"\n  📊 Prediction Summary ({market_regime} market):")
            print(f"     Mean: {np.mean(pred_values):+.3%}")
            print(f"     Std:  {np.std(pred_values):.3%}")
            print(f"     Range: [{min(pred_values):+.3%}, {max(pred_values):+.3%}]")
                
        return predictions
    
    def optimize_portfolio(self, expected_returns, cov_matrix, method='max_sharpe'):
        """Optimize portfolio with different methods"""
        n_assets = len(expected_returns)
        
        if method == 'equal_weight':
            # Equal weight portfolio
            return np.array([1/n_assets] * n_assets)
        
        elif method == 'risk_parity':
            # Risk parity - equal risk contribution
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                # Minimize variance of risk contributions
                return np.var(contrib)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            initial = np.array([1/n_assets] * n_assets)
            
            result = minimize(risk_parity_objective, initial, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x
        
        elif method == 'min_variance':
            # Minimum variance portfolio
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            initial = np.array([1/n_assets] * n_assets)
            
            result = minimize(portfolio_variance, initial, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x
        
        else:  # max_sharpe (default)
            # Maximum Sharpe ratio optimization
            def neg_sharpe(weights):
                ret = np.dot(weights, expected_returns) - self.risk_free_rate
                vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return -ret/vol if vol > 0 else 0
            
            # Constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            initial = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(neg_sharpe, initial, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
            
            return result.x
    
    def calculate_metrics(self, weights):
        """Calculate portfolio metrics"""
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # VaR
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Max Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'return': annual_return,
            'volatility': annual_vol,
            'sharpe': sharpe,
            'var_95': var_95,
            'max_drawdown': max_dd
        }
    
    def run(self, method='max_sharpe'):
        """Main optimization pipeline with method selection"""
        # Get data
        self.fetch_data()
        
        if self.prices is None or self.prices.empty:
            print("❌ No data available for optimization")
            return None
        
        # Get expected returns with DYNAMIC ML predictions
        print("\n📈 Calculating expected returns...")
        ml_predictions = self.train_ml_models()
        
        tickers = self.prices.columns.tolist()
        if ml_predictions:
            expected_returns = np.array([ml_predictions.get(t, 0) for t in tickers]) * 252
        else:
            expected_returns = self.returns.mean().values * 252
        
        # Covariance matrix
        cov_matrix = self.returns.cov().values * 252
        
        # Optimize
        print(f"\n⚖️ Optimizing portfolio using {method} method...")
        weights = self.optimize_portfolio(expected_returns, cov_matrix, method=method)
        
        # Clean weights
        weights = np.round(weights, 4)
        weights = weights / weights.sum()
        
        # Calculate metrics
        metrics = self.calculate_metrics(weights)
        
        return {
            'tickers': tickers,
            'weights': weights,
            'metrics': metrics
        }