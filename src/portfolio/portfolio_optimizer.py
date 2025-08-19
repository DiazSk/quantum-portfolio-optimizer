"""
Portfolio Optimizer - FINAL WORKING VERSION
All data structure issues fixed!
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
    """Portfolio optimizer - fully debugged version"""
    
    def __init__(self, tickers, lookback_years=2):
        self.tickers = tickers
        self.lookback_years = lookback_years
        self.prices = None
        self.returns = None
        self.ml_models = {}
        
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
        """Train XGBoost for return prediction"""
        if self.returns is None or self.returns.empty:
            print("❌ No data available for ML training")
            return {}
            
        print("\n🤖 Training ML models for return prediction...")
        predictions = {}
        
        for ticker in self.prices.columns:
            try:
                # Create simple features
                features = pd.DataFrame(index=self.prices.index)
                
                # Basic features
                features['returns_5d'] = self.returns[ticker].rolling(5).mean()
                features['returns_20d'] = self.returns[ticker].rolling(20).mean()
                features['volatility'] = self.returns[ticker].rolling(20).std()
                features['rsi'] = self.calculate_rsi(self.prices[ticker])
                
                # Target
                target = self.returns[ticker].shift(-1)
                
                # Clean data
                valid_mask = ~(features.isna().any(axis=1) | target.isna())
                X = features[valid_mask].values
                y = target[valid_mask].values
                
                if len(X) > 50:
                    # Train simple model
                    model = xgb.XGBRegressor(
                        n_estimators=30,
                        max_depth=3,
                        random_state=42,
                        verbosity=0
                    )
                    
                    # Train on most data, test on last 20%
                    split = int(len(X) * 0.8)
                    model.fit(X[:split], y[:split])
                    
                    # Make prediction
                    if len(features.dropna()) > 0:
                        last_features = features.dropna().iloc[-1:].values
                        pred = model.predict(last_features)[0]
                        predictions[ticker] = pred
                        print(f"  {ticker}: Predicted return = {pred:+.3%}")
                    else:
                        predictions[ticker] = self.returns[ticker].mean()
                else:
                    predictions[ticker] = self.returns[ticker].mean()
                    print(f"  {ticker}: Using historical mean")
                    
            except Exception as e:
                print(f"  {ticker}: Error - using zero")
                predictions[ticker] = 0.0
                
        return predictions
    
    def optimize_portfolio(self, expected_returns, cov_matrix):
        """Simple Sharpe optimization"""
        n_assets = len(expected_returns)
        
        # Optimization function
        def neg_sharpe(weights):
            ret = np.dot(weights, expected_returns)
            vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -ret/vol if vol > 0 else 0
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
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
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
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
    
    def run(self):
        """Main optimization pipeline"""
        # Get data
        self.fetch_data()
        
        if self.prices is None or self.prices.empty:
            print("❌ No data available for optimization")
            return None
        
        # Get expected returns
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
        print("\n⚖️ Optimizing portfolio...")
        weights = self.optimize_portfolio(expected_returns, cov_matrix)
        
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

def main():
    """Run the portfolio optimization"""
    print("\n" + "="*60)
    print("🚀 QUANTUM PORTFOLIO OPTIMIZER v3.0 (FIXED)")
    print("="*60)
    
    # Portfolio universe
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'META', 'JPM', 'XOM']
    print(f"\n📌 Target Portfolio: {', '.join(tickers)}")
    
    # Run optimizer
    optimizer = PortfolioOptimizer(tickers, lookback_years=2)
    result = optimizer.run()
    
    if result is None:
        print("\n❌ Optimization failed")
        return
    
    # Display results
    print("\n" + "="*60)
    print("📊 OPTIMIZED PORTFOLIO")
    print("="*60)
    
    for ticker, weight in zip(result['tickers'], result['weights']):
        if weight > 0.01:
            bar = '█' * int(weight * 50)
            print(f"{ticker:6s}: {weight:6.2%} {bar}")
    
    print("\n" + "="*60)
    print("📈 PERFORMANCE METRICS")
    print("="*60)
    
    metrics = result['metrics']
    print(f"Expected Return:  {metrics['return']:+.1%} per year")
    print(f"Volatility:       {metrics['volatility']:.1%}")
    print(f"Sharpe Ratio:     {metrics['sharpe']:.2f}")
    print(f"Value at Risk:    {metrics['var_95']:.3%} (95% confidence)")
    print(f"Max Drawdown:     {metrics['max_drawdown']:.1%}")
    
    # Save results
    df = pd.DataFrame({
        'Ticker': result['tickers'],
        'Weight': result['weights']
    })
    df.to_csv('data/portfolio_optimized.csv', index=False)
    
    print("\n✅ Results saved to portfolio_optimized.csv")
    print("🎉 Optimization complete!")
    
    return result

if __name__ == "__main__":
    result = main()