"""
ML-Enhanced Portfolio Optimizer with Alternative Data Integration
Combines traditional optimization with XGBoost predictions and regime detection
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Portfolio optimization
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import HRPOpt, CLA, plotting

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Technical indicators
import pandas_ta as ta

# Risk metrics
from scipy import stats
import riskfolio as rp

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPortfolioOptimizer:
    """Advanced portfolio optimizer combining ML predictions with alternative data"""
    
    def __init__(self, tickers: List[str], 
                 lookback_years: int = 3,
                 risk_free_rate: float = 0.04):
        self.tickers = tickers
        self.lookback_years = lookback_years
        self.risk_free_rate = risk_free_rate
        self.prices = None
        self.returns = None
        self.features = None
        self.ml_models = {}
        self.predictions = {}
        
    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch historical price data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.lookback_years)
        
        logger.info(f"Fetching price data from {start_date} to {end_date}")
        
        # Download price data
        self.prices = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        # Handle single ticker case
        if len(self.tickers) == 1:
            self.prices = pd.DataFrame(self.prices)
            self.prices.columns = self.tickers
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        
        return self.prices
    
    def engineer_features(self) -> pd.DataFrame:
        """Create technical and statistical features for ML models"""
        features_dict = {}
        
        for ticker in self.tickers:
            ticker_features = pd.DataFrame(index=self.prices.index)
            
            # Price-based features
            ticker_features[f'{ticker}_return_1d'] = self.returns[ticker]
            ticker_features[f'{ticker}_return_5d'] = self.returns[ticker].rolling(5).mean()
            ticker_features[f'{ticker}_return_20d'] = self.returns[ticker].rolling(20).mean()
            ticker_features[f'{ticker}_return_60d'] = self.returns[ticker].rolling(60).mean()
            
            # Volatility features
            ticker_features[f'{ticker}_volatility_20d'] = self.returns[ticker].rolling(20).std()
            ticker_features[f'{ticker}_volatility_60d'] = self.returns[ticker].rolling(60).std()
            
            # Technical indicators using pandas_ta
            price_series = self.prices[ticker]
            
            # RSI
            ticker_features[f'{ticker}_rsi'] = ta.rsi(price_series, length=14)
            
            # MACD
            macd = ta.macd(price_series)
            if macd is not None:
                ticker_features[f'{ticker}_macd'] = macd['MACD_12_26_9']
                ticker_features[f'{ticker}_macd_signal'] = macd['MACDs_12_26_9']
            
            # Bollinger Bands
            bbands = ta.bbands(price_series, length=20)
            if bbands is not None:
                ticker_features[f'{ticker}_bb_position'] = (
                    (price_series - bbands['BBL_20_2.0']) / 
                    (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0'])
                ).iloc[:, 0] if len(bbands.shape) > 1 else (price_series - bbands['BBL_20_2.0']) / (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0'])
            
            # Volume-based features (if available)
            try:
                volume = yf.Ticker(ticker).history(period=f"{self.lookback_years}y")['Volume']
                ticker_features[f'{ticker}_volume_ratio'] = volume / volume.rolling(20).mean()
            except:
                pass
            
            # Market regime indicators
            ticker_features[f'{ticker}_trend'] = (
                price_series / price_series.rolling(50).mean() - 1
            )
            
            # Correlation features
            if len(self.tickers) > 1:
                ticker_features[f'{ticker}_corr_market'] = (
                    self.returns[ticker].rolling(60).corr(self.returns.mean(axis=1))
                )
            
            features_dict[ticker] = ticker_features
        
        # Combine all features
        self.features = pd.concat(features_dict.values(), axis=1).dropna()
        
        # Add market-wide features
        self.features['market_return_20d'] = self.returns.mean(axis=1).rolling(20).mean()
        self.features['market_volatility'] = self.returns.mean(axis=1).rolling(20).std()
        
        # VIX proxy (market fear gauge)
        self.features['volatility_regime'] = (
            self.returns.std(axis=1).rolling(20).mean() / 
            self.returns.std(axis=1).rolling(60).mean()
        )
        
        return self.features
    
    def train_return_predictors(self) -> Dict:
        """Train XGBoost models to predict returns for each asset"""
        
        if self.features is None:
            self.engineer_features()
        
        predictions = {}
        
        for ticker in self.tickers:
            logger.info(f"Training ML model for {ticker}")
            
            # Prepare data
            feature_cols = [col for col in self.features.columns 
                          if ticker in col or 'market' in col or 'volatility_regime' in col]
            
            X = self.features[feature_cols].iloc[:-1]  # Features
            y = self.returns[ticker].shift(-1).iloc[:-1]  # Next day returns
            
            # Remove NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                logger.warning(f"Insufficient data for {ticker}, using mean return")
                predictions[ticker] = self.returns[ticker].mean()
                continue
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train XGBoost model
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.01,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation scores
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                xgb_model.fit(X_train, y_train)
                score = xgb_model.score(X_val, y_val)
                cv_scores.append(score)
            
            # Train on full data
            xgb_model.fit(X_scaled, y)
            
            # Store model and make prediction
            self.ml_models[ticker] = {
                'model': xgb_model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'cv_score': np.mean(cv_scores)
            }
            
            # Predict next period return
            last_features = self.features[feature_cols].iloc[-1:].values
            last_features_scaled = scaler.transform(last_features)
            predicted_return = xgb_model.predict(last_features_scaled)[0]
            
            predictions[ticker] = predicted_return
            
            logger.info(f"{ticker} - CV Score: {np.mean(cv_scores):.4f}, "
                       f"Predicted Return: {predicted_return:.4%}")
        
        self.predictions = predictions
        return predictions
    
    def detect_market_regime(self) -> str:
        """Detect current market regime using Hidden Markov Model approach"""
        
        # Simplified regime detection using volatility and returns
        market_returns = self.returns.mean(axis=1)
        recent_vol = market_returns.iloc[-20:].std()
        historical_vol = market_returns.std()
        recent_return = market_returns.iloc[-20:].mean()
        
        if recent_vol > historical_vol * 1.5:
            regime = "high_volatility"
        elif recent_return < -0.001 and recent_vol > historical_vol:
            regime = "bear_market"
        elif recent_return > 0.001 and recent_vol < historical_vol:
            regime = "bull_market"
        else:
            regime = "neutral"
        
        logger.info(f"Detected market regime: {regime}")
        return regime
    
    def optimize_portfolio(self, 
                          method: str = 'max_sharpe',
                          use_ml_predictions: bool = True,
                          alternative_data_scores: Optional[pd.DataFrame] = None) -> Dict:
        """
        Optimize portfolio using various methods with ML predictions
        
        Methods:
        - max_sharpe: Maximum Sharpe Ratio
        - min_volatility: Minimum Volatility
        - hrp: Hierarchical Risk Parity
        - risk_parity: Risk Parity
        - max_diversification: Maximum Diversification
        """
        
        if self.prices is None:
            self.fetch_price_data()
        
        # Calculate expected returns
        if use_ml_predictions and self.predictions:
            # Combine ML predictions with historical returns
            ml_returns = pd.Series(self.predictions)
            historical_returns = expected_returns.mean_historical_return(self.prices)
            
            # Blend ML and historical (70% ML, 30% historical)
            expected_rets = 0.7 * ml_returns + 0.3 * historical_returns
            
            # Adjust for alternative data if provided
            if alternative_data_scores is not None:
                alt_scores = alternative_data_scores.set_index('ticker')['alt_data_score']
                # Boost returns for high alternative data scores
                adjustment = (alt_scores - 0.5) * 0.1  # ±10% adjustment
                expected_rets = expected_rets * (1 + adjustment)
        else:
            expected_rets = expected_returns.mean_historical_return(self.prices)
        
        # Calculate risk model
        cov_matrix = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()
        
        # Detect regime and adjust accordingly
        regime = self.detect_market_regime()
        
        if regime == "high_volatility":
            # Increase risk aversion in high volatility
            risk_adjustment = 1.5
        elif regime == "bear_market":
            # Be more conservative
            risk_adjustment = 2.0
        else:
            risk_adjustment = 1.0
        
        # Initialize optimizer based on method
        if method in ['max_sharpe', 'min_volatility']:
            ef = EfficientFrontier(expected_rets, cov_matrix)
            
            if method == 'max_sharpe':
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            else:  # min_volatility
                weights = ef.min_volatility()
            
            cleaned_weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                verbose=False,
                risk_free_rate=self.risk_free_rate
            )
            
        elif method == 'hrp':
            # Hierarchical Risk Parity
            hrp = HRPOpt(self.returns)
            weights = hrp.optimize()
            cleaned_weights = hrp.clean_weights()
            
            # Calculate performance manually
            portfolio_return = (expected_rets * pd.Series(cleaned_weights)).sum()
            portfolio_vol = np.sqrt(
                np.dot(pd.Series(cleaned_weights).values,
                      np.dot(cov_matrix.values, pd.Series(cleaned_weights).values))
            )
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            performance = (portfolio_return, portfolio_vol, sharpe)
            
        elif method == 'risk_parity':
            # Risk Parity using riskfolio-lib
            port = rp.Portfolio(returns=self.returns)
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            weights_rp = port.rp_optimization(
                model='Classic',
                rm='MV',
                rf=self.risk_free_rate
            )
            
            cleaned_weights = dict(zip(self.tickers, weights_rp.values.flatten()))
            
            # Calculate performance
            portfolio_return = (expected_rets * weights_rp.values.flatten()).sum()
            portfolio_vol = np.sqrt(
                np.dot(weights_rp.values.flatten(),
                      np.dot(cov_matrix.values, weights_rp.values.flatten()))
            )
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            performance = (portfolio_return, portfolio_vol, sharpe)
            
        else:  # max_diversification
            # Maximum Diversification Portfolio
            ef = EfficientFrontier(expected_rets, cov_matrix)
            
            # Use min volatility as proxy for max diversification
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(
                verbose=False,
                risk_free_rate=self.risk_free_rate
            )
        
        # Apply regime-based position sizing
        if regime in ["high_volatility", "bear_market"]:
            # Reduce position sizes in risky regimes
            max_position = 0.25 if regime == "high_volatility" else 0.20
            cleaned_weights = self._apply_position_limits(cleaned_weights, max_position)
        
        # Calculate additional risk metrics
        risk_metrics = self.calculate_risk_metrics(cleaned_weights)
        
        return {
            'weights': cleaned_weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'regime': regime,
            'risk_metrics': risk_metrics,
            'ml_confidence': np.mean([m['cv_score'] for m in self.ml_models.values()]) 
                            if self.ml_models else 0
        }
    
    def _apply_position_limits(self, weights: Dict, max_weight: float) -> Dict:
        """Apply position size limits and renormalize"""
        limited_weights = {}
        
        for ticker, weight in weights.items():
            limited_weights[ticker] = min(weight, max_weight)
        
        # Renormalize
        total = sum(limited_weights.values())
        return {k: v/total for k, v in limited_weights.items()}
    
    def calculate_risk_metrics(self, weights: Dict) -> Dict:
        """Calculate comprehensive risk metrics"""
        weights_array = np.array([weights[ticker] for ticker in self.tickers])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Maximum Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = (portfolio_returns.mean() - self.risk_free_rate/252) / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        calmar = portfolio_returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns)
        }
    
    def backtest_strategy(self, 
                         rebalance_frequency: str = 'monthly',
                         initial_capital: float = 100000) -> pd.DataFrame:
        """Backtest the optimization strategy"""
        
        # This is simplified - in production use vectorized backtesting
        logger.info(f"Backtesting strategy with {rebalance_frequency} rebalancing")
        
        # For now, return mock results
        dates = pd.date_range(
            start=self.prices.index[0],
            end=self.prices.index[-1],
            freq='D'
        )
        
        # Simulate portfolio value
        returns = np.random.normal(0.0008, 0.01, len(dates))
        portfolio_value = initial_capital * (1 + returns).cumprod()
        
        backtest_results = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_value,
            'returns': returns,
            'benchmark': initial_capital * (1 + np.random.normal(0.0005, 0.008, len(dates))).cumprod()
        })
        
        return backtest_results


# Example usage
def main():
    # Define universe
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'GS', 'XOM']
    
    # Initialize optimizer
    optimizer = MLPortfolioOptimizer(tickers, lookback_years=2)
    
    # Fetch data and engineer features
    optimizer.fetch_price_data()
    optimizer.engineer_features()
    
    # Train ML models
    predictions = optimizer.train_return_predictors()
    print("\nML Return Predictions:")
    for ticker, pred in predictions.items():
        print(f"{ticker}: {pred:.4%}")
    
    # Optimize portfolio
    portfolio = optimizer.optimize_portfolio(
        method='max_sharpe',
        use_ml_predictions=True
    )
    
    print("\nOptimized Portfolio:")
    print(f"Weights: {portfolio['weights']}")
    print(f"Expected Return: {portfolio['expected_return']:.2%}")
    print(f"Volatility: {portfolio['volatility']:.2%}")
    print(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}")
    print(f"Market Regime: {portfolio['regime']}")
    
    print("\nRisk Metrics:")
    for metric, value in portfolio['risk_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    return portfolio

if __name__ == "__main__":
    portfolio = main()