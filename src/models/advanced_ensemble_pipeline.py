"""
Advanced Ensemble ML Pipeline for Portfolio Optimization
Story 2.1: Multi-model ensemble with sophisticated feature engineering

This module implements a comprehensive ensemble ML pipeline that combines:
- XGBoost (gradient boosting)
- Random Forest (bagging ensemble)
- Linear Regression (linear models with regularization)
- Advanced feature engineering with technical indicators
- Rigorous validation framework with cross-validation
- Hyperparameter optimization with proper validation splits
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core ML libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# Statistical and optimization libraries
from scipy import stats
from scipy.optimize import minimize
import joblib
import json

# Technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    try:
        import pandas_ta as ta
        PANDAS_TA_AVAILABLE = True
    except ImportError:
        PANDAS_TA_AVAILABLE = False
        print("Warning: No technical analysis library available. Installing pandas-ta...")

import yfinance as yf

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import existing components
from src.utils.professional_logging import get_logger

logger = get_logger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering pipeline with technical indicators,
    rolling statistics, and cross-asset features.
    """
    
    def __init__(self, lookback_windows: List[int] = [5, 10, 20, 50]):
        self.lookback_windows = lookback_windows
        self.scaler = RobustScaler()
        self.feature_names = []
        
    def create_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators for each asset"""
        features = pd.DataFrame(index=prices.index)
        
        for ticker in prices.columns:
            price_series = prices[ticker].dropna()
            
            # Ensure we have enough data for indicators
            if len(price_series) < 50:
                logger.warning(f"Insufficient data for {ticker}, skipping technical indicators")
                continue
                
            try:
                if TALIB_AVAILABLE:
                    # Use TA-Lib if available
                    features[f'{ticker}_rsi_14'] = talib.RSI(price_series.values, timeperiod=14)
                    features[f'{ticker}_rsi_30'] = talib.RSI(price_series.values, timeperiod=30)
                    
                    # MACD
                    macd, macd_signal, macd_hist = talib.MACD(price_series.values)
                    features[f'{ticker}_macd'] = macd
                    features[f'{ticker}_macd_signal'] = macd_signal
                    features[f'{ticker}_macd_histogram'] = macd_hist
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(price_series.values)
                    features[f'{ticker}_bb_upper'] = bb_upper
                    features[f'{ticker}_bb_middle'] = bb_middle
                    features[f'{ticker}_bb_lower'] = bb_lower
                    features[f'{ticker}_bb_width'] = (bb_upper - bb_lower) / bb_middle
                    features[f'{ticker}_bb_position'] = (price_series.values - bb_lower) / (bb_upper - bb_lower)
                    
                    # Moving averages
                    features[f'{ticker}_sma_10'] = talib.SMA(price_series.values, timeperiod=10)
                    features[f'{ticker}_sma_20'] = talib.SMA(price_series.values, timeperiod=20)
                    features[f'{ticker}_sma_50'] = talib.SMA(price_series.values, timeperiod=50)
                    features[f'{ticker}_ema_12'] = talib.EMA(price_series.values, timeperiod=12)
                    features[f'{ticker}_ema_26'] = talib.EMA(price_series.values, timeperiod=26)
                    
                    # Price momentum
                    features[f'{ticker}_momentum_10'] = talib.MOM(price_series.values, timeperiod=10)
                    features[f'{ticker}_momentum_20'] = talib.MOM(price_series.values, timeperiod=20)
                    
                    # Volatility indicators
                    features[f'{ticker}_atr_14'] = talib.ATR(price_series.values, price_series.values, price_series.values, timeperiod=14)
                    
                    # Volume indicators (using price as proxy if volume not available)
                    features[f'{ticker}_obv'] = talib.OBV(price_series.values, price_series.values)
                    
                elif PANDAS_TA_AVAILABLE:
                    # Use pandas-ta as fallback
                    df_temp = pd.DataFrame({'close': price_series})
                    
                    # RSI
                    features[f'{ticker}_rsi_14'] = ta.rsi(df_temp['close'], length=14)
                    features[f'{ticker}_rsi_30'] = ta.rsi(df_temp['close'], length=30)
                    
                    # MACD
                    macd_df = ta.macd(df_temp['close'])
                    if macd_df is not None and not macd_df.empty:
                        features[f'{ticker}_macd'] = macd_df.iloc[:, 0]  # MACD line
                        features[f'{ticker}_macd_signal'] = macd_df.iloc[:, 2]  # Signal line
                        features[f'{ticker}_macd_histogram'] = macd_df.iloc[:, 1]  # Histogram
                    
                    # Bollinger Bands
                    bb_df = ta.bbands(df_temp['close'])
                    if bb_df is not None and not bb_df.empty:
                        features[f'{ticker}_bb_lower'] = bb_df.iloc[:, 0]
                        features[f'{ticker}_bb_middle'] = bb_df.iloc[:, 1]
                        features[f'{ticker}_bb_upper'] = bb_df.iloc[:, 2]
                        features[f'{ticker}_bb_width'] = (bb_df.iloc[:, 2] - bb_df.iloc[:, 0]) / bb_df.iloc[:, 1]
                        features[f'{ticker}_bb_position'] = (price_series - bb_df.iloc[:, 0]) / (bb_df.iloc[:, 2] - bb_df.iloc[:, 0])
                    
                    # Moving averages
                    features[f'{ticker}_sma_10'] = ta.sma(df_temp['close'], length=10)
                    features[f'{ticker}_sma_20'] = ta.sma(df_temp['close'], length=20)
                    features[f'{ticker}_sma_50'] = ta.sma(df_temp['close'], length=50)
                    features[f'{ticker}_ema_12'] = ta.ema(df_temp['close'], length=12)
                    features[f'{ticker}_ema_26'] = ta.ema(df_temp['close'], length=26)
                    
                    # Price momentum
                    features[f'{ticker}_momentum_10'] = ta.mom(df_temp['close'], length=10)
                    features[f'{ticker}_momentum_20'] = ta.mom(df_temp['close'], length=20)
                    
                else:
                    # Fallback to simple calculations
                    logger.warning("No technical analysis library available, using simple indicators")
                    
                    # Simple RSI calculation
                    delta = price_series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    features[f'{ticker}_rsi_14'] = 100 - (100 / (1 + rs))
                    
                    # Simple moving averages
                    features[f'{ticker}_sma_10'] = price_series.rolling(10).mean()
                    features[f'{ticker}_sma_20'] = price_series.rolling(20).mean()
                    features[f'{ticker}_sma_50'] = price_series.rolling(50).mean()
                    
                    # Exponential moving averages
                    features[f'{ticker}_ema_12'] = price_series.ewm(span=12).mean()
                    features[f'{ticker}_ema_26'] = price_series.ewm(span=26).mean()
                    
                    # Price momentum
                    features[f'{ticker}_momentum_10'] = price_series.pct_change(10)
                    features[f'{ticker}_momentum_20'] = price_series.pct_change(20)
                    
                    # Bollinger Bands
                    sma_20 = price_series.rolling(20).mean()
                    std_20 = price_series.rolling(20).std()
                    features[f'{ticker}_bb_upper'] = sma_20 + (std_20 * 2)
                    features[f'{ticker}_bb_middle'] = sma_20
                    features[f'{ticker}_bb_lower'] = sma_20 - (std_20 * 2)
                    features[f'{ticker}_bb_width'] = (features[f'{ticker}_bb_upper'] - features[f'{ticker}_bb_lower']) / features[f'{ticker}_bb_middle']
                    features[f'{ticker}_bb_position'] = (price_series - features[f'{ticker}_bb_lower']) / (features[f'{ticker}_bb_upper'] - features[f'{ticker}_bb_lower'])
                
                logger.debug(f"Created technical indicators for {ticker}")
                
            except Exception as e:
                logger.warning(f"Error creating technical indicators for {ticker}: {e}")
                continue
                
        return features
    
    def create_rolling_statistics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics"""
        features = pd.DataFrame(index=returns.index)
        
        for ticker in returns.columns:
            for window in self.lookback_windows:
                try:
                    # Rolling returns statistics
                    features[f'{ticker}_ret_mean_{window}'] = returns[ticker].rolling(window).mean()
                    features[f'{ticker}_ret_std_{window}'] = returns[ticker].rolling(window).std()
                    features[f'{ticker}_ret_skew_{window}'] = returns[ticker].rolling(window).skew()
                    features[f'{ticker}_ret_kurt_{window}'] = returns[ticker].rolling(window).kurt()
                    
                    # Rolling extremes
                    features[f'{ticker}_ret_max_{window}'] = returns[ticker].rolling(window).max()
                    features[f'{ticker}_ret_min_{window}'] = returns[ticker].rolling(window).min()
                    
                    # Rolling rank (percentile position)
                    features[f'{ticker}_ret_rank_{window}'] = returns[ticker].rolling(window).rank(pct=True)
                    
                    # Rolling Sharpe ratio (annualized)
                    rolling_mean = returns[ticker].rolling(window).mean()
                    rolling_std = returns[ticker].rolling(window).std()
                    features[f'{ticker}_sharpe_{window}'] = (rolling_mean * 252) / (rolling_std * np.sqrt(252))
                    
                except Exception as e:
                    logger.warning(f"Error creating rolling stats for {ticker}, window {window}: {e}")
                    continue
                    
        return features
    
    def create_cross_asset_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset correlation and sector features"""
        features = pd.DataFrame(index=returns.index)
        
        try:
            # Market-wide features
            market_return = returns.mean(axis=1)  # Equal-weighted market return
            market_vol = returns.std(axis=1)      # Cross-sectional volatility
            
            features['market_return'] = market_return
            features['market_volatility'] = market_vol
            features['market_dispersion'] = returns.std(axis=1)
            
                    # Individual asset vs market
            for ticker in returns.columns:
                try:
                    # Beta estimation (rolling window) - fixed duplicate index issue
                    for window in [20, 50]:
                        # Create clean dataframe for correlation calculation
                        temp_df = pd.DataFrame({
                            'asset': returns[ticker],
                            'market': market_return
                        }).dropna()
                        
                        if len(temp_df) >= window:
                            # Rolling covariance and variance for beta calculation
                            rolling_cov = temp_df['asset'].rolling(window).cov(temp_df['market'])
                            rolling_var = temp_df['market'].rolling(window).var()
                            beta = rolling_cov / rolling_var
                            
                            # Align with original index
                            features[f'{ticker}_beta_{window}'] = beta.reindex(returns.index)
                            
                            # Correlation with market
                            corr = temp_df['asset'].rolling(window).corr(temp_df['market'])
                            features[f'{ticker}_market_corr_{window}'] = corr.reindex(returns.index)
                            
                            # Relative strength
                            relative_return = returns[ticker] - market_return
                            features[f'{ticker}_relative_strength_{window}'] = relative_return.rolling(window).mean()
                        
                except Exception as e:
                    logger.warning(f"Error creating cross-asset features for {ticker}: {e}")
                    continue
            
            # Sector momentum (simplified - group similar tickers)
            tech_tickers = [t for t in returns.columns if t in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']]
            financial_tickers = [t for t in returns.columns if t in ['JPM', 'BAC', 'WFC', 'GS']]
            
            if len(tech_tickers) > 1:
                tech_return = returns[tech_tickers].mean(axis=1)
                features['tech_sector_momentum'] = tech_return.rolling(10).mean()
                
            if len(financial_tickers) > 1:
                fin_return = returns[financial_tickers].mean(axis=1)
                features['financial_sector_momentum'] = fin_return.rolling(10).mean()
                
        except Exception as e:
            logger.warning(f"Error creating cross-asset features: {e}")
            
        return features
    
    def create_regime_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create market regime and volatility features"""
        features = pd.DataFrame(index=returns.index)
        
        try:
            market_return = returns.mean(axis=1)
            
            # VIX-like volatility measure
            for window in [10, 20, 30]:
                rolling_vol = market_return.rolling(window).std() * np.sqrt(252)
                features[f'market_vol_{window}'] = rolling_vol
                features[f'vol_regime_{window}'] = (rolling_vol > rolling_vol.rolling(60).quantile(0.75)).astype(int)
            
            # Trend features
            for window in [10, 20, 50]:
                trend = market_return.rolling(window).mean()
                features[f'trend_{window}'] = trend
                features[f'trend_strength_{window}'] = abs(trend) / market_return.rolling(window).std()
            
            # Market stress indicator
            negative_days = (market_return < 0).rolling(10).sum()
            features['market_stress'] = negative_days / 10
            
            # Drawdown features
            cumulative_returns = (1 + market_return).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            features['market_drawdown'] = drawdown
            features['drawdown_duration'] = (drawdown < -0.05).rolling(20).sum()
            
        except Exception as e:
            logger.warning(f"Error creating regime features: {e}")
            
        return features
    
    def engineer_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        logger.info("Starting advanced feature engineering...")
        
        all_features = []
        
        # 1. Technical indicators
        logger.info("Creating technical indicators...")
        tech_features = self.create_technical_indicators(prices)
        if not tech_features.empty:
            all_features.append(tech_features)
        
        # 2. Rolling statistics
        logger.info("Creating rolling statistics...")
        rolling_features = self.create_rolling_statistics(returns)
        if not rolling_features.empty:
            all_features.append(rolling_features)
        
        # 3. Cross-asset features
        logger.info("Creating cross-asset features...")
        cross_features = self.create_cross_asset_features(returns)
        if not cross_features.empty:
            all_features.append(cross_features)
        
        # 4. Regime features
        logger.info("Creating regime features...")
        regime_features = self.create_regime_features(returns)
        if not regime_features.empty:
            all_features.append(regime_features)
        
        # Combine all features
        if all_features:
            features_df = pd.concat(all_features, axis=1)
            features_df = features_df.dropna()
            
            # Store feature names
            self.feature_names = list(features_df.columns)
            
            logger.info(f"Feature engineering complete. Created {len(self.feature_names)} features")
            logger.info(f"Feature matrix shape: {features_df.shape}")
            
            return features_df
        else:
            logger.error("No features created successfully")
            return pd.DataFrame()


class EnsembleModelValidator:
    """
    Comprehensive validation framework for ensemble models with
    time-series cross-validation and statistical testing.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 60):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
    def time_series_cross_validation(self, X: pd.DataFrame, y: pd.Series, model, 
                                   scoring_func=mean_squared_error) -> Dict[str, Any]:
        """Perform time-series cross-validation"""
        scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            scores.append(mse)
            fold_results.append({
                'fold': fold,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_results': fold_results
        }
    
    def compare_models_statistical(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Statistical comparison of model performance"""
        model_names = list(model_results.keys())
        comparison_results = {}
        
        # Pairwise statistical tests
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                scores1 = model_results[model1]['scores']
                scores2 = model_results[model2]['scores']
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                comparison_results[f'{model1}_vs_{model2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'model1_better': t_stat < 0  # Lower scores are better for MSE
                }
        
        return comparison_results
    
    def bootstrap_confidence_intervals(self, scores: List[float], n_bootstrap: int = 1000, 
                                     confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for performance metrics"""
        bootstrap_scores = []
        
        for bootstrap_idx in range(n_bootstrap):
            # Create deterministic bootstrap sample
            bootstrap_sample = []
            for i in range(len(scores)):
                # Deterministic selection based on bootstrap iteration and position
                sample_hash = hash(f"{bootstrap_idx}_{i}") % len(scores)
                bootstrap_sample.append(scores[sample_hash])
            
            bootstrap_scores.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'confidence_interval_lower': np.percentile(bootstrap_scores, lower_percentile),
            'confidence_interval_upper': np.percentile(bootstrap_scores, upper_percentile),
            'confidence_level': confidence_level
        }


class AdvancedEnsembleManager:
    """
    Advanced ensemble ML pipeline combining multiple algorithms
    with sophisticated feature engineering and validation.
    """
    
    def __init__(self, feature_engineer: AdvancedFeatureEngineer = None,
                 validator: EnsembleModelValidator = None,
                 random_state: int = 42):
        self.feature_engineer = feature_engineer or AdvancedFeatureEngineer()
        self.validator = validator or EnsembleModelValidator()
        self.random_state = random_state
        
        # Model configurations
        self.base_models = {}
        self.ensemble_weights = {}
        self.trained_models = {}
        self.feature_importance = {}
        self.validation_results = {}
        
        # Performance tracking
        self.performance_metrics = {}
        self.confidence_intervals = {}
        
        logger.info("Advanced Ensemble Manager initialized")
    
    def _create_base_models(self) -> Dict[str, Any]:
        """Create and configure base models for the ensemble"""
        models = {}
        
        # XGBoost model
        models['xgboost'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Random Forest model
        models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Ridge Regression
        models['ridge'] = Ridge(
            alpha=1.0,
            random_state=self.random_state
        )
        
        # Lasso Regression
        models['lasso'] = Lasso(
            alpha=0.1,
            random_state=self.random_state,
            max_iter=10000
        )
        
        # Elastic Net
        models['elastic_net'] = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=self.random_state,
            max_iter=10000
        )
        
        logger.info(f"Created {len(models)} base models: {list(models.keys())}")
        return models
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                model_name: str, param_grid: Dict) -> Dict[str, Any]:
        """Optimize hyperparameters using time-series cross-validation"""
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        base_model = self._create_base_models()[model_name]
        
        # Use TimeSeriesSplit for hyperparameter optimization
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search with time-series CV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_ensemble(self, prices: pd.DataFrame, returns: pd.DataFrame, 
                      target_column: str) -> Dict[str, Any]:
        """Train the complete ensemble pipeline"""
        logger.info("Starting ensemble training pipeline...")
        
        # Feature engineering
        features = self.feature_engineer.engineer_features(prices, returns)
        
        if features.empty:
            raise ValueError("Feature engineering failed - no features created")
        
        # Prepare target variable
        if target_column not in returns.columns:
            raise ValueError(f"Target column {target_column} not found in returns")
        
        # Align features and target
        common_index = features.index.intersection(returns.index)
        X = features.loc[common_index]
        y = returns.loc[common_index, target_column]
        
        # Remove any remaining NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        # Create and train base models
        self.base_models = self._create_base_models()
        training_results = {}
        
        for model_name, model in self.base_models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X, y)
                self.trained_models[model_name] = model
                
                # Validate using time-series CV
                cv_results = self.validator.time_series_cross_validation(X, y, model)
                self.validation_results[model_name] = cv_results
                
                # Calculate confidence intervals
                self.confidence_intervals[model_name] = self.validator.bootstrap_confidence_intervals(
                    cv_results['scores']
                )
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[model_name] = importance_df
                elif hasattr(model, 'coef_'):
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': np.abs(model.coef_)
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[model_name] = importance_df
                
                training_results[model_name] = {
                    'success': True,
                    'cv_score': cv_results['mean_score'],
                    'cv_std': cv_results['std_score']
                }
                
                logger.info(f"{model_name} training complete. CV Score: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights()
        
        # Model comparison
        model_comparison = self.validator.compare_models_statistical(self.validation_results)
        
        logger.info("Ensemble training complete!")
        
        return {
            'training_results': training_results,
            'model_comparison': model_comparison,
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(X.columns),
            'training_samples': len(X)
        }
    
    def _calculate_ensemble_weights(self):
        """Calculate performance-based weights for ensemble"""
        if not self.validation_results:
            return
        
        # Use inverse of CV scores as weights (lower error = higher weight)
        scores = []
        model_names = []
        
        for model_name, results in self.validation_results.items():
            if 'mean_score' in results:
                scores.append(results['mean_score'])
                model_names.append(model_name)
        
        if scores:
            # Inverse weights (lower error gets higher weight)
            inverse_scores = [1.0 / max(score, 1e-10) for score in scores]
            total_weight = sum(inverse_scores)
            
            # Normalize weights
            self.ensemble_weights = {
                name: weight / total_weight 
                for name, weight in zip(model_names, inverse_scores)
            }
            
            logger.info("Ensemble weights calculated:")
            for name, weight in self.ensemble_weights.items():
                logger.info(f"  {name}: {weight:.4f}")
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions using weighted voting"""
        if not self.trained_models or not self.ensemble_weights:
            raise ValueError("Ensemble not trained yet")
        
        predictions = []
        weights = []
        
        for model_name, model in self.trained_models.items():
            if model_name in self.ensemble_weights:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.ensemble_weights[model_name])
        
        if not predictions:
            raise ValueError("No valid models for prediction")
        
        # Weighted average
        predictions_array = np.array(predictions)
        weights_array = np.array(weights)
        
        ensemble_prediction = np.average(predictions_array, axis=0, weights=weights_array)
        
        return ensemble_prediction
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'model_performance': {},
            'ensemble_weights': self.ensemble_weights,
            'model_comparison': {},
            'feature_importance': {},
            'confidence_intervals': self.confidence_intervals
        }
        
        # Individual model performance
        for model_name, results in self.validation_results.items():
            report['model_performance'][model_name] = {
                'cv_mean': results['mean_score'],
                'cv_std': results['std_score'],
                'fold_details': results['fold_results']
            }
        
        # Feature importance summary
        for model_name, importance_df in self.feature_importance.items():
            report['feature_importance'][model_name] = importance_df.head(10).to_dict('records')
        
        # Model comparison
        if len(self.validation_results) > 1:
            report['model_comparison'] = self.validator.compare_models_statistical(self.validation_results)
        
        return report
    
    def save_ensemble(self, filepath: str):
        """Save the complete ensemble to disk"""
        ensemble_data = {
            'trained_models': self.trained_models,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': {k: v.to_dict() for k, v in self.feature_importance.items()},
            'validation_results': self.validation_results,
            'confidence_intervals': self.confidence_intervals,
            'feature_names': self.feature_engineer.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save using joblib for model objects
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble from disk"""
        ensemble_data = joblib.load(filepath)
        
        self.trained_models = ensemble_data['trained_models']
        self.ensemble_weights = ensemble_data['ensemble_weights']
        self.validation_results = ensemble_data['validation_results']
        self.confidence_intervals = ensemble_data['confidence_intervals']
        self.feature_engineer.feature_names = ensemble_data['feature_names']
        
        # Recreate feature importance DataFrames
        self.feature_importance = {
            k: pd.DataFrame(v) for k, v in ensemble_data['feature_importance'].items()
        }
        
        logger.info(f"Ensemble loaded from {filepath}")


def run_ensemble_pipeline_demo():
    """
    Demonstration of the advanced ensemble ML pipeline
    """
    print("\n" + "="*80)
    print("🚀 ADVANCED ENSEMBLE ML PIPELINE - STORY 2.1 DEMO")
    print("="*80 + "\n")
    
    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
    target_ticker = 'AAPL'  # Predict AAPL returns
    
    try:
        # Download data
        print("📊 Downloading market data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=800)  # ~2+ years
        
        data = yf.download(
            tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        prices = data['Close'].dropna()
        returns = prices.pct_change().dropna()
        
        print(f"✅ Downloaded {len(prices)} days of data for {len(prices.columns)} assets")
        
        # Initialize ensemble
        print("\n🤖 Initializing Advanced Ensemble Pipeline...")
        feature_engineer = AdvancedFeatureEngineer()
        validator = EnsembleModelValidator(n_splits=5)
        ensemble = AdvancedEnsembleManager(feature_engineer, validator)
        
        # Train ensemble
        print("\n🔥 Training Ensemble Models...")
        training_results = ensemble.train_ensemble(prices, returns, target_ticker)
        
        # Display results
        print("\n📊 TRAINING RESULTS SUMMARY")
        print("-" * 50)
        
        for model_name, result in training_results['training_results'].items():
            if result['success']:
                cv_score = result['cv_score']
                cv_std = result['cv_std']
                print(f"✅ {model_name:15s}: {cv_score:.6f} ± {cv_std:.6f}")
            else:
                print(f"❌ {model_name:15s}: FAILED - {result['error']}")
        
        print(f"\n📈 Features Created: {training_results['feature_count']}")
        print(f"📊 Training Samples: {training_results['training_samples']}")
        
        # Ensemble weights
        print("\n🏆 ENSEMBLE WEIGHTS")
        print("-" * 30)
        for model_name, weight in ensemble.ensemble_weights.items():
            print(f"{model_name:15s}: {weight:.4f}")
        
        # Performance report
        performance_report = ensemble.get_performance_report()
        
        print("\n📊 CONFIDENCE INTERVALS (95%)")
        print("-" * 50)
        for model_name, ci in performance_report['confidence_intervals'].items():
            print(f"{model_name:15s}: [{ci['confidence_interval_lower']:.6f}, {ci['confidence_interval_upper']:.6f}]")
        
        # Feature importance (top features)
        print("\n🎯 TOP FEATURES BY MODEL")
        print("-" * 50)
        for model_name, importance in performance_report['feature_importance'].items():
            print(f"\n{model_name} Top 5 Features:")
            for i, feat in enumerate(importance[:5]):
                print(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f}")
        
        # Save ensemble
        ensemble_path = "models/advanced_ensemble.pkl"
        os.makedirs("models", exist_ok=True)
        ensemble.save_ensemble(ensemble_path)
        print(f"\n💾 Ensemble saved to: {ensemble_path}")
        
        # Test prediction
        print("\n🔮 Testing Ensemble Prediction...")
        test_features = feature_engineer.engineer_features(prices, returns)
        if not test_features.empty:
            latest_features = test_features.tail(1)
            prediction = ensemble.predict_ensemble(latest_features)
            print(f"Latest {target_ticker} return prediction: {prediction[0]:.6f}")
        
        print("\n✅ Advanced Ensemble Pipeline Demo Complete!")
        print("🎯 Ready for FAANG technical interviews!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the demo
    success = run_ensemble_pipeline_demo()
    
    if success:
        print("\n🚀 Story 2.1 implementation ready for development team review!")
    else:
        print("\n⚠️ Pipeline needs debugging before Sprint 1 completion")
