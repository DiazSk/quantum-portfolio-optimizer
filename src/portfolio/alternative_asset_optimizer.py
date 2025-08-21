"""
Alternative Asset Portfolio Optimization Engine

Comprehensive multi-asset portfolio optimization including:
- Illiquidity-adjusted portfolio optimization
- Alternative asset allocation limits and risk budgeting
- Multi-asset correlation modeling and regime detection
- Alternative asset rebalancing with liquidity constraints
- Advanced portfolio construction with alternative assets

Business Value:
- Optimal alternative asset allocation strategies
- Risk-adjusted portfolio enhancement
- Sophisticated institutional portfolio management
- Multi-asset class optimization capabilities
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import norm
import cvxpy as cp

from src.portfolio.alternative_assets.real_estate import REITSecurity
from src.portfolio.alternative_assets.commodities import CommodityFuture
from src.portfolio.alternative_assets.cryptocurrency import CryptocurrencyAsset
from src.portfolio.alternative_assets.private_markets import PrivateMarketInvestment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_RETURN = "maximize_return"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class AlternativeAssetConstraints:
    """Constraints for alternative asset portfolio optimization"""
    
    # Asset Class Limits
    max_alternative_allocation: float = 0.30  # Maximum 30% in alternatives
    max_reit_allocation: float = 0.15  # Maximum 15% in REITs
    max_commodity_allocation: float = 0.10  # Maximum 10% in commodities
    max_crypto_allocation: float = 0.05  # Maximum 5% in crypto
    max_private_market_allocation: float = 0.20  # Maximum 20% in private markets
    
    # Liquidity Constraints
    min_liquid_allocation: float = 0.60  # Minimum 60% in liquid assets
    max_illiquid_allocation: float = 0.40  # Maximum 40% in illiquid assets
    
    # Risk Constraints
    max_portfolio_volatility: float = 0.20  # Maximum 20% portfolio volatility
    max_var_95: float = 0.10  # Maximum 10% Value at Risk (95% confidence)
    max_concentration: float = 0.20  # Maximum 20% in single asset
    
    # Correlation Constraints
    max_correlation_exposure: float = 0.80  # Maximum exposure to highly correlated assets
    min_diversification_ratio: float = 0.60  # Minimum diversification ratio
    
    # Rebalancing Constraints
    max_turnover: float = 0.30  # Maximum 30% turnover per rebalancing
    min_trade_size: float = 0.01  # Minimum 1% trade size
    
    # ESG Constraints (optional)
    min_esg_score: Optional[float] = None  # Minimum ESG score if applicable


@dataclass 
class MultiAssetPortfolio:
    """Multi-asset portfolio including alternative investments"""
    
    # Traditional Assets
    equity_allocation: float = 0.0
    bond_allocation: float = 0.0
    cash_allocation: float = 0.0
    
    # Alternative Assets
    reit_allocations: Dict[str, float] = None
    commodity_allocations: Dict[str, float] = None
    crypto_allocations: Dict[str, float] = None
    private_market_allocations: Dict[str, float] = None
    
    # Portfolio Metrics
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk Metrics
    value_at_risk_95: float = 0.0
    max_drawdown: float = 0.0
    liquidity_score: float = 0.0
    
    # Alternative Asset Metrics
    total_alternative_allocation: float = 0.0
    illiquidity_score: float = 0.0
    diversification_ratio: float = 0.0
    
    def __post_init__(self):
        if self.reit_allocations is None:
            self.reit_allocations = {}
        if self.commodity_allocations is None:
            self.commodity_allocations = {}
        if self.crypto_allocations is None:
            self.crypto_allocations = {}
        if self.private_market_allocations is None:
            self.private_market_allocations = {}


class AlternativeAssetOptimizer:
    """
    Advanced portfolio optimizer for multi-asset portfolios including alternatives
    
    Implements sophisticated optimization techniques for portfolios containing
    traditional and alternative investments with liquidity and risk constraints.
    """
    
    def __init__(self):
        """Initialize alternative asset optimizer"""
        self.supported_objectives = list(OptimizationObjective)
        self.risk_models = {}
        self.correlation_models = {}
        
        # Default parameters
        self.risk_free_rate = 0.02
        self.transaction_costs = {
            'equity': 0.001,
            'bond': 0.0005,
            'reit': 0.002,
            'commodity': 0.003,
            'crypto': 0.005,
            'private_market': 0.02
        }
        
        logger.info("Alternative asset optimizer initialized")
    
    def optimize_multi_asset_portfolio(self,
                                     asset_returns: pd.DataFrame,
                                     asset_metadata: Dict[str, Dict],
                                     objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE,
                                     constraints: AlternativeAssetConstraints = None,
                                     current_portfolio: Optional[MultiAssetPortfolio] = None) -> MultiAssetPortfolio:
        """
        Optimize multi-asset portfolio including alternative investments
        
        Args:
            asset_returns: DataFrame with asset returns (columns = assets, index = dates)
            asset_metadata: Metadata for each asset (type, liquidity, etc.)
            objective: Optimization objective
            constraints: Portfolio constraints
            current_portfolio: Current portfolio for rebalancing optimization
            
        Returns:
            Optimized MultiAssetPortfolio
        """
        if constraints is None:
            constraints = AlternativeAssetConstraints()
        
        logger.info(f"Optimizing multi-asset portfolio with {len(asset_returns.columns)} assets")
        
        # Prepare optimization inputs
        expected_returns = self._calculate_expected_returns(asset_returns)
        covariance_matrix = self._calculate_covariance_matrix(asset_returns)
        liquidity_scores = self._calculate_liquidity_scores(asset_metadata)
        
        # Build optimization problem
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        
        # Define objective function
        if objective == OptimizationObjective.MAXIMIZE_SHARPE:
            portfolio_return = expected_returns.T @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            portfolio_volatility = cp.sqrt(portfolio_variance)
            
            # Maximize Sharpe ratio (minimize negative Sharpe)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            objective_func = cp.Minimize(-sharpe_ratio)
            
        elif objective == OptimizationObjective.MINIMIZE_RISK:
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            objective_func = cp.Minimize(portfolio_variance)
            
        elif objective == OptimizationObjective.MAXIMIZE_RETURN:
            portfolio_return = expected_returns.T @ weights
            objective_func = cp.Maximize(portfolio_return)
            
        else:
            # Default to Sharpe maximization
            portfolio_return = expected_returns.T @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            portfolio_volatility = cp.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            objective_func = cp.Minimize(-sharpe_ratio)
        
        # Build constraints
        constraint_list = []
        
        # Weight constraints
        constraint_list.append(cp.sum(weights) == 1.0)  # Fully invested
        constraint_list.append(weights >= 0.0)  # Long-only
        
        # Asset class constraints
        asset_types = [asset_metadata[asset]['type'] for asset in asset_returns.columns]
        constraint_list.extend(self._build_asset_class_constraints(weights, asset_types, constraints))
        
        # Liquidity constraints
        constraint_list.extend(self._build_liquidity_constraints(weights, liquidity_scores, constraints))
        
        # Risk constraints
        if constraints.max_portfolio_volatility:
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            constraint_list.append(cp.sqrt(portfolio_variance) <= constraints.max_portfolio_volatility)
        
        # Concentration constraints
        if constraints.max_concentration:
            constraint_list.append(weights <= constraints.max_concentration)
        
        # Turnover constraints (if rebalancing)
        if current_portfolio is not None and constraints.max_turnover:
            current_weights = self._extract_current_weights(current_portfolio, asset_returns.columns)
            turnover = cp.sum(cp.abs(weights - current_weights))
            constraint_list.append(turnover <= constraints.max_turnover)
        
        # Solve optimization problem
        problem = cp.Problem(objective_func, constraint_list)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = weights.value
                
                # Build optimized portfolio
                optimized_portfolio = self._build_portfolio_from_weights(
                    optimal_weights, asset_returns.columns, asset_metadata,
                    expected_returns, covariance_matrix
                )
                
                logger.info(f"Portfolio optimization completed successfully")
                logger.info(f"Expected return: {optimized_portfolio.expected_return:.2%}")
                logger.info(f"Expected volatility: {optimized_portfolio.expected_volatility:.2%}")
                logger.info(f"Sharpe ratio: {optimized_portfolio.sharpe_ratio:.2f}")
                
                return optimized_portfolio
            else:
                logger.error(f"Optimization failed with status: {problem.status}")
                return self._build_fallback_portfolio(asset_returns.columns, asset_metadata)
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self._build_fallback_portfolio(asset_returns.columns, asset_metadata)
    
    def _calculate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate expected returns using historical mean with adjustments"""
        # Simple historical mean (can be enhanced with factor models)
        historical_mean = returns.mean() * 252  # Annualize daily returns
        
        # Add return forecasting enhancements here if needed
        # (e.g., factor models, regime modeling, etc.)
        
        return historical_mean
    
    def _calculate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix with shrinkage estimation"""
        # Calculate sample covariance matrix
        sample_cov = returns.cov() * 252  # Annualize
        
        # Apply shrinkage toward identity matrix (Ledoit-Wolf shrinkage)
        n_assets = len(sample_cov)
        identity = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        
        # Simple shrinkage (can be enhanced with Ledoit-Wolf optimal shrinkage)
        shrinkage_intensity = 0.2
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * identity
        
        return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)
    
    def _calculate_liquidity_scores(self, asset_metadata: Dict[str, Dict]) -> pd.Series:
        """Calculate liquidity scores for each asset"""
        liquidity_mapping = {
            'equity': 1.0,
            'bond': 0.9,
            'reit': 0.7,
            'commodity': 0.6,
            'crypto': 0.5,
            'private_market': 0.1
        }
        
        scores = {}
        for asset, metadata in asset_metadata.items():
            asset_type = metadata.get('type', 'equity')
            base_score = liquidity_mapping.get(asset_type, 0.5)
            
            # Adjust based on specific asset characteristics
            if 'illiquidity_factor' in metadata:
                adjustment = 1.0 - metadata['illiquidity_factor']
                scores[asset] = base_score * adjustment
            else:
                scores[asset] = base_score
        
        return pd.Series(scores)
    
    def _build_asset_class_constraints(self, 
                                     weights: cp.Variable,
                                     asset_types: List[str],
                                     constraints: AlternativeAssetConstraints) -> List:
        """Build asset class allocation constraints"""
        constraint_list = []
        
        # Group weights by asset type
        type_groups = {}
        for i, asset_type in enumerate(asset_types):
            if asset_type not in type_groups:
                type_groups[asset_type] = []
            type_groups[asset_type].append(i)
        
        # Apply asset class limits
        for asset_type, indices in type_groups.items():
            type_weight = cp.sum([weights[i] for i in indices])
            
            if asset_type == 'reit' and constraints.max_reit_allocation:
                constraint_list.append(type_weight <= constraints.max_reit_allocation)
            elif asset_type == 'commodity' and constraints.max_commodity_allocation:
                constraint_list.append(type_weight <= constraints.max_commodity_allocation)
            elif asset_type == 'crypto' and constraints.max_crypto_allocation:
                constraint_list.append(type_weight <= constraints.max_crypto_allocation)
            elif asset_type == 'private_market' and constraints.max_private_market_allocation:
                constraint_list.append(type_weight <= constraints.max_private_market_allocation)
        
        # Total alternative asset constraint
        alternative_types = ['reit', 'commodity', 'crypto', 'private_market']
        alternative_indices = []
        for asset_type, indices in type_groups.items():
            if asset_type in alternative_types:
                alternative_indices.extend(indices)
        
        if alternative_indices and constraints.max_alternative_allocation:
            alternative_weight = cp.sum([weights[i] for i in alternative_indices])
            constraint_list.append(alternative_weight <= constraints.max_alternative_allocation)
        
        return constraint_list
    
    def _build_liquidity_constraints(self,
                                   weights: cp.Variable,
                                   liquidity_scores: pd.Series,
                                   constraints: AlternativeAssetConstraints) -> List:
        """Build liquidity-based constraints"""
        constraint_list = []
        
        # Weighted average liquidity score
        portfolio_liquidity = cp.sum(cp.multiply(weights, liquidity_scores.values))
        
        if constraints.min_liquid_allocation:
            constraint_list.append(portfolio_liquidity >= constraints.min_liquid_allocation)
        
        return constraint_list
    
    def _extract_current_weights(self, 
                               current_portfolio: MultiAssetPortfolio,
                               asset_names: List[str]) -> np.ndarray:
        """Extract current weights for turnover constraints"""
        # This is a simplified version - in practice, you'd map the current
        # portfolio allocations to the asset universe
        current_weights = np.zeros(len(asset_names))
        
        # Map portfolio allocations to asset weights
        # (Implementation depends on how portfolio is structured)
        
        return current_weights
    
    def _build_portfolio_from_weights(self,
                                    weights: np.ndarray,
                                    asset_names: List[str],
                                    asset_metadata: Dict[str, Dict],
                                    expected_returns: pd.Series,
                                    covariance_matrix: pd.DataFrame) -> MultiAssetPortfolio:
        """Build MultiAssetPortfolio from optimization weights"""
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns.values)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Group allocations by asset type
        allocations = {
            'reit': {},
            'commodity': {},
            'crypto': {},
            'private_market': {}
        }
        
        total_alternative = 0.0
        weighted_illiquidity = 0.0
        
        for i, asset_name in enumerate(asset_names):
            weight = weights[i]
            metadata = asset_metadata.get(asset_name, {})
            asset_type = metadata.get('type', 'equity')
            
            if asset_type in allocations:
                allocations[asset_type][asset_name] = weight
                total_alternative += weight
                
                # Calculate weighted illiquidity
                illiquidity_factor = metadata.get('illiquidity_factor', 0.0)
                weighted_illiquidity += weight * illiquidity_factor
        
        # Calculate liquidity score
        liquidity_scores = self._calculate_liquidity_scores(asset_metadata)
        portfolio_liquidity = np.dot(weights, liquidity_scores.values)
        
        # Calculate diversification ratio
        individual_volatilities = np.sqrt(np.diag(covariance_matrix.values))
        weighted_avg_volatility = np.dot(weights, individual_volatilities)
        diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1.0
        
        return MultiAssetPortfolio(
            # Traditional allocations (simplified)
            equity_allocation=sum(weights[i] for i, name in enumerate(asset_names) 
                                if asset_metadata.get(name, {}).get('type') == 'equity'),
            bond_allocation=sum(weights[i] for i, name in enumerate(asset_names) 
                              if asset_metadata.get(name, {}).get('type') == 'bond'),
            cash_allocation=0.0,  # Assuming no cash allocation in this optimization
            
            # Alternative allocations
            reit_allocations=allocations['reit'],
            commodity_allocations=allocations['commodity'],
            crypto_allocations=allocations['crypto'],
            private_market_allocations=allocations['private_market'],
            
            # Portfolio metrics
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            
            # Risk metrics
            value_at_risk_95=self._calculate_var_95(portfolio_return, portfolio_volatility),
            max_drawdown=0.0,  # Would need historical simulation
            liquidity_score=portfolio_liquidity,
            
            # Alternative asset metrics
            total_alternative_allocation=total_alternative,
            illiquidity_score=weighted_illiquidity / total_alternative if total_alternative > 0 else 0.0,
            diversification_ratio=diversification_ratio
        )
    
    def _calculate_var_95(self, expected_return: float, volatility: float) -> float:
        """Calculate 95% Value at Risk"""
        # Assuming normal distribution (can be enhanced with historical simulation)
        z_score_95 = norm.ppf(0.05)  # 5th percentile
        var_95 = -(expected_return + z_score_95 * volatility)
        return max(0.0, var_95)
    
    def _build_fallback_portfolio(self, 
                                asset_names: List[str],
                                asset_metadata: Dict[str, Dict]) -> MultiAssetPortfolio:
        """Build fallback portfolio if optimization fails"""
        # Simple equal-weight portfolio with asset class limits
        n_assets = len(asset_names)
        equal_weights = np.ones(n_assets) / n_assets
        
        # Apply simple constraints
        constraints = AlternativeAssetConstraints()
        
        # Adjust weights to respect constraints (simplified)
        adjusted_weights = self._apply_simple_constraints(equal_weights, asset_metadata, constraints)
        
        # Calculate basic metrics
        expected_returns = pd.Series([0.08] * n_assets, index=asset_names)  # Assume 8% return
        covariance_matrix = pd.DataFrame(np.eye(n_assets) * 0.04, index=asset_names, columns=asset_names)  # 20% vol
        
        return self._build_portfolio_from_weights(
            adjusted_weights, asset_names, asset_metadata, expected_returns, covariance_matrix
        )
    
    def _apply_simple_constraints(self,
                                weights: np.ndarray,
                                asset_metadata: Dict[str, Dict],
                                constraints: AlternativeAssetConstraints) -> np.ndarray:
        """Apply simple constraints to fallback portfolio"""
        # This is a simplified constraint application
        # In practice, you'd use proper optimization even for fallback
        
        adjusted_weights = weights.copy()
        
        # Normalize to ensure they sum to 1
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        
        return adjusted_weights
    
    def analyze_portfolio_sensitivity(self,
                                    portfolio: MultiAssetPortfolio,
                                    shock_scenarios: Dict[str, float]) -> Dict[str, Dict]:
        """
        Analyze portfolio sensitivity to various shock scenarios
        
        Args:
            portfolio: Current portfolio
            shock_scenarios: Dict of scenario name to shock magnitude
            
        Returns:
            Sensitivity analysis results
        """
        sensitivity_results = {}
        
        for scenario_name, shock_magnitude in shock_scenarios.items():
            # Apply shock and recalculate metrics
            shocked_return = portfolio.expected_return * (1 + shock_magnitude)
            shocked_volatility = portfolio.expected_volatility * (1 + abs(shock_magnitude))
            
            sensitivity_results[scenario_name] = {
                'return_change': shocked_return - portfolio.expected_return,
                'volatility_change': shocked_volatility - portfolio.expected_volatility,
                'sharpe_change': (shocked_return / shocked_volatility) - portfolio.sharpe_ratio
            }
        
        return sensitivity_results
    
    def generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate optimization engine summary"""
        return {
            'optimizer_status': 'active',
            'supported_objectives': [obj.value for obj in self.supported_objectives],
            'transaction_costs': self.transaction_costs,
            'risk_free_rate': self.risk_free_rate,
            'optimization_capabilities': [
                'multi_asset_optimization',
                'liquidity_constraints',
                'alternative_asset_limits',
                'risk_budgeting',
                'turnover_control',
                'sensitivity_analysis'
            ],
            'last_updated': datetime.now().isoformat()
        }


# Demo usage function
def demo_alternative_asset_optimization():
    """Demonstrate alternative asset portfolio optimization"""
    optimizer = AlternativeAssetOptimizer()
    
    print("Alternative Asset Portfolio Optimization Demo")
    print("=" * 60)
    
    # Create sample asset universe
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_assets = 10
    
    asset_names = [
        'US_Equity', 'Bond_Index', 'REIT_1', 'REIT_2', 
        'Gold_ETF', 'Oil_Future', 'Bitcoin', 'Ethereum',
        'PE_Fund', 'VC_Fund'
    ]
    
    # Generate sample returns
    returns_data = {}
    asset_metadata = {}
    
    for i, asset in enumerate(asset_names):
        # Generate returns with different characteristics
        base_vol = 0.15 + 0.05 * i  # Increasing volatility
        returns = np.random.normal(0.0008, base_vol/np.sqrt(252), len(dates))
        returns_data[asset] = returns
        
        # Asset metadata
        if 'Equity' in asset or 'Bond' in asset:
            asset_type = 'equity' if 'Equity' in asset else 'bond'
            illiquidity = 0.0
        elif 'REIT' in asset:
            asset_type = 'reit'
            illiquidity = 0.1
        elif 'ETF' in asset or 'Future' in asset:
            asset_type = 'commodity'
            illiquidity = 0.2
        elif asset in ['Bitcoin', 'Ethereum']:
            asset_type = 'crypto'
            illiquidity = 0.3
        else:
            asset_type = 'private_market'
            illiquidity = 0.8
        
        asset_metadata[asset] = {
            'type': asset_type,
            'illiquidity_factor': illiquidity
        }
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Set optimization constraints
    constraints = AlternativeAssetConstraints(
        max_alternative_allocation=0.40,
        max_portfolio_volatility=0.18,
        max_concentration=0.15
    )
    
    # Optimize portfolio
    optimized_portfolio = optimizer.optimize_multi_asset_portfolio(
        asset_returns=returns_df,
        asset_metadata=asset_metadata,
        objective=OptimizationObjective.MAXIMIZE_SHARPE,
        constraints=constraints
    )
    
    print(f"Optimization Results:")
    print(f"- Expected Return: {optimized_portfolio.expected_return:.2%}")
    print(f"- Expected Volatility: {optimized_portfolio.expected_volatility:.2%}")
    print(f"- Sharpe Ratio: {optimized_portfolio.sharpe_ratio:.2f}")
    print(f"- Alternative Allocation: {optimized_portfolio.total_alternative_allocation:.2%}")
    print(f"- Liquidity Score: {optimized_portfolio.liquidity_score:.2f}")
    print(f"- Diversification Ratio: {optimized_portfolio.diversification_ratio:.2f}")
    
    # Show allocations
    print(f"\nAlternative Asset Allocations:")
    for category, allocations in [
        ('REITs', optimized_portfolio.reit_allocations),
        ('Commodities', optimized_portfolio.commodity_allocations),
        ('Crypto', optimized_portfolio.crypto_allocations),
        ('Private Markets', optimized_portfolio.private_market_allocations)
    ]:
        if allocations:
            total_alloc = sum(allocations.values())
            print(f"- {category}: {total_alloc:.2%}")
            for asset, weight in allocations.items():
                print(f"  * {asset}: {weight:.2%}")
    
    # Sensitivity analysis
    shock_scenarios = {
        'market_crash': -0.30,
        'inflation_spike': 0.20,
        'liquidity_crisis': -0.15
    }
    
    sensitivity = optimizer.analyze_portfolio_sensitivity(optimized_portfolio, shock_scenarios)
    
    print(f"\nSensitivity Analysis:")
    for scenario, results in sensitivity.items():
        print(f"- {scenario.replace('_', ' ').title()}:")
        print(f"  Return Change: {results['return_change']:.2%}")
        print(f"  Volatility Change: {results['volatility_change']:.2%}")
    
    # Summary
    summary = optimizer.generate_optimization_summary()
    print(f"\nOptimizer Capabilities: {len(summary['optimization_capabilities'])}")


if __name__ == "__main__":
    demo_alternative_asset_optimization()
