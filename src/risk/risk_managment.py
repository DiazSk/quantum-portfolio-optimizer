"""
Advanced Risk Management System
Implements VaR, CVaR, stress testing, and real-time risk monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
import scipy.stats as ss

# Risk metrics
from pypfopt import risk_models
import riskfolio as rp

# Monte Carlo
from numpy.random import multivariate_normal

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for portfolio risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    beta: float
    tracking_error: float
    information_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    tail_ratio: float


class RiskManager:
    """
    Comprehensive risk management system for portfolio optimization
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.risk_limits = self._set_default_limits()
        self.stress_scenarios = self._define_stress_scenarios()
        self.risk_history = []
        
    def _set_default_limits(self) -> Dict:
        """Set default risk limits for portfolio"""
        return {
            'max_var_95': 0.05,  # 5% VaR at 95% confidence
            'max_var_99': 0.10,  # 10% VaR at 99% confidence
            'max_drawdown': 0.15,  # 15% maximum drawdown
            'max_leverage': 1.5,  # 150% gross exposure
            'max_concentration': 0.30,  # 30% max single position
            'min_liquidity_ratio': 0.20,  # 20% in liquid assets
            'max_correlation': 0.70  # 70% max correlation between assets
        }
    
    def _define_stress_scenarios(self) -> Dict:
        """Define stress test scenarios"""
        return {
            'market_crash_2008': {
                'equity_shock': -0.40,
                'credit_spread': 0.05,
                'volatility_shock': 2.5,
                'correlation_shock': 0.30
            },
            'covid_2020': {
                'equity_shock': -0.35,
                'credit_spread': 0.03,
                'volatility_shock': 3.0,
                'correlation_shock': 0.40
            },
            'tech_bubble_2000': {
                'equity_shock': -0.45,
                'credit_spread': 0.02,
                'volatility_shock': 2.0,
                'correlation_shock': 0.25
            },
            'black_monday_1987': {
                'equity_shock': -0.22,
                'credit_spread': 0.01,
                'volatility_shock': 5.0,
                'correlation_shock': 0.50
            },
            'euro_crisis_2011': {
                'equity_shock': -0.20,
                'credit_spread': 0.04,
                'volatility_shock': 1.8,
                'correlation_shock': 0.35
            }
        }
    
    def calculate_var_cvar(self, returns: pd.DataFrame, 
                          weights: np.ndarray,
                          method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk
        
        Args:
            returns: Historical returns
            weights: Portfolio weights
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            Dictionary with VaR and CVaR at different confidence levels
        """
        portfolio_returns = (returns @ weights).dropna()
        
        results = {}
        
        if method == 'historical':
            for conf in self.confidence_levels:
                var = np.percentile(portfolio_returns, (1 - conf) * 100)
                cvar = portfolio_returns[portfolio_returns <= var].mean()
                results[f'var_{int(conf*100)}'] = abs(var)
                results[f'cvar_{int(conf*100)}'] = abs(cvar)
                
        elif method == 'parametric':
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            
            for conf in self.confidence_levels:
                z_score = stats.norm.ppf(1 - conf)
                var = -(mean + z_score * std)
                # CVaR formula for normal distribution
                cvar = -mean + std * stats.norm.pdf(z_score) / (1 - conf)
                results[f'var_{int(conf*100)}'] = var
                results[f'cvar_{int(conf*100)}'] = cvar
                
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            n_days = 252  # Annual
            
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Generate scenarios
            random_returns = multivariate_normal(
                mean_returns, cov_matrix, (n_simulations, n_days)
            )
            
            # Calculate portfolio returns for each scenario
            portfolio_scenarios = random_returns @ weights
            final_returns = portfolio_scenarios.sum(axis=1)
            
            for conf in self.confidence_levels:
                var = np.percentile(final_returns, (1 - conf) * 100)
                cvar = final_returns[final_returns <= var].mean()
                results[f'var_{int(conf*100)}'] = abs(var)
                results[f'cvar_{int(conf*100)}'] = abs(cvar)
        
        return results
    
    def calculate_drawdown(self, prices: pd.DataFrame, 
                          weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate drawdown metrics
        
        Args:
            prices: Historical prices
            weights: Portfolio weights
        
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate portfolio value
        portfolio_value = (prices * weights).sum(axis=1)
        
        # Calculate running maximum
        running_max = portfolio_value.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_value - running_max) / running_max
        
        return {
            'current_drawdown': drawdown.iloc[-1],
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0,
            'drawdown_duration': self._calculate_drawdown_duration(drawdown),
            'recovery_time': self._calculate_recovery_time(drawdown)
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate longest drawdown duration in days"""
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_duration = 0
        
        for is_drawdown in in_drawdown:
            if is_drawdown:
                current_duration += 1
            elif current_duration > 0:
                drawdown_periods.append(current_duration)
                current_duration = 0
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate average recovery time from drawdowns"""
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_times.append(i - drawdown_start)
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def stress_test_portfolio(self, returns: pd.DataFrame,
                            weights: np.ndarray,
                            scenario: str = 'market_crash_2008') -> Dict:
        """
        Perform stress testing on portfolio
        
        Args:
            returns: Historical returns
            weights: Portfolio weights
            scenario: Stress scenario to apply
        
        Returns:
            Dictionary with stress test results
        """
        if scenario not in self.stress_scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        scenario_params = self.stress_scenarios[scenario]
        
        # Apply shocks to returns
        stressed_returns = returns.copy()
        
        # Apply equity shock
        equity_shock = scenario_params['equity_shock']
        stressed_returns = stressed_returns * (1 + equity_shock)
        
        # Increase correlation
        correlation_shock = scenario_params['correlation_shock']
        cov_matrix = returns.cov()
        correlation_matrix = returns.corr()
        
        # Increase correlations (move towards 1)
        stressed_correlation = correlation_matrix + (1 - correlation_matrix) * correlation_shock
        np.fill_diagonal(stressed_correlation.values, 1)
        
        # Reconstruct covariance matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        stressed_cov = np.outer(std_devs, std_devs) * stressed_correlation
        
        # Calculate stressed portfolio metrics
        stressed_portfolio_return = (stressed_returns @ weights).mean()
        stressed_portfolio_std = np.sqrt(weights @ stressed_cov @ weights)
        
        # Calculate losses
        normal_return = (returns @ weights).mean()
        stress_loss = stressed_portfolio_return - normal_return
        
        # Calculate stressed VaR
        stressed_var = self.calculate_var_cvar(stressed_returns, weights)
        
        return {
            'scenario': scenario,
            'stress_loss': stress_loss,
            'stressed_return': stressed_portfolio_return,
            'stressed_volatility': stressed_portfolio_std,
            'stressed_var_95': stressed_var['var_95'],
            'stressed_var_99': stressed_var['var_99'],
            'impact_percentage': (stress_loss / normal_return) * 100
        }
    
    def calculate_risk_metrics(self, returns: pd.DataFrame,
                              weights: np.ndarray,
                              benchmark_returns: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.02) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Asset returns
            weights: Portfolio weights
            benchmark_returns: Benchmark returns for relative metrics
            risk_free_rate: Risk-free rate for Sharpe/Sortino
        
        Returns:
            RiskMetrics object with all risk measures
        """
        portfolio_returns = (returns @ weights).dropna()
        
        # VaR and CVaR
        var_cvar = self.calculate_var_cvar(returns, weights)
        
        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Beta (if benchmark provided)
        if benchmark_returns is not None:
            covariance = portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Tracking error
            tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
            
            # Information ratio
            excess_returns = portfolio_returns - benchmark_returns
            information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
        else:
            beta = 1.0
            tracking_error = 0
            information_ratio = 0
        
        # Downside deviation (for Sortino)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        excess_return = portfolio_returns.mean() * 252 - risk_free_rate
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        annual_return = portfolio_returns.mean() * 252
        calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0
        
        # Tail ratio
        percentile_95 = np.percentile(portfolio_returns, 95)
        percentile_5 = np.percentile(portfolio_returns, 5)
        tail_ratio = abs(percentile_95 / percentile_5) if percentile_5 != 0 else 0
        
        return RiskMetrics(
            var_95=var_cvar['var_95'],
            var_99=var_cvar['var_99'],
            cvar_95=var_cvar['cvar_95'],
            cvar_99=var_cvar['cvar_99'],
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            tail_ratio=tail_ratio
        )
    
    def check_risk_limits(self, risk_metrics: RiskMetrics,
                         weights: np.ndarray) -> Dict[str, bool]:
        """
        Check if portfolio violates risk limits
        
        Args:
            risk_metrics: Calculated risk metrics
            weights: Portfolio weights
        
        Returns:
            Dictionary with limit violations
        """
        violations = {}
        
        # VaR limits
        violations['var_95_exceeded'] = risk_metrics.var_95 > self.risk_limits['max_var_95']
        violations['var_99_exceeded'] = risk_metrics.var_99 > self.risk_limits['max_var_99']
        
        # Drawdown limit
        violations['drawdown_exceeded'] = risk_metrics.max_drawdown > self.risk_limits['max_drawdown']
        
        # Concentration limit
        max_weight = weights.max()
        violations['concentration_exceeded'] = max_weight > self.risk_limits['max_concentration']
        
        # Log violations
        for limit, violated in violations.items():
            if violated:
                logger.warning(f"Risk limit violated: {limit}")
        
        return violations
    
    def calculate_risk_contribution(self, returns: pd.DataFrame,
                                  weights: np.ndarray) -> pd.DataFrame:
        """
        Calculate risk contribution of each asset
        
        Args:
            returns: Asset returns
            weights: Portfolio weights
        
        Returns:
            DataFrame with risk contributions
        """
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Marginal risk contribution
        marginal_contrib = (cov_matrix @ weights) / portfolio_std
        
        # Component risk contribution
        component_contrib = weights * marginal_contrib
        
        # Percentage contribution
        pct_contrib = component_contrib / portfolio_std
        
        return pd.DataFrame({
            'Weight': weights,
            'Marginal_Risk': marginal_contrib,
            'Component_Risk': component_contrib,
            'Risk_Contribution_%': pct_contrib * 100
        }, index=returns.columns)
    
    def generate_risk_report(self, returns: pd.DataFrame,
                           weights: np.ndarray,
                           prices: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk report
        
        Args:
            returns: Asset returns
            weights: Portfolio weights
            prices: Asset prices
        
        Returns:
            Dictionary with complete risk analysis
        """
        # Calculate all risk metrics
        risk_metrics = self.calculate_risk_metrics(returns, weights)
        
        # Drawdown analysis
        drawdown_analysis = self.calculate_drawdown(prices, weights)
        
        # Risk contributions
        risk_contributions = self.calculate_risk_contribution(returns, weights)
        
        # Stress testing
        stress_results = {}
        for scenario in self.stress_scenarios.keys():
            stress_results[scenario] = self.stress_test_portfolio(
                returns, weights, scenario
            )
        
        # Check risk limits
        limit_violations = self.check_risk_limits(risk_metrics, weights)
        
        # Correlation analysis
        correlation_matrix = returns.corr()
        max_correlation = correlation_matrix.values[np.triu_indices_from(
            correlation_matrix.values, k=1)].max()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': {
                'VaR_95': f"{risk_metrics.var_95:.2%}",
                'VaR_99': f"{risk_metrics.var_99:.2%}",
                'CVaR_95': f"{risk_metrics.cvar_95:.2%}",
                'CVaR_99': f"{risk_metrics.cvar_99:.2%}",
                'Volatility': f"{risk_metrics.volatility:.2%}",
                'Max_Drawdown': f"{risk_metrics.max_drawdown:.2%}",
                'Beta': f"{risk_metrics.beta:.3f}",
                'Sortino_Ratio': f"{risk_metrics.sortino_ratio:.3f}",
                'Calmar_Ratio': f"{risk_metrics.calmar_ratio:.3f}"
            },
            'drawdown_analysis': drawdown_analysis,
            'risk_contributions': risk_contributions.to_dict(),
            'stress_test_summary': {
                scenario: {
                    'loss': f"{result['stress_loss']:.2%}",
                    'impact': f"{result['impact_percentage']:.1f}%"
                }
                for scenario, result in stress_results.items()
            },
            'limit_violations': limit_violations,
            'max_correlation': f"{max_correlation:.2f}",
            'risk_score': self._calculate_risk_score(risk_metrics, limit_violations)
        }
        
        # Store in history
        self.risk_history.append(report)
        
        return report
    
    def _calculate_risk_score(self, risk_metrics: RiskMetrics,
                            violations: Dict[str, bool]) -> float:
        """
        Calculate overall risk score (0-100, lower is better)
        
        Args:
            risk_metrics: Risk metrics
            violations: Risk limit violations
        
        Returns:
            Risk score
        """
        score = 0
        
        # Penalty for VaR
        score += min(risk_metrics.var_95 * 100, 30)  # Max 30 points
        
        # Penalty for drawdown
        score += min(risk_metrics.max_drawdown * 100, 25)  # Max 25 points
        
        # Penalty for volatility
        score += min(risk_metrics.volatility * 50, 20)  # Max 20 points
        
        # Penalty for violations
        score += sum(violations.values()) * 5  # 5 points per violation
        
        # Bonus for good Sharpe/Sortino
        if risk_metrics.sortino_ratio > 1.5:
            score -= 10
        elif risk_metrics.sortino_ratio > 1.0:
            score -= 5
        
        return max(0, min(100, score))
    
    def plot_risk_dashboard(self, report: Dict) -> go.Figure:
        """
        Create interactive risk dashboard
        
        Args:
            report: Risk report dictionary
        
        Returns:
            Plotly figure with risk dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Metrics', 'Stress Test Results', 
                          'Risk Contributions', 'Risk Score'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'pie'}, {'type': 'indicator'}]]
        )
        
        # Risk Metrics
        metrics = report['risk_metrics']
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=[float(v.strip('%')) for v in metrics.values() if '%' in v],
                name='Risk Metrics',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Stress Test Results
        stress_data = report['stress_test_summary']
        scenarios = list(stress_data.keys())
        losses = [float(stress_data[s]['loss'].strip('%')) for s in scenarios]
        
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=losses,
                name='Stress Losses',
                marker_color='coral'
            ),
            row=1, col=2
        )
        
        # Risk Contributions (pie chart)
        risk_contrib = report['risk_contributions']['Risk_Contribution_%']
        fig.add_trace(
            go.Pie(
                labels=list(risk_contrib.keys()),
                values=list(risk_contrib.values()),
                name='Risk Contribution'
            ),
            row=2, col=1
        )
        
        # Risk Score (gauge)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=report['risk_score'],
                title={'text': "Risk Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 66
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Portfolio Risk Dashboard",
            showlegend=False,
            height=600
        )
        
        return fig


# Regulatory Compliance Module
class RegulatoryCompliance:
    """
    Ensure portfolio compliance with regulatory requirements
    MiFID II, SEC Rule 613, UCITS, etc.
    """
    
    def __init__(self):
        self.regulations = self._load_regulations()
        self.compliance_log = []
        
    def _load_regulations(self) -> Dict:
        """Load regulatory requirements"""
        return {
            'mifid_ii': {
                'max_leverage': 2.0,
                'min_diversification': 16,  # Min 16 holdings for UCITS
                'max_single_position': 0.10,  # 10% for UCITS
                'reporting_frequency': 'daily',
                'transaction_reporting': True
            },
            'sec_rule_613': {
                'cat_reporting': True,
                'audit_trail': True,
                'timestamp_precision': 'microsecond'
            },
            'volcker_rule': {
                'proprietary_trading': False,
                'covered_funds': False
            }
        }
    
    def check_mifid_compliance(self, portfolio: Dict) -> Dict[str, bool]:
        """Check MiFID II compliance"""
        mifid = self.regulations['mifid_ii']
        
        checks = {
            'leverage_compliant': portfolio.get('leverage', 1) <= mifid['max_leverage'],
            'diversification_compliant': len(portfolio.get('holdings', [])) >= mifid['min_diversification'],
            'concentration_compliant': all(
                w <= mifid['max_single_position'] 
                for w in portfolio.get('weights', {}).values()
            )
        }
        
        # Log compliance check
        self.compliance_log.append({
            'timestamp': datetime.now().isoformat(),
            'regulation': 'MiFID II',
            'checks': checks,
            'compliant': all(checks.values())
        })
        
        return checks
    
    def generate_regulatory_report(self, portfolio: Dict, 
                                  trades: List[Dict]) -> Dict:
        """Generate regulatory compliance report"""
        report = {
            'report_date': datetime.now().isoformat(),
            'portfolio_id': portfolio.get('id', 'N/A'),
            'mifid_compliance': self.check_mifid_compliance(portfolio),
            'trade_count': len(trades),
            'total_volume': sum(t.get('value', 0) for t in trades),
            'compliance_score': self._calculate_compliance_score(portfolio)
        }
        
        return report
    
    def _calculate_compliance_score(self, portfolio: Dict) -> float:
        """Calculate overall compliance score"""
        checks = self.check_mifid_compliance(portfolio)
        return sum(checks.values()) / len(checks) * 100


# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager()
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_days = 252
    
    # Simulate returns
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            [0.0001] * n_assets,
            np.eye(n_assets) * 0.01,
            n_days
        ),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Simulate prices
    prices = (1 + returns).cumprod() * 100
    
    # Random weights
    weights = np.random.dirichlet(np.ones(n_assets))
    
    # Generate risk report
    print("Generating risk report...")
    report = risk_manager.generate_risk_report(returns, weights, prices)
    
    # Display results
    print("\n" + "="*60)
    print("RISK MANAGEMENT REPORT")
    print("="*60)
    
    print("\n--- Risk Metrics ---")
    for metric, value in report['risk_metrics'].items():
        print(f"{metric:15s}: {value}")
    
    print("\n--- Stress Test Summary ---")
    for scenario, result in report['stress_test_summary'].items():
        print(f"{scenario:20s}: Loss={result['loss']}, Impact={result['impact']}")
    
    print("\n--- Risk Limit Violations ---")
    violations = report['limit_violations']
    if any(violations.values()):
        for limit, violated in violations.items():
            if violated:
                print(f"⚠️  {limit}")
    else:
        print("✓ All risk limits satisfied")
    
    print(f"\n--- Overall Risk Score: {report['risk_score']:.1f}/100 ---")
    
    # Test regulatory compliance
    print("\n--- Regulatory Compliance ---")
    compliance = RegulatoryCompliance()
    
    portfolio = {
        'id': 'PORT_001',
        'leverage': 1.2,
        'holdings': list(range(20)),
        'weights': dict(zip(returns.columns, weights))
    }
    
    compliance_report = compliance.generate_regulatory_report(portfolio, [])
    print(f"MiFID II Compliance: {compliance_report['mifid_compliance']}")
    print(f"Compliance Score: {compliance_report['compliance_score']:.1f}%")