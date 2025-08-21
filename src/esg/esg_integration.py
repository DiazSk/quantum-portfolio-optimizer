"""
ESG (Environmental, Social, Governance) Integration System
Epic 6.3: Enterprise ESG scoring, ratings, and sustainability analytics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import requests
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ESGScore:
    """ESG scoring data structure"""
    ticker: str
    overall_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    sustainability_rank: int
    carbon_intensity: float
    last_updated: datetime
    data_source: str
    controversy_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['last_updated'] = data['last_updated'].isoformat()
        return data

@dataclass
class ESGReport:
    """ESG analysis report"""
    report_id: str
    portfolio_data: Dict[str, Any]
    esg_analysis: Dict[str, Any]
    sustainability_metrics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    improvement_recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['generated_at'] = data['generated_at'].isoformat()
        return data

class ESGDataProvider(ABC):
    """Abstract base class for ESG data providers"""
    
    @abstractmethod
    async def get_esg_score(self, ticker: str) -> Optional[ESGScore]:
        """Get ESG score for a ticker"""
        pass
    
    @abstractmethod
    async def get_industry_benchmarks(self, industry: str) -> Dict[str, float]:
        """Get industry ESG benchmarks"""
        pass

class MockESGProvider(ESGDataProvider):
    """Mock ESG data provider for development/testing"""
    
    def __init__(self):
        logger.info("Using Mock ESG Provider for development")
        
        # Mock ESG scores for common tickers
        self.mock_scores = {
            'AAPL': ESGScore(
                ticker='AAPL',
                overall_score=8.2,
                environmental_score=7.8,
                social_score=8.5,
                governance_score=8.3,
                sustainability_rank=15,
                carbon_intensity=12.5,
                last_updated=datetime.now(),
                data_source='MockProvider',
                controversy_level='Low'
            ),
            'MSFT': ESGScore(
                ticker='MSFT',
                overall_score=8.7,
                environmental_score=8.9,
                social_score=8.6,
                governance_score=8.6,
                sustainability_rank=8,
                carbon_intensity=8.2,
                last_updated=datetime.now(),
                data_source='MockProvider',
                controversy_level='Low'
            ),
            'GOOGL': ESGScore(
                ticker='GOOGL',
                overall_score=7.9,
                environmental_score=8.1,
                social_score=7.5,
                governance_score=8.1,
                sustainability_rank=22,
                carbon_intensity=15.8,
                last_updated=datetime.now(),
                data_source='MockProvider',
                controversy_level='Medium'
            ),
            'TSLA': ESGScore(
                ticker='TSLA',
                overall_score=6.8,
                environmental_score=9.2,
                social_score=5.1,
                governance_score=6.1,
                sustainability_rank=45,
                carbon_intensity=3.2,
                last_updated=datetime.now(),
                data_source='MockProvider',
                controversy_level='High'
            ),
            'AMZN': ESGScore(
                ticker='AMZN',
                overall_score=7.1,
                environmental_score=6.8,
                social_score=7.2,
                governance_score=7.3,
                sustainability_rank=38,
                carbon_intensity=22.1,
                last_updated=datetime.now(),
                data_source='MockProvider',
                controversy_level='Medium'
            )
        }
        
        # Mock industry benchmarks
        self.industry_benchmarks = {
            'Technology': {
                'overall_score': 7.8,
                'environmental_score': 7.5,
                'social_score': 7.9,
                'governance_score': 8.0,
                'carbon_intensity': 12.5
            },
            'Energy': {
                'overall_score': 5.2,
                'environmental_score': 4.1,
                'social_score': 5.8,
                'governance_score': 5.7,
                'carbon_intensity': 45.2
            },
            'Healthcare': {
                'overall_score': 6.9,
                'environmental_score': 6.5,
                'social_score': 7.8,
                'governance_score': 6.4,
                'carbon_intensity': 8.9
            },
            'Financial': {
                'overall_score': 6.5,
                'environmental_score': 5.8,
                'social_score': 6.8,
                'governance_score': 7.0,
                'carbon_intensity': 5.2
            }
        }
    
    async def get_esg_score(self, ticker: str) -> Optional[ESGScore]:
        """Get ESG score for a ticker"""
        await asyncio.sleep(0.1)  # Simulate API call
        
        if ticker in self.mock_scores:
            return self.mock_scores[ticker]
        
        # Generate mock score for unknown ticker
        return ESGScore(
            ticker=ticker,
            overall_score=np.random.normal(6.5, 1.5),
            environmental_score=np.random.normal(6.8, 1.2),
            social_score=np.random.normal(6.3, 1.4),
            governance_score=np.random.normal(6.7, 1.1),
            sustainability_rank=np.random.randint(1, 100),
            carbon_intensity=np.random.normal(18.5, 8.2),
            last_updated=datetime.now(),
            data_source='MockProvider',
            controversy_level=np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
        )
    
    async def get_industry_benchmarks(self, industry: str) -> Dict[str, float]:
        """Get industry ESG benchmarks"""
        await asyncio.sleep(0.05)
        
        if industry in self.industry_benchmarks:
            return self.industry_benchmarks[industry]
        
        # Default benchmarks
        return {
            'overall_score': 6.0,
            'environmental_score': 6.0,
            'social_score': 6.0,
            'governance_score': 6.0,
            'carbon_intensity': 20.0
        }

class RefinitivESGProvider(ESGDataProvider):
    """Refinitiv (LSEG) ESG data provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.refinitiv.com/data/environmental-social-governance/v1/"
        
        if not api_key:
            logger.warning("Refinitiv API key not provided - falling back to mock data")
            self.mock_provider = MockESGProvider()
    
    async def get_esg_score(self, ticker: str) -> Optional[ESGScore]:
        """Get ESG score from Refinitiv API"""
        if not self.api_key:
            return await self.mock_provider.get_esg_score(ticker)
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.base_url}views/scores-full"
            params = {'universe': ticker}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('data'):
                return None
            
            score_data = data['data'][0]
            
            return ESGScore(
                ticker=ticker,
                overall_score=score_data.get('ESGScore', 0),
                environmental_score=score_data.get('EnvironmentPillarScore', 0),
                social_score=score_data.get('SocialPillarScore', 0),
                governance_score=score_data.get('GovernancePillarScore', 0),
                sustainability_rank=score_data.get('ESGPercentile', 0),
                carbon_intensity=score_data.get('CarbonIntensity', 0),
                last_updated=datetime.now(),
                data_source='Refinitiv',
                controversy_level=score_data.get('ESGControversiesScore', 'Unknown')
            )
            
        except Exception as e:
            logger.error(f"Error fetching ESG data from Refinitiv: {e}")
            return await self.mock_provider.get_esg_score(ticker)
    
    async def get_industry_benchmarks(self, industry: str) -> Dict[str, float]:
        """Get industry benchmarks from Refinitiv"""
        if not self.api_key:
            return await self.mock_provider.get_industry_benchmarks(industry)
        
        # Implementation would use Refinitiv industry benchmark API
        # For now, fall back to mock data
        return await self.mock_provider.get_industry_benchmarks(industry)

class ESGIntegration:
    """Main ESG integration system"""
    
    def __init__(self, esg_provider: Optional[ESGDataProvider] = None):
        """Initialize ESG integration system"""
        self.esg_provider = esg_provider or MockESGProvider()
        self.esg_cache = {}
        self.cache_ttl = timedelta(hours=6)  # Cache for 6 hours
        
        logger.info("ESG Integration System initialized")
    
    async def get_portfolio_esg_scores(self, holdings: Dict[str, float]) -> Dict[str, ESGScore]:
        """Get ESG scores for all portfolio holdings"""
        logger.info(f"Fetching ESG scores for {len(holdings)} holdings")
        
        scores = {}
        for ticker, shares in holdings.items():
            # Check cache first
            if self._is_cached(ticker):
                scores[ticker] = self.esg_cache[ticker]['score']
                continue
            
            # Fetch from provider
            esg_score = await self.esg_provider.get_esg_score(ticker)
            if esg_score:
                scores[ticker] = esg_score
                self._cache_score(ticker, esg_score)
        
        return scores
    
    async def calculate_portfolio_esg_metrics(self, 
                                            holdings: Dict[str, float], 
                                            market_values: Dict[str, float]) -> Dict[str, Any]:
        """Calculate weighted ESG metrics for portfolio"""
        logger.info("Calculating portfolio ESG metrics")
        
        esg_scores = await self.get_portfolio_esg_scores(holdings)
        
        if not esg_scores:
            return {
                'weighted_esg_score': 0,
                'portfolio_carbon_intensity': 0,
                'sustainability_rank': 0,
                'controversy_exposure': 'Unknown',
                'sector_analysis': {}
            }
        
        total_value = sum(market_values.values())
        
        # Calculate weighted scores
        weighted_overall = 0
        weighted_environmental = 0
        weighted_social = 0
        weighted_governance = 0
        weighted_carbon = 0
        
        controversy_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        
        for ticker, score in esg_scores.items():
            if ticker in market_values:
                weight = market_values[ticker] / total_value
                
                weighted_overall += score.overall_score * weight
                weighted_environmental += score.environmental_score * weight
                weighted_social += score.social_score * weight
                weighted_governance += score.governance_score * weight
                weighted_carbon += score.carbon_intensity * weight
                
                controversy_counts[score.controversy_level] += weight
        
        # Determine overall controversy level
        max_controversy = max(controversy_counts.items(), key=lambda x: x[1])
        
        return {
            'weighted_esg_score': round(weighted_overall, 2),
            'environmental_score': round(weighted_environmental, 2),
            'social_score': round(weighted_social, 2),
            'governance_score': round(weighted_governance, 2),
            'portfolio_carbon_intensity': round(weighted_carbon, 2),
            'controversy_exposure': max_controversy[0],
            'holdings_coverage': len(esg_scores) / len(holdings),
            'esg_distribution': {
                'high_performers': len([s for s in esg_scores.values() if s.overall_score >= 8.0]),
                'medium_performers': len([s for s in esg_scores.values() if 6.0 <= s.overall_score < 8.0]),
                'low_performers': len([s for s in esg_scores.values() if s.overall_score < 6.0])
            }
        }
    
    async def generate_esg_report(self, 
                                portfolio_data: Dict[str, Any],
                                market_values: Dict[str, float]) -> ESGReport:
        """Generate comprehensive ESG analysis report"""
        logger.info("Generating ESG analysis report")
        
        report_id = f"ESG_RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        holdings = portfolio_data.get('holdings', {})
        
        # Calculate portfolio ESG metrics
        esg_metrics = await self.calculate_portfolio_esg_metrics(holdings, market_values)
        
        # Get individual scores for analysis
        esg_scores = await self.get_portfolio_esg_scores(holdings)
        
        # Sustainability analysis
        sustainability_metrics = await self._analyze_sustainability(esg_scores, market_values)
        
        # Risk assessment
        risk_assessment = await self._assess_esg_risks(esg_scores, esg_metrics)
        
        # Generate improvement recommendations
        recommendations = await self._generate_recommendations(esg_metrics, esg_scores)
        
        return ESGReport(
            report_id=report_id,
            portfolio_data=portfolio_data,
            esg_analysis=esg_metrics,
            sustainability_metrics=sustainability_metrics,
            risk_assessment=risk_assessment,
            improvement_recommendations=recommendations,
            generated_at=datetime.now()
        )
    
    async def screen_investments(self, 
                               tickers: List[str], 
                               esg_criteria: Dict[str, float]) -> Dict[str, bool]:
        """Screen investments based on ESG criteria"""
        logger.info(f"Screening {len(tickers)} investments with ESG criteria")
        
        results = {}
        
        for ticker in tickers:
            esg_score = await self.esg_provider.get_esg_score(ticker)
            
            if not esg_score:
                results[ticker] = False
                continue
            
            # Check against criteria
            passes_screen = True
            
            if 'min_overall_score' in esg_criteria:
                if esg_score.overall_score < esg_criteria['min_overall_score']:
                    passes_screen = False
            
            if 'min_environmental_score' in esg_criteria:
                if esg_score.environmental_score < esg_criteria['min_environmental_score']:
                    passes_screen = False
            
            if 'max_carbon_intensity' in esg_criteria:
                if esg_score.carbon_intensity > esg_criteria['max_carbon_intensity']:
                    passes_screen = False
            
            if 'exclude_controversies' in esg_criteria:
                excluded_levels = esg_criteria['exclude_controversies']
                if esg_score.controversy_level in excluded_levels:
                    passes_screen = False
            
            results[ticker] = passes_screen
        
        return results
    
    async def get_esg_leaders(self, universe: List[str], top_n: int = 10) -> List[Tuple[str, ESGScore]]:
        """Get top ESG performers from investment universe"""
        logger.info(f"Finding top {top_n} ESG leaders from {len(universe)} securities")
        
        scored_securities = []
        
        for ticker in universe:
            esg_score = await self.esg_provider.get_esg_score(ticker)
            if esg_score:
                scored_securities.append((ticker, esg_score))
        
        # Sort by overall ESG score
        scored_securities.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return scored_securities[:top_n]
    
    async def _analyze_sustainability(self, 
                                    esg_scores: Dict[str, ESGScore], 
                                    market_values: Dict[str, float]) -> Dict[str, Any]:
        """Analyze portfolio sustainability metrics"""
        
        if not esg_scores:
            return {'carbon_footprint': 0, 'green_revenue_exposure': 0}
        
        total_value = sum(market_values.values())
        
        # Calculate carbon footprint
        portfolio_carbon = 0
        for ticker, score in esg_scores.items():
            if ticker in market_values:
                weight = market_values[ticker] / total_value
                portfolio_carbon += score.carbon_intensity * weight
        
        # Estimate green revenue exposure (simplified)
        green_exposure = 0
        for ticker, score in esg_scores.items():
            if ticker in market_values and score.environmental_score >= 8.0:
                weight = market_values[ticker] / total_value
                green_exposure += weight * 0.3  # Assume 30% green revenue for high E score
        
        return {
            'carbon_footprint': round(portfolio_carbon, 2),
            'green_revenue_exposure': round(green_exposure * 100, 1),
            'sustainability_trend': 'Improving' if portfolio_carbon < 15 else 'Stable',
            'climate_alignment': 'Paris Agreement Compatible' if portfolio_carbon < 10 else 'Needs Improvement'
        }
    
    async def _assess_esg_risks(self, 
                              esg_scores: Dict[str, ESGScore], 
                              esg_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ESG-related investment risks"""
        
        risk_level = 'Low'
        risk_factors = []
        
        # High controversy exposure
        if esg_metrics.get('controversy_exposure') == 'High':
            risk_level = 'High'
            risk_factors.append('High controversy exposure')
        
        # Low overall ESG score
        if esg_metrics.get('weighted_esg_score', 0) < 5.0:
            risk_level = 'High'
            risk_factors.append('Low portfolio ESG score')
        
        # High carbon intensity
        if esg_metrics.get('portfolio_carbon_intensity', 0) > 25:
            if risk_level == 'Low':
                risk_level = 'Medium'
            risk_factors.append('High carbon intensity')
        
        # Governance concerns
        if esg_metrics.get('governance_score', 0) < 6.0:
            if risk_level == 'Low':
                risk_level = 'Medium'
            risk_factors.append('Governance concerns')
        
        return {
            'overall_risk_level': risk_level,
            'risk_factors': risk_factors,
            'regulatory_risk': 'Medium' if esg_metrics.get('portfolio_carbon_intensity', 0) > 20 else 'Low',
            'reputational_risk': esg_metrics.get('controversy_exposure', 'Low'),
            'transition_risk': 'High' if esg_metrics.get('environmental_score', 0) < 6.0 else 'Medium'
        }
    
    async def _generate_recommendations(self, 
                                      esg_metrics: Dict[str, Any], 
                                      esg_scores: Dict[str, ESGScore]) -> List[str]:
        """Generate ESG improvement recommendations"""
        
        recommendations = []
        
        # Overall ESG score recommendations
        if esg_metrics.get('weighted_esg_score', 0) < 7.0:
            recommendations.append("Consider increasing allocation to high-ESG scoring securities (8.0+)")
        
        # Environmental recommendations
        if esg_metrics.get('portfolio_carbon_intensity', 0) > 20:
            recommendations.append("Reduce portfolio carbon intensity by divesting from high-carbon sectors")
            recommendations.append("Consider green bonds or renewable energy investments")
        
        # Social recommendations
        if esg_metrics.get('social_score', 0) < 6.5:
            recommendations.append("Increase exposure to companies with strong labor practices and community impact")
        
        # Governance recommendations
        if esg_metrics.get('governance_score', 0) < 7.0:
            recommendations.append("Focus on companies with strong board diversity and executive compensation alignment")
        
        # Controversy recommendations
        if esg_metrics.get('controversy_exposure') in ['Medium', 'High']:
            recommendations.append("Review holdings with ESG controversies and consider divestment")
        
        # Diversification recommendations
        distribution = esg_metrics.get('esg_distribution', {})
        if distribution.get('low_performers', 0) > distribution.get('high_performers', 0):
            recommendations.append("Rebalance portfolio to increase high-ESG performers concentration")
        
        # If portfolio is already strong
        if not recommendations:
            recommendations.append("Portfolio demonstrates strong ESG characteristics - maintain current approach")
            recommendations.append("Consider ESG impact measurement and reporting to stakeholders")
        
        return recommendations
    
    def _is_cached(self, ticker: str) -> bool:
        """Check if ESG score is cached and still valid"""
        if ticker not in self.esg_cache:
            return False
        
        cache_entry = self.esg_cache[ticker]
        return datetime.now() - cache_entry['timestamp'] < self.cache_ttl
    
    def _cache_score(self, ticker: str, score: ESGScore):
        """Cache ESG score"""
        self.esg_cache[ticker] = {
            'score': score,
            'timestamp': datetime.now()
        }

# Factory function for creating ESG provider
def create_esg_provider(provider_type: str = 'mock', **kwargs) -> ESGDataProvider:
    """Create ESG data provider instance"""
    
    if provider_type.lower() == 'refinitiv':
        return RefinitivESGProvider(kwargs.get('api_key'))
    elif provider_type.lower() == 'mock':
        return MockESGProvider()
    else:
        raise ValueError(f"Unknown ESG provider type: {provider_type}")

# Example usage
if __name__ == "__main__":
    async def demo_esg_system():
        print("🌱 ESG Integration System Demo")
        print("=" * 50)
        
        # Initialize ESG system
        esg_system = ESGIntegration()
        
        # Mock portfolio data
        portfolio_holdings = {
            'AAPL': 100,
            'MSFT': 150,
            'GOOGL': 75,
            'TSLA': 50,
            'AMZN': 80
        }
        
        market_values = {
            'AAPL': 1500000,
            'MSFT': 2000000,
            'GOOGL': 1200000,
            'TSLA': 800000,
            'AMZN': 1300000
        }
        
        # Calculate portfolio ESG metrics
        print("📊 Calculating Portfolio ESG Metrics...")
        esg_metrics = await esg_system.calculate_portfolio_esg_metrics(
            portfolio_holdings, market_values
        )
        
        print(f"   Weighted ESG Score: {esg_metrics['weighted_esg_score']}/10")
        print(f"   Carbon Intensity: {esg_metrics['portfolio_carbon_intensity']} tCO2/$M")
        print(f"   Controversy Exposure: {esg_metrics['controversy_exposure']}")
        
        # Generate ESG report
        print("\n📋 Generating ESG Report...")
        portfolio_data = {'holdings': portfolio_holdings, 'total_value': sum(market_values.values())}
        esg_report = await esg_system.generate_esg_report(portfolio_data, market_values)
        
        print(f"   Report ID: {esg_report.report_id}")
        print(f"   Recommendations: {len(esg_report.improvement_recommendations)}")
        
        # Screen investments
        print("\n🔍 ESG Investment Screening...")
        screening_criteria = {
            'min_overall_score': 7.0,
            'min_environmental_score': 7.5,
            'max_carbon_intensity': 15.0,
            'exclude_controversies': ['High']
        }
        
        screen_results = await esg_system.screen_investments(
            list(portfolio_holdings.keys()), screening_criteria
        )
        
        passed_screen = [ticker for ticker, passed in screen_results.items() if passed]
        print(f"   Passed ESG Screen: {len(passed_screen)}/{len(portfolio_holdings)} holdings")
        print(f"   Qualified Holdings: {', '.join(passed_screen)}")
        
        print("\n🚀 Epic 6.3: ESG Integration System - OPERATIONAL")
    
    # Run demo
    asyncio.run(demo_esg_system())
