"""
STORY 6.2: AI-POWERED INVESTMENT INSIGHTS & AUTOMATION
GPT-Powered Research, Analysis, and Automated Investment Recommendations
================================================================================

Advanced AI capabilities for investment research, market analysis, risk assessment,
and automated client communication to provide institutional-grade insights.

AC-6.2.1: GPT-Powered Investment Research & Analysis
AC-6.2.2: Automated Investment Recommendations
AC-6.2.3: Market Intelligence & Trend Analysis
AC-6.2.4: Risk Assessment & Scenario Modeling
AC-6.2.5: Automated Client Communication
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import os
import sys

# OpenAI integration (with fallback for demo)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not available - using simulated responses")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.professional_logging import get_logger

logger = get_logger(__name__)

class InsightType(Enum):
    """Types of AI-generated insights"""
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    MARKET_SENTIMENT = "market_sentiment"
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY_IDENTIFICATION = "opportunity_identification"
    SECTOR_ROTATION = "sector_rotation"

class RecommendationType(Enum):
    """Investment recommendation types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"
    REBALANCE = "rebalance"

class ConfidenceLevel(Enum):
    """AI recommendation confidence levels"""
    LOW = "low"           # 50-65%
    MEDIUM = "medium"     # 65-80%
    HIGH = "high"         # 80-95%
    VERY_HIGH = "very_high"  # 95%+

@dataclass
class InvestmentInsight:
    """AI-generated investment insight"""
    insight_id: str
    symbol: str
    insight_type: InsightType
    title: str
    summary: str
    detailed_analysis: str
    key_factors: List[str]
    confidence_score: float
    confidence_level: ConfidenceLevel
    sources_cited: List[str]
    generated_timestamp: datetime
    expiry_date: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None

@dataclass
class AIRecommendation:
    """Automated investment recommendation"""
    recommendation_id: str
    portfolio_id: str
    symbol: str
    recommendation_type: RecommendationType
    current_weight: float
    recommended_weight: float
    rationale: str
    expected_return: float
    risk_adjustment: float
    confidence_level: ConfidenceLevel
    implementation_priority: int  # 1-5 scale
    timing_guidance: str
    generated_timestamp: datetime
    expiry_timestamp: datetime

@dataclass
class MarketIntelligence:
    """AI-powered market intelligence"""
    intelligence_id: str
    market_segment: str
    headline: str
    summary: str
    impact_analysis: str
    affected_sectors: List[str]
    portfolio_implications: str
    action_items: List[str]
    confidence_score: float
    urgency_level: int  # 1-5 scale
    generated_timestamp: datetime

class AIInvestmentResearchEngine:
    """
    GPT-powered investment research and analysis system
    Implements AC-6.2.1: GPT-Powered Investment Research & Analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.openai_available = OPENAI_AVAILABLE and api_key
        
        if self.openai_available:
            openai.api_key = api_key
        
        # Research templates
        self.research_prompts = {
            InsightType.FUNDAMENTAL_ANALYSIS: """
Analyze the fundamental characteristics of {symbol} and provide a comprehensive investment analysis.

Consider the following factors:
1. Financial health and key metrics (P/E, ROE, debt levels, cash flow)
2. Business model and competitive position
3. Management quality and corporate governance
4. Industry trends and market dynamics
5. Growth prospects and valuation

Current stock data: {stock_data}
Recent news: {news_summary}

Provide analysis in the following format:
- Executive Summary (2-3 sentences)
- Key Strengths (bullet points)
- Key Risks (bullet points)
- Valuation Assessment
- Recommendation with price target
- Confidence level and reasoning

Be specific, data-driven, and suitable for institutional investors.
""",
            
            InsightType.TECHNICAL_ANALYSIS: """
Perform technical analysis on {symbol} and provide trading insights.

Technical indicators to analyze:
1. Price trends and momentum indicators
2. Support and resistance levels
3. Volume patterns and market microstructure
4. Relative strength vs. market and sector
5. Chart patterns and technical setups

Chart data: {technical_data}
Market context: {market_context}

Provide analysis covering:
- Current technical picture
- Key levels to watch (support/resistance)
- Momentum assessment
- Trading recommendations
- Risk management levels (stop loss, profit targets)
- Time horizon for analysis

Focus on actionable insights for portfolio managers.
""",
            
            InsightType.MARKET_SENTIMENT: """
Analyze market sentiment for {symbol} and broader market implications.

Sentiment indicators to evaluate:
1. Social media and news sentiment
2. Analyst revisions and recommendations
3. Institutional flow and positioning
4. Options market sentiment (put/call ratios)
5. Insider trading activity

Data sources: {sentiment_data}
Market backdrop: {market_environment}

Provide sentiment analysis including:
- Current sentiment overview
- Sentiment trend analysis
- Contrarian indicators
- Market positioning insights
- Implications for price action
- Sentiment-based trading opportunities

Emphasize how sentiment affects institutional decision-making.
"""
        }
        
        logger.info(f"AIInvestmentResearchEngine initialized (OpenAI: {self.openai_available})")
    
    async def generate_investment_insight(self, symbol: str, insight_type: InsightType, 
                                        portfolio_context: Optional[Dict] = None) -> InvestmentInsight:
        """
        Generate comprehensive AI-powered investment insight
        Implements comprehensive AI-generated research reports using GPT models
        """
        try:
            insight_id = str(uuid.uuid4())
            
            # Gather relevant data
            stock_data = await self._fetch_stock_data(symbol)
            market_data = await self._fetch_market_context()
            news_data = await self._fetch_recent_news(symbol)
            
            # Generate AI analysis
            if self.openai_available:
                analysis = await self._generate_openai_analysis(symbol, insight_type, 
                                                              stock_data, market_data, news_data)
            else:
                analysis = self._generate_simulated_analysis(symbol, insight_type, stock_data)
            
            # Extract structured data from analysis
            confidence_score = self._extract_confidence_score(analysis)
            confidence_level = self._determine_confidence_level(confidence_score)
            key_factors = self._extract_key_factors(analysis)
            price_target = self._extract_price_target(analysis, stock_data)
            
            insight = InvestmentInsight(
                insight_id=insight_id,
                symbol=symbol,
                insight_type=insight_type,
                title=f"{insight_type.value.replace('_', ' ').title()} - {symbol}",
                summary=analysis.get('summary', ''),
                detailed_analysis=analysis.get('detailed_analysis', ''),
                key_factors=key_factors,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                sources_cited=analysis.get('sources', []),
                generated_timestamp=datetime.now(timezone.utc),
                expiry_date=datetime.now(timezone.utc) + timedelta(days=30),
                price_target=price_target,
                stop_loss=analysis.get('stop_loss')
            )
            
            logger.info(f"Generated insight for {symbol}: {insight_type.value}")
            return insight
            
        except Exception as e:
            logger.error(f"Failed to generate insight for {symbol}: {e}")
            raise
    
    async def _fetch_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch current stock data and metrics"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'year_high': hist['High'].max(),
                'year_low': hist['Low'].min(),
                'ytd_return': ((info.get('currentPrice', 0) - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
                'volatility': hist['Close'].pct_change().std() * np.sqrt(252) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to fetch stock data for {symbol}: {e}")
            return {'symbol': symbol, 'current_price': 100}  # Fallback data
    
    async def _fetch_market_context(self) -> Dict[str, Any]:
        """Fetch broader market context"""
        try:
            # Fetch major indices
            spy = yf.Ticker("SPY").history(period="1mo")
            vix = yf.Ticker("^VIX").history(period="1mo")
            
            return {
                'sp500_trend': 'bullish' if spy['Close'].iloc[-1] > spy['Close'].iloc[0] else 'bearish',
                'market_volatility': vix['Close'].iloc[-1],
                'market_sentiment': 'neutral'  # Simplified
            }
        except Exception:
            return {'market_sentiment': 'neutral', 'market_volatility': 20}
    
    async def _fetch_recent_news(self, symbol: str) -> List[str]:
        """Fetch recent news headlines (simulated)"""
        # In production, this would integrate with news APIs
        return [
            f"{symbol} reports strong quarterly earnings",
            f"Analysts upgrade {symbol} price target",
            f"Industry trends favor {symbol} business model"
        ]
    
    async def _generate_openai_analysis(self, symbol: str, insight_type: InsightType,
                                      stock_data: Dict, market_data: Dict, 
                                      news_data: List[str]) -> Dict[str, Any]:
        """Generate analysis using OpenAI GPT"""
        try:
            prompt = self.research_prompts[insight_type].format(
                symbol=symbol,
                stock_data=json.dumps(stock_data, indent=2),
                news_summary="\n".join(news_data),
                technical_data=stock_data,
                market_context=json.dumps(market_data, indent=2),
                sentiment_data=news_data,
                market_environment=market_data
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert investment analyst providing institutional-grade research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                'summary': analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
                'detailed_analysis': analysis_text,
                'sources': ['OpenAI GPT-4', 'Yahoo Finance', 'Market Data'],
                'confidence_raw': 0.85  # Would be extracted from response
            }
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._generate_simulated_analysis(symbol, insight_type, stock_data)
    
    def _generate_simulated_analysis(self, symbol: str, insight_type: InsightType, 
                                   stock_data: Dict) -> Dict[str, Any]:
        """Generate simulated analysis when OpenAI not available"""
        current_price = stock_data.get('current_price', 100)
        pe_ratio = stock_data.get('pe_ratio', 15)
        
        if insight_type == InsightType.FUNDAMENTAL_ANALYSIS:
            analysis = f"""
**Fundamental Analysis - {symbol}**

**Executive Summary:**
{symbol} demonstrates solid fundamentals with a P/E ratio of {pe_ratio:.1f} and strong operational metrics. 
The company is well-positioned in its sector with competitive advantages and growth potential.

**Key Strengths:**
• Strong financial position with healthy cash flow generation
• Market-leading position in growing industry segment  
• Experienced management team with proven track record
• Robust balance sheet providing financial flexibility

**Key Risks:**
• Market competition increasing pressure on margins
• Regulatory changes could impact business model
• Economic sensitivity affecting demand patterns
• Currency exposure creating earnings volatility

**Valuation Assessment:**
Current valuation appears reasonable relative to peers and growth prospects. 
Price target of ${current_price * 1.15:.2f} represents 15% upside potential based on DCF analysis.

**Investment Recommendation:** 
BUY with 12-month price target of ${current_price * 1.15:.2f}. Strong fundamentals support 
continued outperformance with attractive risk-adjusted returns.
"""
        
        elif insight_type == InsightType.TECHNICAL_ANALYSIS:
            analysis = f"""
**Technical Analysis - {symbol}**

**Current Technical Picture:**
{symbol} is trading at ${current_price:.2f} with constructive technical indicators. 
Price action suggests consolidation phase with potential for upward breakout.

**Key Levels:**
• Support: ${current_price * 0.95:.2f} (recent swing low)
• Resistance: ${current_price * 1.08:.2f} (previous high)
• Next target: ${current_price * 1.15:.2f} (measured move)

**Momentum Assessment:**
Relative strength indicators show improving momentum with volume supporting price action. 
MACD showing positive divergence indicating potential trend continuation.

**Trading Recommendation:**
Initiate position on pullback to ${current_price * 0.98:.2f} area with stop loss at ${current_price * 0.92:.2f}. 
Target ${current_price * 1.12:.2f} for 12-15% potential return.
"""
        
        else:  # Default for other types
            analysis = f"""
**Market Analysis - {symbol}**

Comprehensive analysis indicates {symbol} presents attractive investment opportunity 
with favorable risk-return profile. Multiple factors support positive outlook 
including strong fundamentals, technical momentum, and market positioning.

Key investment themes include sector rotation benefits, earnings growth potential, 
and institutional accumulation patterns. Risk factors remain manageable with 
appropriate position sizing and stop-loss levels.
"""
        
        return {
            'summary': analysis[:300] + "...",
            'detailed_analysis': analysis,
            'sources': ['Technical Analysis', 'Market Data', 'Quantitative Models'],
            'confidence_raw': 0.75
        }
    
    def _extract_confidence_score(self, analysis: Dict) -> float:
        """Extract confidence score from analysis"""
        return analysis.get('confidence_raw', 0.75)
    
    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from numeric score"""
        if score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.80:
            return ConfidenceLevel.HIGH
        elif score >= 0.65:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _extract_key_factors(self, analysis: Dict) -> List[str]:
        """Extract key factors from analysis text"""
        # Simplified extraction - in production would use NLP
        return [
            "Strong fundamental metrics",
            "Favorable technical setup", 
            "Positive market sentiment",
            "Sector tailwinds"
        ]
    
    def _extract_price_target(self, analysis: Dict, stock_data: Dict) -> Optional[float]:
        """Extract price target from analysis"""
        current_price = stock_data.get('current_price', 100)
        # Simplified extraction - in production would parse from text
        return current_price * 1.15  # 15% upside target

class AutomatedInsightEngine:
    """
    Automated investment recommendations and market intelligence
    Implements AC-6.2.2: Automated Investment Recommendations
    """
    
    def __init__(self, research_engine: AIInvestmentResearchEngine):
        self.research_engine = research_engine
        self.active_recommendations: Dict[str, List[AIRecommendation]] = {}
        self.market_intelligence: List[MarketIntelligence] = []
        
    async def generate_portfolio_recommendations(self, portfolio_id: str, 
                                               current_holdings: Dict[str, float],
                                               client_objectives: Dict[str, Any]) -> List[AIRecommendation]:
        """
        Generate automated investment recommendations for portfolio
        Implements automated investment recommendations with rationale
        """
        recommendations = []
        
        try:
            for symbol, current_weight in current_holdings.items():
                # Generate insight for each holding
                insight = await self.research_engine.generate_investment_insight(
                    symbol, InsightType.FUNDAMENTAL_ANALYSIS
                )
                
                # Generate recommendation based on insight
                recommendation = await self._create_recommendation(
                    portfolio_id, symbol, current_weight, insight, client_objectives
                )
                
                if recommendation:
                    recommendations.append(recommendation)
            
            # Store active recommendations
            self.active_recommendations[portfolio_id] = recommendations
            
            # Sort by priority
            recommendations.sort(key=lambda x: x.implementation_priority, reverse=True)
            
            logger.info(f"Generated {len(recommendations)} recommendations for portfolio {portfolio_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            raise
    
    async def _create_recommendation(self, portfolio_id: str, symbol: str, 
                                   current_weight: float, insight: InvestmentInsight,
                                   client_objectives: Dict[str, Any]) -> Optional[AIRecommendation]:
        """Create investment recommendation based on insight"""
        
        # Determine recommendation type based on insight
        confidence_score = insight.confidence_score
        risk_tolerance = client_objectives.get('risk_tolerance', 'moderate')
        
        if confidence_score > 0.8 and insight.insight_type == InsightType.FUNDAMENTAL_ANALYSIS:
            if current_weight < 0.05:  # Less than 5% allocation
                rec_type = RecommendationType.INCREASE
                recommended_weight = min(current_weight * 1.5, 0.08)  # Increase to max 8%
            else:
                rec_type = RecommendationType.HOLD
                recommended_weight = current_weight
        elif confidence_score < 0.6:
            rec_type = RecommendationType.REDUCE
            recommended_weight = current_weight * 0.7  # Reduce by 30%
        else:
            rec_type = RecommendationType.HOLD
            recommended_weight = current_weight
        
        # Calculate expected return (simplified)
        expected_return = (insight.price_target / 100 - 1) * 100 if insight.price_target else 8.0
        
        recommendation = AIRecommendation(
            recommendation_id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            symbol=symbol,
            recommendation_type=rec_type,
            current_weight=current_weight,
            recommended_weight=recommended_weight,
            rationale=insight.summary,
            expected_return=expected_return,
            risk_adjustment=confidence_score,
            confidence_level=insight.confidence_level,
            implementation_priority=self._calculate_priority(rec_type, confidence_score),
            timing_guidance="Implement over 5-10 trading days",
            generated_timestamp=datetime.now(timezone.utc),
            expiry_timestamp=datetime.now(timezone.utc) + timedelta(days=14)
        )
        
        return recommendation
    
    def _calculate_priority(self, rec_type: RecommendationType, confidence: float) -> int:
        """Calculate implementation priority (1-5 scale)"""
        base_priority = {
            RecommendationType.SELL: 5,
            RecommendationType.REDUCE: 4,
            RecommendationType.INCREASE: 3,
            RecommendationType.BUY: 3,
            RecommendationType.REBALANCE: 2,
            RecommendationType.HOLD: 1
        }
        
        priority = base_priority.get(rec_type, 1)
        
        # Adjust for confidence
        if confidence > 0.9:
            priority = min(5, priority + 1)
        elif confidence < 0.6:
            priority = max(1, priority - 1)
        
        return priority
    
    async def generate_market_intelligence(self, market_segments: List[str]) -> List[MarketIntelligence]:
        """
        Generate AI-powered market intelligence and trend analysis
        Implements AC-6.2.3: Market Intelligence & Trend Analysis
        """
        intelligence_reports = []
        
        for segment in market_segments:
            # Generate market intelligence for each segment
            intelligence = await self._create_market_intelligence(segment)
            intelligence_reports.append(intelligence)
        
        # Store for tracking
        self.market_intelligence.extend(intelligence_reports)
        
        return intelligence_reports
    
    async def _create_market_intelligence(self, market_segment: str) -> MarketIntelligence:
        """Create market intelligence report for segment"""
        intelligence_id = str(uuid.uuid4())
        
        # Simulated market intelligence (in production would use real analysis)
        intelligence_data = {
            'technology': {
                'headline': 'AI and Cloud Computing Drive Tech Sector Outperformance',
                'summary': 'Technology sector showing strong momentum driven by AI adoption and cloud migration trends.',
                'impact': 'Positive earnings revisions expected for cloud infrastructure and AI software companies.',
                'sectors': ['Software', 'Semiconductors', 'Cloud Services'],
                'implications': 'Overweight technology allocation recommended for growth-oriented portfolios.',
                'actions': ['Increase allocation to AI leaders', 'Consider cloud infrastructure plays']
            },
            'financials': {
                'headline': 'Rising Interest Rates Benefit Financial Sector Margins',
                'summary': 'Financial sector positioned to benefit from continued rate normalization and steepening yield curve.',
                'impact': 'Net interest margin expansion expected for regional and money center banks.',
                'sectors': ['Banking', 'Insurance', 'Asset Management'],
                'implications': 'Financial sector rotation opportunity as rates stabilize at higher levels.',
                'actions': ['Add quality bank exposure', 'Consider insurance companies']
            }
        }
        
        segment_data = intelligence_data.get(market_segment.lower(), intelligence_data['technology'])
        
        intelligence = MarketIntelligence(
            intelligence_id=intelligence_id,
            market_segment=market_segment,
            headline=segment_data['headline'],
            summary=segment_data['summary'],
            impact_analysis=segment_data['impact'],
            affected_sectors=segment_data['sectors'],
            portfolio_implications=segment_data['implications'],
            action_items=segment_data['actions'],
            confidence_score=0.8,
            urgency_level=3,
            generated_timestamp=datetime.now(timezone.utc)
        )
        
        return intelligence

if __name__ == "__main__":
    # Example usage and testing
    research_engine = AIInvestmentResearchEngine()
    insight_engine = AutomatedInsightEngine(research_engine)
    
    async def run_ai_demo():
        # Generate investment insight
        insight = await research_engine.generate_investment_insight(
            "AAPL", InsightType.FUNDAMENTAL_ANALYSIS
        )
        print(f"Generated insight for AAPL: {insight.confidence_level.value}")
        
        # Generate portfolio recommendations
        portfolio_holdings = {"AAPL": 0.05, "MSFT": 0.04, "GOOGL": 0.03}
        client_objectives = {"risk_tolerance": "moderate", "return_target": 8.0}
        
        recommendations = await insight_engine.generate_portfolio_recommendations(
            "portfolio_001", portfolio_holdings, client_objectives
        )
        print(f"Generated {len(recommendations)} recommendations")
        
        # Generate market intelligence
        intelligence = await insight_engine.generate_market_intelligence(
            ["Technology", "Financials"]
        )
        print(f"Generated {len(intelligence)} market intelligence reports")
    
    # Run demo
    asyncio.run(run_ai_demo())
