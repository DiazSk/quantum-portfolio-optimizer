"""
Alternative Data Collection Pipeline
Collects data from Reddit, Twitter, News APIs, and Google Trends
"""

import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncpraw
import tweepy
from newsapi import NewsApiClient
from pytrends.request import TrendReq
import yfinance as yf
from textblob import TextBlob
import aiohttp
import json
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AlternativeDataCollector:
    """Collects and processes alternative data for portfolio optimization"""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.data_cache = {}
        
        # Initialize API clients (use mock data if no keys)
        self._init_apis()
        
    def _init_apis(self):
        """Initialize API connections with fallback to mock data"""
        try:
            # Reddit API (will be initialized async)
            self.reddit_credentials = {
                'client_id': os.getenv('REDDIT_CLIENT_ID', 'mock'),
                'client_secret': os.getenv('REDDIT_CLIENT_SECRET', 'mock'),
                'user_agent': os.getenv('REDDIT_USER_AGENT', 'PortfolioOptimizer/1.0')
            }
            
            # News API
            self.newsapi = NewsApiClient(
                api_key=os.getenv('NEWS_API_KEY', 'mock')
            )
            
            # Google Trends
            self.pytrends = TrendReq(hl='en-US', tz=360)
            
        except Exception as e:
            logger.warning(f"API initialization failed, using mock data: {e}")
    
    async def collect_reddit_sentiment(self, ticker: str, limit: int = 100) -> Dict:
        """Collect Reddit sentiment from r/wallstreetbets and r/stocks"""
        sentiments = []
        
        try:
            # Initialize async Reddit instance
            async with asyncpraw.Reddit(
                client_id=self.reddit_credentials['client_id'],
                client_secret=self.reddit_credentials['client_secret'],
                user_agent=self.reddit_credentials['user_agent']
            ) as reddit:
                
                subreddits = ['wallstreetbets', 'stocks', 'investing']
                
                for subreddit_name in subreddits:
                    subreddit = await reddit.subreddit(subreddit_name)
                    
                    # Search for ticker mentions
                    async for submission in subreddit.search(ticker, limit=limit//3):
                        # Analyze title sentiment
                        blob = TextBlob(submission.title)
                        sentiments.append({
                            'source': f'reddit_{subreddit_name}',
                            'timestamp': datetime.fromtimestamp(submission.created_utc),
                            'sentiment': blob.sentiment.polarity,
                            'subjectivity': blob.sentiment.subjectivity,
                            'score': submission.score,
                            'num_comments': submission.num_comments
                        })
                    
        except Exception as e:
            logger.info(f"Using mock Reddit data for {ticker}: {e}")
            # Generate mock data for testing
            sentiments = self._generate_mock_sentiment(ticker, 'reddit')
        
        return self._aggregate_sentiment(sentiments)
    
    def collect_news_sentiment(self, ticker: str) -> Dict:
        """Collect news sentiment from NewsAPI"""
        try:
            # Get company name for better search
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            
            # Fetch news articles
            articles = self.newsapi.get_everything(
                q=company_name,
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=50
            )
            
            sentiments = []
            for article in articles.get('articles', []):
                if article['title'] and article['description']:
                    text = article['title'] + ' ' + article['description']
                    blob = TextBlob(text)
                    sentiments.append({
                        'source': 'news',
                        'timestamp': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                        'sentiment': blob.sentiment.polarity,
                        'subjectivity': blob.sentiment.subjectivity,
                        'source_name': article['source']['name']
                    })
                    
        except Exception as e:
            logger.info(f"Using mock news data for {ticker}: {e}")
            sentiments = self._generate_mock_sentiment(ticker, 'news')
            
        return self._aggregate_sentiment(sentiments)
    
    def collect_google_trends(self, ticker: str) -> Dict:
        """Collect Google Trends data"""
        try:
            # Get company name
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            
            # Build payload
            self.pytrends.build_payload(
                [company_name, ticker],
                timeframe='today 3-m'
            )
            
            # Get interest over time
            trends_data = self.pytrends.interest_over_time()
            
            if not trends_data.empty:
                recent_trend = trends_data[company_name].iloc[-7:].mean()
                trend_momentum = (trends_data[company_name].iloc[-7:].mean() / 
                                trends_data[company_name].iloc[-30:].mean() - 1)
                
                return {
                    'trend_score': recent_trend,
                    'trend_momentum': trend_momentum,
                    'trend_volatility': trends_data[company_name].iloc[-30:].std()
                }
                
        except Exception as e:
            logger.info(f"Using mock trends data for {ticker}: {e}")
            
        # Return mock data
        return {
            'trend_score': np.random.uniform(40, 80),
            'trend_momentum': np.random.uniform(-0.2, 0.3),
            'trend_volatility': np.random.uniform(5, 20)
        }
    
    async def collect_satellite_data_proxy(self, ticker: str) -> Dict:
        """
        Simulate satellite data analysis (parking lot car counts, shipping traffic)
        In production, this would connect to real satellite data providers
        """
        # For retail companies, simulate parking lot traffic
        # For shipping companies, simulate port activity
        
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', '')
        
        if 'Retail' in sector or 'Consumer' in sector:
            # Simulate parking lot occupancy data
            base_occupancy = 0.7
            trend = np.random.uniform(-0.1, 0.1)
            volatility = np.random.uniform(0.05, 0.15)
            
            return {
                'data_type': 'parking_occupancy',
                'current_level': base_occupancy + trend,
                'trend_30d': trend,
                'volatility': volatility,
                'locations_tracked': np.random.randint(50, 200)
            }
            
        elif 'Industrial' in sector or 'Energy' in sector:
            # Simulate shipping/port activity
            return {
                'data_type': 'shipping_activity',
                'vessel_count': np.random.randint(100, 500),
                'port_congestion': np.random.uniform(0.3, 0.9),
                'trend_30d': np.random.uniform(-0.15, 0.15)
            }
        
        else:
            # Generic alternative metric
            return {
                'data_type': 'generic_activity',
                'activity_index': np.random.uniform(0.4, 0.8),
                'trend_30d': np.random.uniform(-0.1, 0.1)
            }
    
    def _generate_mock_sentiment(self, ticker: str, source: str) -> List[Dict]:
        """Generate mock sentiment data for testing"""
        sentiments = []
        for i in range(30):
            sentiments.append({
                'source': source,
                'timestamp': datetime.now() - timedelta(days=i),
                'sentiment': np.random.uniform(-0.5, 0.5),
                'subjectivity': np.random.uniform(0.3, 0.8),
                'score': np.random.randint(0, 1000) if source == 'reddit' else 0
            })
        return sentiments
    
    def _aggregate_sentiment(self, sentiments: List[Dict]) -> Dict:
        """Aggregate sentiment scores with time decay"""
        if not sentiments:
            return {'sentiment_score': 0, 'sentiment_momentum': 0, 'sentiment_volume': 0}
        
        df = pd.DataFrame(sentiments)
        
        # Time decay weight (recent sentiments matter more)
        now = datetime.now()
        df['days_ago'] = (now - df['timestamp']).dt.days
        df['weight'] = np.exp(-df['days_ago'] / 7)  # 7-day half-life
        
        # Weighted sentiment
        weighted_sentiment = (df['sentiment'] * df['weight']).sum() / df['weight'].sum()
        
        # Sentiment momentum (recent vs older)
        recent = df[df['days_ago'] <= 3]['sentiment'].mean()
        older = df[df['days_ago'] > 3]['sentiment'].mean()
        momentum = recent - older if not pd.isna(older) and older != 0 else 0
        
        return {
            'sentiment_score': weighted_sentiment,
            'sentiment_momentum': momentum,
            'sentiment_volume': len(sentiments),
            'sentiment_std': df['sentiment'].std()
        }
    
    async def collect_all_alternative_data(self) -> pd.DataFrame:
        """Collect all alternative data for all tickers"""
        all_data = []
        
        for ticker in self.tickers:
            logger.info(f"Collecting alternative data for {ticker}")
            
            # Collect from all sources
            reddit_data = await self.collect_reddit_sentiment(ticker)
            news_data = self.collect_news_sentiment(ticker)
            trends_data = self.collect_google_trends(ticker)
            satellite_data = await self.collect_satellite_data_proxy(ticker)
            
            # Combine all data
            ticker_data = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                **{f'reddit_{k}': v for k, v in reddit_data.items()},
                **{f'news_{k}': v for k, v in news_data.items()},
                **{f'google_{k}': v for k, v in trends_data.items()},
                **{f'satellite_{k}': v for k, v in satellite_data.items()}
            }
            
            all_data.append(ticker_data)
        
        return pd.DataFrame(all_data)
    
    def calculate_alternative_data_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite alternative data score for each ticker"""
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert satellite data fields to numeric if they exist
        satellite_cols = [col for col in df.columns if 'satellite_' in col]
        for col in satellite_cols:
            if col in df.columns:
                # Handle the 'data_type' column which contains strings
                if 'data_type' in col:
                    # Map data types to numeric scores
                    data_type_scores = {
                        'parking_occupancy': 0.8,
                        'shipping_activity': 0.7,
                        'generic_activity': 0.5
                    }
                    if col in df.columns:
                        df[col + '_score'] = df[col].map(data_type_scores).fillna(0.5)
                else:
                    # Convert other satellite columns to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col not in ['ticker', 'timestamp']]
        
        # Normalize numeric features to 0-1 scale
        for feature in features:
            if feature in df.columns:
                col_std = df[feature].std()
                if col_std > 0:
                    col_min = df[feature].min()
                    col_max = df[feature].max()
                    if col_max > col_min:
                        df[f'{feature}_norm'] = (df[feature] - col_min) / (col_max - col_min)
                    else:
                        df[f'{feature}_norm'] = 0.5
                else:
                    df[f'{feature}_norm'] = 0.5
        
        # Calculate weighted combination with error handling
        weights = {
            'sentiment': 0.3,
            'momentum': 0.25,
            'trends': 0.25,
            'satellite': 0.2
        }
        
        # Initialize score with default value
        df['alt_data_score'] = 0.5
        
        # Safely calculate each component
        score_components = []
        
        # Sentiment component
        if 'reddit_sentiment_score_norm' in df.columns and 'news_sentiment_score_norm' in df.columns:
            sentiment_score = weights['sentiment'] * (
                df['reddit_sentiment_score_norm'] * 0.5 + 
                df['news_sentiment_score_norm'] * 0.5
            )
            score_components.append(sentiment_score)
        
        # Momentum component
        if 'reddit_sentiment_momentum_norm' in df.columns and 'news_sentiment_momentum_norm' in df.columns:
            momentum_score = weights['momentum'] * (
                df['reddit_sentiment_momentum_norm'] * 0.5 + 
                df['news_sentiment_momentum_norm'] * 0.5
            )
            score_components.append(momentum_score)
        
        # Trends component
        if 'google_trend_score_norm' in df.columns:
            trends_score = weights['trends'] * df['google_trend_score_norm']
            score_components.append(trends_score)
        
        # Satellite component (use any available satellite metric)
        satellite_norm_cols = [col for col in df.columns if 'satellite_' in col and '_norm' in col]
        if satellite_norm_cols:
            # Average all normalized satellite metrics
            satellite_avg = df[satellite_norm_cols].mean(axis=1)
            satellite_score = weights['satellite'] * satellite_avg
            score_components.append(satellite_score)
        
        # Combine all available components
        if score_components:
            df['alt_data_score'] = sum(score_components) / len(score_components) * len(weights)
        
        # Add confidence score based on data availability
        df['alt_data_confidence'] = len(score_components) / len(weights)
        
        return df[['ticker', 'alt_data_score', 'alt_data_confidence', 'timestamp']]


# Example usage
async def main():
    # Top tech stocks for testing
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD']
    
    collector = AlternativeDataCollector(tickers)
    
    # Collect all data
    alt_data = await collector.collect_all_alternative_data()
    
    # Calculate scores
    scores = collector.calculate_alternative_data_score(alt_data)
    
    print("\nAlternative Data Scores:")
    print(scores.sort_values('alt_data_score', ascending=False))
    
    # Save data
    alt_data.to_csv('data/alternative_data.csv', index=False)
    scores.to_csv('data/alt_data_scores.csv', index=False)
    
    return alt_data, scores

if __name__ == "__main__":
    asyncio.run(main())