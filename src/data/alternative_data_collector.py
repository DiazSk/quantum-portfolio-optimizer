"""
Enhanced Alternative Data Collector with Real API Integration
Uses actual Reddit, News API, and Alpha Vantage endpoints
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import praw
from newsapi import NewsApiClient
from pytrends.request import TrendReq
import yfinance as yf
from textblob import TextBlob
import json
from dotenv import load_dotenv
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EnhancedAlternativeDataCollector:
    """
    Production-ready alternative data collector with real API integration
    """
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.data_cache = {}
        self.cache_expiry = 3600  # 1 hour cache
        
        # Initialize API clients
        self._init_apis()
        
    def _init_apis(self):
        """Initialize API connections"""
        # Reddit API
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='QuantumPortfolioOptimizer/1.0 by /u/your_username'
        )
        
        # News API
        self.newsapi = NewsApiClient(
            api_key=os.getenv('NEWS_API_KEY')
        )
        
        # Alpha Vantage
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Google Trends
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
        logger.info("✅ APIs initialized successfully")
    
    def collect_reddit_sentiment(self, ticker: str, limit: int = 100) -> Dict:
        """
        Collect real Reddit sentiment from multiple subreddits
        """
        logger.info(f"📱 Collecting Reddit sentiment for {ticker}")
        
        try:
            sentiments = []
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket', 'options']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for ticker mentions in hot posts
                    for submission in subreddit.hot(limit=20):
                        # Check if ticker is mentioned
                        text = f"{submission.title} {submission.selftext}"
                        if ticker.upper() in text.upper() or f"${ticker.upper()}" in text.upper():
                            # Analyze sentiment
                            blob = TextBlob(submission.title)
                            sentiments.append({
                                'source': f'reddit_{subreddit_name}',
                                'timestamp': datetime.fromtimestamp(submission.created_utc),
                                'sentiment': blob.sentiment.polarity,
                                'subjectivity': blob.sentiment.subjectivity,
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'upvote_ratio': submission.upvote_ratio
                            })
                            
                            # Also analyze top comments
                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments[:5]:
                                if hasattr(comment, 'body'):
                                    comment_blob = TextBlob(comment.body)
                                    sentiments.append({
                                        'source': f'reddit_{subreddit_name}_comment',
                                        'timestamp': datetime.fromtimestamp(comment.created_utc),
                                        'sentiment': comment_blob.sentiment.polarity,
                                        'subjectivity': comment_blob.sentiment.subjectivity,
                                        'score': comment.score,
                                        'num_comments': 0,
                                        'upvote_ratio': 0
                                    })
                    
                    # Also search for ticker directly
                    for submission in subreddit.search(ticker, limit=10, time_filter='week'):
                        blob = TextBlob(submission.title)
                        sentiments.append({
                            'source': f'reddit_{subreddit_name}_search',
                            'timestamp': datetime.fromtimestamp(submission.created_utc),
                            'sentiment': blob.sentiment.polarity,
                            'subjectivity': blob.sentiment.subjectivity,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'upvote_ratio': submission.upvote_ratio
                        })
                    
                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                    continue
            
            if sentiments:
                return self._aggregate_sentiment(sentiments)
            else:
                logger.warning(f"No Reddit sentiment found for {ticker}")
                return {'sentiment_score': 0, 'sentiment_momentum': 0, 'sentiment_volume': 0}
                
        except Exception as e:
            logger.error(f"Reddit API error for {ticker}: {e}")
            return {'sentiment_score': 0, 'sentiment_momentum': 0, 'sentiment_volume': 0}
    
    def collect_news_sentiment(self, ticker: str) -> Dict:
        """
        Collect real news sentiment from NewsAPI
        """
        logger.info(f"📰 Collecting news sentiment for {ticker}")
        
        try:
            # Get company info for better search
            stock = yf.Ticker(ticker)
            company_info = stock.info
            company_name = company_info.get('longName', ticker)
            
            # Search for news articles
            articles = self.newsapi.get_everything(
                q=f"{company_name} OR {ticker}",
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            sentiments = []
            for article in articles.get('articles', []):
                if article['title'] and article['description']:
                    # Combine title and description for analysis
                    text = f"{article['title']} {article['description']}"
                    
                    # Skip if ticker not actually mentioned
                    if ticker.upper() not in text.upper() and company_name.upper() not in text.upper():
                        continue
                    
                    blob = TextBlob(text)
                    sentiments.append({
                        'source': article['source']['name'],
                        'timestamp': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                        'sentiment': blob.sentiment.polarity,
                        'subjectivity': blob.sentiment.subjectivity,
                        'url': article['url']
                    })
            
            if sentiments:
                return self._aggregate_sentiment(sentiments)
            else:
                logger.warning(f"No news sentiment found for {ticker}")
                return {'sentiment_score': 0, 'sentiment_momentum': 0, 'sentiment_volume': 0}
                
        except Exception as e:
            logger.error(f"News API error for {ticker}: {e}")
            return {'sentiment_score': 0, 'sentiment_momentum': 0, 'sentiment_volume': 0}
    
    def collect_alpha_vantage_data(self, ticker: str) -> Dict:
        """
        Collect technical indicators from Alpha Vantage
        """
        logger.info(f"📈 Collecting Alpha Vantage data for {ticker}")
        
        try:
            indicators = {}
            
            # Get RSI
            url = f"https://www.alphavantage.co/query?function=RSI&symbol={ticker}&interval=daily&time_period=14&series_type=close&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'Technical Analysis: RSI' in data:
                    latest_date = list(data['Technical Analysis: RSI'].keys())[0]
                    indicators['rsi'] = float(data['Technical Analysis: RSI'][latest_date]['RSI'])
            
            time.sleep(12)  # Alpha Vantage rate limit: 5 calls per minute
            
            # Get MACD
            url = f"https://www.alphavantage.co/query?function=MACD&symbol={ticker}&interval=daily&series_type=close&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'Technical Analysis: MACD' in data:
                    latest_date = list(data['Technical Analysis: MACD'].keys())[0]
                    indicators['macd'] = float(data['Technical Analysis: MACD'][latest_date]['MACD'])
                    indicators['macd_signal'] = float(data['Technical Analysis: MACD'][latest_date]['MACD_Signal'])
            
            time.sleep(12)
            
            # Get SMA
            url = f"https://www.alphavantage.co/query?function=SMA&symbol={ticker}&interval=daily&time_period=20&series_type=close&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'Technical Analysis: SMA' in data:
                    latest_date = list(data['Technical Analysis: SMA'].keys())[0]
                    indicators['sma_20'] = float(data['Technical Analysis: SMA'][latest_date]['SMA'])
            
            return indicators
            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
            return {}
    
    def collect_google_trends(self, ticker: str) -> Dict:
        """
        Collect real Google Trends data
        """
        logger.info(f"🔍 Collecting Google Trends for {ticker}")
        
        try:
            # Get company name
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            
            # Build payload with both ticker and company name
            keywords = [ticker, company_name]
            
            self.pytrends.build_payload(
                keywords,
                timeframe='today 3-m',
                geo='US'
            )
            
            # Get interest over time
            trends_data = self.pytrends.interest_over_time()
            
            if not trends_data.empty:
                # Calculate metrics for both keywords
                results = {}
                for keyword in keywords:
                    if keyword in trends_data.columns:
                        recent_trend = trends_data[keyword].iloc[-7:].mean()
                        older_trend = trends_data[keyword].iloc[-30:-7].mean()
                        trend_momentum = (recent_trend / older_trend - 1) if older_trend > 0 else 0
                        
                        results[keyword] = {
                            'trend_score': recent_trend,
                            'trend_momentum': trend_momentum,
                            'trend_volatility': trends_data[keyword].iloc[-30:].std()
                        }
                
                # Combine results (use max score)
                if results:
                    max_key = max(results.keys(), key=lambda k: results[k]['trend_score'])
                    return results[max_key]
            
            return {
                'trend_score': 50,
                'trend_momentum': 0,
                'trend_volatility': 10
            }
            
        except Exception as e:
            logger.error(f"Google Trends error for {ticker}: {e}")
            return {
                'trend_score': 50,
                'trend_momentum': 0,
                'trend_volatility': 10
            }
    
    def collect_yahoo_finance_metrics(self, ticker: str) -> Dict:
        """
        Collect additional metrics from Yahoo Finance
        """
        logger.info(f"💹 Collecting Yahoo Finance metrics for {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metrics = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'volume_ratio': info.get('volume', 0) / info.get('averageVolume', 1) if info.get('averageVolume', 0) > 0 else 1,
                'beta': info.get('beta', 1),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'analyst_rating': self._convert_analyst_rating(info.get('recommendationKey', 'hold'))
            }
            
            # Get recent price action
            hist = stock.history(period='1mo')
            if not hist.empty:
                metrics['price_momentum_1m'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                metrics['volatility_1m'] = hist['Close'].pct_change().std()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {ticker}: {e}")
            return {}
    
    def _convert_analyst_rating(self, rating: str) -> float:
        """Convert analyst rating to numeric score"""
        ratings = {
            'strong_buy': 1.0,
            'buy': 0.75,
            'hold': 0.5,
            'sell': 0.25,
            'strong_sell': 0.0
        }
        return ratings.get(rating, 0.5)
    
    def _aggregate_sentiment(self, sentiments: List[Dict]) -> Dict:
        """Aggregate sentiment scores with time decay and source weighting"""
        if not sentiments:
            return {'sentiment_score': 0, 'sentiment_momentum': 0, 'sentiment_volume': 0}
        
        df = pd.DataFrame(sentiments)
        
        # Time decay weight (recent sentiments matter more)
        now = datetime.now()
        if 'timestamp' in df.columns:
            df['days_ago'] = (now - df['timestamp']).dt.days
            df['time_weight'] = np.exp(-df['days_ago'] / 3)  # 3-day half-life
        else:
            df['time_weight'] = 1
        
        # Source weight (higher engagement = more weight)
        if 'score' in df.columns:
            df['engagement_weight'] = np.log1p(df['score']) / np.log1p(df['score'].max() + 1)
        else:
            df['engagement_weight'] = 1
        
        # Combined weight
        df['weight'] = df['time_weight'] * df['engagement_weight']
        
        # Weighted sentiment
        weighted_sentiment = (df['sentiment'] * df['weight']).sum() / df['weight'].sum()
        
        # Sentiment momentum (recent vs older)
        if len(df) > 5:
            recent = df.nlargest(5, 'time_weight')['sentiment'].mean()
            older = df.nsmallest(len(df) - 5, 'time_weight')['sentiment'].mean()
            momentum = recent - older
        else:
            momentum = 0
        
        # Volume and confidence
        volume = len(sentiments)
        confidence = 1 - df['subjectivity'].mean() if 'subjectivity' in df.columns else 0.5
        
        return {
            'sentiment_score': weighted_sentiment,
            'sentiment_momentum': momentum,
            'sentiment_volume': volume,
            'sentiment_confidence': confidence,
            'sentiment_std': df['sentiment'].std()
        }
    
    def collect_all_alternative_data(self) -> pd.DataFrame:
        """
        Collect all alternative data for all tickers with parallel processing
        """
        all_data = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for ticker in self.tickers:
                logger.info(f"🎯 Processing {ticker}")
                
                # Submit tasks in parallel
                future_reddit = executor.submit(self.collect_reddit_sentiment, ticker)
                future_news = executor.submit(self.collect_news_sentiment, ticker)
                future_trends = executor.submit(self.collect_google_trends, ticker)
                future_yahoo = executor.submit(self.collect_yahoo_finance_metrics, ticker)
                
                futures.append({
                    'ticker': ticker,
                    'reddit': future_reddit,
                    'news': future_news,
                    'trends': future_trends,
                    'yahoo': future_yahoo
                })
            
            # Collect results
            for future_set in futures:
                ticker = future_set['ticker']
                
                # Get results with timeout
                try:
                    reddit_data = future_set['reddit'].result(timeout=30)
                except:
                    reddit_data = {'sentiment_score': 0, 'sentiment_momentum': 0}
                
                try:
                    news_data = future_set['news'].result(timeout=30)
                except:
                    news_data = {'sentiment_score': 0, 'sentiment_momentum': 0}
                
                try:
                    trends_data = future_set['trends'].result(timeout=30)
                except:
                    trends_data = {'trend_score': 50, 'trend_momentum': 0}
                
                try:
                    yahoo_data = future_set['yahoo'].result(timeout=30)
                except:
                    yahoo_data = {}
                
                # Combine all data
                ticker_data = {
                    'ticker': ticker,
                    'timestamp': datetime.now(),
                    **{f'reddit_{k}': v for k, v in reddit_data.items()},
                    **{f'news_{k}': v for k, v in news_data.items()},
                    **{f'google_{k}': v for k, v in trends_data.items()},
                    **{f'yahoo_{k}': v for k, v in yahoo_data.items()}
                }
                
                all_data.append(ticker_data)
        
        return pd.DataFrame(all_data)
    
    def calculate_alternative_data_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite alternative data score with ML-based weighting
        """
        # Normalize features
        features_to_normalize = [col for col in df.columns if col not in ['ticker', 'timestamp']]
        
        for feature in features_to_normalize:
            if feature in df.columns and df[feature].std() > 0:
                # Z-score normalization
                df[f'{feature}_zscore'] = (df[feature] - df[feature].mean()) / df[feature].std()
                # Min-max normalization to 0-1
                min_val = df[f'{feature}_zscore'].min()
                max_val = df[f'{feature}_zscore'].max()
                if max_val > min_val:
                    df[f'{feature}_norm'] = (df[f'{feature}_zscore'] - min_val) / (max_val - min_val)
                else:
                    df[f'{feature}_norm'] = 0.5
        
        # Calculate weighted composite score
        weights = {
            'sentiment': 0.30,  # Reddit + News sentiment
            'momentum': 0.20,   # Price and sentiment momentum
            'trends': 0.20,     # Google trends
            'fundamentals': 0.30  # Yahoo Finance metrics
        }
        
        # Sentiment component
        sentiment_cols = [col for col in df.columns if 'sentiment_score_norm' in col]
        if sentiment_cols:
            df['sentiment_component'] = df[sentiment_cols].mean(axis=1)
        else:
            df['sentiment_component'] = 0.5
        
        # Momentum component
        momentum_cols = [col for col in df.columns if 'momentum_norm' in col]
        if momentum_cols:
            df['momentum_component'] = df[momentum_cols].mean(axis=1)
        else:
            df['momentum_component'] = 0.5
        
        # Trends component
        if 'google_trend_score_norm' in df.columns:
            df['trends_component'] = df['google_trend_score_norm']
        else:
            df['trends_component'] = 0.5
        
        # Fundamentals component (if Yahoo data available)
        fundamental_cols = ['yahoo_volume_ratio_norm', 'yahoo_analyst_rating_norm', 
                          'yahoo_revenue_growth_norm', 'yahoo_profit_margin_norm']
        available_fundamentals = [col for col in fundamental_cols if col in df.columns]
        if available_fundamentals:
            df['fundamentals_component'] = df[available_fundamentals].mean(axis=1)
        else:
            df['fundamentals_component'] = 0.5
        
        # Calculate final score
        df['alt_data_score'] = (
            weights['sentiment'] * df['sentiment_component'] +
            weights['momentum'] * df['momentum_component'] +
            weights['trends'] * df['trends_component'] +
            weights['fundamentals'] * df['fundamentals_component']
        )
        
        # Add confidence metric
        df['alt_data_confidence'] = df[[col for col in df.columns if 'volume' in col]].sum(axis=1) / df[[col for col in df.columns if 'volume' in col]].sum(axis=1).max()
        
        return df[['ticker', 'alt_data_score', 'alt_data_confidence', 'timestamp']]

# Example usage
def main():
    # Portfolio universe
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
    
    collector = EnhancedAlternativeDataCollector(tickers)
    
    # Collect all data
    print("🚀 Starting alternative data collection...")
    alt_data = collector.collect_all_alternative_data()
    
    # Calculate scores
    scores = collector.calculate_alternative_data_score(alt_data)
    
    print("\n📊 Alternative Data Scores:")
    print(scores.sort_values('alt_data_score', ascending=False).to_string(index=False))
    
    # Save data
    alt_data.to_csv('data/alternative_data_detailed.csv', index=False)
    scores.to_csv('data/alternative_data_scores.csv', index=False)
    
    print("\n✅ Data saved to data/ directory")
    
    return alt_data, scores

if __name__ == "__main__":
    main()