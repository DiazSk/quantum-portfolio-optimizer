#!/usr/bin/env python3
"""
Test script for real API integration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, '.')

def test_api_integration():
    """Test real API integration"""
    try:
        from src.data.alternative_data_collector import EnhancedAlternativeDataCollector
        
        print('🚀 Testing real API integration for AAPL...')
        collector = EnhancedAlternativeDataCollector(['AAPL'])
        
        # Test Reddit API (quick test)
        print('📱 Testing Reddit API...')
        reddit_data = collector.collect_reddit_sentiment('AAPL')
        print(f'Reddit sentiment score: {reddit_data.get("sentiment_score", "N/A")}')
        
        # Test News API (quick test) 
        print('📰 Testing News API...')
        news_data = collector.collect_news_sentiment('AAPL')
        print(f'News sentiment score: {news_data.get("sentiment_score", "N/A")}')
        
        print('✅ Real API test successful!')
        return True
        
    except Exception as e:
        print(f'❌ Real API test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_api_integration()
