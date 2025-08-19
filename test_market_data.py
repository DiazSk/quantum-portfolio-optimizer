#!/usr/bin/env python3
"""
Test real-time market data fetching
"""

import yfinance as yf
from datetime import datetime

def test_market_data():
    """Test fetching real market data"""
    print("🌍 Testing Real-Time Market Data")
    print("=" * 40)
    
    tickers = {
        "SPY": "S&P 500",
        "^VIX": "VIX",
        "^TNX": "10Y Treasury", 
        "DX-Y.NYB": "Dollar Index",
        "BTC-USD": "Bitcoin"
    }
    
    for ticker, name in tickers.items():
        try:
            print(f"\n📊 Fetching {name} ({ticker})...")
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                change = float(data['Close'].pct_change().iloc[-1])
                
                if ticker == "^TNX":
                    print(f"✅ {name}: {price:.2f}% ({change:+.2%})")
                elif ticker == "BTC-USD":
                    print(f"✅ {name}: ${price:,.0f} ({change:+.2%})")
                else:
                    print(f"✅ {name}: {price:.2f} ({change:+.2%})")
            else:
                print(f"❌ {name}: No data available")
                
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print(f"\n🕐 Data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_market_data()
