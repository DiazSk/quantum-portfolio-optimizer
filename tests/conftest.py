"""
Shared fixtures for all tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_tickers():
    """Standard test tickers"""
    return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']


@pytest.fixture(scope="session")
def sample_prices():
    """Generate sample price data"""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    data = {}
    for ticker in tickers:
        np.random.seed(hash(ticker) % 1000)
        prices = 100 * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
        data[ticker] = prices
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="function")
def mock_yfinance():
    """Mock yfinance for testing"""
    with pytest.mock.patch('yfinance.Ticker') as mock:
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105],
            'Volume': [1000000, 1100000, 900000, 1200000, 1050000]
        })
        ticker_mock.info = {
            'shortName': 'Test Company',
            'sector': 'Technology',
            'marketCap': 1000000000
        }
        mock.return_value = ticker_mock
        yield mock


@pytest.fixture(scope="function")
def mock_api_responses():
    """Mock external API responses"""
    return {
        'reddit': {
            'sentiment': 0.7,
            'mentions': 150,
            'trending': True
        },
        'news': {
            'sentiment': 0.6,
            'articles': 25,
            'positive_ratio': 0.68
        },
        'trends': {
            'interest': 75,
            'rising': True
        }
    }


@pytest.fixture(scope="function")
def portfolio_config():
    """Standard portfolio configuration"""
    return {
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'initial_capital': 100000,
        'max_position': 0.40,
        'min_position': 0.05,
        'risk_free_rate': 0.04,
        'rebalance_frequency': 'monthly'
    }


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests"""
    # Add any singleton resets here
    yield
    # Cleanup after test


@pytest.fixture(scope="function")
def temp_data_dir(tmp_path):
    """Create temporary data directory"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# Performance tracking fixture
@pytest.fixture
def benchmark(request):
    """Benchmark test performance"""
    import time
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"\n[TIME] {request.node.name} took {elapsed:.3f}s")
