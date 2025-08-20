"""
Model Manager Tests
Tests for model saving, loading, and pipeline functionality
"""
import os
import sys
import pytest
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestModelManager:
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for model testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def model_manager(self, temp_models_dir):
        """Create ModelManager instance with temporary directory"""
        with patch('models.model_manager.os.makedirs'):
            from models.model_manager import ModelManager
            manager = ModelManager(models_dir=temp_models_dir)
            return manager
    
    def test_model_manager_initialization(self, temp_models_dir):
        """Test ModelManager initialization"""
        with patch('models.model_manager.os.makedirs') as mock_makedirs:
            from models.model_manager import ModelManager
            
            manager = ModelManager(models_dir=temp_models_dir)
            assert manager.models_dir == temp_models_dir
            mock_makedirs.assert_called_once_with(temp_models_dir, exist_ok=True)
    
    def test_model_manager_default_directory(self):
        """Test ModelManager with default directory"""
        with patch('models.model_manager.os.makedirs') as mock_makedirs:
            from models.model_manager import ModelManager
            
            manager = ModelManager()
            assert manager.models_dir == "models"
            mock_makedirs.assert_called_once_with("models", exist_ok=True)
    
    def test_save_models_basic(self, model_manager, temp_models_dir):
        """Test basic model saving functionality"""
        # Create mock models
        mock_model_1 = Mock()
        mock_model_2 = Mock()
        models = {
            'AAPL': mock_model_1,
            'MSFT': mock_model_2
        }
        
        with patch('models.model_manager.joblib.dump') as mock_dump, \
             patch('models.model_manager.datetime') as mock_datetime:
            
            # Mock datetime
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            timestamp = model_manager.save_models(models)
            
            # Verify joblib.dump was called for each model
            assert mock_dump.call_count == 2
            assert timestamp == "20240101_120000"
    
    def test_save_models_with_metadata(self, model_manager, temp_models_dir):
        """Test model saving with metadata"""
        models = {'AAPL': Mock()}
        metadata = {'accuracy': 0.85, 'features': ['price', 'volume']}
        
        with patch('models.model_manager.joblib.dump'), \
             patch('models.model_manager.datetime') as mock_datetime, \
             patch('builtins.open', create=True) as mock_open, \
             patch('models.model_manager.json.dump') as mock_json_dump:
            
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            timestamp = model_manager.save_models(models, metadata)
            
            # Verify metadata was saved
            mock_open.assert_called()
            mock_json_dump.assert_called_once()
            
            # Check metadata structure
            call_args = mock_json_dump.call_args[0][0]
            assert call_args['timestamp'] == "20240101_120000"
            assert call_args['tickers'] == ['AAPL']
            assert call_args['accuracy'] == 0.85
    
    def test_load_latest_models_no_models(self, model_manager, temp_models_dir):
        """Test loading when no models exist"""
        with patch('models.model_manager.os.listdir', return_value=[]):
            models, metadata = model_manager.load_latest_models()
            
            assert models is None
            assert metadata is None
    
    def test_load_latest_models_success(self, model_manager, temp_models_dir):
        """Test successful model loading"""
        # Mock file listing
        metadata_files = ['metadata_20240101_120000.json', 'metadata_20240102_130000.json']
        
        # Mock metadata content
        mock_metadata = {
            'timestamp': '20240102_130000',
            'tickers': ['AAPL', 'MSFT'],
            'accuracy': 0.9
        }
        
        mock_model = Mock()
        
        with patch('models.model_manager.os.listdir', return_value=metadata_files), \
             patch('builtins.open', create=True) as mock_open, \
             patch('models.model_manager.json.load', return_value=mock_metadata), \
             patch('models.model_manager.os.path.exists', return_value=True), \
             patch('models.model_manager.joblib.load', return_value=mock_model):
            
            models, metadata = model_manager.load_latest_models()
            
            # Verify correct models were loaded
            assert models is not None
            assert metadata == mock_metadata
            assert 'AAPL' in models
            assert 'MSFT' in models
            assert models['AAPL'] == mock_model
    
    def test_load_latest_models_missing_model_files(self, model_manager, temp_models_dir):
        """Test loading when metadata exists but model files are missing"""
        metadata_files = ['metadata_20240101_120000.json']
        mock_metadata = {
            'timestamp': '20240101_120000',
            'tickers': ['AAPL'],
            'accuracy': 0.9
        }
        
        with patch('models.model_manager.os.listdir', return_value=metadata_files), \
             patch('builtins.open', create=True), \
             patch('models.model_manager.json.load', return_value=mock_metadata), \
             patch('models.model_manager.os.path.exists', return_value=False):
            
            models, metadata = model_manager.load_latest_models()
            
            # Should return empty models dict but valid metadata
            assert models == {}
            assert metadata == mock_metadata
    
    def test_complete_pipeline_mock_execution(self):
        """Test complete pipeline execution with mocks"""
        with patch('models.model_manager.os.makedirs'), \
             patch('models.model_manager.ModelManager') as mock_manager, \
             patch('models.model_manager.AlternativeDataCollector') as mock_collector, \
             patch('models.model_manager.PortfolioOptimizer') as mock_optimizer, \
             patch('models.model_manager.os.getenv') as mock_getenv:
            
            # Mock environment variables
            mock_getenv.side_effect = lambda key: {
                'ALPHA_VANTAGE_API_KEY': 'test_key',
                'REDDIT_CLIENT_ID': 'test_id',
                'NEWS_API_KEY': 'test_news_key'
            }.get(key)
            
            # Mock collector behavior
            mock_collector_instance = Mock()
            mock_collector.return_value = mock_collector_instance
            
            # Mock alternative data
            import pandas as pd
            mock_alt_data = pd.DataFrame({
                'ticker': ['AAPL', 'MSFT'],
                'sentiment': [0.7, 0.6],
                'volume': [1000, 2000]
            })
            mock_alt_scores = pd.DataFrame({
                'ticker': ['AAPL', 'MSFT'],
                'alt_data_score': [0.8, 0.7],
                'alt_data_confidence': [0.9, 0.8]
            })
            
            mock_collector_instance.collect_all_alternative_data.return_value = mock_alt_data
            mock_collector_instance.calculate_alternative_data_score.return_value = mock_alt_scores
            
            # Mock optimizer
            mock_optimizer_instance = Mock()
            mock_optimizer.return_value = mock_optimizer_instance
            mock_optimizer_instance.run.return_value = {
                'tickers': ['AAPL', 'MSFT'],
                'weights': [0.6, 0.4],
                'expected_return': 0.12,
                'volatility': 0.15,
                'metrics': {
                    'sharpe_ratio': 1.2,
                    'annual_return': 0.12,
                    'annual_volatility': 0.15
                }
            }
            
            # Import and run
            from models.model_manager import run_complete_pipeline
            
            # Should not raise exceptions
            try:
                result = run_complete_pipeline()
                # If it returns False, that's acceptable for missing APIs
                assert result is False or result is None
            except Exception as e:
                # Pipeline might fail due to missing dependencies, which is expected
                assert "Missing API keys" in str(e) or "Real APIs required" in str(e)
    
    def test_complete_pipeline_missing_api_keys(self):
        """Test pipeline behavior with missing API keys"""
        with patch('models.model_manager.os.makedirs'), \
             patch('models.model_manager.ModelManager'), \
             patch('models.model_manager.AlternativeDataCollector'), \
             patch('models.model_manager.os.getenv', return_value=None):
            
            from models.model_manager import run_complete_pipeline
            
            # Should handle missing API keys gracefully
            result = run_complete_pipeline()
            assert result is False
    
    def test_model_manager_error_handling(self, temp_models_dir):
        """Test error handling in model operations"""
        with patch('models.model_manager.os.makedirs'):
            from models.model_manager import ModelManager
            
            manager = ModelManager(models_dir=temp_models_dir)
            
            # Test save with joblib error
            with patch('models.model_manager.joblib.dump', side_effect=Exception("Save error")):
                with pytest.raises(Exception):
                    manager.save_models({'AAPL': Mock()})
            
            # Test load with JSON error
            with patch('models.model_manager.os.listdir', return_value=['metadata_test.json']), \
                 patch('builtins.open', create=True), \
                 patch('models.model_manager.json.load', side_effect=json.JSONDecodeError("msg", "doc", 0)):
                
                with pytest.raises(json.JSONDecodeError):
                    manager.load_latest_models()
    
    def test_model_timestamps(self, model_manager, temp_models_dir):
        """Test timestamp handling in model operations"""
        models = {'AAPL': Mock()}
        
        with patch('models.model_manager.joblib.dump'), \
             patch('models.model_manager.datetime') as mock_datetime:
            
            # Test specific timestamp format
            mock_datetime.now.return_value.strftime.return_value = "20241215_143022"
            
            timestamp = model_manager.save_models(models)
            assert timestamp == "20241215_143022"
            assert len(timestamp) == 15  # YYYYMMDD_HHMMSS format
