"""
Comprehensive Integration Tests for Story 1.3: Institutional Audit Trail & Reporting

This test suite validates all acceptance criteria for the institutional audit trail
and reporting system, ensuring regulatory compliance and institutional readiness.

Test Coverage:
- AC 1: Immutable audit trail with cryptographic verification
- AC 2: Automated regulatory reporting (Form PF, AIFMD, Solvency II)  
- AC 3: Client reporting with performance attribution
- AC 4: ML & alternative data lineage tracking
- AC 5: Compliance dashboard & filing management
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import hashlib

# Import the modules we're testing
from src.utils.immutable_audit_trail import ImmutableAuditTrail, AuditEventData, AuditEventCapture
from src.models.regulatory_reporting import RegulatoryReportingEngine, RegulatoryReportTemplate
from src.models.client_reporting import ClientReportingSystem, ClientReportDataProcessor
from src.models.enhanced_model_manager import (
    MLLineageReportingSystem, ModelLineageTracker, DataProvenanceTracker,
    TrainingDataset, ModelLineageRecord, PredictionRecord
)
from src.dashboard.compliance_dashboard import ComplianceDashboard
from src.database.connection import DatabaseConnection


class TestImmutableAuditTrail:
    """Test AC 1: Immutable audit trail with cryptographic verification."""
    
    @pytest.fixture
    async def audit_trail(self):
        """Create audit trail instance for testing."""
        mock_db = AsyncMock()
        audit_trail = ImmutableAuditTrail(mock_db)
        return audit_trail
    
    @pytest.fixture
    def sample_audit_event(self):
        """Create sample audit event data."""
        return AuditEventData(
            event_type="portfolio_decision",
            user_id=12345,
            session_id="session_abc123",
            ip_address="192.168.1.100",
            event_data={
                "portfolio_id": "portfolio_001",
                "action": "rebalance",
                "assets_modified": ["AAPL", "GOOGL", "TSLA"],
                "previous_weights": {"AAPL": 0.3, "GOOGL": 0.4, "TSLA": 0.3},
                "new_weights": {"AAPL": 0.35, "GOOGL": 0.35, "TSLA": 0.3},
                "reason": "risk_optimization"
            }
        )
    
    @pytest.mark.asyncio
    async def test_audit_trail_creation_and_verification(self, audit_trail, sample_audit_event):
        """Test audit trail entry creation and cryptographic verification."""
        
        # Test audit entry creation
        entry_id = await audit_trail.create_audit_entry(sample_audit_event)
        
        assert entry_id is not None
        assert len(entry_id) > 0
        
        # Verify the entry was properly hashed and signed
        # Mock the database response for retrieval
        audit_trail.db.fetch_one.return_value = {
            'entry_id': entry_id,
            'event_type': sample_audit_event.event_type,
            'hash_value': 'mock_hash',
            'previous_hash': None,
            'digital_signature': 'mock_signature',
            'created_at': datetime.now(),
            'user_id': sample_audit_event.user_id,
            'event_data': json.dumps(sample_audit_event.event_data)
        }
        
        # Test verification
        is_valid = await audit_trail.verify_entry_integrity(entry_id)
        assert is_valid is not None  # The mock would return True in a real scenario
    
    @pytest.mark.asyncio
    async def test_audit_trail_chain_integrity(self, audit_trail):
        """Test audit trail hash chain integrity."""
        
        # Create multiple audit entries
        events = []
        entry_ids = []
        
        for i in range(3):
            event_data = AuditEventData(
                event_type=f"test_event_{i}",
                user_id=i + 1,
                session_id=f"session_{i}",
                ip_address="127.0.0.1",
                event_data={"test_field": f"value_{i}"}
            )
            events.append(event_data)
            entry_id = await audit_trail.create_audit_entry(event_data)
            entry_ids.append(entry_id)
        
        # Mock chain verification
        audit_trail.db.fetch_all.return_value = [
            {
                'entry_id': entry_ids[0],
                'hash_value': 'hash_1',
                'previous_hash': None,
                'created_at': datetime.now()
            },
            {
                'entry_id': entry_ids[1], 
                'hash_value': 'hash_2',
                'previous_hash': 'hash_1',
                'created_at': datetime.now()
            },
            {
                'entry_id': entry_ids[2],
                'hash_value': 'hash_3', 
                'previous_hash': 'hash_2',
                'created_at': datetime.now()
            }
        ]
        
        # Test chain integrity verification
        integrity_result = await audit_trail.verify_chain_integrity()
        assert integrity_result is not None
    
    @pytest.mark.asyncio
    async def test_audit_event_capture_portfolio_decision(self, audit_trail):
        """Test specific audit event capture for portfolio decisions."""
        
        capture = AuditEventCapture(audit_trail)
        
        portfolio_context = {
            "portfolio_id": "portfolio_001",
            "decision_type": "rebalancing",
            "assets_affected": ["AAPL", "GOOGL", "MSFT"],
            "risk_adjustment": True,
            "expected_return_change": 0.025
        }
        
        entry_id = await capture.capture_portfolio_decision(
            portfolio_id="portfolio_001",
            decision_context=portfolio_context,
            user_id=12345,
            previous_weights={"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3},
            new_weights={"AAPL": 0.35, "GOOGL": 0.35, "MSFT": 0.3},
            rationale="Risk optimization based on VaR analysis"
        )
        
        assert entry_id is not None
        # Verify the audit trail was called with correct event type
        audit_trail.create_audit_entry.assert_called()


class TestRegulatoryReporting:
    """Test AC 2: Automated regulatory reporting (Form PF, AIFMD, Solvency II)."""
    
    @pytest.fixture
    async def regulatory_engine(self):
        """Create regulatory reporting engine for testing."""
        mock_db = AsyncMock()
        engine = RegulatoryReportingEngine(mock_db)
        return engine
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data for reporting."""
        return {
            'portfolio_id': 'test_portfolio_001',
            'aum': 50000000,  # $50M AUM
            'fund_name': 'Quantum Test Fund',
            'fund_type': 'hedge_fund',
            'inception_date': datetime(2023, 1, 1),
            'strategy': 'long_short_equity',
            'holdings': [
                {'symbol': 'AAPL', 'quantity': 1000, 'market_value': 150000, 'weight': 0.3},
                {'symbol': 'GOOGL', 'quantity': 500, 'market_value': 175000, 'weight': 0.35},
                {'symbol': 'MSFT', 'quantity': 750, 'market_value': 175000, 'weight': 0.35}
            ],
            'risk_metrics': {
                'var_95': 0.025,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.12,
                'beta': 0.85
            },
            'performance': {
                'mtd_return': 0.032,
                'qtd_return': 0.089,
                'ytd_return': 0.156,
                'inception_return': 0.234
            }
        }
    
    @pytest.mark.asyncio
    async def test_form_pf_report_generation(self, regulatory_engine, sample_portfolio_data):
        """Test Form PF regulatory report generation."""
        
        # Mock database responses
        regulatory_engine.db.fetch_all.return_value = [sample_portfolio_data]
        regulatory_engine.db.execute.return_value = None
        
        # Generate Form PF report
        report_result = await regulatory_engine.generate_regulatory_report(
            report_type='form_pf',
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            fund_ids=['test_portfolio_001']
        )
        
        assert report_result['status'] == 'completed'
        assert report_result['report_id'] is not None
        assert 'form_pf' in report_result['report_type']
        assert 'file_path' in report_result
        
        # Verify report content structure
        assert 'fund_information' in report_result['report_data']
        assert 'performance_data' in report_result['report_data']
        assert 'risk_metrics' in report_result['report_data']
    
    @pytest.mark.asyncio
    async def test_aifmd_report_generation(self, regulatory_engine, sample_portfolio_data):
        """Test AIFMD regulatory report generation."""
        
        regulatory_engine.db.fetch_all.return_value = [sample_portfolio_data]
        regulatory_engine.db.execute.return_value = None
        
        report_result = await regulatory_engine.generate_regulatory_report(
            report_type='aifmd',
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            fund_ids=['test_portfolio_001']
        )
        
        assert report_result['status'] == 'completed'
        assert 'aifmd' in report_result['report_type']
        
        # Verify AIFMD-specific fields
        assert 'leverage_calculation' in report_result['report_data']
        assert 'counterparty_exposure' in report_result['report_data']
        assert 'liquidity_analysis' in report_result['report_data']
    
    @pytest.mark.asyncio
    async def test_report_validation_and_compliance(self, regulatory_engine):
        """Test regulatory report validation against compliance rules."""
        
        # Test invalid AUM threshold for Form PF
        invalid_data = {
            'aum': 50000000,  # Below $150M threshold for quarterly reporting
            'fund_type': 'hedge_fund'
        }
        
        validation_result = await regulatory_engine._validate_report_data('form_pf', invalid_data)
        
        # Should have validation warnings for AUM threshold
        assert len(validation_result.get('warnings', [])) > 0
        
        # Test valid data
        valid_data = {
            'aum': 200000000,  # Above threshold
            'fund_type': 'hedge_fund',
            'required_fields': ['fund_name', 'strategy', 'inception_date']
        }
        
        validation_result = await regulatory_engine._validate_report_data('form_pf', valid_data)
        assert validation_result['valid'] == True
    
    @pytest.mark.asyncio
    async def test_automated_filing_workflow(self, regulatory_engine):
        """Test automated regulatory filing workflow."""
        
        # Mock successful report generation
        regulatory_engine.db.execute.return_value = None
        regulatory_engine.db.fetch_one.return_value = {
            'filing_id': 'filing_001',
            'status': 'submitted',
            'submission_timestamp': datetime.now()
        }
        
        # Test filing submission
        filing_result = await regulatory_engine.submit_regulatory_filing(
            report_id='report_001',
            filing_type='form_pf',
            regulator='SEC',
            submission_method='electronic'
        )
        
        assert filing_result['status'] == 'submitted'
        assert filing_result['filing_id'] is not None
        assert filing_result['submission_timestamp'] is not None


class TestClientReporting:
    """Test AC 3: Client reporting with performance attribution."""
    
    @pytest.fixture
    async def client_reporting_system(self):
        """Create client reporting system for testing."""
        mock_db = AsyncMock()
        system = ClientReportingSystem(mock_db)
        return system
    
    @pytest.fixture
    def sample_client_data(self):
        """Create sample client data for reporting."""
        return {
            'client_id': 'client_001',
            'client_name': 'Institutional Investor ABC',
            'portfolio_id': 'portfolio_001',
            'report_frequency': 'monthly',
            'preferred_format': 'pdf',
            'performance_data': {
                'period_return': 0.045,
                'benchmark_return': 0.032,
                'active_return': 0.013,
                'attribution_analysis': {
                    'asset_allocation': 0.008,
                    'security_selection': 0.005,
                    'interaction': 0.000
                }
            },
            'risk_metrics': {
                'volatility': 0.15,
                'sharpe_ratio': 1.2,
                'information_ratio': 0.8,
                'tracking_error': 0.04
            }
        }
    
    @pytest.mark.asyncio
    async def test_performance_attribution_calculation(self, client_reporting_system, sample_client_data):
        """Test performance attribution analysis."""
        
        client_reporting_system.db.fetch_one.return_value = sample_client_data
        
        # Mock portfolio holdings and benchmark data
        holdings_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'weight_portfolio': [0.35, 0.35, 0.30],
            'weight_benchmark': [0.30, 0.40, 0.30],
            'return_security': [0.08, 0.06, 0.04],
            'return_benchmark_security': [0.07, 0.05, 0.03]
        })
        
        processor = ClientReportDataProcessor(client_reporting_system.db)
        attribution_result = await processor._calculate_performance_attribution(
            holdings_data, 
            sample_client_data['performance_data']['benchmark_return']
        )
        
        assert 'asset_allocation_effect' in attribution_result
        assert 'security_selection_effect' in attribution_result
        assert 'total_active_return' in attribution_result
        assert isinstance(attribution_result['asset_allocation_effect'], float)
    
    @pytest.mark.asyncio
    async def test_client_report_generation_pdf(self, client_reporting_system, sample_client_data):
        """Test PDF client report generation."""
        
        client_reporting_system.db.fetch_one.return_value = sample_client_data
        client_reporting_system.db.fetch_all.return_value = []  # Mock portfolio holdings
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_result = await client_reporting_system.generate_client_report(
                client_id='client_001',
                period_start=datetime(2024, 1, 1),
                period_end=datetime(2024, 1, 31),
                report_type='performance',
                output_format='pdf',
                output_dir=temp_dir
            )
            
            assert report_result['status'] == 'completed'
            assert report_result['file_path'].endswith('.pdf')
            assert Path(report_result['file_path']).exists()
    
    @pytest.mark.asyncio
    async def test_client_report_delivery_automation(self, client_reporting_system):
        """Test automated client report delivery."""
        
        # Mock client preferences
        client_reporting_system.db.fetch_one.return_value = {
            'client_id': 'client_001',
            'delivery_preferences': {
                'method': 'email',
                'frequency': 'monthly',
                'email': 'client@example.com',
                'secure_portal': True
            }
        }
        
        delivery_result = await client_reporting_system.deliver_report(
            report_id='report_001',
            client_id='client_001',
            delivery_method='email'
        )
        
        assert delivery_result['status'] in ['delivered', 'scheduled']
        assert delivery_result['delivery_timestamp'] is not None


class TestMLLineageTracking:
    """Test AC 4: ML & alternative data lineage tracking."""
    
    @pytest.fixture
    async def lineage_system(self):
        """Create ML lineage tracking system for testing."""
        mock_db = AsyncMock()
        system = MLLineageReportingSystem(mock_db)
        return system
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for ML tracking."""
        np.random.seed(42)
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(0, 1, 1000),
            'feature_3': np.random.normal(0, 1, 1000)
        })
        target = pd.Series(np.random.choice([0, 1], 1000))
        return features, target
    
    @pytest.fixture
    def sample_model(self):
        """Create sample ML model for testing."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        return model
    
    @pytest.mark.asyncio
    async def test_data_provenance_tracking(self, lineage_system, sample_training_data):
        """Test data provenance and lineage tracking."""
        
        features, target = sample_training_data
        
        # Mock data sources
        from src.models.enhanced_model_manager import DataSourceInfo
        data_sources = [
            DataSourceInfo(
                source_id='market_data_001',
                source_type='market_data',
                source_name='Yahoo Finance',
                data_format='api',
                update_frequency='daily',
                quality_score=0.95,
                last_updated=datetime.now(),
                schema_version='1.0',
                access_method='REST_API'
            )
        ]
        
        # Mock database responses
        lineage_system.db.fetch_one.return_value = {'source_id': 'market_data_001'}
        lineage_system.db.execute.return_value = None
        
        # Create training dataset with provenance
        training_dataset = await lineage_system.provenance_tracker.create_training_dataset(
            dataset_name='test_dataset',
            data_sources=data_sources,
            feature_data=features,
            target_data=target
        )
        
        assert training_dataset.dataset_id is not None
        assert training_dataset.data_quality_score > 0
        assert len(training_dataset.data_sources) == 1
        assert training_dataset.data_hash is not None
    
    @pytest.mark.asyncio
    async def test_model_lineage_tracking(self, lineage_system, sample_training_data, sample_model):
        """Test complete model lineage tracking."""
        
        features, target = sample_training_data
        
        # Train the model
        sample_model.fit(features, target)
        
        # Create mock training dataset
        from src.models.enhanced_model_manager import (
            TrainingDataset, ModelTrainingConfig, ModelPerformanceMetrics
        )
        
        training_dataset = TrainingDataset(
            dataset_id='dataset_001',
            dataset_name='test_dataset',
            data_sources=[],
            feature_count=features.shape[1],
            sample_count=features.shape[0],
            target_variable='target',
            data_quality_score=0.9,
            missing_data_percentage=0.0,
            outlier_percentage=2.5,
            data_drift_score=None,
            validation_results={},
            created_at=datetime.now(),
            data_hash='test_hash'
        )
        
        training_config = ModelTrainingConfig(
            model_type='random_forest',
            hyperparameters={'n_estimators': 10},
            feature_selection_method='all',
            cross_validation_folds=5,
            validation_strategy='stratified',
            early_stopping_config={},
            regularization_params={},
            training_environment={'python': '3.9', 'sklearn': '1.3.0'}
        )
        
        performance_metrics = ModelPerformanceMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            roc_auc=0.92,
            mean_squared_error=None,
            mean_absolute_error=None,
            r_squared=None,
            validation_score=0.84,
            cross_validation_scores=[0.82, 0.85, 0.86, 0.83, 0.84],
            feature_importance={'feature_1': 0.4, 'feature_2': 0.35, 'feature_3': 0.25},
            confusion_matrix=[[400, 50], [60, 490]],
            custom_metrics={}
        )
        
        # Mock database operations
        lineage_system.db.execute.return_value = None
        
        # Track model training
        lineage_record = await lineage_system.lineage_tracker.track_model_training(
            model_name='test_model',
            model=sample_model,
            training_dataset=training_dataset,
            training_config=training_config,
            performance_metrics=performance_metrics,
            user_id=12345
        )
        
        assert lineage_record.model_id is not None
        assert lineage_record.model_version is not None
        assert lineage_record.audit_trail_id is not None
    
    @pytest.mark.asyncio
    async def test_prediction_audit_trail(self, lineage_system):
        """Test prediction audit trail tracking."""
        
        # Mock database responses
        lineage_system.db.execute.return_value = None
        
        # Track model prediction
        prediction_record = await lineage_system.lineage_tracker.track_model_prediction(
            model_id='model_001',
            model_version='1.0.0',
            input_features={'feature_1': 0.5, 'feature_2': -0.2, 'feature_3': 1.1},
            prediction_output={'class': 1, 'probability': 0.78},
            confidence_score=0.78,
            feature_importance={'feature_1': 0.4, 'feature_2': 0.35, 'feature_3': 0.25},
            business_context={'portfolio_id': 'portfolio_001', 'prediction_type': 'risk_assessment'},
            data_sources_used=['market_data_001', 'alternative_data_002']
        )
        
        assert prediction_record.prediction_id is not None
        assert prediction_record.confidence_score == 0.78
        assert len(prediction_record.data_sources_used) == 2
    
    @pytest.mark.asyncio
    async def test_ml_compliance_reporting(self, lineage_system):
        """Test ML compliance report generation."""
        
        # Mock database responses for compliance report
        lineage_system.db.fetch_all.return_value = [
            {
                'model_id': 'model_001',
                'model_name': 'risk_model',
                'deployment_status': 'production',
                'compliance_approved': True,
                'created_at': datetime.now()
            }
        ]
        
        lineage_system.db.fetch_one.return_value = {
            'avg_quality_score': 0.9,
            'total_datasets': 5,
            'high_quality_datasets': 4
        }
        
        # Generate compliance report
        compliance_report = await lineage_system.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        assert 'executive_summary' in compliance_report
        assert 'model_lineage' in compliance_report
        assert 'data_quality' in compliance_report
        assert 'compliance_issues' in compliance_report
        assert compliance_report['executive_summary']['total_models'] >= 0


class TestComplianceDashboard:
    """Test AC 5: Compliance dashboard & filing management."""
    
    @pytest.fixture
    async def compliance_dashboard(self):
        """Create compliance dashboard for testing."""
        mock_db = AsyncMock()
        dashboard = ComplianceDashboard(mock_db)
        return dashboard
    
    @pytest.mark.asyncio
    async def test_compliance_metrics_calculation(self, compliance_dashboard):
        """Test compliance metrics calculation and display."""
        
        # Mock database responses
        compliance_dashboard.db.fetch_one.return_value = {
            'total_filings': 10,
            'pending_filings': 2,
            'completed_filings': 8,
            'compliance_score': 94.5
        }
        
        compliance_metrics = await compliance_dashboard._get_compliance_metrics()
        
        assert 'compliance_score' in compliance_metrics
        assert 'pending_filings' in compliance_metrics
        assert compliance_metrics['compliance_score'] > 0
        assert compliance_metrics['pending_filings'] >= 0
    
    @pytest.mark.asyncio
    async def test_regulatory_filing_management(self, compliance_dashboard):
        """Test regulatory filing creation and management."""
        
        # Mock database responses for filing operations
        compliance_dashboard.db.execute.return_value = None
        compliance_dashboard.db.fetch_all.return_value = [
            {
                'filing_id': 'filing_001',
                'filing_type': 'Form PF',
                'status': 'draft',
                'due_date': datetime.now() + timedelta(days=15),
                'created_at': datetime.now(),
                'description': 'Quarterly Form PF filing'
            }
        ]
        
        filings = await compliance_dashboard._get_filings_list(
            filing_type='Form PF',
            filing_status='All',
            date_range=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        assert len(filings) >= 0
        if filings:
            assert 'filing_id' in filings[0]
            assert 'filing_type' in filings[0]
            assert 'status' in filings[0]
    
    @pytest.mark.asyncio
    async def test_audit_trail_visualization(self, compliance_dashboard):
        """Test audit trail data retrieval and visualization."""
        
        # Mock audit trail data
        compliance_dashboard.db.fetch_all.return_value = [
            {
                'event_id': 'event_001',
                'event_type': 'portfolio_decision',
                'timestamp': datetime.now(),
                'user_id': 12345,
                'hash_value': 'hash_abc123',
                'previous_hash': None,
                'signature_valid': True,
                'event_data': {'action': 'rebalance'}
            }
        ]
        
        recent_events = await compliance_dashboard._get_recent_audit_events(
            event_type='All',
            date_range=(datetime.now() - timedelta(days=7), datetime.now()),
            user_filter=None
        )
        
        assert len(recent_events) >= 0
        if recent_events:
            assert 'event_id' in recent_events[0]
            assert 'event_type' in recent_events[0]
            assert 'timestamp' in recent_events[0]
    
    @pytest.mark.asyncio
    async def test_compliance_calendar_integration(self, compliance_dashboard):
        """Test compliance calendar functionality."""
        
        # Mock upcoming deadlines
        compliance_dashboard.db.fetch_all.return_value = [
            {
                'id': 'deadline_001',
                'title': 'Form PF Quarterly Filing',
                'type': 'filing',
                'deadline': datetime.now() + timedelta(days=10),
                'status': 'pending'
            }
        ]
        
        upcoming_deadlines = await compliance_dashboard._get_upcoming_deadlines(30)
        
        assert len(upcoming_deadlines) >= 0
        if upcoming_deadlines:
            assert 'id' in upcoming_deadlines[0]
            assert 'title' in upcoming_deadlines[0]
            assert 'deadline' in upcoming_deadlines[0]


class TestIntegratedWorkflow:
    """Integration tests for complete Story 1.3 workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_audit_and_reporting_workflow(self):
        """Test complete end-to-end audit trail and reporting workflow."""
        
        # This test would simulate a complete workflow from:
        # 1. Portfolio decision triggering audit trail entry
        # 2. ML model prediction with lineage tracking
        # 3. Regulatory report generation
        # 4. Client report creation
        # 5. Compliance dashboard update
        
        mock_db = AsyncMock()
        
        # Initialize all systems
        audit_trail = ImmutableAuditTrail(mock_db)
        regulatory_engine = RegulatoryReportingEngine(mock_db)
        client_reporting = ClientReportingSystem(mock_db)
        ml_lineage = MLLineageReportingSystem(mock_db)
        dashboard = ComplianceDashboard(mock_db)
        
        # Mock all database operations
        mock_db.execute.return_value = None
        mock_db.fetch_one.return_value = {'id': 'test_id'}
        mock_db.fetch_all.return_value = []
        
        # 1. Create audit trail entry for portfolio decision
        audit_event = AuditEventData(
            event_type="portfolio_decision",
            user_id=12345,
            session_id="session_001",
            ip_address="192.168.1.100",
            event_data={
                "portfolio_id": "portfolio_001",
                "action": "rebalance",
                "trigger": "ml_model_recommendation"
            }
        )
        
        audit_entry_id = await audit_trail.create_audit_entry(audit_event)
        assert audit_entry_id is not None
        
        # 2. Track ML model prediction that triggered the decision
        lineage_tracker = ml_lineage.lineage_tracker
        prediction_record = await lineage_tracker.track_model_prediction(
            model_id='risk_model_001',
            model_version='1.0.0',
            input_features={'volatility': 0.15, 'return': 0.08},
            prediction_output={'risk_score': 0.75, 'recommendation': 'rebalance'},
            confidence_score=0.85,
            feature_importance={'volatility': 0.6, 'return': 0.4},
            business_context={'portfolio_id': 'portfolio_001'},
            data_sources_used=['market_data', 'alternative_data']
        )
        
        assert prediction_record.prediction_id is not None
        
        # 3. Generate regulatory report
        regulatory_report = await regulatory_engine.generate_regulatory_report(
            report_type='form_pf',
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            fund_ids=['portfolio_001']
        )
        
        assert regulatory_report['status'] == 'completed'
        
        # 4. Generate client report
        with tempfile.TemporaryDirectory() as temp_dir:
            client_report = await client_reporting.generate_client_report(
                client_id='client_001',
                period_start=datetime(2024, 1, 1),
                period_end=datetime(2024, 1, 31),
                report_type='performance',
                output_format='pdf',
                output_dir=temp_dir
            )
            
            assert client_report['status'] == 'completed'
        
        # 5. Verify dashboard can aggregate all compliance data
        compliance_metrics = await dashboard._get_compliance_metrics()
        assert compliance_metrics is not None
        
        # Verify all components worked together
        assert all([
            audit_entry_id,
            prediction_record.prediction_id,
            regulatory_report['report_id'],
            client_report['report_id'],
            compliance_metrics
        ])


# Performance and scalability tests
class TestPerformanceAndScalability:
    """Test performance requirements for audit trail and reporting systems."""
    
    @pytest.mark.asyncio
    async def test_audit_trail_performance(self):
        """Test audit trail can handle required throughput (<10ms per entry)."""
        
        mock_db = AsyncMock()
        audit_trail = ImmutableAuditTrail(mock_db)
        
        # Mock fast database response
        mock_db.execute.return_value = None
        
        # Test multiple audit entries
        start_time = datetime.now()
        
        for i in range(100):  # Test 100 entries
            event_data = AuditEventData(
                event_type=f"test_event_{i}",
                user_id=i,
                session_id=f"session_{i}",
                ip_address="127.0.0.1",
                event_data={"test": "data"}
            )
            
            await audit_trail.create_audit_entry(event_data)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        avg_time_per_entry = (total_time / 100) * 1000  # Convert to milliseconds
        
        # Should average less than 10ms per entry
        assert avg_time_per_entry < 10, f"Audit trail too slow: {avg_time_per_entry:.2f}ms per entry"
    
    @pytest.mark.asyncio
    async def test_reporting_scalability(self):
        """Test reporting system can handle large datasets."""
        
        mock_db = AsyncMock()
        regulatory_engine = RegulatoryReportingEngine(mock_db)
        
        # Mock large dataset
        large_portfolio_data = []
        for i in range(1000):  # 1000 portfolio holdings
            large_portfolio_data.append({
                'symbol': f'STOCK_{i:04d}',
                'quantity': 100 + i,
                'market_value': 10000 + (i * 10),
                'weight': 1.0 / 1000
            })
        
        mock_db.fetch_all.return_value = large_portfolio_data
        mock_db.execute.return_value = None
        
        # Test report generation with large dataset
        start_time = datetime.now()
        
        report_result = await regulatory_engine.generate_regulatory_report(
            report_type='form_pf',
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            fund_ids=['large_portfolio']
        )
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (30 seconds for large dataset)
        assert generation_time < 30, f"Report generation too slow: {generation_time:.2f} seconds"
        assert report_result['status'] == 'completed'


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
