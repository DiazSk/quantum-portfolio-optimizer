"""
Immutable Audit Trail System for Quantum Portfolio Optimizer

This module provides blockchain-style immutable audit logging with cryptographic
verification for regulatory compliance and institutional audit requirements.

Features:
- Immutable audit trail with hash chain verification
- Digital signature support for audit integrity
- Comprehensive event tracking across all system components
- Regulatory compliance support (SOX, GDPR, institutional standards)
- High-performance logging with minimal overhead (<10ms)
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import asyncio
import logging

from ..database.connection import DatabaseConnection
from ..utils.professional_logging import get_logger

logger = get_logger(__name__)


@dataclass
class AuditEventData:
    """Structured audit event data with standardized fields."""
    event_type: str
    entity_type: str  # 'portfolio', 'model', 'trade', 'user', 'system'
    entity_id: Optional[str]
    action: str  # 'create', 'update', 'delete', 'execute', 'override'
    details: Dict[str, Any]
    user_id: Optional[int]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    business_context: Optional[Dict[str, Any]] = None
    regulatory_flags: Optional[List[str]] = None


@dataclass
class AuditEntry:
    """Immutable audit trail entry with cryptographic verification."""
    id: Optional[int]
    event_id: str
    previous_hash: Optional[str]
    current_hash: str
    event_data: AuditEventData
    portfolio_id: Optional[int]
    model_version: Optional[str]
    data_sources: Optional[Dict[str, Any]]
    digital_signature: Optional[str]
    created_at: datetime
    is_verified: bool = False


class AuditHashCalculator:
    """Cryptographic hash calculation for audit trail integrity."""
    
    @staticmethod
    def calculate_entry_hash(
        event_id: str,
        previous_hash: Optional[str],
        event_data: AuditEventData,
        created_at: datetime
    ) -> str:
        """Calculate SHA-256 hash for audit entry."""
        # Create deterministic hash input
        hash_input = {
            "event_id": event_id,
            "previous_hash": previous_hash or "",
            "event_data": asdict(event_data),
            "timestamp": created_at.isoformat()
        }
        
        # Convert to canonical JSON string
        canonical_json = json.dumps(hash_input, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    @staticmethod
    def verify_hash_chain(entries: List[AuditEntry]) -> bool:
        """Verify integrity of audit trail hash chain."""
        if not entries:
            return True
            
        # Sort by creation time to ensure proper order
        sorted_entries = sorted(entries, key=lambda x: x.created_at)
        
        for i, entry in enumerate(sorted_entries):
            expected_previous = sorted_entries[i-1].current_hash if i > 0 else None
            
            # Verify previous hash reference
            if entry.previous_hash != expected_previous:
                logger.error(f"Hash chain broken at entry {entry.event_id}")
                return False
            
            # Verify current hash calculation
            expected_hash = AuditHashCalculator.calculate_entry_hash(
                entry.event_id,
                entry.previous_hash,
                entry.event_data,
                entry.created_at
            )
            
            if entry.current_hash != expected_hash:
                logger.error(f"Hash verification failed for entry {entry.event_id}")
                return False
        
        return True


class DigitalSignatureManager:
    """Digital signature management for audit trail entries."""
    
    def __init__(self):
        self._private_key = None
        self._public_key = None
        self._load_or_generate_keys()
    
    def _load_or_generate_keys(self):
        """Load existing keys or generate new RSA key pair."""
        try:
            # In production, load from secure key storage
            # For now, generate new keys (should be persistent in production)
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self._public_key = self._private_key.public_key()
            logger.info("Generated new RSA key pair for audit signatures")
        except Exception as e:
            logger.error(f"Failed to load/generate audit signing keys: {e}")
            raise
    
    def sign_audit_entry(self, entry_hash: str) -> str:
        """Create digital signature for audit entry hash."""
        try:
            signature = self._private_key.sign(
                entry_hash.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            logger.error(f"Failed to sign audit entry: {e}")
            raise
    
    def verify_signature(self, entry_hash: str, signature_hex: str) -> bool:
        """Verify digital signature for audit entry."""
        try:
            signature = bytes.fromhex(signature_hex)
            self._public_key.verify(
                signature,
                entry_hash.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class ImmutableAuditTrail:
    """
    Immutable audit trail system with cryptographic verification.
    
    Provides blockchain-style audit logging for regulatory compliance
    with tamper-evident storage and verification capabilities.
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.signature_manager = DigitalSignatureManager()
        self._last_hash_cache = None
        self._cache_lock = asyncio.Lock()
    
    async def log_audit_event(
        self,
        event_data: AuditEventData,
        portfolio_id: Optional[int] = None,
        model_version: Optional[str] = None,
        data_sources: Optional[Dict[str, Any]] = None,
        auto_sign: bool = True
    ) -> str:
        """
        Log immutable audit event with cryptographic verification.
        
        Args:
            event_data: Structured audit event information
            portfolio_id: Associated portfolio ID (if applicable)
            model_version: Model version used (if applicable)
            data_sources: Data sources involved (if applicable)
            auto_sign: Whether to automatically create digital signature
            
        Returns:
            event_id: Unique identifier for the audit entry
            
        Raises:
            AuditError: If audit logging fails
        """
        try:
            async with self._cache_lock:
                # Generate unique event ID
                event_id = str(uuid.uuid4())
                created_at = datetime.now(timezone.utc)
                
                # Get previous hash for chain integrity
                previous_hash = await self._get_last_hash()
                
                # Calculate current hash
                current_hash = AuditHashCalculator.calculate_entry_hash(
                    event_id, previous_hash, event_data, created_at
                )
                
                # Create digital signature if requested
                digital_signature = None
                if auto_sign:
                    digital_signature = self.signature_manager.sign_audit_entry(current_hash)
                
                # Create audit entry
                entry = AuditEntry(
                    id=None,
                    event_id=event_id,
                    previous_hash=previous_hash,
                    current_hash=current_hash,
                    event_data=event_data,
                    portfolio_id=portfolio_id,
                    model_version=model_version,
                    data_sources=data_sources,
                    digital_signature=digital_signature,
                    created_at=created_at,
                    is_verified=auto_sign
                )
                
                # Store in database
                await self._store_audit_entry(entry)
                
                # Update cache
                self._last_hash_cache = current_hash
                
                logger.info(
                    f"Audit event logged successfully",
                    extra={
                        "event_id": event_id,
                        "event_type": event_data.event_type,
                        "entity_type": event_data.entity_type,
                        "action": event_data.action,
                        "user_id": event_data.user_id,
                        "portfolio_id": portfolio_id
                    }
                )
                
                return event_id
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise AuditError(f"Audit logging failed: {e}")
    
    async def _get_last_hash(self) -> Optional[str]:
        """Get hash of most recent audit entry for chain integrity."""
        if self._last_hash_cache:
            return self._last_hash_cache
            
        query = """
            SELECT current_hash 
            FROM audit_trail_entries 
            ORDER BY created_at DESC, id DESC 
            LIMIT 1
        """
        
        result = await self.db.fetch_one(query)
        last_hash = result['current_hash'] if result else None
        self._last_hash_cache = last_hash
        return last_hash
    
    async def _store_audit_entry(self, entry: AuditEntry):
        """Store audit entry in database."""
        query = """
            INSERT INTO audit_trail_entries (
                event_id, previous_hash, current_hash, event_type,
                event_data, user_id, portfolio_id, model_version,
                data_sources, digital_signature, created_at, is_verified
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
            )
        """
        
        await self.db.execute(
            query,
            entry.event_id,
            entry.previous_hash,
            entry.current_hash,
            entry.event_data.event_type,
            json.dumps(asdict(entry.event_data)),
            entry.event_data.user_id,
            entry.portfolio_id,
            entry.model_version,
            json.dumps(entry.data_sources) if entry.data_sources else None,
            entry.digital_signature,
            entry.created_at,
            entry.is_verified
        )
    
    async def get_audit_trail(
        self,
        event_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_id: Optional[int] = None,
        portfolio_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Query audit trail with filtering and pagination.
        
        Args:
            event_type: Filter by event type
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            user_id: Filter by user ID
            portfolio_id: Filter by portfolio ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of audit entries matching criteria
        """
        conditions = []
        params = []
        param_count = 0
        
        if event_type:
            param_count += 1
            conditions.append(f"event_type = ${param_count}")
            params.append(event_type)
        
        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
        
        if portfolio_id:
            param_count += 1
            conditions.append(f"portfolio_id = ${param_count}")
            params.append(portfolio_id)
        
        if start_date:
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(end_date)
        
        if entity_type or entity_id:
            if entity_type:
                param_count += 1
                conditions.append(f"event_data->>'entity_type' = ${param_count}")
                params.append(entity_type)
            
            if entity_id:
                param_count += 1
                conditions.append(f"event_data->>'entity_id' = ${param_count}")
                params.append(entity_id)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        param_count += 1
        limit_param = f"${param_count}"
        params.append(limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(offset)
        
        query = f"""
            SELECT * FROM audit_trail_entries
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT {limit_param} OFFSET {offset_param}
        """
        
        rows = await self.db.fetch_all(query, *params)
        
        entries = []
        for row in rows:
            event_data_dict = json.loads(row['event_data'])
            event_data = AuditEventData(**event_data_dict)
            
            entry = AuditEntry(
                id=row['id'],
                event_id=row['event_id'],
                previous_hash=row['previous_hash'],
                current_hash=row['current_hash'],
                event_data=event_data,
                portfolio_id=row['portfolio_id'],
                model_version=row['model_version'],
                data_sources=json.loads(row['data_sources']) if row['data_sources'] else None,
                digital_signature=row['digital_signature'],
                created_at=row['created_at'],
                is_verified=row['is_verified']
            )
            entries.append(entry)
        
        return entries
    
    async def verify_integrity(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Verify integrity of audit trail using hash chain and digital signatures.
        
        This method provides comprehensive integrity verification for compliance.
        
        Args:
            start_date: Start date for verification range
            end_date: End date for verification range
            
        Returns:
            Dict containing verification results and any issues found
        """
        return await self.verify_audit_integrity(start_date, end_date)
        """
        Verify integrity of audit trail using hash chain and digital signatures.
        
        Args:
            start_date: Start date for verification range
            end_date: End date for verification range
            
        Returns:
            Dict containing verification results and any issues found
        """
        try:
            # Get audit entries for verification
            entries = await self.get_audit_trail(
                start_date=start_date,
                end_date=end_date,
                limit=10000  # Verify in batches if needed
            )
            
            if not entries:
                return {
                    "status": "success",
                    "entries_verified": 0,
                    "hash_chain_valid": True,
                    "signatures_valid": True,
                    "issues": []
                }
            
            # Verify hash chain integrity
            hash_chain_valid = AuditHashCalculator.verify_hash_chain(entries)
            
            # Verify digital signatures
            signature_issues = []
            valid_signatures = 0
            total_signatures = 0
            
            for entry in entries:
                if entry.digital_signature:
                    total_signatures += 1
                    if self.signature_manager.verify_signature(
                        entry.current_hash, 
                        entry.digital_signature
                    ):
                        valid_signatures += 1
                    else:
                        signature_issues.append({
                            "event_id": entry.event_id,
                            "issue": "Invalid digital signature",
                            "created_at": entry.created_at.isoformat()
                        })
            
            signatures_valid = len(signature_issues) == 0
            
            result = {
                "status": "success" if hash_chain_valid and signatures_valid else "failure",
                "entries_verified": len(entries),
                "hash_chain_valid": hash_chain_valid,
                "signatures_valid": signatures_valid,
                "signatures_verified": f"{valid_signatures}/{total_signatures}",
                "issues": signature_issues if signature_issues else [],
                "verification_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if not hash_chain_valid:
                result["issues"].append({
                    "issue": "Hash chain integrity verification failed",
                    "severity": "critical"
                })
            
            logger.info(
                f"Audit integrity verification completed",
                extra={
                    "entries_verified": len(entries),
                    "hash_chain_valid": hash_chain_valid,
                    "signatures_valid": signatures_valid,
                    "issues_found": len(result["issues"])
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Audit integrity verification failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "verification_timestamp": datetime.now(timezone.utc).isoformat()
            }


class AuditEventCapture:
    """
    Comprehensive audit event capture across all system components.
    
    Provides convenient methods for capturing standardized audit events
    from different parts of the quantum portfolio optimizer system.
    """
    
    def __init__(self, audit_trail: ImmutableAuditTrail):
        self.audit_trail = audit_trail
    
    async def capture_portfolio_decision(
        self,
        portfolio_id: int,
        decision_type: str,
        optimization_params: Dict[str, Any],
        model_version: str,
        user_id: int,
        session_context: Dict[str, Any]
    ) -> str:
        """Capture portfolio optimization decision audit event."""
        event_data = AuditEventData(
            event_type="portfolio_decision",
            entity_type="portfolio",
            entity_id=str(portfolio_id),
            action="optimize",
            details={
                "decision_type": decision_type,
                "optimization_params": optimization_params,
                "result_metrics": session_context.get("result_metrics", {}),
                "constraints_applied": session_context.get("constraints", [])
            },
            user_id=user_id,
            session_id=session_context.get("session_id"),
            ip_address=session_context.get("ip_address"),
            user_agent=session_context.get("user_agent"),
            business_context={
                "decision_confidence": session_context.get("confidence_score"),
                "risk_adjusted": session_context.get("risk_adjustment_applied"),
                "alternative_data_used": session_context.get("alternative_data_sources", [])
            },
            regulatory_flags=["fiduciary_decision", "investment_advice"]
        )
        
        return await self.audit_trail.log_audit_event(
            event_data,
            portfolio_id=portfolio_id,
            model_version=model_version,
            data_sources=session_context.get("data_sources")
        )
    
    async def capture_trade_execution(
        self,
        trade_id: str,
        portfolio_id: int,
        trade_details: Dict[str, Any],
        execution_context: Dict[str, Any],
        user_id: int
    ) -> str:
        """Capture trade execution audit event."""
        event_data = AuditEventData(
            event_type="trade_execution",
            entity_type="trade",
            entity_id=trade_id,
            action="execute",
            details={
                "trade_details": trade_details,
                "execution_price": execution_context.get("execution_price"),
                "execution_time": execution_context.get("execution_time"),
                "slippage": execution_context.get("slippage"),
                "commission": execution_context.get("commission")
            },
            user_id=user_id,
            session_id=execution_context.get("session_id"),
            ip_address=execution_context.get("ip_address"),
            user_agent=execution_context.get("user_agent"),
            business_context={
                "market_conditions": execution_context.get("market_conditions"),
                "execution_algorithm": execution_context.get("execution_algorithm"),
                "risk_checks_passed": execution_context.get("risk_checks", [])
            },
            regulatory_flags=["trade_execution", "best_execution"]
        )
        
        return await self.audit_trail.log_audit_event(
            event_data,
            portfolio_id=portfolio_id
        )
    
    async def capture_risk_override(
        self,
        portfolio_id: int,
        override_type: str,
        original_limits: Dict[str, Any],
        new_limits: Dict[str, Any],
        justification: str,
        user_id: int,
        approval_context: Dict[str, Any]
    ) -> str:
        """Capture risk limit override audit event."""
        event_data = AuditEventData(
            event_type="risk_override",
            entity_type="portfolio",
            entity_id=str(portfolio_id),
            action="override",
            details={
                "override_type": override_type,
                "original_limits": original_limits,
                "new_limits": new_limits,
                "justification": justification,
                "approval_required": approval_context.get("approval_required", False),
                "approver_id": approval_context.get("approver_id")
            },
            user_id=user_id,
            session_id=approval_context.get("session_id"),
            ip_address=approval_context.get("ip_address"),
            user_agent=approval_context.get("user_agent"),
            business_context={
                "risk_escalation": True,
                "override_duration": approval_context.get("override_duration"),
                "monitoring_enhanced": approval_context.get("enhanced_monitoring", True)
            },
            regulatory_flags=["risk_override", "limit_breach", "supervisory_review"]
        )
        
        return await self.audit_trail.log_audit_event(
            event_data,
            portfolio_id=portfolio_id
        )
    
    async def capture_ml_prediction(
        self,
        model_id: str,
        model_version: str,
        prediction_context: Dict[str, Any],
        input_data_sources: List[str],
        confidence_score: float,
        feature_importance: Dict[str, float]
    ) -> str:
        """Capture ML model prediction audit event."""
        event_data = AuditEventData(
            event_type="ml_prediction",
            entity_type="model",
            entity_id=model_id,
            action="predict",
            details={
                "prediction_type": prediction_context.get("prediction_type"),
                "input_features": prediction_context.get("input_features", []),
                "prediction_value": prediction_context.get("prediction_value"),
                "confidence_score": confidence_score,
                "feature_importance": feature_importance,
                "model_performance_metrics": prediction_context.get("performance_metrics", {})
            },
            user_id=prediction_context.get("user_id"),
            session_id=prediction_context.get("session_id"),
            ip_address=prediction_context.get("ip_address"),
            user_agent=prediction_context.get("user_agent"),
            business_context={
                "prediction_usage": prediction_context.get("usage_context"),
                "alternative_data_weight": prediction_context.get("alt_data_weight", 0.0),
                "model_drift_score": prediction_context.get("drift_score")
            },
            regulatory_flags=["automated_decision", "model_prediction", "data_lineage"]
        )
        
        return await self.audit_trail.log_audit_event(
            event_data,
            model_version=model_version,
            data_sources={
                "input_sources": input_data_sources,
                "training_data_hash": prediction_context.get("training_data_hash"),
                "feature_sources": prediction_context.get("feature_sources", {})
            }
        )


class AuditError(Exception):
    """Custom exception for audit trail operations."""
    pass


# Convenience function for quick audit logging
async def log_audit_event(
    event_type: str,
    entity_type: str,
    action: str,
    details: Dict[str, Any],
    user_id: Optional[int] = None,
    entity_id: Optional[str] = None,
    portfolio_id: Optional[int] = None,
    model_version: Optional[str] = None,
    session_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function for logging audit events.
    
    This function creates a database connection and audit trail instance
    for one-off audit logging when the caller doesn't have these objects.
    """
    db = DatabaseConnection()
    audit_trail = ImmutableAuditTrail(db)
    
    event_data = AuditEventData(
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        details=details,
        user_id=user_id,
        session_id=session_context.get("session_id") if session_context else None,
        ip_address=session_context.get("ip_address") if session_context else None,
        user_agent=session_context.get("user_agent") if session_context else None,
        business_context=session_context.get("business_context") if session_context else None,
        regulatory_flags=session_context.get("regulatory_flags") if session_context else None
    )
    
    return await audit_trail.log_audit_event(
        event_data,
        portfolio_id=portfolio_id,
        model_version=model_version,
        data_sources=session_context.get("data_sources") if session_context else None
    )
