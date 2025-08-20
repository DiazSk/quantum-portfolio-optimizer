"""
Database Connection Management for Quantum Portfolio Optimizer

This module provides asynchronous database connection management
for PostgreSQL using asyncpg for high-performance operations.
"""

import asyncio
import asyncpg
import os
from typing import Dict, Any, List, Optional
import logging
from contextlib import asynccontextmanager

from ..utils.professional_logging import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Asynchronous PostgreSQL database connection manager.
    
    Provides connection pooling and transaction management for
    high-performance database operations in the audit trail system.
    """
    
    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url or self._get_connection_url()
        self.pool = None
        self._initialized = False
    
    def _get_connection_url(self) -> str:
        """Get database connection URL from environment variables."""
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'quantum_portfolio')
        username = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', 'password')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize database connection pool."""
        if self._initialized:
            return
            
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self._initialized = True
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("Database connection pool closed")
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query that doesn't return data (INSERT, UPDATE, DELETE)."""
        if not self._initialized:
            await self.initialize()
            
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and return a single row."""
        if not self._initialized:
            await self.initialize()
            
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return all rows."""
        if not self._initialized:
            await self.initialize()
            
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(query, *args)
            return [dict(row) for row in rows]
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self._initialized:
            await self.initialize()
            
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                yield connection


# Singleton instance for global use
_db_instance = None


def get_database_connection() -> DatabaseConnection:
    """Get singleton database connection instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


async def initialize_database():
    """Initialize the global database connection."""
    db = get_database_connection()
    await db.initialize()


async def close_database():
    """Close the global database connection."""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None
