"""
Database Connection Manager
Centralized database connection and query management

Provides secure database connections with:
- Connection pooling and optimization
- Transaction management
- Query execution with error handling
- Database migration support
"""

import sqlite3
import os
import logging
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import threading
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration parameters"""
    database_path: str
    max_connections: int = 10
    timeout: float = 30.0
    check_same_thread: bool = False


class DatabaseManager:
    """
    Centralized database connection manager for the quantum portfolio platform
    
    Manages SQLite connections with connection pooling, transaction management,
    and query optimization for enterprise-grade performance.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database manager"""
        if not hasattr(self, 'initialized'):
            self.config = DatabaseConfig(
                database_path=os.getenv('DATABASE_PATH', 'quantum_portfolio.db')
            )
            self.connection_pool = []
            self.pool_lock = threading.Lock()
            self.initialized = True
            
            # Initialize database
            self._initialize_database()
            logger.info("Database manager initialized")
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get database connection from pool or create new one
        
        Returns:
            SQLite connection object
        """
        with self.pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
        
        # Create new connection
        conn = sqlite3.connect(
            self.config.database_path,
            timeout=self.config.timeout,
            check_same_thread=self.config.check_same_thread
        )
        
        # Configure connection
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        
        return conn
    
    def return_connection(self, conn: sqlite3.Connection):
        """
        Return connection to pool
        
        Args:
            conn: Database connection to return
        """
        with self.pool_lock:
            if len(self.connection_pool) < self.config.max_connections:
                self.connection_pool.append(conn)
            else:
                conn.close()
    
    @contextmanager
    def get_connection_context(self):
        """
        Context manager for database connections
        
        Yields:
            Database connection
        """
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
        """
        Execute SELECT query and return results
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results as list of rows
        """
        try:
            with self.get_connection_context() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection_context() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            raise
    
    def execute_batch(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute batch INSERT/UPDATE operations
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise
    
    @contextmanager
    def transaction(self):
        """
        Transaction context manager
        
        Yields:
            Database connection with transaction
        """
        conn = self.get_connection()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def _initialize_database(self):
        """Initialize database with core tables"""
        try:
            with self.get_connection_context() as conn:
                cursor = conn.cursor()
                
                # Core tables initialization
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS database_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Check current version
                cursor.execute("SELECT MAX(version) FROM database_version")
                current_version = cursor.fetchone()[0] or 0
                
                # Apply migrations if needed
                if current_version < 1:
                    self._apply_migration_v1(cursor)
                    cursor.execute("INSERT INTO database_version (version) VALUES (1)")
                
                conn.commit()
                logger.info(f"Database initialized at version {current_version}")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _apply_migration_v1(self, cursor):
        """Apply version 1 database migration"""
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Portfolios table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                base_currency TEXT DEFAULT 'USD',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # Portfolio holdings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_holdings (
                holding_id TEXT PRIMARY KEY,
                portfolio_id TEXT NOT NULL,
                asset_symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                quantity DECIMAL(15,6) NOT NULL,
                average_cost DECIMAL(15,6),
                current_price DECIMAL(15,6),
                market_value DECIMAL(15,2),
                allocation_percentage DECIMAL(5,4),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
        """)
        
        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                data_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                price DECIMAL(15,6) NOT NULL,
                volume BIGINT,
                timestamp TIMESTAMP NOT NULL,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                metric_id TEXT PRIMARY KEY,
                portfolio_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_value DECIMAL(15,6) NOT NULL,
                calculation_date TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
        """)
        
        logger.info("Applied database migration v1")
    
    def backup_database(self, backup_path: str):
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file
        """
        try:
            with self.get_connection_context() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
                
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Database statistics
        """
        try:
            with self.get_connection_context() as conn:
                cursor = conn.cursor()
                
                # Get table information
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                # Get table sizes
                table_stats = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_stats[table] = count
                
                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size = page_count * page_size
                
                return {
                    'database_path': self.config.database_path,
                    'database_size_bytes': db_size,
                    'table_count': len(tables),
                    'table_stats': table_stats,
                    'connection_pool_size': len(self.connection_pool)
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close_all_connections(self):
        """Close all connections in pool"""
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool.clear()
        
        logger.info("All database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    return db_manager


# Demo usage
def demo_database_manager():
    """Demonstrate database manager capabilities"""
    dm = DatabaseManager()
    
    print("🗄️ Database Manager Demo")
    print("=" * 30)
    
    # Test connection
    try:
        with dm.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT datetime('now')")
            current_time = cursor.fetchone()[0]
            print(f"✅ Database connected: {current_time}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Test query execution
    try:
        results = dm.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"✅ Found {len(results)} tables")
    except Exception as e:
        print(f"❌ Query failed: {e}")
    
    # Get database stats
    stats = dm.get_database_stats()
    print(f"✅ Database size: {stats.get('database_size_bytes', 0):,} bytes")
    print(f"✅ Tables: {stats.get('table_count', 0)}")
    
    print("🗄️ Database Manager Ready!")


if __name__ == "__main__":
    demo_database_manager()
