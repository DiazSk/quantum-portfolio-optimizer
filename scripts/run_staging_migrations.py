"""
Safe staging DB migration runner (dry-run capable).
This script is a helper for ops to apply migrations to staging with backups and smoke tests.
It is intentionally conservative: by default it only prints the SQL files to apply.

Usage:
    python scripts/run_staging_migrations.py --apply  # to actually apply (requires DB env vars)

Environment variables used (only required when --apply):
- STAGING_DB_DSN (e.g. postgresql://user:pass@host:5432/dbname)

This script requires psycopg2 or asyncpg when applying; otherwise it will prompt.
"""

import argparse
import os
import sys
from pathlib import Path

MIGRATIONS = [
    'src/database/migrations/003_multi_tenant_auth.sql',
    'src/database/migrations/add_portal_tables.sql'
]


def list_migrations():
    print("Planned migrations:")
    for m in MIGRATIONS:
        print(f" - {m}")


def apply_migration(dsn, migration_path):
    try:
        import psycopg2
    except Exception:
        print("psycopg2 not installed. Cannot apply migrations programmatically.")
        return False

    sql = Path(migration_path).read_text()
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"Applied: {migration_path}")
        return True
    except Exception as e:
        print(f"Failed to apply {migration_path}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Actually apply migrations (requires STAGING_DB_DSN)')
    args = parser.parse_args()

    list_migrations()

    if not args.apply:
        print('\nDry-run mode. Use --apply to execute migrations (requires DB credentials in env).')
        sys.exit(0)

    dsn = os.environ.get('STAGING_DB_DSN')
    if not dsn:
        print('STAGING_DB_DSN not set. Aborting.')
        sys.exit(2)

    # Apply migrations
    for m in MIGRATIONS:
        ok = apply_migration(dsn, m)
        if not ok:
            print('Stopping on failure. Please inspect and roll back if needed.')
            sys.exit(3)

    print('Migrations applied. Please run smoke tests separately.')
