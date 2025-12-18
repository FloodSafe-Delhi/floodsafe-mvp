"""
Migration: Add archived_at column to reports table

This migration adds support for archiving flood reports after 3 days.
Reports are auto-archived based on their timestamp (3 days old) OR
can be manually archived by setting archived_at.

Run with:
    python -m apps.backend.src.scripts.migrate_add_report_archive

This is safe to run multiple times - it checks if column exists first.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

# Load environment variables
load_dotenv(project_root / "apps" / "backend" / ".env")


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable not set")
    return url


def run_migration():
    """Add archived_at column to reports table."""
    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.connect() as conn:
        # Check if column already exists
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('reports')]

        if 'archived_at' in columns:
            print("[OK] Column 'archived_at' already exists in reports table")
            return True

        print("Adding 'archived_at' column to reports table...")

        # Add the column
        conn.execute(text("""
            ALTER TABLE reports
            ADD COLUMN archived_at TIMESTAMP NULL
        """))

        # Add index for efficient filtering
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reports_archived_at
            ON reports (archived_at)
        """))

        # Add composite index for common query pattern (active reports by time)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reports_timestamp_archived
            ON reports (timestamp DESC, archived_at)
        """))

        conn.commit()
        print("[OK] Successfully added 'archived_at' column with indexes")

        # Show stats
        result = conn.execute(text("SELECT COUNT(*) FROM reports"))
        total = result.scalar()
        print(f"  Total reports in database: {total}")

        return True


def rollback_migration():
    """Remove archived_at column (for rollback if needed)."""
    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.connect() as conn:
        print("Rolling back: Removing 'archived_at' column...")

        conn.execute(text("DROP INDEX IF EXISTS idx_reports_archived_at"))
        conn.execute(text("DROP INDEX IF EXISTS idx_reports_timestamp_archived"))
        conn.execute(text("ALTER TABLE reports DROP COLUMN IF EXISTS archived_at"))

        conn.commit()
        print("[OK] Rollback complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate reports table for archive feature")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    args = parser.parse_args()

    try:
        if args.rollback:
            rollback_migration()
        else:
            run_migration()
    except Exception as e:
        print(f"[FAIL] Migration failed: {e}")
        sys.exit(1)
