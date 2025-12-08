"""
Migration: Add location_verified column to reports table

This migration adds the location_verified column to the reports table.
The column is used to flag whether a report's photo GPS matches the reported location.

Run with: python -m src.scripts.migrate_add_location_verified
"""

import sys
import os

# Add the parent directory to sys.path to allow importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sqlalchemy import text
from src.infrastructure.database import engine

def migrate():
    """Add location_verified column to reports table if it doesn't exist."""
    print("Starting migration: add location_verified to reports table...")

    with engine.connect() as conn:
        # Check if column already exists
        check_query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'reports'
            AND column_name = 'location_verified'
        """)
        result = conn.execute(check_query)
        column_exists = result.fetchone() is not None

        if column_exists:
            print("Column 'location_verified' already exists. Skipping migration.")
            return

        # Add the column with default value TRUE
        # All existing reports are assumed to be location verified
        add_column_query = text("""
            ALTER TABLE reports
            ADD COLUMN location_verified BOOLEAN DEFAULT TRUE NOT NULL
        """)

        try:
            conn.execute(add_column_query)
            conn.commit()
            print("Successfully added 'location_verified' column to reports table.")
            print("Default value: TRUE (all existing reports marked as location verified)")
        except Exception as e:
            print(f"Error adding column: {e}")
            conn.rollback()
            raise

def rollback():
    """Remove location_verified column from reports table."""
    print("Rolling back migration: removing location_verified from reports table...")

    with engine.connect() as conn:
        # Check if column exists before trying to drop
        check_query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'reports'
            AND column_name = 'location_verified'
        """)
        result = conn.execute(check_query)
        column_exists = result.fetchone() is not None

        if not column_exists:
            print("Column 'location_verified' does not exist. Nothing to rollback.")
            return

        drop_column_query = text("""
            ALTER TABLE reports
            DROP COLUMN location_verified
        """)

        try:
            conn.execute(drop_column_query)
            conn.commit()
            print("Successfully removed 'location_verified' column from reports table.")
        except Exception as e:
            print(f"Error removing column: {e}")
            conn.rollback()
            raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        rollback()
    else:
        migrate()
