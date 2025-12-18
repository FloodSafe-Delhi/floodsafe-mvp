"""
Migration: Add email/password authentication fields to users table.

This migration adds support for email/password authentication:
- password_hash: bcrypt hash of user's password (NULL for OAuth/Phone users)
- email_verified: boolean flag for email verification status

Run with:
    python -m apps.backend.src.scripts.migrate_add_password_auth

This is safe to run multiple times - it checks if columns exist first.
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
    """Add password_hash and email_verified columns to users table."""
    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.connect() as conn:
        # Check existing columns
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('users')]

        changes_made = False

        # Add password_hash column
        if 'password_hash' not in columns:
            print("Adding 'password_hash' column to users table...")
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN password_hash VARCHAR NULL
            """))
            changes_made = True
            print("[OK] Added password_hash column")
        else:
            print("[OK] Column 'password_hash' already exists")

        # Add email_verified column
        if 'email_verified' not in columns:
            print("Adding 'email_verified' column to users table...")
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN email_verified BOOLEAN DEFAULT FALSE
            """))
            changes_made = True
            print("[OK] Added email_verified column")
        else:
            print("[OK] Column 'email_verified' already exists")

        if changes_made:
            conn.commit()
            print("\n[SUCCESS] Migration completed successfully!")
        else:
            print("\n[INFO] No changes needed - columns already exist")

        # Show stats
        result = conn.execute(text("SELECT COUNT(*) FROM users"))
        total = result.scalar()
        print(f"  Total users in database: {total}")

        # Show users with password_hash set (if any)
        result = conn.execute(text(
            "SELECT COUNT(*) FROM users WHERE password_hash IS NOT NULL"
        ))
        with_password = result.scalar()
        print(f"  Users with email/password auth: {with_password}")

        return True


def rollback_migration():
    """Remove password_hash and email_verified columns (for rollback if needed)."""
    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.connect() as conn:
        print("Rolling back: Removing password auth columns...")

        conn.execute(text(
            "ALTER TABLE users DROP COLUMN IF EXISTS password_hash"
        ))
        conn.execute(text(
            "ALTER TABLE users DROP COLUMN IF EXISTS email_verified"
        ))

        conn.commit()
        print("[OK] Rollback complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate users table for email/password authentication"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback the migration"
    )
    args = parser.parse_args()

    try:
        if args.rollback:
            rollback_migration()
        else:
            run_migration()
    except Exception as e:
        print(f"[FAIL] Migration failed: {e}")
        sys.exit(1)
