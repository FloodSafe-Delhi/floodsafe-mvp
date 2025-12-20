"""
Migration script to add role system enhancement.

Adds columns to users table:
- verified_reporter_since: Timestamp when user became verified reporter
- moderator_since: Timestamp when user became moderator

Creates role_history table for audit trail:
- Tracks all role changes with reason and who changed it

Run: python -m apps.backend.src.scripts.migrate_add_role_system
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from sqlalchemy import text
from apps.backend.src.infrastructure.database import engine


def run_migration():
    """Run the role system migration."""
    print("Starting role system migration...")

    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'users'
            AND column_name IN ('verified_reporter_since', 'moderator_since')
        """))
        existing_columns = [row[0] for row in result.fetchall()]

        # Add verified_reporter_since column if not exists
        if 'verified_reporter_since' not in existing_columns:
            print("Adding verified_reporter_since column to users table...")
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN verified_reporter_since TIMESTAMP NULL
            """))
            print("  Added verified_reporter_since column")
        else:
            print("  verified_reporter_since column already exists")

        # Add moderator_since column if not exists
        if 'moderator_since' not in existing_columns:
            print("Adding moderator_since column to users table...")
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN moderator_since TIMESTAMP NULL
            """))
            print("  Added moderator_since column")
        else:
            print("  moderator_since column already exists")

        # Check if role_history table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'role_history'
            )
        """))
        table_exists = result.scalar()

        if not table_exists:
            print("Creating role_history table...")
            conn.execute(text("""
                CREATE TABLE role_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    old_role VARCHAR(50) NOT NULL,
                    new_role VARCHAR(50) NOT NULL,
                    changed_by UUID REFERENCES users(id) ON DELETE SET NULL,
                    reason VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            print("  Created role_history table")

            # Create index for faster lookups
            conn.execute(text("""
                CREATE INDEX ix_role_history_user_id ON role_history(user_id)
            """))
            print("  Created index on role_history.user_id")

            conn.execute(text("""
                CREATE INDEX ix_role_history_created_at ON role_history(created_at)
            """))
            print("  Created index on role_history.created_at")
        else:
            print("  role_history table already exists")

        conn.commit()

    print("\nRole system migration completed successfully!")


def rollback_migration():
    """Rollback the migration (for development/testing)."""
    print("Rolling back role system migration...")

    with engine.connect() as conn:
        # Drop role_history table
        conn.execute(text("DROP TABLE IF EXISTS role_history CASCADE"))
        print("  Dropped role_history table")

        # Remove columns from users table
        conn.execute(text("""
            ALTER TABLE users
            DROP COLUMN IF EXISTS verified_reporter_since,
            DROP COLUMN IF EXISTS moderator_since
        """))
        print("  Removed role columns from users table")

        conn.commit()

    print("Rollback completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Role system migration")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    args = parser.parse_args()

    if args.rollback:
        rollback_migration()
    else:
        run_migration()
