"""
Migration: Add onboarding fields to users table and create daily_routes table

Adds:
- users.city_preference (VARCHAR)
- users.profile_complete (BOOLEAN)
- users.onboarding_step (INTEGER)
- daily_routes table (full schema)

Run: python -m apps.backend.src.scripts.migrate_add_onboarding_fields
Rollback: python -m apps.backend.src.scripts.migrate_add_onboarding_fields --rollback
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from apps.backend.src.infrastructure.database import engine


def migrate():
    """Run migration to add onboarding fields and daily_routes table."""
    print("Starting migration...")

    with engine.connect() as conn:
        # Add new columns to users table
        try:
            print("Adding onboarding fields to users table...")
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS city_preference VARCHAR(50),
                ADD COLUMN IF NOT EXISTS profile_complete BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS onboarding_step INTEGER;
            """))
            conn.commit()
            print("[OK] Added onboarding fields to users table")
        except Exception as e:
            print(f"[ERROR] Error adding user fields: {e}")
            conn.rollback()
            raise

        # Create daily_routes table
        try:
            print("Creating daily_routes table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_routes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name VARCHAR(100) NOT NULL,
                    origin_latitude DOUBLE PRECISION NOT NULL,
                    origin_longitude DOUBLE PRECISION NOT NULL,
                    destination_latitude DOUBLE PRECISION NOT NULL,
                    destination_longitude DOUBLE PRECISION NOT NULL,
                    transport_mode VARCHAR(20) DEFAULT 'driving',
                    notify_on_flood BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_daily_routes_user_id ON daily_routes(user_id);
            """))
            conn.commit()
            print("[OK] Created daily_routes table with index")
        except Exception as e:
            print(f"[ERROR] Error creating daily_routes table: {e}")
            conn.rollback()
            raise

    print("\n[SUCCESS] Migration completed successfully!")
    print("\nAdded fields:")
    print("  - users.city_preference (VARCHAR)")
    print("  - users.profile_complete (BOOLEAN, default FALSE)")
    print("  - users.onboarding_step (INTEGER, nullable)")
    print("\nCreated table:")
    print("  - daily_routes (with user_id foreign key)")


def rollback():
    """Rollback migration - remove onboarding fields and daily_routes table."""
    print("Starting rollback...")

    with engine.connect() as conn:
        try:
            print("Dropping daily_routes table...")
            conn.execute(text("DROP TABLE IF EXISTS daily_routes CASCADE;"))
            conn.commit()
            print("[OK] Dropped daily_routes table")
        except Exception as e:
            print(f"[ERROR] Error dropping daily_routes table: {e}")
            conn.rollback()
            raise

        try:
            print("Removing onboarding fields from users table...")
            conn.execute(text("""
                ALTER TABLE users
                DROP COLUMN IF EXISTS city_preference,
                DROP COLUMN IF EXISTS profile_complete,
                DROP COLUMN IF EXISTS onboarding_step;
            """))
            conn.commit()
            print("[OK] Removed onboarding fields from users table")
        except Exception as e:
            print(f"✗ Rollback error: {e}")
            conn.rollback()
            raise

    print("\n[SUCCESS] Rollback completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate onboarding fields")
    parser.add_argument('--rollback', action='store_true', help='Rollback migration')
    args = parser.parse_args()

    if args.rollback:
        confirm = input("[WARNING] This will remove onboarding fields and daily_routes table. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            rollback()
        else:
            print("Rollback cancelled")
    else:
        migrate()
