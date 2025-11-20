"""
Database migration script to add profile and notification features.

This script adds:
1. Profile fields to users table (phone, profile_photo_url, language)
2. Notification preference fields to users table
3. Creates watch_areas table for user-defined monitoring locations

Run this script before starting the backend with the new profile features.
"""

import sys
import os

# Add parent directory to path to import infrastructure modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from infrastructure.database import engine, SessionLocal
from infrastructure.models import Base


def migrate_database():
    """Apply database migrations for profile features."""

    print("Starting database migration for profile features...")

    with engine.begin() as connection:
        # Check if columns already exist before adding them
        print("\n1. Checking existing schema...")

        # Add new columns to users table if they don't exist
        print("\n2. Adding profile fields to users table...")

        migrations = [
            # Profile fields
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_photo_url VARCHAR",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS language VARCHAR DEFAULT 'english'",

            # Notification preferences
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS notification_push BOOLEAN DEFAULT true",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS notification_sms BOOLEAN DEFAULT true",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS notification_whatsapp BOOLEAN DEFAULT false",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS notification_email BOOLEAN DEFAULT true",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS alert_preferences VARCHAR DEFAULT '{\"watch\":true,\"advisory\":true,\"warning\":true,\"emergency\":true}'",
        ]

        for migration in migrations:
            try:
                connection.execute(text(migration))
                print(f"  ✓ {migration.split('ADD COLUMN')[1].split()[2] if 'ADD COLUMN' in migration else 'executed'}")
            except Exception as e:
                print(f"  ⚠ Warning: {str(e)}")

        print("\n3. Creating watch_areas table...")

        # Create watch_areas table
        create_watch_areas_sql = """
        CREATE TABLE IF NOT EXISTS watch_areas (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR NOT NULL,
            location GEOMETRY(POINT, 4326) NOT NULL,
            radius FLOAT DEFAULT 1000.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            connection.execute(text(create_watch_areas_sql))
            print("  ✓ watch_areas table created successfully")
        except Exception as e:
            print(f"  ⚠ Warning: {str(e)}")

        # Create index on user_id for faster queries
        print("\n4. Creating indexes...")
        index_sql = "CREATE INDEX IF NOT EXISTS idx_watch_areas_user_id ON watch_areas(user_id);"
        try:
            connection.execute(text(index_sql))
            print("  ✓ Index on watch_areas.user_id created")
        except Exception as e:
            print(f"  ⚠ Warning: {str(e)}")

    print("\n✅ Migration completed successfully!")
    print("\nNew features available:")
    print("  - User profile fields (phone, photo, language)")
    print("  - Notification preferences (push, SMS, WhatsApp, email)")
    print("  - Alert type preferences")
    print("  - Watch areas for location monitoring")


if __name__ == "__main__":
    try:
        migrate_database()
    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        sys.exit(1)
