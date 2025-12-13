"""
Migration: Create external_alerts table for multi-source flood warnings

Creates:
- external_alerts table with indexes for efficient querying
- Supports sources: IMD, CWC, Twitter, RSS, Telegram

Run: python -m apps.backend.src.scripts.migrate_add_external_alerts
Rollback: python -m apps.backend.src.scripts.migrate_add_external_alerts --rollback
Verify: python -m apps.backend.src.scripts.migrate_add_external_alerts --verify
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from apps.backend.src.infrastructure.database import engine


def migrate():
    """Run migration to create external_alerts table."""
    print("Starting migration: external_alerts table...")

    with engine.connect() as conn:
        # Create external_alerts table
        try:
            print("Creating external_alerts table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS external_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source VARCHAR(50) NOT NULL,
                    source_id VARCHAR(255) UNIQUE,
                    source_name VARCHAR(100),
                    city VARCHAR(50) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    message TEXT NOT NULL,
                    severity VARCHAR(20),
                    url VARCHAR(2048),
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    raw_data JSONB,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """))
            conn.commit()
            print("[OK] Created external_alerts table")
        except Exception as e:
            print(f"[ERROR] Error creating external_alerts table: {e}")
            conn.rollback()
            raise

        # Create indexes
        try:
            print("Creating indexes...")

            # Index on source for filtering by source type
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_external_alerts_source
                ON external_alerts(source);
            """))

            # Index on city for filtering by city
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_external_alerts_city
                ON external_alerts(city);
            """))

            # Index on created_at for sorting by recency
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_external_alerts_created_at
                ON external_alerts(created_at DESC);
            """))

            # Composite index for city + created_at (common query pattern)
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_external_alerts_city_created
                ON external_alerts(city, created_at DESC);
            """))

            # Composite index for source + city (filtering by source within a city)
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_external_alerts_source_city
                ON external_alerts(source, city);
            """))

            conn.commit()
            print("[OK] Created all indexes")
        except Exception as e:
            print(f"[ERROR] Error creating indexes: {e}")
            conn.rollback()
            raise

    print("\n[SUCCESS] Migration completed successfully!")
    print("\nCreated table: external_alerts")
    print("  Columns:")
    print("    - id (UUID, PK)")
    print("    - source (VARCHAR, indexed)")
    print("    - source_id (VARCHAR, unique - for deduplication)")
    print("    - source_name (VARCHAR)")
    print("    - city (VARCHAR, indexed)")
    print("    - title (VARCHAR)")
    print("    - message (TEXT)")
    print("    - severity (VARCHAR)")
    print("    - url (VARCHAR)")
    print("    - latitude, longitude (DOUBLE PRECISION)")
    print("    - raw_data (JSONB)")
    print("    - expires_at (TIMESTAMP)")
    print("    - created_at (TIMESTAMP, indexed)")
    print("\n  Indexes:")
    print("    - ix_external_alerts_source")
    print("    - ix_external_alerts_city")
    print("    - ix_external_alerts_created_at")
    print("    - ix_external_alerts_city_created")
    print("    - ix_external_alerts_source_city")


def rollback():
    """Rollback migration - drop external_alerts table."""
    print("Starting rollback: external_alerts table...")

    with engine.connect() as conn:
        try:
            print("Dropping external_alerts table...")
            conn.execute(text("DROP TABLE IF EXISTS external_alerts CASCADE;"))
            conn.commit()
            print("[OK] Dropped external_alerts table")
        except Exception as e:
            print(f"[ERROR] Error dropping external_alerts table: {e}")
            conn.rollback()
            raise

    print("\n[SUCCESS] Rollback completed successfully!")


def verify():
    """Verify migration - check table and indexes exist."""
    print("Verifying migration: external_alerts table...")

    with engine.connect() as conn:
        # Check table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'external_alerts'
            );
        """))
        table_exists = result.scalar()
        print(f"[{'OK' if table_exists else 'FAIL'}] Table 'external_alerts' exists: {table_exists}")

        if not table_exists:
            print("\n[FAIL] Migration not applied. Run migration first.")
            return False

        # Check columns
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'external_alerts'
            ORDER BY ordinal_position;
        """))
        columns = result.fetchall()
        print(f"\n[OK] Found {len(columns)} columns:")
        for col in columns:
            print(f"    - {col[0]} ({col[1]}, nullable={col[2]})")

        # Check indexes
        result = conn.execute(text("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'external_alerts';
        """))
        indexes = [row[0] for row in result.fetchall()]
        print(f"\n[OK] Found {len(indexes)} indexes:")
        for idx in indexes:
            print(f"    - {idx}")

        # Test insert/select
        try:
            print("\n[TEST] Testing insert/select...")
            conn.execute(text("""
                INSERT INTO external_alerts (source, city, title, message)
                VALUES ('test', 'delhi', 'Test Alert', 'This is a test alert')
                ON CONFLICT DO NOTHING;
            """))

            result = conn.execute(text("""
                SELECT COUNT(*) FROM external_alerts WHERE source = 'test';
            """))
            count = result.scalar()
            print(f"[OK] Insert/select works. Test records: {count}")

            # Clean up test data
            conn.execute(text("DELETE FROM external_alerts WHERE source = 'test';"))
            conn.commit()
            print("[OK] Cleaned up test data")
        except Exception as e:
            print(f"[ERROR] Insert/select test failed: {e}")
            conn.rollback()
            return False

    print("\n[SUCCESS] Verification completed successfully!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate external_alerts table")
    parser.add_argument('--rollback', action='store_true', help='Rollback migration')
    parser.add_argument('--verify', action='store_true', help='Verify migration')
    args = parser.parse_args()

    if args.rollback:
        confirm = input("[WARNING] This will DROP the external_alerts table and ALL its data. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            rollback()
        else:
            print("Rollback cancelled")
    elif args.verify:
        verify()
    else:
        migrate()
