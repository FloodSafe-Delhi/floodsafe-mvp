"""
Migration: Create whatsapp_sessions table for WhatsApp conversation state

Creates:
- whatsapp_sessions table for tracking multi-step WhatsApp conversations

Run (from project root): python -m apps.backend.src.scripts.migrate_add_whatsapp_sessions
Run (in Docker): docker-compose exec backend python -m src.scripts.migrate_add_whatsapp_sessions
Rollback: Add --rollback flag
Verify: Add --verify flag
"""
import sys
from pathlib import Path

# Add project root to path for local development
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text

# Try importing from Docker path first, then local path
try:
    from src.infrastructure.database import engine
except ImportError:
    from apps.backend.src.infrastructure.database import engine


def migrate():
    """Run migration to create whatsapp_sessions table."""
    print("Starting migration: whatsapp_sessions table...")

    with engine.connect() as conn:
        # Create whatsapp_sessions table
        try:
            print("\n1. Creating whatsapp_sessions table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS whatsapp_sessions (
                    phone VARCHAR(20) PRIMARY KEY,
                    state VARCHAR(50) DEFAULT 'idle',
                    data JSONB DEFAULT '{}',
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """))
            conn.commit()
            print("[OK] Created whatsapp_sessions table")
        except Exception as e:
            print(f"[ERROR] Error creating whatsapp_sessions table: {e}")
            conn.rollback()
            raise

        # Create index for user_id lookup (find sessions by user)
        try:
            print("   Creating index for user_id lookup...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_whatsapp_sessions_user_id
                ON whatsapp_sessions(user_id);
            """))
            conn.commit()
            print("[OK] Created index ix_whatsapp_sessions_user_id")
        except Exception as e:
            print(f"[ERROR] Error creating index: {e}")
            conn.rollback()
            raise

        # Create index for updated_at (cleanup stale sessions)
        try:
            print("   Creating index for session cleanup...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_whatsapp_sessions_updated_at
                ON whatsapp_sessions(updated_at);
            """))
            conn.commit()
            print("[OK] Created index ix_whatsapp_sessions_updated_at")
        except Exception as e:
            print(f"[ERROR] Error creating index: {e}")
            conn.rollback()
            raise

    print("\n" + "=" * 60)
    print("[SUCCESS] Migration completed successfully!")
    print("=" * 60)
    print("\nCreated table: whatsapp_sessions")
    print("   Purpose: Track WhatsApp conversation state for multi-step flows")
    print("   Columns:")
    print("     - phone (VARCHAR(20), PK) - E.164 format: +919876543210")
    print("     - state (VARCHAR(50)) - Conversation state (idle, awaiting_choice, etc.)")
    print("     - data (JSONB) - Arbitrary session data (email attempts, temp values)")
    print("     - user_id (UUID, FK -> users, SET NULL) - Linked FloodSafe account")
    print("     - created_at (TIMESTAMP)")
    print("     - updated_at (TIMESTAMP)")
    print("   Indexes:")
    print("     - ix_whatsapp_sessions_user_id (for user session lookup)")
    print("     - ix_whatsapp_sessions_updated_at (for cleanup of stale sessions)")
    print("\n   States:")
    print("     - idle: Ready for new command")
    print("     - awaiting_choice: Asked user '1. Create account' or '2. Submit anonymously'")
    print("     - awaiting_email: User chose to create account, waiting for email")
    print("     - sos_active: User in active SOS flow")


def rollback():
    """Rollback migration - drop whatsapp_sessions table."""
    print("Starting rollback: whatsapp_sessions table...")

    with engine.connect() as conn:
        try:
            print("Dropping whatsapp_sessions table...")
            conn.execute(text("DROP TABLE IF EXISTS whatsapp_sessions CASCADE;"))
            conn.commit()
            print("[OK] Dropped whatsapp_sessions table")
        except Exception as e:
            print(f"[ERROR] Error dropping whatsapp_sessions table: {e}")
            conn.rollback()
            raise

    print("\n[SUCCESS] Rollback completed successfully!")


def verify():
    """Verify migration - check table and indexes exist."""
    print("Verifying migration: whatsapp_sessions table...")

    with engine.connect() as conn:
        # Check table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'whatsapp_sessions'
            );
        """))
        table_exists = result.scalar()
        print(f"\n[{'OK' if table_exists else 'FAIL'}] Table 'whatsapp_sessions' exists: {table_exists}")

        if not table_exists:
            print("\n[FAIL] Migration not applied. Run migration first.")
            return False

        # Check columns
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'whatsapp_sessions'
            ORDER BY ordinal_position;
        """))
        columns = result.fetchall()
        print(f"\n[OK] whatsapp_sessions has {len(columns)} columns:")
        for col in columns:
            print(f"    - {col[0]} ({col[1]})")

        # Check indexes
        result = conn.execute(text("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'whatsapp_sessions';
        """))
        indexes = result.fetchall()
        print(f"\n[OK] Found {len(indexes)} indexes:")
        for idx in indexes:
            print(f"    - {idx[0]}")

    print("\n[SUCCESS] Verification completed successfully!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate whatsapp_sessions table")
    parser.add_argument('--rollback', action='store_true', help='Rollback migration')
    parser.add_argument('--verify', action='store_true', help='Verify migration')
    args = parser.parse_args()

    if args.rollback:
        confirm = input("[WARNING] This will DROP whatsapp_sessions table and ALL its data. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            rollback()
        else:
            print("Rollback cancelled")
    elif args.verify:
        verify()
    else:
        migrate()
