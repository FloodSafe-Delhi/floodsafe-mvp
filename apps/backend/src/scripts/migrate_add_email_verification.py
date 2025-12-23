"""
Migration: Create email_verification_tokens table for email verification

Creates:
- email_verification_tokens table for storing verification tokens

Run: python -m apps.backend.src.scripts.migrate_add_email_verification
Rollback: python -m apps.backend.src.scripts.migrate_add_email_verification --rollback
Verify: python -m apps.backend.src.scripts.migrate_add_email_verification --verify
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from apps.backend.src.infrastructure.database import engine


def migrate():
    """Run migration to create email_verification_tokens table."""
    print("Starting migration: email_verification_tokens table...")

    with engine.connect() as conn:
        # Create email_verification_tokens table
        try:
            print("\n1. Creating email_verification_tokens table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS email_verification_tokens (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash VARCHAR NOT NULL UNIQUE,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    used_at TIMESTAMP DEFAULT NULL
                );
            """))
            conn.commit()
            print("[OK] Created email_verification_tokens table")
        except Exception as e:
            print(f"[ERROR] Error creating email_verification_tokens table: {e}")
            conn.rollback()
            raise

        # Create index for token lookup
        try:
            print("   Creating index for token lookup...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_email_verification_tokens_token_hash
                ON email_verification_tokens(token_hash);
            """))
            conn.commit()
            print("[OK] Created index ix_email_verification_tokens_token_hash")
        except Exception as e:
            print(f"[ERROR] Error creating index: {e}")
            conn.rollback()
            raise

        # Create index for user_id lookup (for resend rate limiting)
        try:
            print("   Creating index for user token lookup...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_email_verification_tokens_user_id
                ON email_verification_tokens(user_id, created_at);
            """))
            conn.commit()
            print("[OK] Created index ix_email_verification_tokens_user_id")
        except Exception as e:
            print(f"[ERROR] Error creating index: {e}")
            conn.rollback()
            raise

    print("\n" + "=" * 60)
    print("[SUCCESS] Migration completed successfully!")
    print("=" * 60)
    print("\nCreated table: email_verification_tokens")
    print("   Columns:")
    print("     - id (UUID, PK)")
    print("     - user_id (UUID, FK -> users, CASCADE DELETE)")
    print("     - token_hash (VARCHAR, UNIQUE) - hashed verification token")
    print("     - expires_at (TIMESTAMP) - 24hr expiry")
    print("     - created_at (TIMESTAMP)")
    print("     - used_at (TIMESTAMP, nullable) - NULL = unused")
    print("   Indexes:")
    print("     - ix_email_verification_tokens_token_hash (for token lookup)")
    print("     - ix_email_verification_tokens_user_id (for resend rate limiting)")


def rollback():
    """Rollback migration - drop email_verification_tokens table."""
    print("Starting rollback: email_verification_tokens table...")

    with engine.connect() as conn:
        try:
            print("Dropping email_verification_tokens table...")
            conn.execute(text("DROP TABLE IF EXISTS email_verification_tokens CASCADE;"))
            conn.commit()
            print("[OK] Dropped email_verification_tokens table")
        except Exception as e:
            print(f"[ERROR] Error dropping email_verification_tokens table: {e}")
            conn.rollback()
            raise

    print("\n[SUCCESS] Rollback completed successfully!")


def verify():
    """Verify migration - check table and indexes exist."""
    print("Verifying migration: email_verification_tokens table...")

    with engine.connect() as conn:
        # Check table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'email_verification_tokens'
            );
        """))
        table_exists = result.scalar()
        print(f"\n[{'OK' if table_exists else 'FAIL'}] Table 'email_verification_tokens' exists: {table_exists}")

        if not table_exists:
            print("\n[FAIL] Migration not applied. Run migration first.")
            return False

        # Check columns
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'email_verification_tokens'
            ORDER BY ordinal_position;
        """))
        columns = result.fetchall()
        print(f"\n[OK] email_verification_tokens has {len(columns)} columns:")
        for col in columns:
            print(f"    - {col[0]} ({col[1]})")

        # Check indexes
        result = conn.execute(text("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'email_verification_tokens';
        """))
        indexes = result.fetchall()
        print(f"\n[OK] Found {len(indexes)} indexes:")
        for idx in indexes:
            print(f"    - {idx[0]}")

    print("\n[SUCCESS] Verification completed successfully!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate email verification tokens table")
    parser.add_argument('--rollback', action='store_true', help='Rollback migration')
    parser.add_argument('--verify', action='store_true', help='Verify migration')
    args = parser.parse_args()

    if args.rollback:
        confirm = input("[WARNING] This will DROP email_verification_tokens table and ALL its data. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            rollback()
        else:
            print("Rollback cancelled")
    elif args.verify:
        verify()
    else:
        migrate()
