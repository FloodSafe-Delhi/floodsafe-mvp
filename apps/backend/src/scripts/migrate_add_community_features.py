"""
Migration: Create tables for community report feedback features

Creates:
- report_votes table for vote deduplication (one vote per user per report)
- comments table for community discussions on flood reports

Run: python -m apps.backend.src.scripts.migrate_add_community_features
Rollback: python -m apps.backend.src.scripts.migrate_add_community_features --rollback
Verify: python -m apps.backend.src.scripts.migrate_add_community_features --verify
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from apps.backend.src.infrastructure.database import engine


def migrate():
    """Run migration to create report_votes and comments tables."""
    print("Starting migration: community features tables...")

    with engine.connect() as conn:
        # Create report_votes table
        try:
            print("\n1. Creating report_votes table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS report_votes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
                    vote_type VARCHAR(10) NOT NULL CHECK (vote_type IN ('upvote', 'downvote')),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """))
            conn.commit()
            print("[OK] Created report_votes table")
        except Exception as e:
            print(f"[ERROR] Error creating report_votes table: {e}")
            conn.rollback()
            raise

        # Create unique index for vote deduplication
        try:
            print("   Creating unique index for vote deduplication...")
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS ix_report_votes_user_report
                ON report_votes(user_id, report_id);
            """))
            conn.commit()
            print("[OK] Created unique index ix_report_votes_user_report")
        except Exception as e:
            print(f"[ERROR] Error creating index: {e}")
            conn.rollback()
            raise

        # Create comments table
        try:
            print("\n2. Creating comments table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS comments (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    content VARCHAR(500) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """))
            conn.commit()
            print("[OK] Created comments table")
        except Exception as e:
            print(f"[ERROR] Error creating comments table: {e}")
            conn.rollback()
            raise

        # Create index for comments lookup by report
        try:
            print("   Creating index for comments lookup...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_comments_report_created
                ON comments(report_id, created_at);
            """))
            conn.commit()
            print("[OK] Created index ix_comments_report_created")
        except Exception as e:
            print(f"[ERROR] Error creating index: {e}")
            conn.rollback()
            raise

    print("\n" + "=" * 60)
    print("[SUCCESS] Migration completed successfully!")
    print("=" * 60)
    print("\nCreated tables:")
    print("\n1. report_votes")
    print("   Columns:")
    print("     - id (UUID, PK)")
    print("     - user_id (UUID, FK -> users)")
    print("     - report_id (UUID, FK -> reports)")
    print("     - vote_type (VARCHAR, 'upvote' or 'downvote')")
    print("     - created_at (TIMESTAMP)")
    print("   Indexes:")
    print("     - ix_report_votes_user_report (UNIQUE - prevents duplicate votes)")
    print("\n2. comments")
    print("   Columns:")
    print("     - id (UUID, PK)")
    print("     - report_id (UUID, FK -> reports)")
    print("     - user_id (UUID, FK -> users)")
    print("     - content (VARCHAR(500))")
    print("     - created_at (TIMESTAMP)")
    print("   Indexes:")
    print("     - ix_comments_report_created (for efficient comment lookup)")


def rollback():
    """Rollback migration - drop report_votes and comments tables."""
    print("Starting rollback: community features tables...")

    with engine.connect() as conn:
        try:
            print("Dropping comments table...")
            conn.execute(text("DROP TABLE IF EXISTS comments CASCADE;"))
            conn.commit()
            print("[OK] Dropped comments table")
        except Exception as e:
            print(f"[ERROR] Error dropping comments table: {e}")
            conn.rollback()
            raise

        try:
            print("Dropping report_votes table...")
            conn.execute(text("DROP TABLE IF EXISTS report_votes CASCADE;"))
            conn.commit()
            print("[OK] Dropped report_votes table")
        except Exception as e:
            print(f"[ERROR] Error dropping report_votes table: {e}")
            conn.rollback()
            raise

    print("\n[SUCCESS] Rollback completed successfully!")


def verify():
    """Verify migration - check tables and indexes exist."""
    print("Verifying migration: community features tables...")

    with engine.connect() as conn:
        # Check report_votes table
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'report_votes'
            );
        """))
        votes_exists = result.scalar()
        print(f"\n[{'OK' if votes_exists else 'FAIL'}] Table 'report_votes' exists: {votes_exists}")

        # Check comments table
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'comments'
            );
        """))
        comments_exists = result.scalar()
        print(f"[{'OK' if comments_exists else 'FAIL'}] Table 'comments' exists: {comments_exists}")

        if not votes_exists or not comments_exists:
            print("\n[FAIL] Migration not applied. Run migration first.")
            return False

        # Check report_votes columns
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'report_votes'
            ORDER BY ordinal_position;
        """))
        columns = result.fetchall()
        print(f"\n[OK] report_votes has {len(columns)} columns:")
        for col in columns:
            print(f"    - {col[0]} ({col[1]})")

        # Check comments columns
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'comments'
            ORDER BY ordinal_position;
        """))
        columns = result.fetchall()
        print(f"\n[OK] comments has {len(columns)} columns:")
        for col in columns:
            print(f"    - {col[0]} ({col[1]})")

        # Check indexes
        result = conn.execute(text("""
            SELECT tablename, indexname FROM pg_indexes
            WHERE tablename IN ('report_votes', 'comments');
        """))
        indexes = result.fetchall()
        print(f"\n[OK] Found {len(indexes)} indexes:")
        for idx in indexes:
            print(f"    - {idx[0]}.{idx[1]}")

        # Test unique constraint on report_votes
        try:
            print("\n[TEST] Testing unique vote constraint...")
            # This should only insert once due to ON CONFLICT
            conn.execute(text("""
                INSERT INTO report_votes (user_id, report_id, vote_type)
                SELECT u.id, r.id, 'upvote'
                FROM users u, reports r
                LIMIT 1
                ON CONFLICT (user_id, report_id) DO NOTHING;
            """))
            conn.rollback()  # Don't actually insert
            print("[OK] Unique constraint works")
        except Exception as e:
            print(f"[WARN] Could not test constraint (no users/reports?): {e}")
            conn.rollback()

    print("\n[SUCCESS] Verification completed successfully!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate community features tables")
    parser.add_argument('--rollback', action='store_true', help='Rollback migration')
    parser.add_argument('--verify', action='store_true', help='Verify migration')
    args = parser.parse_args()

    if args.rollback:
        confirm = input("[WARNING] This will DROP report_votes and comments tables and ALL their data. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            rollback()
        else:
            print("Rollback cancelled")
    elif args.verify:
        verify()
    else:
        migrate()
