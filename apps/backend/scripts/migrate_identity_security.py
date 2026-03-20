"""
Identity + Security Hardening Migration
=========================================
Applies schema changes required for WhatsApp-native login and account lifecycle:

  1. Make users.email nullable (phone-only users have no email)
  2. Fix watch_areas FK cascade (pre-existing gap: user_id had no ON DELETE CASCADE)
  3. Create whatsapp_login_challenges table (OTP challenge lifecycle)
  4. Create indexes on whatsapp_login_challenges for fast lookup

Usage:
  # Direct DB connection (local or Supabase via DATABASE_URL)
  DATABASE_URL=postgresql://user:password@localhost:5432/floodsafe python scripts/migrate_identity_security.py

  # Supabase Management API (if host is IPv6-only)
  DATABASE_URL=postgresql://... python scripts/migrate_identity_security.py supabase

  # Verify only (no schema changes)
  DATABASE_URL=... python scripts/migrate_identity_security.py --verify

  # Rollback
  DATABASE_URL=... python scripts/migrate_identity_security.py --rollback

Notes:
  - All steps run in a single transaction via engine.begin()
  - Rollback documentation: re-add NOT NULL to users.email manually if needed
    (only safe after confirming all phone-only users have been back-filled)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── SQL Definitions ──────────────────────────────────────────────────────────

MIGRATION_STEPS = [
    (
        "Make users.email nullable (supports phone-only accounts)",
        "ALTER TABLE users ALTER COLUMN email DROP NOT NULL;",
    ),
    (
        "Drop existing watch_areas user_id FK (no cascade)",
        "ALTER TABLE watch_areas DROP CONSTRAINT IF EXISTS watch_areas_user_id_fkey;",
    ),
    (
        "Re-add watch_areas user_id FK with ON DELETE CASCADE",
        """ALTER TABLE watch_areas ADD CONSTRAINT watch_areas_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;""",
    ),
    (
        "Create whatsapp_login_challenges table",
        """CREATE TABLE IF NOT EXISTS whatsapp_login_challenges (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    phone       VARCHAR(20)  NOT NULL,
    code        VARCHAR(10)  NOT NULL,
    session_id  UUID         NOT NULL UNIQUE,
    expires_at  TIMESTAMP    NOT NULL,
    verified    BOOLEAN      DEFAULT FALSE,
    verified_at TIMESTAMP,
    user_id     UUID,
    tokens_issued BOOLEAN    DEFAULT FALSE,
    created_at  TIMESTAMP    DEFAULT NOW()
);""",
    ),
    (
        "Create index on (phone, code) for unverified challenges",
        "CREATE INDEX IF NOT EXISTS idx_wlc_phone_code ON whatsapp_login_challenges(phone, code) WHERE verified = FALSE;",
    ),
    (
        "Create index on session_id for unverified challenges",
        "CREATE INDEX IF NOT EXISTS idx_wlc_session ON whatsapp_login_challenges(session_id) WHERE verified = FALSE;",
    ),
]

ROLLBACK_STEPS = [
    (
        "Drop whatsapp_login_challenges table",
        "DROP TABLE IF EXISTS whatsapp_login_challenges CASCADE;",
    ),
    (
        "Drop watch_areas cascade FK",
        "ALTER TABLE watch_areas DROP CONSTRAINT IF EXISTS watch_areas_user_id_fkey;",
    ),
    (
        "Restore watch_areas FK without cascade",
        """ALTER TABLE watch_areas ADD CONSTRAINT watch_areas_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(id);""",
    ),
    # NOTE: Restoring NOT NULL on users.email is intentionally omitted.
    # Only do this manually after confirming no phone-only users exist (email IS NULL = 0).
]

# ─── Verification Queries ─────────────────────────────────────────────────────

VERIFY_CHECKS = [
    (
        "users.email is nullable",
        """SELECT is_nullable FROM information_schema.columns
           WHERE table_schema = 'public' AND table_name = 'users' AND column_name = 'email';""",
        lambda rows: rows and rows[0][0] == "YES",
    ),
    (
        "users.username is NOT nullable (unchanged)",
        """SELECT is_nullable FROM information_schema.columns
           WHERE table_schema = 'public' AND table_name = 'users' AND column_name = 'username';""",
        lambda rows: rows and rows[0][0] == "NO",
    ),
    (
        "whatsapp_login_challenges table exists",
        """SELECT COUNT(*) FROM information_schema.tables
           WHERE table_schema = 'public' AND table_name = 'whatsapp_login_challenges';""",
        lambda rows: rows and int(rows[0][0]) == 1,
    ),
    (
        "No existing users lost their email (email IS NULL = 0)",
        "SELECT COUNT(*) FROM users WHERE email IS NULL;",
        lambda rows: rows and int(rows[0][0]) == 0,
    ),
]


# ─── Runner ───────────────────────────────────────────────────────────────────

def get_engine():
    """Build SQLAlchemy engine from DATABASE_URL environment variable."""
    from sqlalchemy import create_engine

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable is not set.")
        print(
            "  Example: DATABASE_URL=postgresql://user:password@localhost:5432/floodsafe"
            " python scripts/migrate_identity_security.py"
        )
        sys.exit(1)

    # Detect remote Supabase host and add SSL
    connect_args = {}
    host = database_url.split("@")[1].split(":")[0] if "@" in database_url else ""
    if "localhost" not in host and "127.0.0.1" not in host and host != "db":
        connect_args = {
            "sslmode": "require",
            "options": "-c search_path=public",
        }

    return create_engine(database_url, connect_args=connect_args)


def run_migration(engine):
    """Execute all migration steps in a single transaction."""
    from sqlalchemy import text

    print("Running migration steps in a single transaction...")
    with engine.begin() as conn:
        for i, (description, sql) in enumerate(MIGRATION_STEPS, start=1):
            print(f"\n  [{i}/{len(MIGRATION_STEPS)}] {description}")
            conn.execute(text(sql))
            print(f"       OK")
    print("\nAll migration steps committed.")


def run_rollback(engine):
    """Rollback schema changes in a single transaction."""
    from sqlalchemy import text

    print("Rolling back identity security migration...")
    with engine.begin() as conn:
        for i, (description, sql) in enumerate(ROLLBACK_STEPS, start=1):
            print(f"\n  [{i}/{len(ROLLBACK_STEPS)}] {description}")
            conn.execute(text(sql))
            print(f"       OK")
    print("\nRollback committed.")
    print(
        "NOTE: users.email NOT NULL was NOT restored. "
        "Restore manually only after confirming no phone-only users exist."
    )


def run_verify(engine):
    """Run all verification checks and print PASS/FAIL for each."""
    from sqlalchemy import text

    print("\nRunning verification checks...")
    all_passed = True

    with engine.connect() as conn:
        for label, sql, check_fn in VERIFY_CHECKS:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            passed = check_fn(rows)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {label}")
            if not passed:
                all_passed = False
                print(f"         Got rows: {rows}")

    if all_passed:
        print("\nAll checks passed.")
    else:
        print("\nOne or more checks FAILED. Review output above.")
        sys.exit(1)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    rollback = "--rollback" in args or "rollback" in args
    verify_only = "--verify" in args or "verify" in args

    engine = get_engine()

    if rollback:
        run_rollback(engine)
        run_verify(engine)
        return

    if verify_only:
        run_verify(engine)
        return

    run_migration(engine)
    run_verify(engine)


if __name__ == "__main__":
    main()
