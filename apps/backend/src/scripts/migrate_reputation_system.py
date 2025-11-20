"""
Migration script for Reputation System
Adds new tables and columns for the reputation/gamification system
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy import text
from src.infrastructure.database import engine, SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """
    Run the migration to add reputation system tables and columns
    """
    db = SessionLocal()

    try:
        logger.info("Starting reputation system migration...")

        # ========================================================================
        # 1. ADD NEW COLUMNS TO USERS TABLE
        # ========================================================================
        logger.info("Adding new columns to users table...")

        user_columns = [
            # Reputation system
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS reputation_score INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS streak_days INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_activity_date TIMESTAMP",

            # Privacy controls
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS leaderboard_visible BOOLEAN DEFAULT true",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_public BOOLEAN DEFAULT true",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR",
        ]

        for sql in user_columns:
            try:
                db.execute(text(sql))
                logger.info(f"✓ {sql.split('ADD COLUMN IF NOT EXISTS')[1].split(' ')[1]}")
            except Exception as e:
                logger.warning(f"Column may already exist: {e}")

        db.commit()

        # ========================================================================
        # 2. ADD NEW COLUMNS TO REPORTS TABLE
        # ========================================================================
        logger.info("Adding new columns to reports table...")

        report_columns = [
            "ALTER TABLE reports ADD COLUMN IF NOT EXISTS downvotes INTEGER DEFAULT 0",
            "ALTER TABLE reports ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0",
            "ALTER TABLE reports ADD COLUMN IF NOT EXISTS verified_at TIMESTAMP",
        ]

        for sql in report_columns:
            try:
                db.execute(text(sql))
                logger.info(f"✓ {sql.split('ADD COLUMN IF NOT EXISTS')[1].split(' ')[1]}")
            except Exception as e:
                logger.warning(f"Column may already exist: {e}")

        db.commit()

        # ========================================================================
        # 3. CREATE REPUTATION_HISTORY TABLE
        # ========================================================================
        logger.info("Creating reputation_history table...")

        create_reputation_history = text("""
        CREATE TABLE IF NOT EXISTS reputation_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            action VARCHAR NOT NULL,
            points_change INTEGER DEFAULT 0,
            new_total INTEGER NOT NULL,
            reason VARCHAR,
            metadata VARCHAR DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        db.execute(create_reputation_history)
        db.commit()
        logger.info("✓ reputation_history table created")

        # Create indexes
        try:
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_reputation_history_user "
                "ON reputation_history(user_id)"
            ))
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_reputation_history_created "
                "ON reputation_history(created_at DESC)"
            ))
            logger.info("✓ reputation_history indexes created")
        except Exception as e:
            logger.warning(f"Indexes may already exist: {e}")

        db.commit()

        # ========================================================================
        # 4. CREATE BADGES TABLE
        # ========================================================================
        logger.info("Creating badges table...")

        create_badges = text("""
        CREATE TABLE IF NOT EXISTS badges (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key VARCHAR UNIQUE NOT NULL,
            name VARCHAR NOT NULL,
            description VARCHAR,
            icon VARCHAR DEFAULT '🏆',
            category VARCHAR DEFAULT 'achievement',
            requirement_type VARCHAR NOT NULL,
            requirement_value INTEGER NOT NULL,
            points_reward INTEGER DEFAULT 0,
            sort_order INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        db.execute(create_badges)
        db.commit()
        logger.info("✓ badges table created")

        # Create indexes
        try:
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_badges_key "
                "ON badges(key)"
            ))
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_badges_active "
                "ON badges(is_active)"
            ))
            logger.info("✓ badges indexes created")
        except Exception as e:
            logger.warning(f"Indexes may already exist: {e}")

        db.commit()

        # ========================================================================
        # 5. CREATE USER_BADGES TABLE
        # ========================================================================
        logger.info("Creating user_badges table...")

        create_user_badges = text("""
        CREATE TABLE IF NOT EXISTS user_badges (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            badge_id UUID NOT NULL REFERENCES badges(id) ON DELETE CASCADE,
            earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, badge_id)
        )
        """)

        db.execute(create_user_badges)
        db.commit()
        logger.info("✓ user_badges table created")

        # Create indexes
        try:
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_user_badges_user "
                "ON user_badges(user_id)"
            ))
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_user_badges_earned "
                "ON user_badges(earned_at DESC)"
            ))
            logger.info("✓ user_badges indexes created")
        except Exception as e:
            logger.warning(f"Indexes may already exist: {e}")

        db.commit()

        logger.info("✅ Migration completed successfully!")

        return {
            'status': 'success',
            'message': 'Reputation system migration completed'
        }

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        db.rollback()
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        db.close()


if __name__ == "__main__":
    result = run_migration()
    print(f"\n{'='*60}")
    print(f"Migration Result: {result['status'].upper()}")
    print(f"Message: {result['message']}")
    print(f"{'='*60}\n")
