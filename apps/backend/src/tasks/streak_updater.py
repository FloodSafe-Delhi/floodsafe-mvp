"""
Daily streak updater task
Resets streaks for users who haven't been active
"""
from datetime import datetime, timedelta
import logging

from ..infrastructure.database import SessionLocal
from ..infrastructure import models

logger = logging.getLogger(__name__)


def reset_inactive_streaks():
    """
    Check all users for broken streaks
    Runs once per day (should be called via cron or scheduler)
    """
    db = SessionLocal()
    try:
        yesterday = datetime.utcnow().date() - timedelta(days=1)
        two_days_ago = yesterday - timedelta(days=1)

        # Find users whose last activity was 2+ days ago (streak broken)
        users = db.query(models.User).filter(
            models.User.last_activity_date < two_days_ago,
            models.User.streak_days > 0
        ).all()

        reset_count = 0
        for user in users:
            logger.info(f"Resetting streak for user {user.id} (was {user.streak_days} days)")
            user.streak_days = 0
            reset_count += 1

        db.commit()
        logger.info(f"Reset streaks for {reset_count} inactive users")

        return {
            'status': 'success',
            'users_reset': reset_count
        }

    except Exception as e:
        logger.error(f"Error resetting streaks: {e}")
        db.rollback()
        return {
            'status': 'error',
            'error': str(e)
        }
    finally:
        db.close()


def run_daily_tasks():
    """
    Run all daily maintenance tasks
    """
    logger.info("Starting daily streak maintenance...")
    result = reset_inactive_streaks()
    logger.info(f"Daily tasks completed: {result}")
    return result


if __name__ == "__main__":
    # Can be run manually for testing
    logging.basicConfig(level=logging.INFO)
    run_daily_tasks()
