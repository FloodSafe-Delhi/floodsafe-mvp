"""
Seed script for initial badges
Creates all the default badges for the reputation system
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.infrastructure.database import SessionLocal
from src.infrastructure import models
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Badge definitions
INITIAL_BADGES = [
    # ========================================================================
    # MILESTONE BADGES (Progress-based)
    # ========================================================================
    {
        'key': 'first_report',
        'name': 'First Report',
        'description': 'Submit your first flood report',
        'icon': '⭐',
        'category': 'milestone',
        'requirement_type': 'reports_count',
        'requirement_value': 1,
        'points_reward': 5,
        'sort_order': 1
    },
    {
        'key': 'reporter',
        'name': 'Reporter',
        'description': 'Submit 5 flood reports',
        'icon': '📝',
        'category': 'milestone',
        'requirement_type': 'reports_count',
        'requirement_value': 5,
        'points_reward': 10,
        'sort_order': 2
    },
    {
        'key': 'guardian',
        'name': 'Guardian',
        'description': 'Get 10 reports verified',
        'icon': '🛡️',
        'category': 'milestone',
        'requirement_type': 'verified_count',
        'requirement_value': 10,
        'points_reward': 25,
        'sort_order': 3
    },
    {
        'key': 'hero',
        'name': 'Hero',
        'description': 'Get 25 reports verified',
        'icon': '🦸',
        'category': 'milestone',
        'requirement_type': 'verified_count',
        'requirement_value': 25,
        'points_reward': 50,
        'sort_order': 4
    },
    {
        'key': 'legend',
        'name': 'Legend',
        'description': 'Get 50 reports verified',
        'icon': '👑',
        'category': 'milestone',
        'requirement_type': 'verified_count',
        'requirement_value': 50,
        'points_reward': 100,
        'sort_order': 5
    },

    # ========================================================================
    # ACHIEVEMENT BADGES (Streak-based)
    # ========================================================================
    {
        'key': 'streak_7',
        'name': 'Dedicated',
        'description': 'Maintain a 7 day activity streak',
        'icon': '🔥',
        'category': 'achievement',
        'requirement_type': 'streak_days',
        'requirement_value': 7,
        'points_reward': 20,
        'sort_order': 6
    },
    {
        'key': 'streak_30',
        'name': 'Committed',
        'description': 'Maintain a 30 day activity streak',
        'icon': '⚡',
        'category': 'achievement',
        'requirement_type': 'streak_days',
        'requirement_value': 30,
        'points_reward': 100,
        'sort_order': 7
    },

    # ========================================================================
    # LEVEL BADGES (Level-based)
    # ========================================================================
    {
        'key': 'level_5',
        'name': 'Rising Star',
        'description': 'Reach level 5',
        'icon': '🌟',
        'category': 'achievement',
        'requirement_type': 'level',
        'requirement_value': 5,
        'points_reward': 15,
        'sort_order': 8
    },
    {
        'key': 'level_10',
        'name': 'Expert',
        'description': 'Reach level 10',
        'icon': '💎',
        'category': 'achievement',
        'requirement_type': 'level',
        'requirement_value': 10,
        'points_reward': 50,
        'sort_order': 9
    },
    {
        'key': 'level_25',
        'name': 'Master',
        'description': 'Reach level 25',
        'icon': '🏆',
        'category': 'achievement',
        'requirement_type': 'level',
        'requirement_value': 25,
        'points_reward': 150,
        'sort_order': 10
    },

    # ========================================================================
    # SPECIAL BADGES (Points-based)
    # ========================================================================
    {
        'key': 'points_1000',
        'name': 'High Achiever',
        'description': 'Earn 1000 points',
        'icon': '🎯',
        'category': 'achievement',
        'requirement_type': 'points',
        'requirement_value': 1000,
        'points_reward': 50,
        'sort_order': 11
    },
    {
        'key': 'points_5000',
        'name': 'Champion',
        'description': 'Earn 5000 points',
        'icon': '🥇',
        'category': 'achievement',
        'requirement_type': 'points',
        'requirement_value': 5000,
        'points_reward': 200,
        'sort_order': 12
    },
]


def seed_badges():
    """
    Seed the database with initial badges
    """
    db = SessionLocal()

    try:
        logger.info("Starting badge seeding...")

        created_count = 0
        updated_count = 0
        skipped_count = 0

        for badge_data in INITIAL_BADGES:
            # Check if badge already exists
            existing_badge = db.query(models.Badge).filter(
                models.Badge.key == badge_data['key']
            ).first()

            if existing_badge:
                # Update existing badge
                for key, value in badge_data.items():
                    setattr(existing_badge, key, value)
                updated_count += 1
                logger.info(f"✓ Updated badge: {badge_data['name']}")
            else:
                # Create new badge
                new_badge = models.Badge(**badge_data)
                db.add(new_badge)
                created_count += 1
                logger.info(f"✓ Created badge: {badge_data['name']}")

        db.commit()

        logger.info(f"""
{'='*60}
Badge Seeding Complete!
{'='*60}
Created: {created_count} badges
Updated: {updated_count} badges
Total Badges: {len(INITIAL_BADGES)}
{'='*60}
""")

        return {
            'status': 'success',
            'created': created_count,
            'updated': updated_count,
            'total': len(INITIAL_BADGES)
        }

    except Exception as e:
        logger.error(f"❌ Badge seeding failed: {e}")
        db.rollback()
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        db.close()


if __name__ == "__main__":
    result = seed_badges()
    if result['status'] == 'error':
        print(f"\n❌ Error: {result['message']}\n")
        sys.exit(1)
