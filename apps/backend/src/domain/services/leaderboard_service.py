from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from uuid import UUID
from datetime import datetime, timedelta
import json
import logging

from ...infrastructure import models

logger = logging.getLogger(__name__)


class LeaderboardService:
    """
    Service for managing leaderboards with privacy controls
    """

    def __init__(self, db: Session):
        self.db = db

    def get_leaderboard_data(self, user: models.User) -> Optional[Dict]:
        """
        Return appropriate data based on privacy settings
        Returns None if user opted out of leaderboards
        """
        if not user.leaderboard_visible:
            return None  # Don't include in leaderboard

        # Determine what name to show
        if user.display_name:
            # User chose anonymous name
            display_name = user.display_name
            show_profile_photo = False
        elif user.profile_public:
            # Show real info
            display_name = user.username
            show_profile_photo = True
        else:
            # Auto-generate anonymous name
            display_name = f"User_{str(user.id)[:8]}"
            show_profile_photo = False

        # Parse badges
        try:
            badges_list = json.loads(user.badges or '[]')
        except json.JSONDecodeError:
            badges_list = []

        return {
            'display_name': display_name,
            'profile_photo_url': user.profile_photo_url if show_profile_photo else None,
            'points': user.points,
            'level': user.level,
            'reputation_score': user.reputation_score,
            'verified_reports': user.verified_reports_count,
            'badges_count': len(badges_list),
            'is_anonymous': bool(user.display_name) or not user.profile_public
        }

    def get_leaderboard(
        self,
        leaderboard_type: str = 'global',
        limit: int = 100,
        current_user_id: Optional[UUID] = None
    ) -> Dict:
        """
        Get leaderboard with privacy controls
        Types: global, weekly, monthly
        """
        # Base query: only users who opted in
        query = self.db.query(models.User).filter(
            models.User.leaderboard_visible == True
        )

        # Filter by time period
        if leaderboard_type == 'weekly':
            # Get users with activity in last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = query.filter(
                models.User.last_activity_date >= week_ago
            )
        elif leaderboard_type == 'monthly':
            # Get users with activity in last 30 days
            month_ago = datetime.utcnow() - timedelta(days=30)
            query = query.filter(
                models.User.last_activity_date >= month_ago
            )

        # Order by points
        users = query.order_by(models.User.points.desc()).limit(limit).all()

        # Format with privacy controls
        entries = []
        current_user_rank = None

        for rank, user in enumerate(users, 1):
            entry_data = self.get_leaderboard_data(user)
            if entry_data:  # Respects privacy settings
                entry_data['rank'] = rank
                entries.append(entry_data)

                # Track current user's rank
                if current_user_id and user.id == current_user_id:
                    current_user_rank = rank

        # If current user not in top, find their rank
        if current_user_id and current_user_rank is None:
            user_rank = self._get_user_rank(current_user_id, leaderboard_type)
            if user_rank > 0:
                current_user_rank = user_rank

        return {
            'leaderboard_type': leaderboard_type,
            'updated_at': datetime.utcnow(),
            'entries': entries,
            'current_user_rank': current_user_rank
        }

    def _get_user_rank(
        self,
        user_id: UUID,
        leaderboard_type: str
    ) -> int:
        """
        Get a specific user's rank in the leaderboard
        Returns 0 if user not found or opted out
        """
        user = self.db.query(models.User).filter(
            models.User.id == user_id
        ).first()

        if not user or not user.leaderboard_visible:
            return 0

        # Base query
        query = self.db.query(models.User).filter(
            models.User.leaderboard_visible == True
        )

        # Filter by time period
        if leaderboard_type == 'weekly':
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = query.filter(
                models.User.last_activity_date >= week_ago
            )
        elif leaderboard_type == 'monthly':
            month_ago = datetime.utcnow() - timedelta(days=30)
            query = query.filter(
                models.User.last_activity_date >= month_ago
            )

        # Count users with more points
        rank = query.filter(
            models.User.points > user.points
        ).count() + 1

        return rank

    def get_top_users_summary(self, limit: int = 10) -> List[Dict]:
        """
        Get top users summary (simplified, for dashboard widgets)
        """
        users = self.db.query(models.User).filter(
            models.User.leaderboard_visible == True
        ).order_by(
            models.User.points.desc()
        ).limit(limit).all()

        result = []
        for rank, user in enumerate(users, 1):
            entry = self.get_leaderboard_data(user)
            if entry:
                entry['rank'] = rank
                result.append(entry)

        return result
