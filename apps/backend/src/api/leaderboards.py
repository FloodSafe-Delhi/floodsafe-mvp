from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional
import logging

from ..infrastructure.database import get_db
from ..domain.reputation_models import LeaderboardResponse
from ..domain.services.leaderboard_service import LeaderboardService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=LeaderboardResponse)
def get_leaderboard(
    type: str = Query("global", regex="^(global|weekly|monthly)$"),
    limit: int = Query(100, ge=1, le=500),
    user_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get leaderboard with privacy controls.

    Leaderboard types:
    - global: All-time top users
    - weekly: Top users from last 7 days
    - monthly: Top users from last 30 days

    Privacy:
    - Only shows users who opted in (leaderboard_visible=true)
    - Respects display_name settings
    - Anonymous users shown with custom names
    """
    try:
        service = LeaderboardService(db)
        leaderboard = service.get_leaderboard(
            leaderboard_type=type,
            limit=limit,
            current_user_id=user_id
        )
        return leaderboard
    except Exception as e:
        logger.error(f"Error fetching leaderboard ({type}): {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch leaderboard")


@router.get("/top", response_model=list)
def get_top_users(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Get top users summary (for dashboard widgets).

    Returns simplified leaderboard with top N users.
    Respects all privacy settings.
    """
    try:
        service = LeaderboardService(db)
        top_users = service.get_top_users_summary(limit)
        return top_users
    except Exception as e:
        logger.error(f"Error fetching top users: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch top users")
