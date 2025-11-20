from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.reputation_models import (
    ReputationSummaryResponse,
    ReputationHistoryResponse,
    PrivacySettingsUpdate
)
from ..domain.services.reputation_service import ReputationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{user_id}", response_model=ReputationSummaryResponse)
def get_reputation_summary(
    user_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get complete reputation summary for a user.

    Returns:
    - Points, level, reputation score
    - Accuracy rate, streak days
    - Progress to next level
    - Badge counts
    """
    try:
        service = ReputationService(db)
        summary = service.get_reputation_summary(user_id)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting reputation summary for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch reputation summary")


@router.get("/{user_id}/history", response_model=List[ReputationHistoryResponse])
def get_reputation_history(
    user_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get user's reputation history with pagination.

    Shows all reputation changes:
    - Reports verified/rejected
    - Badges earned
    - Streak bonuses
    - Level ups
    """
    try:
        service = ReputationService(db)
        history = service.get_reputation_history(user_id, limit, offset)
        return history
    except Exception as e:
        logger.error(f"Error getting reputation history for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch reputation history")


@router.patch("/{user_id}/privacy")
def update_privacy_settings(
    user_id: UUID,
    settings: PrivacySettingsUpdate,
    db: Session = Depends(get_db)
):
    """
    Update user's privacy settings for reputation/leaderboard.

    Privacy options:
    - leaderboard_visible: Show on leaderboards
    - profile_public: Public profile page
    - display_name: Anonymous name for leaderboards
    """
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update privacy settings
        update_data = settings.model_dump(exclude_unset=True)

        # Validate display_name if provided
        if 'display_name' in update_data and update_data['display_name']:
            display_name = update_data['display_name']

            # Check if display name is already taken
            existing = db.query(models.User).filter(
                models.User.display_name == display_name,
                models.User.id != user_id
            ).first()

            if existing:
                raise HTTPException(
                    status_code=400,
                    detail="Display name already taken"
                )

        for field, value in update_data.items():
            setattr(user, field, value)

        db.commit()
        db.refresh(user)

        return {
            'message': 'Privacy settings updated',
            'settings': {
                'leaderboard_visible': user.leaderboard_visible,
                'profile_public': user.profile_public,
                'display_name': user.display_name
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating privacy settings for {user_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update privacy settings")
