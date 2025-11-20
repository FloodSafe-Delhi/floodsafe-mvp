from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.reputation_models import BadgeResponse, BadgeWithProgressResponse
from ..domain.services.reputation_service import ReputationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=List[BadgeResponse])
def get_all_badges(
    db: Session = Depends(get_db)
):
    """
    Get all available badges in the system.

    Returns all active badges ordered by sort_order.
    """
    try:
        badges = db.query(models.Badge).filter(
            models.Badge.is_active == True
        ).order_by(models.Badge.sort_order).all()

        return badges
    except Exception as e:
        logger.error(f"Error fetching badges: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch badges")


@router.get("/user/{user_id}")
def get_user_badges(
    user_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get user's badges with progress on locked badges.

    Returns:
    - earned: List of badges user has earned
    - in_progress: List of badges user is working toward (with progress %)
    """
    try:
        service = ReputationService(db)
        badges_data = service.get_badges_with_progress(user_id)

        # Format the response
        earned = [
            {
                'badge_id': item['badge'].id,
                'key': item['badge'].key,
                'name': item['badge'].name,
                'description': item['badge'].description,
                'icon': item['badge'].icon,
                'earned_at': item['earned_at']
            }
            for item in badges_data['earned']
        ]

        in_progress = [
            {
                'badge_id': item['badge'].id,
                'key': item['badge'].key,
                'name': item['badge'].name,
                'description': item['badge'].description,
                'icon': item['badge'].icon,
                'current_value': item['current_value'],
                'required_value': item['required_value'],
                'progress_percent': item['progress_percent']
            }
            for item in badges_data['in_progress']
        ]

        return {
            'earned': earned,
            'in_progress': in_progress
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching user badges for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user badges")


@router.get("/{badge_id}", response_model=BadgeResponse)
def get_badge(
    badge_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific badge.
    """
    try:
        badge = db.query(models.Badge).filter(
            models.Badge.id == badge_id
        ).first()

        if not badge:
            raise HTTPException(status_code=404, detail="Badge not found")

        return badge
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching badge {badge_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch badge")
