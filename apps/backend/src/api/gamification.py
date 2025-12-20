"""
Gamification API endpoints.

Exposes reputation, badges, and achievements via REST API.
Uses existing ReputationService methods - no duplication.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.services.reputation_service import ReputationService
from .deps import get_current_user

router = APIRouter()


# ============================================================================
# AUTHENTICATED USER ENDPOINTS
# ============================================================================

@router.get("/me/badges")
async def get_my_badges(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get authenticated user's badges with progress on locked ones.

    Returns:
        {
            "earned": [{"badge": {...}, "earned_at": "..."}],
            "in_progress": [{"badge": {...}, "current_value": 5, "required_value": 10, "progress_percent": 50}]
        }
    """
    service = ReputationService(db)
    result = service.get_badges_with_progress(current_user.id)

    # Convert badge models to dicts for JSON serialization
    earned = []
    for item in result['earned']:
        badge = item['badge']
        earned.append({
            'badge': {
                'key': badge.key,
                'name': badge.name,
                'description': badge.description,
                'icon': badge.icon,
                'category': badge.category,
                'points_reward': badge.points_reward
            },
            'earned_at': item['earned_at'].isoformat() if item['earned_at'] else None
        })

    in_progress = []
    for item in result['in_progress']:
        badge = item['badge']
        in_progress.append({
            'badge': {
                'key': badge.key,
                'name': badge.name,
                'description': badge.description,
                'icon': badge.icon,
                'category': badge.category,
                'points_reward': badge.points_reward
            },
            'current_value': item['current_value'],
            'required_value': item['required_value'],
            'progress_percent': item['progress_percent']
        })

    return {
        'earned': earned,
        'in_progress': in_progress
    }


@router.get("/me/reputation")
async def get_my_reputation(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get authenticated user's reputation summary.

    Returns:
        {
            "user_id": "...",
            "points": 150,
            "level": 2,
            "reputation_score": 85,
            "accuracy_rate": 90.5,
            "streak_days": 7,
            "next_level_points": 50,
            "badges_earned": 3,
            "total_badges": 10
        }
    """
    service = ReputationService(db)
    summary = service.get_reputation_summary(current_user.id)

    # Convert UUID to string for JSON
    return {
        'user_id': str(summary['user_id']),
        'points': summary['points'],
        'level': summary['level'],
        'reputation_score': summary['reputation_score'],
        'accuracy_rate': summary['accuracy_rate'],
        'streak_days': summary['streak_days'],
        'next_level_points': summary['next_level_points'],
        'badges_earned': summary['badges_earned'],
        'total_badges': summary['total_badges']
    }


@router.get("/me/reputation/history")
async def get_my_reputation_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get authenticated user's reputation history (point changes).

    Returns list of:
        {
            "action": "report_verified",
            "points_change": 15,
            "new_total": 165,
            "reason": "Report verified (quality: 85)",
            "created_at": "2024-01-15T10:30:00"
        }
    """
    service = ReputationService(db)
    history = service.get_reputation_history(current_user.id, limit, offset)

    return [
        {
            'action': h.action,
            'points_change': h.points_change,
            'new_total': h.new_total,
            'reason': h.reason,
            'created_at': h.created_at.isoformat() if h.created_at else None
        }
        for h in history
    ]


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@router.get("/badges/catalog")
async def get_badges_catalog(
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get all available badges (public endpoint).

    Returns list of all active badges sorted by order.
    """
    badges = db.query(models.Badge).filter(
        models.Badge.is_active == True
    ).order_by(models.Badge.sort_order).all()

    return [
        {
            'key': b.key,
            'name': b.name,
            'description': b.description,
            'icon': b.icon,
            'category': b.category,
            'requirement_type': b.requirement_type,
            'requirement_value': b.requirement_value,
            'points_reward': b.points_reward
        }
        for b in badges
    ]
