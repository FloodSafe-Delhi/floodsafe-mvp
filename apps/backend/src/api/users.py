from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_
from uuid import UUID
from datetime import datetime, timedelta
from typing import List
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import UserCreate, UserResponse
from geoalchemy2.functions import ST_DWithin, ST_MakePoint

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    """
    try:
        # Check if user already exists
        existing_user = db.query(models.User).filter(
            (models.User.email == user.email) | (models.User.username == user.username)
        ).first()

        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already registered")

        new_user = models.User(
            username=user.username,
            email=user.email,
            role=user.role
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return new_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create user")

@router.get("/", response_model=List[UserResponse])
def list_users(db: Session = Depends(get_db)):
    """
    List all users.
    """
    try:
        users = db.query(models.User).all()
        return users
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Failed to list users")

@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    """
    Get user profile by ID.
    """
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user")

@router.get("/stats/active-reporters", response_model=dict)
def get_active_reporters_count(db: Session = Depends(get_db)):
    """
    Get count of users who have made reports in the past 7 days.
    Active reporters must have reports_count > 0 AND have made at least one report in past week.
    """
    try:
        seven_days_ago = datetime.utcnow() - timedelta(days=7)

        # Get users who have made reports in the past 7 days
        active_reporter_ids = db.query(models.Report.user_id).filter(
            models.Report.timestamp >= seven_days_ago
        ).distinct().all()

        active_reporter_ids = [uid[0] for uid in active_reporter_ids]

        # Count users with reports_count > 0 AND who made reports recently
        count = db.query(models.User).filter(
            and_(
                models.User.reports_count > 0,
                models.User.id.in_(active_reporter_ids) if active_reporter_ids else False
            )
        ).count()

        return {"count": count, "period_days": 7}
    except Exception as e:
        logger.error(f"Error counting active reporters: {e}")
        raise HTTPException(status_code=500, detail="Failed to count active reporters")

@router.get("/stats/nearby-reporters", response_model=dict)
def get_nearby_reporters_count(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(5.0, gt=0, le=50),
    db: Session = Depends(get_db)
):
    """
    Get count of users who have made reports within radius_km of the given location.
    Uses PostGIS ST_DWithin for efficient spatial queries.
    Radius is in kilometers, converted to meters for PostGIS.
    """
    try:
        radius_meters = radius_km * 1000  # Convert km to meters

        # Create a point for the query location (PostGIS format: POINT(lng lat))
        query_point = ST_MakePoint(longitude, latitude)

        # Find all reports within radius
        nearby_reports = db.query(models.Report.user_id).filter(
            ST_DWithin(
                models.Report.location,
                query_point,
                radius_meters,
                True  # Use spheroid for accurate distance calculation
            )
        ).distinct().all()

        nearby_user_ids = [uid[0] for uid in nearby_reports]

        # Count unique users who made those reports
        count = len(nearby_user_ids)

        return {
            "count": count,
            "radius_km": radius_km,
            "center": {"latitude": latitude, "longitude": longitude}
        }
    except Exception as e:
        logger.error(f"Error counting nearby reporters: {e}")
        raise HTTPException(status_code=500, detail="Failed to count nearby reporters")

@router.get("/leaderboard/top", response_model=List[UserResponse])
def get_leaderboard(limit: int = 10, db: Session = Depends(get_db)):
    """
    Get top users by points.
    """
    try:
        users = db.query(models.User).order_by(models.User.points.desc()).limit(limit).all()
        return users
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch leaderboard")
