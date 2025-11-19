from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import UserCreate, UserResponse

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

@router.get("/leaderboard/top", response_model=list[UserResponse])
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
