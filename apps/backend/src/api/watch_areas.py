from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import WatchAreaCreate, WatchAreaResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=WatchAreaResponse)
def create_watch_area(watch_area: WatchAreaCreate, db: Session = Depends(get_db)):
    """
    Create a new watch area for a user.
    Watch areas allow users to monitor specific locations for flood alerts.
    """
    try:
        # Verify user exists
        user = db.query(models.User).filter(models.User.id == watch_area.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Create PostGIS point from lat/lng
        point_wkt = f"POINT({watch_area.longitude} {watch_area.latitude})"

        new_watch_area = models.WatchArea(
            user_id=watch_area.user_id,
            name=watch_area.name,
            location=point_wkt,
            radius=watch_area.radius
        )

        db.add(new_watch_area)
        db.commit()
        db.refresh(new_watch_area)

        # Return response with extracted lat/lng
        return WatchAreaResponse(
            id=new_watch_area.id,
            user_id=new_watch_area.user_id,
            name=new_watch_area.name,
            latitude=new_watch_area.latitude,
            longitude=new_watch_area.longitude,
            radius=new_watch_area.radius,
            created_at=new_watch_area.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating watch area: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create watch area")


@router.get("/user/{user_id}", response_model=list[WatchAreaResponse])
def get_user_watch_areas(user_id: UUID, db: Session = Depends(get_db)):
    """
    Get all watch areas for a specific user.
    """
    try:
        # Verify user exists
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        watch_areas = db.query(models.WatchArea).filter(
            models.WatchArea.user_id == user_id
        ).all()

        # Convert to response format with lat/lng
        response = []
        for wa in watch_areas:
            response.append(WatchAreaResponse(
                id=wa.id,
                user_id=wa.user_id,
                name=wa.name,
                latitude=wa.latitude,
                longitude=wa.longitude,
                radius=wa.radius,
                created_at=wa.created_at
            ))

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching watch areas for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch watch areas")


@router.get("/{watch_area_id}", response_model=WatchAreaResponse)
def get_watch_area(watch_area_id: UUID, db: Session = Depends(get_db)):
    """
    Get a specific watch area by ID.
    """
    try:
        watch_area = db.query(models.WatchArea).filter(
            models.WatchArea.id == watch_area_id
        ).first()

        if not watch_area:
            raise HTTPException(status_code=404, detail="Watch area not found")

        return WatchAreaResponse(
            id=watch_area.id,
            user_id=watch_area.user_id,
            name=watch_area.name,
            latitude=watch_area.latitude,
            longitude=watch_area.longitude,
            radius=watch_area.radius,
            created_at=watch_area.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching watch area {watch_area_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch watch area")


@router.delete("/{watch_area_id}")
def delete_watch_area(watch_area_id: UUID, db: Session = Depends(get_db)):
    """
    Delete a watch area.
    """
    try:
        watch_area = db.query(models.WatchArea).filter(
            models.WatchArea.id == watch_area_id
        ).first()

        if not watch_area:
            raise HTTPException(status_code=404, detail="Watch area not found")

        db.delete(watch_area)
        db.commit()

        return {"message": "Watch area deleted successfully", "id": str(watch_area_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting watch area {watch_area_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete watch area")
