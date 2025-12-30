from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
import logging
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from geoalchemy2 import WKTElement

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import WatchAreaCreate, WatchAreaResponse
from ..domain.services.watch_area_risk_service import WatchAreaRiskService

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

        # Create PostGIS point from lat/lng using WKTElement for proper geometry conversion
        point_wkt = f"POINT({watch_area.longitude} {watch_area.latitude})"

        new_watch_area = models.WatchArea(
            user_id=watch_area.user_id,
            name=watch_area.name,
            location=WKTElement(point_wkt, srid=4326),
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


# Pydantic models for risk assessment response
class HotspotInWatchAreaResponse(BaseModel):
    """Hotspot within a watch area."""
    id: int
    name: str
    fhi_score: float
    fhi_level: str
    fhi_color: str
    distance_meters: float


class WatchAreaRiskAssessmentResponse(BaseModel):
    """Risk assessment for a watch area."""
    watch_area_id: UUID
    watch_area_name: str
    latitude: float
    longitude: float
    radius: float
    nearby_hotspots: List[HotspotInWatchAreaResponse]
    nearby_hotspots_count: int
    critical_hotspots_count: int
    average_fhi: float
    max_fhi: float
    max_fhi_level: str
    is_at_risk: bool
    risk_flag_reason: Optional[str]
    last_calculated: datetime


@router.get("/user/{user_id}/risk-assessment", response_model=List[WatchAreaRiskAssessmentResponse])
async def get_user_watch_area_risks(user_id: UUID, db: Session = Depends(get_db)):
    """
    Get risk assessment for all user's watch areas based on nearby hotspots.

    Analyzes each watch area for:
    - Nearby hotspots within radius
    - Average and maximum FHI scores
    - Critical hotspots (HIGH/EXTREME)
    - Risk flag if average FHI > 0.5 OR any HIGH/EXTREME hotspot present

    Returns:
        List of risk assessments, one per watch area
    """
    try:
        # Verify user exists
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Calculate risk assessments
        service = WatchAreaRiskService(db)
        assessments = await service.calculate_risk_for_user_watch_areas(user_id)

        # Convert dataclasses to Pydantic models
        response = []
        for assessment in assessments:
            response.append(WatchAreaRiskAssessmentResponse(
                watch_area_id=assessment.watch_area_id,
                watch_area_name=assessment.watch_area_name,
                latitude=assessment.latitude,
                longitude=assessment.longitude,
                radius=assessment.radius,
                nearby_hotspots=[
                    HotspotInWatchAreaResponse(
                        id=h.id,
                        name=h.name,
                        fhi_score=h.fhi_score,
                        fhi_level=h.fhi_level,
                        fhi_color=h.fhi_color,
                        distance_meters=h.distance_meters
                    )
                    for h in assessment.nearby_hotspots
                ],
                nearby_hotspots_count=assessment.nearby_hotspots_count,
                critical_hotspots_count=assessment.critical_hotspots_count,
                average_fhi=assessment.average_fhi,
                max_fhi=assessment.max_fhi,
                max_fhi_level=assessment.max_fhi_level,
                is_at_risk=assessment.is_at_risk,
                risk_flag_reason=assessment.risk_flag_reason,
                last_calculated=assessment.last_calculated
            ))

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating watch area risks for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate watch area risks")
