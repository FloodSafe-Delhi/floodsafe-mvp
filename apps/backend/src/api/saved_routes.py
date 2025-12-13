"""
Saved Routes API - CRUD operations for user's saved route bookmarks.

Provides:
- GET /saved-routes/user/{user_id} - Get all saved routes for a user
- POST /saved-routes/ - Create a new saved route
- PUT /saved-routes/{route_id} - Update a saved route
- DELETE /saved-routes/{route_id} - Delete a saved route
- POST /saved-routes/{route_id}/increment - Increment use count
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime

from ..infrastructure.database import get_db
from ..infrastructure import models


router = APIRouter(prefix="/saved-routes", tags=["saved-routes"])


# Request/Response Models
class SavedRouteCreate(BaseModel):
    user_id: str
    name: str = Field(..., min_length=1, max_length=100)
    origin_latitude: float = Field(..., ge=-90, le=90)
    origin_longitude: float = Field(..., ge=-180, le=180)
    origin_name: str | None = None
    destination_latitude: float = Field(..., ge=-90, le=90)
    destination_longitude: float = Field(..., ge=-180, le=180)
    destination_name: str | None = None
    transport_mode: str = Field(default="driving", pattern="^(driving|walking|cycling)$")


class SavedRouteUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    origin_latitude: float | None = Field(None, ge=-90, le=90)
    origin_longitude: float | None = Field(None, ge=-180, le=180)
    origin_name: str | None = None
    destination_latitude: float | None = Field(None, ge=-90, le=90)
    destination_longitude: float | None = Field(None, ge=-180, le=180)
    destination_name: str | None = None
    transport_mode: str | None = Field(None, pattern="^(driving|walking|cycling)$")


class SavedRouteResponse(BaseModel):
    id: str
    user_id: str
    name: str
    origin_latitude: float
    origin_longitude: float
    origin_name: str | None
    destination_latitude: float
    destination_longitude: float
    destination_name: str | None
    transport_mode: str
    use_count: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


@router.get("/user/{user_id}", response_model=List[SavedRouteResponse])
def get_user_saved_routes(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all saved routes for a specific user.

    Routes are ordered by use_count (most used first).
    """
    routes = db.query(models.SavedRoute).filter(
        models.SavedRoute.user_id == user_id
    ).order_by(desc(models.SavedRoute.use_count)).all()

    # Convert datetime to ISO string
    return [
        SavedRouteResponse(
            id=str(route.id),
            user_id=str(route.user_id),
            name=route.name,
            origin_latitude=route.origin_latitude,
            origin_longitude=route.origin_longitude,
            origin_name=route.origin_name,
            destination_latitude=route.destination_latitude,
            destination_longitude=route.destination_longitude,
            destination_name=route.destination_name,
            transport_mode=route.transport_mode,
            use_count=route.use_count,
            created_at=route.created_at.isoformat() if route.created_at else "",
            updated_at=route.updated_at.isoformat() if route.updated_at else ""
        )
        for route in routes
    ]


@router.post("/", response_model=SavedRouteResponse, status_code=status.HTTP_201_CREATED)
def create_saved_route(
    route_data: SavedRouteCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new saved route for a user.

    The route is initialized with use_count = 0.
    """
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == route_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {route_data.user_id} not found"
        )

    # Create new saved route
    db_route = models.SavedRoute(
        user_id=route_data.user_id,
        name=route_data.name,
        origin_latitude=route_data.origin_latitude,
        origin_longitude=route_data.origin_longitude,
        origin_name=route_data.origin_name,
        destination_latitude=route_data.destination_latitude,
        destination_longitude=route_data.destination_longitude,
        destination_name=route_data.destination_name,
        transport_mode=route_data.transport_mode,
        use_count=0
    )

    db.add(db_route)
    db.commit()
    db.refresh(db_route)

    return SavedRouteResponse(
        id=str(db_route.id),
        user_id=str(db_route.user_id),
        name=db_route.name,
        origin_latitude=db_route.origin_latitude,
        origin_longitude=db_route.origin_longitude,
        origin_name=db_route.origin_name,
        destination_latitude=db_route.destination_latitude,
        destination_longitude=db_route.destination_longitude,
        destination_name=db_route.destination_name,
        transport_mode=db_route.transport_mode,
        use_count=db_route.use_count,
        created_at=db_route.created_at.isoformat() if db_route.created_at else "",
        updated_at=db_route.updated_at.isoformat() if db_route.updated_at else ""
    )


@router.put("/{route_id}", response_model=SavedRouteResponse)
def update_saved_route(
    route_id: str,
    route_data: SavedRouteUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an existing saved route.

    Only provided fields are updated; others remain unchanged.
    """
    db_route = db.query(models.SavedRoute).filter(
        models.SavedRoute.id == route_id
    ).first()

    if not db_route:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved route {route_id} not found"
        )

    # Update only provided fields
    update_data = route_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_route, field, value)

    db_route.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_route)

    return SavedRouteResponse(
        id=str(db_route.id),
        user_id=str(db_route.user_id),
        name=db_route.name,
        origin_latitude=db_route.origin_latitude,
        origin_longitude=db_route.origin_longitude,
        origin_name=db_route.origin_name,
        destination_latitude=db_route.destination_latitude,
        destination_longitude=db_route.destination_longitude,
        destination_name=db_route.destination_name,
        transport_mode=db_route.transport_mode,
        use_count=db_route.use_count,
        created_at=db_route.created_at.isoformat() if db_route.created_at else "",
        updated_at=db_route.updated_at.isoformat() if db_route.updated_at else ""
    )


@router.delete("/{route_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_route(
    route_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a saved route.
    """
    db_route = db.query(models.SavedRoute).filter(
        models.SavedRoute.id == route_id
    ).first()

    if not db_route:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved route {route_id} not found"
        )

    db.delete(db_route)
    db.commit()

    return None


@router.post("/{route_id}/increment", response_model=SavedRouteResponse)
def increment_route_usage(
    route_id: str,
    db: Session = Depends(get_db)
):
    """
    Increment the use_count for a saved route.

    Call this when a user loads a saved route for navigation.
    """
    db_route = db.query(models.SavedRoute).filter(
        models.SavedRoute.id == route_id
    ).first()

    if not db_route:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved route {route_id} not found"
        )

    db_route.use_count += 1
    db_route.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_route)

    return SavedRouteResponse(
        id=str(db_route.id),
        user_id=str(db_route.user_id),
        name=db_route.name,
        origin_latitude=db_route.origin_latitude,
        origin_longitude=db_route.origin_longitude,
        origin_name=db_route.origin_name,
        destination_latitude=db_route.destination_latitude,
        destination_longitude=db_route.destination_longitude,
        destination_name=db_route.destination_name,
        transport_mode=db_route.transport_mode,
        use_count=db_route.use_count,
        created_at=db_route.created_at.isoformat() if db_route.created_at else "",
        updated_at=db_route.updated_at.isoformat() if db_route.updated_at else ""
    )
