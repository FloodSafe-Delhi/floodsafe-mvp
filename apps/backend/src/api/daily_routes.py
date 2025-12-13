"""
Daily Routes API - CRUD endpoints for user's daily commute routes

Allows users to save regular routes (e.g., home to work) and get flood alerts
along those routes.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List

from ..infrastructure.database import get_db
from ..infrastructure.models import DailyRoute, User
from ..domain.models import DailyRouteCreate, DailyRouteResponse

router = APIRouter()


@router.post("/", response_model=DailyRouteResponse, status_code=201)
def create_daily_route(
    route_data: DailyRouteCreate,
    db: Session = Depends(get_db)
):
    """Create a new daily route for a user"""
    # Verify user exists
    user = db.query(User).filter(User.id == route_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create route
    db_route = DailyRoute(
        user_id=route_data.user_id,
        name=route_data.name,
        origin_latitude=route_data.origin_latitude,
        origin_longitude=route_data.origin_longitude,
        destination_latitude=route_data.destination_latitude,
        destination_longitude=route_data.destination_longitude,
        transport_mode=route_data.transport_mode,
        notify_on_flood=route_data.notify_on_flood
    )
    db.add(db_route)
    db.commit()
    db.refresh(db_route)
    return db_route


@router.get("/user/{user_id}", response_model=List[DailyRouteResponse])
def get_user_daily_routes(
    user_id: UUID,
    db: Session = Depends(get_db)
):
    """Get all daily routes for a specific user"""
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    routes = db.query(DailyRoute).filter(DailyRoute.user_id == user_id).all()
    return routes


@router.get("/{route_id}", response_model=DailyRouteResponse)
def get_daily_route(
    route_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific daily route by ID"""
    route = db.query(DailyRoute).filter(DailyRoute.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Daily route not found")
    return route


@router.patch("/{route_id}", response_model=DailyRouteResponse)
def update_daily_route(
    route_id: UUID,
    route_data: DailyRouteCreate,
    db: Session = Depends(get_db)
):
    """Update a daily route"""
    route = db.query(DailyRoute).filter(DailyRoute.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Daily route not found")

    # Update fields
    route.name = route_data.name
    route.origin_latitude = route_data.origin_latitude
    route.origin_longitude = route_data.origin_longitude
    route.destination_latitude = route_data.destination_latitude
    route.destination_longitude = route_data.destination_longitude
    route.transport_mode = route_data.transport_mode
    route.notify_on_flood = route_data.notify_on_flood

    db.commit()
    db.refresh(route)
    return route


@router.delete("/{route_id}", status_code=204)
def delete_daily_route(
    route_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a daily route"""
    route = db.query(DailyRoute).filter(DailyRoute.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Daily route not found")

    db.delete(route)
    db.commit()
    return None
