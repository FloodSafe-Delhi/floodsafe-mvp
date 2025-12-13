"""
Safe Route Navigation API Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..infrastructure.database import get_db
from ..domain.services.routing_service import RoutingService
from pydantic import BaseModel


router = APIRouter(prefix="/routes", tags=["routing"])


# Request/Response Models
class LocationPoint(BaseModel):
    lat: float
    lng: float


class RouteCalculationRequest(BaseModel):
    origin: LocationPoint
    destination: LocationPoint
    mode: str = "driving"  # driving, walking, cycling
    city: str = "BLR"
    avoid_ml_risk: bool = False


@router.post("/calculate")
async def calculate_route(
    request: RouteCalculationRequest,
    db: Session = Depends(get_db)
):
    """
    Calculate safe routes avoiding flood zones.

    Returns multiple route options (safe, fast, balanced) with flood zones GeoJSON.
    """
    service = RoutingService(db)

    try:
        result = await service.calculate_safe_routes(
            origin=(request.origin.lng, request.origin.lat),
            destination=(request.destination.lng, request.destination.lat),
            city_code=request.city,
            mode=request.mode,
            avoid_ml_risk=request.avoid_ml_risk
        )

        if not result or not result.get("routes"):
            raise HTTPException(
                status_code=404,
                detail="No routes found between the specified locations"
            )

        return result  # Returns {"routes": [...], "flood_zones": {...}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nearby-metros")
async def get_nearby_metros(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    city: str = Query("BLR", description="City code (BLR/DEL)"),
    radius_km: float = Query(2.0, description="Search radius in kilometers"),
    db: Session = Depends(get_db)
):
    """
    Find nearby metro stations within specified radius.

    Returns list of metro stations with distance and walking time.
    """
    service = RoutingService(db)

    try:
        stations = await service.get_nearby_metros(lat, lng, city, radius_km)
        return {"metros": stations, "count": len(stations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WalkingRouteRequest(BaseModel):
    origin: LocationPoint
    destination: LocationPoint


@router.post("/walking-route")
async def calculate_walking_route(
    request: WalkingRouteRequest,
    db: Session = Depends(get_db)
):
    """
    Calculate walking route between two points.
    Used for routes to metro stations.
    """
    service = RoutingService(db)

    try:
        route = await service.calculate_walking_route(
            origin=(request.origin.lng, request.origin.lat),
            destination=(request.destination.lng, request.destination.lat)
        )

        if not route:
            raise HTTPException(
                status_code=404,
                detail="Could not calculate walking route"
            )

        return route
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RouteComparisonRequest(BaseModel):
    """Request for comparing normal vs FloodSafe routes"""
    origin: LocationPoint
    destination: LocationPoint
    mode: str = "driving"  # driving, walking, cycling
    city: str = "BLR"
    test_fhi_override: str = None  # Testing: 'high', 'extreme', or 'mixed'


@router.post("/compare")
async def compare_routes(
    request: RouteComparisonRequest,
    db: Session = Depends(get_db)
):
    """
    Compare normal route vs FloodSafe route.

    Returns both routes with comparison metrics including:
    - Time penalty for taking the safe route
    - Number of flood zones avoided
    - Estimated stuck time if taking normal route
    - Risk breakdown (reports, sensors, ML predictions)
    - Recommendation message
    - Hotspot analysis with HARD AVOID warnings (Delhi only)

    Test Mode:
    - Set test_fhi_override to 'high', 'extreme', or 'mixed' to simulate
      HIGH/EXTREME FHI hotspots for testing the HARD AVOID routing logic.
    """
    service = RoutingService(db)

    try:
        result = await service.compare_routes(
            origin=(request.origin.lng, request.origin.lat),
            destination=(request.destination.lng, request.destination.lat),
            city_code=request.city,
            mode=request.mode,
            test_fhi_override=request.test_fhi_override,
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for routing service"""
    return {"status": "ok", "service": "routing"}
