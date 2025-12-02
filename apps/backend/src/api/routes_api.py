"""
Safe Route Navigation API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..infrastructure.database import get_db
from ..domain.services.routing_service import RoutingService
from ..domain.models import RouteRequest, RouteResponse

router = APIRouter(prefix="/routes", tags=["routing"])


@router.post("/calculate", response_model=RouteResponse)
async def calculate_route(
    request: RouteRequest,
    db: Session = Depends(get_db)
):
    """
    Calculate safe routes avoiding flood zones.

    Returns multiple route options (safe, fast, balanced) with flood information.
    """
    service = RoutingService(db)

    try:
        routes = await service.calculate_safe_routes(
            origin=(request.origin.lng, request.origin.lat),
            destination=(request.destination.lng, request.destination.lat),
            city_code=request.city,
            mode=request.mode
        )

        if not routes:
            raise HTTPException(
                status_code=404,
                detail="No routes found between the specified locations"
            )

        return RouteResponse(
            routes=routes,
            city=request.city,
            warnings=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for routing service"""
    return {"status": "ok", "service": "routing"}
