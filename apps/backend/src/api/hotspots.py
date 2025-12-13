"""
Backend API wrapper for waterlogging hotspot predictions.

Proxies requests to the ML service with caching.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from datetime import datetime
import httpx
import logging

from ..core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple in-memory cache
_hotspots_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes (hotspots change with rainfall)


def _is_cache_valid(cache_entry: Dict[str, Any]) -> bool:
    """Check if cache entry is still valid."""
    if "timestamp" not in cache_entry:
        return False
    age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
    return age < CACHE_TTL_SECONDS


@router.get("/all")
async def get_all_hotspots(
    include_rainfall: bool = Query(True, description="Include current rainfall factor"),
    test_fhi_override: str = Query(None, description="Override FHI for testing: 'high', 'extreme', or 'mixed'"),
):
    """
    Get all 62 Delhi waterlogging hotspots with current risk levels.

    Returns GeoJSON FeatureCollection with:
    - Point features for each hotspot
    - Properties: id, name, zone, risk_probability, risk_level, risk_color

    Risk is dynamically adjusted based on current rainfall.
    """
    if not settings.ML_SERVICE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ML service is not enabled",
        )

    # Check cache (skip cache in test mode to ensure fresh test values)
    cache_key = f"hotspots_all_{include_rainfall}"
    if not test_fhi_override and cache_key in _hotspots_cache:
        cache_entry = _hotspots_cache[cache_key]
        if _is_cache_valid(cache_entry):
            logger.info("Cache hit for hotspots")
            return cache_entry["data"]

    # Call ML service
    try:
        params = {"include_rainfall": include_rainfall}
        if test_fhi_override:
            params["test_fhi_override"] = test_fhi_override
            logger.info(f"Using test FHI override: {test_fhi_override}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{settings.ML_SERVICE_URL}/api/v1/hotspots/all",
                params=params,
            )

            if response.status_code != 200:
                logger.error(f"ML service error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ML service error: {response.text}",
                )

            result = response.json()

            # Cache the result (but not in test mode)
            if not test_fhi_override:
                _hotspots_cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.now(),
                }

            logger.info(f"Hotspots fetched: {len(result.get('features', []))} locations" +
                       (f" (TEST MODE: {test_fhi_override})" if test_fhi_override else ""))
            return result

    except httpx.TimeoutException:
        logger.error("ML service timeout")
        raise HTTPException(
            status_code=504,
            detail="ML service request timed out",
        )
    except httpx.RequestError as e:
        logger.error(f"ML service request failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"ML service unavailable: {str(e)}",
        )


@router.get("/hotspot/{hotspot_id}")
async def get_hotspot_risk(hotspot_id: int):
    """
    Get risk details for a specific hotspot by ID.

    Args:
        hotspot_id: Hotspot identifier (1-62)

    Returns:
        Hotspot details with current risk assessment
    """
    if not settings.ML_SERVICE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ML service is not enabled",
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.ML_SERVICE_URL}/api/v1/hotspots/hotspot/{hotspot_id}",
            )

            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Hotspot {hotspot_id} not found",
                )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ML service error: {response.text}",
                )

            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"ML service unavailable: {str(e)}")


@router.post("/risk-at-point")
async def get_risk_at_point(
    lat: float = Query(..., ge=28.3, le=29.0, description="Latitude"),
    lng: float = Query(..., ge=76.7, le=77.5, description="Longitude"),
):
    """
    Get flood risk for any point in Delhi.

    Uses proximity to known hotspots and current rainfall.

    Args:
        lat: Latitude (must be within Delhi bounds)
        lng: Longitude (must be within Delhi bounds)

    Returns:
        Risk assessment for the point
    """
    if not settings.ML_SERVICE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ML service is not enabled",
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ML_SERVICE_URL}/api/v1/hotspots/risk-at-point",
                json={"latitude": lat, "longitude": lng},
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ML service error: {response.text}",
                )

            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"ML service unavailable: {str(e)}")


@router.get("/health")
async def hotspots_health():
    """Check hotspots service health."""
    if not settings.ML_SERVICE_ENABLED:
        return {
            "status": "disabled",
            "ml_service_enabled": False,
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{settings.ML_SERVICE_URL}/api/v1/hotspots/health"
            )
            ml_health = response.json()

            return {
                "status": "healthy",
                "ml_service_enabled": True,
                "hotspots_loaded": ml_health.get("hotspots_loaded"),
                "total_hotspots": ml_health.get("total_hotspots"),
                "model_trained": ml_health.get("model_trained"),
            }

    except Exception as e:
        return {
            "status": "degraded",
            "ml_service_enabled": True,
            "error": str(e),
        }
