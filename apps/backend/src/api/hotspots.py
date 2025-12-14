"""
Backend API wrapper for waterlogging hotspot predictions.

Proxies requests to the ML service with caching.
Falls back to static data when ML service is disabled.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import httpx
import logging
import json
import os

from ..core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# Path to static hotspot data (for when ML service is disabled)
def _get_static_hotspots_path() -> Optional[Path]:
    """Get path to static hotspots data file."""
    # Check backend's data directory first
    backend_data = Path(__file__).resolve().parent.parent.parent / "data" / "delhi_waterlogging_hotspots.json"
    if backend_data.exists():
        return backend_data

    # Fallback to ml-service data
    ml_data = Path(__file__).resolve().parent.parent.parent.parent / "ml-service" / "data" / "delhi_waterlogging_hotspots.json"
    if ml_data.exists():
        return ml_data

    return None


def _load_static_hotspots() -> Dict[str, Any]:
    """
    Load static hotspot data when ML service is disabled.
    Returns GeoJSON FeatureCollection with baseline risk levels.
    """
    data_path = _get_static_hotspots_path()
    if not data_path:
        raise HTTPException(
            status_code=503,
            detail="Hotspot data file not found. Deploy with data files or enable ML service.",
        )

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading static hotspots: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading hotspot data: {e}")

    # Handle both formats: raw array or {metadata, hotspots} object
    if isinstance(raw_data, dict) and "hotspots" in raw_data:
        hotspots_list = raw_data["hotspots"]
    elif isinstance(raw_data, list):
        hotspots_list = raw_data
    else:
        logger.error(f"Unexpected hotspots data format: {type(raw_data)}")
        raise HTTPException(status_code=500, detail="Invalid hotspot data format")

    # Convert to GeoJSON FeatureCollection format
    features = []
    for hotspot in hotspots_list:
        # Support both property name conventions
        severity = hotspot.get("severity_history") or hotspot.get("historical_severity", "moderate")
        severity = severity.lower() if severity else "moderate"

        # Map severity to risk levels
        if severity in ["high", "severe"]:
            risk_prob = 0.6
            risk_level = "high"
            risk_color = "#f97316"  # orange
        elif severity in ["critical", "extreme"]:
            risk_prob = 0.8
            risk_level = "extreme"
            risk_color = "#ef4444"  # red
        else:  # moderate, low
            risk_prob = 0.4
            risk_level = "moderate"
            risk_color = "#eab308"  # yellow

        # Support both coordinate conventions (lat/lng vs latitude/longitude)
        lng = hotspot.get("lng") or hotspot.get("longitude")
        lat = hotspot.get("lat") or hotspot.get("latitude")

        if lng is None or lat is None:
            logger.warning(f"Skipping hotspot without coordinates: {hotspot.get('id')}")
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lng, lat]
            },
            "properties": {
                "id": hotspot.get("id", 0),
                "name": hotspot.get("name", "Unknown"),
                "zone": hotspot.get("zone", "Unknown"),
                "description": hotspot.get("description", ""),
                "risk_probability": risk_prob,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "fhi": None,  # No live FHI without ML service
                "fhi_color": None,
                "historical_severity": severity,
                "elevation_m": hotspot.get("elevation_m"),
                "static_data": True,  # Flag indicating this is static data
            }
        }
        features.append(feature)

    logger.info(f"Loaded {len(features)} static hotspots from {data_path}")

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_hotspots": len(features),
            "source": "static",
            "ml_service_enabled": False,
            "fhi_available": False,
            "generated_at": datetime.now().isoformat(),
            "note": "Live FHI calculations unavailable. Showing baseline risk from historical data."
        }
    }

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

    Risk is dynamically adjusted based on current rainfall when ML service is enabled.
    Falls back to static baseline data when ML service is disabled.
    """
    # If ML service is disabled, return static data
    if not settings.ML_SERVICE_ENABLED:
        logger.info("ML service disabled, serving static hotspot data")
        return _load_static_hotspots()

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
        # Check if static data is available
        static_path = _get_static_hotspots_path()
        return {
            "status": "static_fallback",
            "ml_service_enabled": False,
            "static_data_available": static_path is not None,
            "static_data_path": str(static_path) if static_path else None,
            "note": "Serving baseline hotspot data. Live FHI requires ML service.",
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
