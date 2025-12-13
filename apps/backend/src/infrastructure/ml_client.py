"""
ML Service Client for Backend Integration.

HTTP client for communicating with the ML prediction service.
"""

import httpx
import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MLServiceClient:
    """
    Client for FloodSafe ML Service.

    Provides methods to get flood predictions and risk assessments.
    """

    def __init__(self, base_url: str = "http://localhost:8002", enabled: bool = True):
        """
        Initialize ML service client.

        Args:
            base_url: Base URL of ML service
            enabled: Whether ML service is enabled
        """
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.timeout = 30.0  # seconds

    async def health_check(self) -> Optional[Dict]:
        """
        Check ML service health.

        Returns:
            Dict with health status or None if unavailable
        """
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/predictions/health",
                    timeout=5.0,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"ML service health check failed: {e}")
            return None

    async def get_flood_forecast(
        self,
        latitude: float,
        longitude: float,
        horizon_days: int = 7,
        include_uncertainty: bool = False,
    ) -> Optional[Dict]:
        """
        Get flood probability forecast for a location.

        Args:
            latitude: Latitude
            longitude: Longitude
            horizon_days: Number of days to forecast (1-30)
            include_uncertainty: Include individual model contributions

        Returns:
            Dict with forecast data or None if service unavailable
        """
        if not self.enabled:
            logger.info("ML service disabled, skipping prediction")
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/predictions/forecast",
                    json={
                        "latitude": latitude,
                        "longitude": longitude,
                        "horizon_days": horizon_days,
                        "include_uncertainty": include_uncertainty,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"ML service returned error {e.response.status_code}: {e.response.text}")
            return None
        except httpx.TimeoutException:
            logger.error(f"ML service request timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"ML service request failed: {e}")
            return None

    async def get_risk_assessment(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
    ) -> Optional[Dict]:
        """
        Get static flood risk assessment for a location.

        Args:
            latitude: Latitude
            longitude: Longitude
            radius_km: Buffer radius in kilometers

        Returns:
            Dict with risk assessment or None if unavailable
        """
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/predictions/risk-assessment",
                    json={
                        "latitude": latitude,
                        "longitude": longitude,
                        "radius_km": radius_km,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"ML service risk assessment failed: {e}")
            return None

    async def get_model_info(self) -> Optional[Dict]:
        """
        Get information about loaded ML models.

        Returns:
            Dict with model info or None if unavailable
        """
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/predictions/models/info",
                    timeout=5.0,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return None

    def get_highest_risk_prediction(self, forecast: Dict) -> Optional[Dict]:
        """
        Extract the highest risk prediction from forecast.

        Args:
            forecast: Forecast response from get_flood_forecast()

        Returns:
            Dict with date and probability of highest risk day
        """
        if not forecast or "predictions" not in forecast:
            return None

        predictions = forecast["predictions"]
        if not predictions:
            return None

        highest = max(predictions, key=lambda p: p.get("flood_probability", 0))
        return {
            "date": highest.get("date"),
            "probability": highest.get("flood_probability"),
            "risk_level": highest.get("risk_level"),
        }


# Global instance (configure in main.py startup)
ml_client: Optional[MLServiceClient] = None


def get_ml_client() -> Optional[MLServiceClient]:
    """Get the global ML service client instance."""
    return ml_client


def init_ml_client(base_url: str, enabled: bool = True) -> MLServiceClient:
    """
    Initialize the global ML service client.

    Args:
        base_url: ML service base URL
        enabled: Whether ML service is enabled

    Returns:
        Initialized client instance
    """
    global ml_client
    ml_client = MLServiceClient(base_url=base_url, enabled=enabled)
    return ml_client
