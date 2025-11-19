from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from ..models import Report, Sensor

class IPredictionService(ABC):
    """
    Interface for AI Prediction Service (e.g., Prophet).
    Future Implementation: Will consume historical sensor data and predict flood levels.
    """
    @abstractmethod
    async def predict_flood_level(self, location_lat: float, location_lng: float, hours_ahead: int) -> float:
        pass

    @abstractmethod
    async def get_risk_score(self, location_lat: float, location_lng: float) -> float:
        pass

class INotificationService(ABC):
    """
    Interface for Notification Service (SMS, WhatsApp, Push).
    Future Implementation: Will integrate with Twilio/WhatsApp Business API.
    """
    @abstractmethod
    async def send_alert(self, user_id: UUID, message: str, channel: str = "sms"):
        pass

    @abstractmethod
    async def broadcast_emergency(self, geofence_polygon: dict, message: str):
        """Send alert to all users within a specific area."""
        pass

class IRoutingService(ABC):
    """
    Interface for Routing Service.
    Future Implementation: Will integrate with OSRM/Google Routes to provide flood-safe paths.
    """
    @abstractmethod
    async def get_safe_route(self, start_lat: float, start_lng: float, end_lat: float, end_lng: float):
        pass
