from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4

# ============================================================================
# BASE MODELS (Full entity representations matching infrastructure layer)
# ============================================================================

class User(BaseModel):
    """User entity with gamification features"""
    id: UUID = Field(default_factory=uuid4)
    username: str
    email: str
    role: str = "user"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Gamification fields
    points: int = 0
    level: int = 1
    reports_count: int = 0
    verified_reports_count: int = 0
    badges: str = "[]"  # JSON string array

    model_config = ConfigDict(from_attributes=True)


class Sensor(BaseModel):
    """IoT Sensor entity"""
    id: UUID = Field(default_factory=uuid4)
    location_lat: float
    location_lng: float
    status: str = "active"
    last_ping: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class Reading(BaseModel):
    """Water level reading from sensor"""
    id: UUID = Field(default_factory=uuid4)
    sensor_id: UUID
    water_level: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class Report(BaseModel):
    """Flood report from user"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    description: str
    location_lat: float
    location_lng: float
    media_url: Optional[str] = None
    media_type: str = "image"  # image, video
    media_metadata: str = "{}"  # JSON string
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    verified: bool = False
    verification_score: int = 0  # Computed from upvotes/user reputation
    upvotes: int = 0

    model_config = ConfigDict(from_attributes=True)


class FloodZone(BaseModel):
    """Flood risk zone polygon"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    risk_level: str  # low, medium, high, critical
    geometry: dict  # GeoJSON

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# REQUEST DTOs (For API input validation)
# ============================================================================

class UserCreate(BaseModel):
    """Request DTO for user registration"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    role: str = "user"

    model_config = ConfigDict(from_attributes=True)


class ReportCreate(BaseModel):
    """Request DTO for creating flood report"""
    user_id: UUID
    description: str = Field(..., min_length=10, max_length=500)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    media_type: str = "image"

    model_config = ConfigDict(from_attributes=True)


class SensorReading(BaseModel):
    """Request DTO for IoT sensor data ingestion"""
    sensor_id: UUID
    water_level: float = Field(..., ge=0)  # Water level in meters
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# RESPONSE DTOs (For API output)
# ============================================================================

class UserResponse(BaseModel):
    """Response DTO for user data (excludes sensitive fields)"""
    id: UUID
    username: str
    email: str
    role: str
    points: int
    level: int
    reports_count: int
    verified_reports_count: int
    badges: List[str]  # Parsed JSON array

    model_config = ConfigDict(from_attributes=True)


class ReportResponse(BaseModel):
    """Response DTO for flood report"""
    id: UUID
    description: str
    latitude: float
    longitude: float
    media_url: Optional[str]
    verified: bool
    verification_score: int
    upvotes: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class SensorResponse(BaseModel):
    """Response DTO for sensor data"""
    id: UUID
    latitude: float
    longitude: float
    status: str
    last_ping: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class ReadingResponse(BaseModel):
    """Response DTO for water level reading"""
    id: UUID
    sensor_id: UUID
    water_level: float
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
