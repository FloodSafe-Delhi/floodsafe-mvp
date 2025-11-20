from pydantic import BaseModel, Field, ConfigDict, field_validator
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

    # Reputation system
    reputation_score: int = 0
    streak_days: int = 0
    last_activity_date: Optional[datetime] = None

    # Privacy controls
    leaderboard_visible: bool = True
    profile_public: bool = True
    display_name: Optional[str] = None

    # Profile fields
    phone: Optional[str] = None
    profile_photo_url: Optional[str] = None
    language: str = "english"

    # Notification preferences
    notification_push: bool = True
    notification_sms: bool = True
    notification_whatsapp: bool = False
    notification_email: bool = True
    alert_preferences: str = '{"watch":true,"advisory":true,"warning":true,"emergency":true}'  # JSON string

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
    downvotes: int = 0
    quality_score: float = 0.0
    verified_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class FloodZone(BaseModel):
    """Flood risk zone polygon"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    risk_level: str  # low, medium, high, critical
    geometry: dict  # GeoJSON

    model_config = ConfigDict(from_attributes=True)


class WatchArea(BaseModel):
    """User-defined area to monitor for flood alerts"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    name: str
    location_lat: float
    location_lng: float
    radius: float = 1000.0  # meters
    created_at: datetime = Field(default_factory=datetime.utcnow)

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


class UserUpdate(BaseModel):
    """Request DTO for updating user profile"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    phone: Optional[str] = None
    profile_photo_url: Optional[str] = None
    language: Optional[str] = None
    notification_push: Optional[bool] = None
    notification_sms: Optional[bool] = None
    notification_whatsapp: Optional[bool] = None
    notification_email: Optional[bool] = None
    alert_preferences: Optional[str] = None  # JSON string

    # Privacy controls
    leaderboard_visible: Optional[bool] = None
    profile_public: Optional[bool] = None
    display_name: Optional[str] = Field(None, min_length=3, max_length=50)

    model_config = ConfigDict(from_attributes=True)


class WatchAreaCreate(BaseModel):
    """Request DTO for creating a watch area"""
    user_id: UUID
    name: str = Field(..., min_length=3, max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius: float = Field(default=1000.0, ge=100, le=10000)  # 100m to 10km

    model_config = ConfigDict(from_attributes=True)


class ReportCreate(BaseModel):
    """Request DTO for creating flood report"""
    user_id: UUID
    description: str = Field(..., min_length=10, max_length=500)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    media_type: str = "image"

    model_config = ConfigDict(from_attributes=True)


class SensorCreate(BaseModel):
    """Request DTO for registering a new sensor"""
    location_lat: float = Field(..., ge=-90, le=90)
    location_lng: float = Field(..., ge=-180, le=180)
    status: str = "active"

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
    created_at: datetime
    points: int
    level: int
    reports_count: int
    verified_reports_count: int
    badges: List[str]  # Parsed JSON array

    # Reputation system
    reputation_score: int = 0
    streak_days: int = 0
    last_activity_date: Optional[datetime] = None

    # Privacy controls
    leaderboard_visible: bool = True
    profile_public: bool = True
    display_name: Optional[str] = None

    # Profile fields
    phone: Optional[str] = None
    profile_photo_url: Optional[str] = None
    language: str = "english"

    # Notification preferences
    notification_push: bool = True
    notification_sms: bool = True
    notification_whatsapp: bool = False
    notification_email: bool = True
    alert_preferences: dict  # Parsed JSON object

    @field_validator('badges', mode='before')
    @classmethod
    def parse_badges(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v

    @field_validator('alert_preferences', mode='before')
    @classmethod
    def parse_alert_preferences(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {"watch": True, "advisory": True, "warning": True, "emergency": True}
        return v

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


class WatchAreaResponse(BaseModel):
    """Response DTO for watch area"""
    id: UUID
    user_id: UUID
    name: str
    latitude: float
    longitude: float
    radius: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
