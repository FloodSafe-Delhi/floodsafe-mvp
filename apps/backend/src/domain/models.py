from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
    status: str = "active"
    last_ping: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

class Reading(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    sensor_id: UUID
    water_level: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(from_attributes=True)

class Report(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    description: str
    location_lat: float
    location_lng: float
    media_url: Optional[str] = None
    media_type: str = "image" # image, video
    media_metadata: dict = {} # EXIF, Geotags, Device Info
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    verified: bool = False
    verification_score: int = 0 # Computed from upvotes/user reputation
    upvotes: int = 0
    
    model_config = ConfigDict(from_attributes=True)

class FloodZone(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    risk_level: str # low, medium, high, critical
    geometry: dict # GeoJSON
    
    model_config = ConfigDict(from_attributes=True)
