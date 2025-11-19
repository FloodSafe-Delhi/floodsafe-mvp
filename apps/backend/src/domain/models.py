from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4

class User(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    username: str
    email: str
    role: str = "user"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(from_attributes=True)

class Sensor(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    location_lat: float
    location_lng: float
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
    image_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    verified: bool = False
    
    model_config = ConfigDict(from_attributes=True)

class FloodZone(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    risk_level: str # low, medium, high, critical
    geometry: dict # GeoJSON
    
    model_config = ConfigDict(from_attributes=True)
