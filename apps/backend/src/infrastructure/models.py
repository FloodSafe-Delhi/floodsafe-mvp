from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from .database import Base
import uuid
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)

class Sensor(Base):
    __tablename__ = "sensors"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location = Column(Geometry('POINT', srid=4326))
    status = Column(String, default="active")
    last_ping = Column(DateTime, nullable=True)
    readings = relationship("Reading", back_populates="sensor")

class Reading(Base):
    __tablename__ = "readings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sensor_id = Column(UUID(as_uuid=True), ForeignKey("sensors.id"))
    water_level = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sensor = relationship("Sensor", back_populates="readings")

class Report(BaseModel):
    __tablename__ = "reports"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    description = Column(String)
    location = Column(Geometry('POINT', srid=4326))
    media_url = Column(String, nullable=True)
    media_type = Column(String, default="image")
    media_metadata = Column(String, default="{}") # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    verified = Column(Boolean, default=False)
    verification_score = Column(Integer, default=0)
    upvotes = Column(Integer, default=0)

class FloodZone(Base):
    __tablename__ = "flood_zones"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    risk_level = Column(String)
    geometry = Column(Geometry('POLYGON', srid=4326))
