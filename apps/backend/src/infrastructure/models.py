from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, object_session
from sqlalchemy.ext.hybrid import hybrid_property
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_X, ST_Y
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

    # Gamification
    points = Column(Integer, default=0)
    level = Column(Integer, default=1)
    reports_count = Column(Integer, default=0)
    verified_reports_count = Column(Integer, default=0)
    badges = Column(String, default="[]") # JSON string

    # Profile fields
    phone = Column(String, nullable=True)
    profile_photo_url = Column(String, nullable=True)
    language = Column(String, default="english")

    # Notification preferences
    notification_push = Column(Boolean, default=True)
    notification_sms = Column(Boolean, default=True)
    notification_whatsapp = Column(Boolean, default=False)
    notification_email = Column(Boolean, default=True)
    alert_preferences = Column(String, default='{"watch":true,"advisory":true,"warning":true,"emergency":true}') # JSON string

class Sensor(Base):
    __tablename__ = "sensors"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location = Column(Geometry('POINT', srid=4326))
    status = Column(String, default="active")
    last_ping = Column(DateTime, nullable=True)
    readings = relationship("Reading", back_populates="sensor")

    @hybrid_property
    def latitude(self):
        """Extract latitude from PostGIS POINT geometry"""
        if self.location is not None:
            session = object_session(self)
            if session:
                result = session.scalar(ST_Y(self.location))
                return float(result) if result is not None else None
        return None

    @hybrid_property
    def longitude(self):
        """Extract longitude from PostGIS POINT geometry"""
        if self.location is not None:
            session = object_session(self)
            if session:
                result = session.scalar(ST_X(self.location))
                return float(result) if result is not None else None
        return None

class Reading(Base):
    __tablename__ = "readings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sensor_id = Column(UUID(as_uuid=True), ForeignKey("sensors.id"))
    water_level = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sensor = relationship("Sensor", back_populates="readings")

class Report(Base):
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

    @hybrid_property
    def latitude(self):
        """Extract latitude from PostGIS POINT geometry"""
        if self.location is not None:
            session = object_session(self)
            if session:
                result = session.scalar(ST_Y(self.location))
                return float(result) if result is not None else None
        return None

    @hybrid_property
    def longitude(self):
        """Extract longitude from PostGIS POINT geometry"""
        if self.location is not None:
            session = object_session(self)
            if session:
                result = session.scalar(ST_X(self.location))
                return float(result) if result is not None else None
        return None

class FloodZone(Base):
    __tablename__ = "flood_zones"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    risk_level = Column(String)
    geometry = Column(Geometry('POLYGON', srid=4326))

class WatchArea(Base):
    __tablename__ = "watch_areas"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    name = Column(String)
    location = Column(Geometry('POINT', srid=4326))
    radius = Column(Float, default=1000.0) # meters
    created_at = Column(DateTime, default=datetime.utcnow)

    @hybrid_property
    def latitude(self):
        """Extract latitude from PostGIS POINT geometry"""
        if self.location is not None:
            session = object_session(self)
            if session:
                result = session.scalar(ST_Y(self.location))
                return float(result) if result is not None else None
        return None

    @hybrid_property
    def longitude(self):
        """Extract longitude from PostGIS POINT geometry"""
        if self.location is not None:
            session = object_session(self)
            if session:
                result = session.scalar(ST_X(self.location))
                return float(result) if result is not None else None
        return None
