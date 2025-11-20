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

    # Reputation system
    reputation_score = Column(Integer, default=0)
    streak_days = Column(Integer, default=0)
    last_activity_date = Column(DateTime, nullable=True)

    # Privacy controls
    leaderboard_visible = Column(Boolean, default=True)
    profile_public = Column(Boolean, default=True)
    display_name = Column(String, nullable=True)

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
    downvotes = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    verified_at = Column(DateTime, nullable=True)

    # Community reporting fields
    phone_number = Column(String(20), nullable=True)
    phone_verified = Column(Boolean, default=False)
    water_depth = Column(String(20), nullable=True)  # ankle, knee, waist, impassable
    vehicle_passability = Column(String(30), nullable=True)  # all, high-clearance, none
    iot_validation_score = Column(Integer, default=0)  # 0-100
    nearby_sensor_ids = Column(String, default="[]")  # JSON array
    prophet_prediction_match = Column(Boolean, nullable=True)  # Future: Prophet integration

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

class ReputationHistory(Base):
    __tablename__ = "reputation_history"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    action = Column(String, nullable=False)
    points_change = Column(Integer, default=0)
    new_total = Column(Integer, nullable=False)
    reason = Column(String, nullable=True)
    extra_metadata = Column(String, default="{}") # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

class Badge(Base):
    __tablename__ = "badges"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    icon = Column(String, default="🏆")
    category = Column(String, default="achievement")
    requirement_type = Column(String, nullable=False)
    requirement_value = Column(Integer, nullable=False)
    points_reward = Column(Integer, default=0)
    sort_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserBadge(Base):
    __tablename__ = "user_badges"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    badge_id = Column(UUID(as_uuid=True), ForeignKey("badges.id", ondelete="CASCADE"), nullable=False)
    earned_at = Column(DateTime, default=datetime.utcnow)
