from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from .db import Base

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String, nullable=False, index=True)
    hotspot_id = Column(String, index=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    water_level = Column(Float, nullable=True)
    soil_moisture = Column(Float, nullable=True)
    raw = Column(Text)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    hotspot_id = Column(String, index=True)
    ds = Column(DateTime(timezone=True), nullable=False, index=True)
    yhat = Column(Float)
    yhat_lower = Column(Float)
    yhat_upper = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class RoadSegment(Base):
    __tablename__ = "road_segments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    geojson = Column(Text, nullable=True)  # GeoJSON string for the road shape
    status = Column(String, default="normal")  # normal, warning, flooded
    last_update = Column(DateTime(timezone=True), server_default=func.now())
