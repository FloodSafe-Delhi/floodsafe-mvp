from sqlalchemy import Column, Integer, String, DateTime, func, Text, ForeignKey
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from src.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    email = Column(String(256), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(256))
    description = Column(Text)
    geom = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    image_path = Column(String(512))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User")
