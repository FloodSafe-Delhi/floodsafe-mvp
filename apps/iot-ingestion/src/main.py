import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text
import os
import uuid

# --- Infrastructure (Duplicated for MVP isolation) ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/floodsafe")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Reading(Base):
    __tablename__ = "readings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sensor_id = Column(UUID(as_uuid=True)) # No FK constraint check in raw ingestion for speed
    water_level = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# --- API ---
app = FastAPI(title="FloodSafe IoT Ingestion")

class SensorData(BaseModel):
    sensor_id: str
    water_level: float
    timestamp: datetime = None

@app.post("/ingest")
async def ingest_data(data: SensorData):
    """
    High-throughput ingestion endpoint.
    """
    db = SessionLocal()
    try:
        # 1. Store Reading
        reading = Reading(
            sensor_id=uuid.UUID(data.sensor_id),
            water_level=data.water_level,
            timestamp=data.timestamp or datetime.utcnow()
        )
        db.add(reading)
        
        # 2. Update Sensor Status (Raw SQL for speed)
        # We update the last_ping time directly
        stmt = text("UPDATE sensors SET last_ping = :ts WHERE id = :sid")
        db.execute(stmt, {"ts": reading.timestamp, "sid": reading.sensor_id})
        
        db.commit()
        return {"status": "stored", "id": str(reading.id)}
    except Exception as e:
        print(f"Ingestion Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Ingestion failed")
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
