"""
FloodSafe IoT Ingestion Service
================================

High-throughput ingestion endpoint for ESP32 sensor readings.

Features:
- Optional API key authentication (backwards compatible)
- Extended sensor reading fields (water_segments, distance_mm, etc.)
- Rate limiting per sensor
- Efficient bulk operations

Run: python src/main.py
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from typing import Optional
from collections import defaultdict
import os
import uuid
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Infrastructure ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/floodsafe")
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Rate Limiting ---
# Simple in-memory rate limiter (per sensor)
rate_limit_store: dict = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 30
RATE_LIMIT_MAX_REQUESTS = 2  # Max 2 readings per 30 seconds per sensor


def is_rate_limited(sensor_id: str) -> bool:
    """Check if sensor is rate limited."""
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    # Clean old entries
    rate_limit_store[sensor_id] = [
        t for t in rate_limit_store[sensor_id] if t > window_start
    ]

    if len(rate_limit_store[sensor_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return True

    rate_limit_store[sensor_id].append(now)
    return False


# --- API Key Validation ---
def validate_api_key(api_key: str) -> Optional[str]:
    """
    Validate API key and return sensor_id if valid.
    Returns None if key is invalid.
    """
    if not api_key:
        return None

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT id FROM sensors WHERE api_key_hash = :hash"),
            {"hash": key_hash}
        )
        row = result.fetchone()
        if row:
            return str(row[0])
        return None
    finally:
        db.close()


# --- API ---
app = FastAPI(
    title="FloodSafe IoT Ingestion",
    description="High-throughput sensor data ingestion for ESP32 flood sensors",
    version="1.1.0"
)


class SensorData(BaseModel):
    """Sensor reading payload from ESP32 firmware."""
    sensor_id: Optional[str] = None  # Optional if using API key auth
    water_level: float  # Legacy field - percentage or raw value
    timestamp: Optional[datetime] = None

    # Extended fields from ESP32 firmware (optional for backward compat)
    water_segments: Optional[int] = None  # 0-20 from Grove sensor
    distance_mm: Optional[float] = None  # VL53L0X raw reading
    water_height_mm: Optional[float] = None  # Calculated water height
    water_percent_strips: Optional[float] = None  # % from strip sensor
    water_percent_distance: Optional[float] = None  # % from distance sensor
    is_warning: Optional[bool] = None
    is_flood: Optional[bool] = None


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/Kubernetes probes."""
    return {"status": "healthy", "service": "iot-ingestion"}


@app.post("/ingest")
async def ingest_data(
    data: SensorData,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    High-throughput ingestion endpoint for sensor readings.

    Authentication:
    - If X-API-Key header is provided, sensor_id is derived from the key
    - If no header, sensor_id must be in the payload (backwards compatible)

    Rate Limiting:
    - Max 2 readings per 30 seconds per sensor
    - Prevents accidental flooding from misconfigured devices

    Returns:
    - 200: Reading stored successfully
    - 401: Invalid API key
    - 429: Rate limited
    - 500: Server error
    """
    # Determine sensor_id from API key or payload
    sensor_id_str = None

    if x_api_key:
        # API key authentication
        sensor_id_str = validate_api_key(x_api_key)
        if not sensor_id_str:
            raise HTTPException(status_code=401, detail="Invalid API key")
        logger.debug(f"Authenticated sensor via API key: {sensor_id_str}")
    elif data.sensor_id:
        # Legacy: sensor_id in payload
        sensor_id_str = data.sensor_id
    else:
        raise HTTPException(
            status_code=400,
            detail="Either X-API-Key header or sensor_id in payload is required"
        )

    # Validate UUID format
    try:
        sensor_uuid = uuid.UUID(sensor_id_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid sensor_id format")

    # Rate limiting
    if is_rate_limited(sensor_id_str):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited: max {RATE_LIMIT_MAX_REQUESTS} readings per {RATE_LIMIT_WINDOW_SECONDS}s"
        )

    # Store reading
    db = SessionLocal()
    try:
        reading_id = uuid.uuid4()
        reading_timestamp = data.timestamp or datetime.utcnow()

        # Build insert statement with all fields
        insert_stmt = text("""
            INSERT INTO readings (
                id, sensor_id, water_level, timestamp,
                water_segments, distance_mm, water_height_mm,
                water_percent_strips, water_percent_distance,
                is_warning, is_flood
            ) VALUES (
                :id, :sensor_id, :water_level, :timestamp,
                :water_segments, :distance_mm, :water_height_mm,
                :water_percent_strips, :water_percent_distance,
                :is_warning, :is_flood
            )
        """)

        db.execute(insert_stmt, {
            "id": reading_id,
            "sensor_id": sensor_uuid,
            "water_level": data.water_level,
            "timestamp": reading_timestamp,
            "water_segments": data.water_segments,
            "distance_mm": data.distance_mm,
            "water_height_mm": data.water_height_mm,
            "water_percent_strips": data.water_percent_strips,
            "water_percent_distance": data.water_percent_distance,
            "is_warning": data.is_warning,
            "is_flood": data.is_flood,
        })

        # Update sensor last_ping
        update_stmt = text("UPDATE sensors SET last_ping = :ts WHERE id = :sid")
        db.execute(update_stmt, {"ts": reading_timestamp, "sid": sensor_uuid})

        db.commit()

        logger.info(f"Stored reading {reading_id} for sensor {sensor_id_str}")

        return {
            "status": "stored",
            "id": str(reading_id),
            "sensor_id": sensor_id_str,
            "timestamp": reading_timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Ingestion Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Ingestion failed")
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Starting FloodSafe IoT Ingestion Service on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
