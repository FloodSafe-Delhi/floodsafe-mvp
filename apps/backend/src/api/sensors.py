from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
import logging
import secrets
import hashlib

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import SensorCreate, SensorResponse, SensorReading, ReadingResponse, ApiKeyResponse
from .deps import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=SensorResponse)
def create_sensor(sensor: SensorCreate, db: Session = Depends(get_db)):
    """
    Register a new IoT sensor.
    """
    try:
        # PostGIS Point: POINT(lng lat)
        location_wkt = f"POINT({sensor.location_lng} {sensor.location_lat})"
        
        new_sensor = models.Sensor(
            location=location_wkt,
            status=sensor.status
        )
        
        db.add(new_sensor)
        db.commit()
        db.refresh(new_sensor)
        
        return new_sensor
    except Exception as e:
        logger.error(f"Error creating sensor: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create sensor")

@router.get("/", response_model=List[SensorResponse])
def list_sensors(db: Session = Depends(get_db)):
    """
    List all sensors.
    """
    try:
        sensors = db.query(models.Sensor).all()
        return sensors
    except Exception as e:
        logger.error(f"Error listing sensors: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sensors")

@router.post("/{sensor_id}/readings", response_model=ReadingResponse)
def record_reading(sensor_id: UUID, reading: SensorReading, db: Session = Depends(get_db)):
    """
    Record a new water level reading for a sensor.
    """
    try:
        # Verify sensor exists
        sensor = db.query(models.Sensor).filter(models.Sensor.id == sensor_id).first()
        if not sensor:
            raise HTTPException(status_code=404, detail="Sensor not found")

        new_reading = models.Reading(
            sensor_id=sensor_id,
            water_level=reading.water_level,
            timestamp=reading.timestamp
        )
        
        # Update sensor last_ping
        sensor.last_ping = reading.timestamp
        
        db.add(new_reading)
        db.commit()
        db.refresh(new_reading)
        
        return new_reading
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording reading: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to record reading")

@router.get("/{sensor_id}/readings", response_model=List[ReadingResponse])
def get_readings(sensor_id: UUID, db: Session = Depends(get_db)):
    """
    Get reading history for a sensor.
    """
    try:
        readings = db.query(models.Reading).filter(
            models.Reading.sensor_id == sensor_id
        ).order_by(models.Reading.timestamp.desc()).limit(100).all()
        return readings
    except Exception as e:
        logger.error(f"Error fetching readings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch readings")


@router.post("/{sensor_id}/generate-key", response_model=ApiKeyResponse)
async def generate_api_key(
    sensor_id: UUID,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Generate an API key for a sensor.

    - The API key is shown ONCE in the response
    - Only the SHA256 hash is stored in the database
    - If called again, generates a NEW key (invalidates old one)
    - Associates sensor with current user if not already owned

    Security:
    - Requires authentication
    - Only sensor owner can regenerate key
    - Key is cryptographically random (256 bits)
    """
    try:
        # Fetch sensor
        sensor = db.query(models.Sensor).filter(models.Sensor.id == sensor_id).first()
        if not sensor:
            raise HTTPException(status_code=404, detail="Sensor not found")

        # Check ownership
        if sensor.user_id is not None and sensor.user_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized - sensor belongs to another user"
            )

        # Assign ownership if not set
        if sensor.user_id is None:
            sensor.user_id = current_user.id
            logger.info(f"Assigned sensor {sensor_id} to user {current_user.id}")

        # Generate cryptographically secure API key (32 bytes = 256 bits)
        api_key = secrets.token_urlsafe(32)

        # Store only the hash
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        sensor.api_key_hash = key_hash

        db.commit()

        logger.info(f"Generated new API key for sensor {sensor_id}")

        return ApiKeyResponse(
            sensor_id=sensor_id,
            api_key=api_key,
            message="Save this API key securely - it cannot be retrieved again. "
                    "Use it in the X-API-Key header when sending sensor readings."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating API key: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to generate API key")


@router.patch("/{sensor_id}/name")
async def update_sensor_name(
    sensor_id: UUID,
    name: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Update the human-readable name of a sensor.
    Only the sensor owner can update the name.
    """
    try:
        sensor = db.query(models.Sensor).filter(models.Sensor.id == sensor_id).first()
        if not sensor:
            raise HTTPException(status_code=404, detail="Sensor not found")

        # Check ownership
        if sensor.user_id is None:
            # Allow claiming unowned sensor
            sensor.user_id = current_user.id
        elif sensor.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Validate name length
        if len(name) > 100:
            raise HTTPException(status_code=400, detail="Name must be 100 characters or less")

        sensor.name = name
        db.commit()

        return {"sensor_id": str(sensor_id), "name": name, "message": "Sensor name updated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating sensor name: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update sensor name")
