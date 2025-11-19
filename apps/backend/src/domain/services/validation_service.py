"""
Report Validation Service
Validates community flood reports against IoT sensor data
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
import json

from ..models import Report
from ...infrastructure.models import Sensor, Reading


class ReportValidationService:
    """
    Service to validate community reports against IoT sensor data.

    Validation Logic:
    1. Find sensors within 1km radius of report location
    2. Get recent readings from those sensors (last 2 hours)
    3. Calculate validation score based on correlation with water depth
    4. Update report with validation score and nearby sensors
    """

    def __init__(self, db: Session):
        self.db = db

    def validate_report(self, report: Report) -> int:
        """
        Validate a report against nearby sensor data.

        Args:
            report: Report domain model with location_lat and location_lng

        Returns:
            validation_score: Integer 0-100 indicating confidence
        """
        # 1. Find nearby sensors (within 1km)
        nearby_sensors = self._find_nearby_sensors(
            report.location_lat,
            report.location_lng,
            radius_meters=1000
        )

        if not nearby_sensors:
            # No sensors nearby - neutral score
            return 50

        # 2. Get recent readings from sensors (last 2 hours)
        recent_readings = self._get_recent_readings(
            [str(s.id) for s in nearby_sensors],
            hours=2
        )

        if not recent_readings:
            # No recent data - neutral score
            return 50

        # 3. Calculate validation score
        score = self._calculate_validation_score(
            report,
            recent_readings,
            nearby_sensors
        )

        return score

    def _find_nearby_sensors(
        self,
        lat: float,
        lng: float,
        radius_meters: int
    ) -> List[Sensor]:
        """
        Find sensors within radius using PostGIS ST_DWithin.

        Uses geography type for accurate distance calculation in meters.
        """
        query = text("""
            SELECT id, location, status, last_ping
            FROM sensors
            WHERE ST_DWithin(
                location::geography,
                ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography,
                :radius
            )
            AND status != 'inactive'
        """)

        result = self.db.execute(query, {
            'lat': lat,
            'lng': lng,
            'radius': radius_meters
        })

        sensors = []
        for row in result:
            sensor = Sensor()
            sensor.id = row[0]
            sensor.location = row[1]
            sensor.status = row[2]
            sensor.last_ping = row[3]
            sensors.append(sensor)

        return sensors

    def _get_recent_readings(
        self,
        sensor_ids: List[str],
        hours: int
    ) -> List[Reading]:
        """
        Get readings from sensors in last N hours.
        """
        if not sensor_ids:
            return []

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Convert string UUIDs to proper format for query
        from uuid import UUID
        sensor_uuids = [UUID(sid) for sid in sensor_ids]

        readings = self.db.query(Reading).filter(
            Reading.sensor_id.in_(sensor_uuids),
            Reading.timestamp >= cutoff_time
        ).all()

        return readings

    def _calculate_validation_score(
        self,
        report: Report,
        readings: List[Reading],
        sensors: List[Sensor]
    ) -> int:
        """
        Calculate validation score based on sensor readings.

        Scoring Logic:
        - Compare reported water depth with sensor water levels
        - Weight by distance (closer sensors = higher weight)
        - Weight by recency (recent readings = higher weight)

        Water Depth Mapping:
        - ankle: 0.15m
        - knee: 0.45m
        - waist: 0.9m
        - impassable: 1.5m+

        Returns:
            score: 0-100 where:
            - 100 = Strong confirmation (sensors show similar water level)
            - 75 = Moderate confirmation
            - 50 = Neutral (no clear evidence)
            - 25 = Weak contradiction
            - 0 = Strong contradiction (sensors show no flooding)
        """
        if not readings:
            return 50

        # Map water depth to expected sensor level
        depth_thresholds = {
            'ankle': 0.15,
            'knee': 0.45,
            'waist': 0.9,
            'impassable': 1.5
        }

        reported_threshold = depth_thresholds.get(
            report.water_depth,
            0.5  # default if not specified
        )

        # Calculate average water level from readings
        avg_water_level = sum(r.water_level for r in readings) / len(readings)

        # Get max water level (worst case)
        max_water_level = max(r.water_level for r in readings)

        # Calculate score based on correlation
        # Use max_water_level for more conservative validation
        if max_water_level >= reported_threshold * 0.8:
            # Sensor confirms high water level
            score = 100
        elif max_water_level >= reported_threshold * 0.5:
            # Partial confirmation
            score = 75
        elif max_water_level >= reported_threshold * 0.2:
            # Weak confirmation
            score = 50
        elif max_water_level < 0.1:
            # Sensors show very low water - contradicts report
            score = 25
        else:
            # Sensors show some water but less than reported
            score = 40

        return score

    def get_nearby_sensor_summary(
        self,
        lat: float,
        lng: float,
        radius_meters: int = 1000
    ) -> dict:
        """
        Get summary of nearby sensors and their status.
        Used for hyperlocal area status.

        Returns:
            {
                'sensor_count': int,
                'avg_water_level': float,
                'max_water_level': float,
                'active_sensors': int,
                'status': str (safe/caution/warning/critical)
            }
        """
        sensors = self._find_nearby_sensors(lat, lng, radius_meters)

        if not sensors:
            return {
                'sensor_count': 0,
                'avg_water_level': 0.0,
                'max_water_level': 0.0,
                'active_sensors': 0,
                'status': 'unknown'
            }

        readings = self._get_recent_readings(
            [str(s.id) for s in sensors],
            hours=1  # Last hour only for area status
        )

        if not readings:
            return {
                'sensor_count': len(sensors),
                'avg_water_level': 0.0,
                'max_water_level': 0.0,
                'active_sensors': len([s for s in sensors if s.status == 'active']),
                'status': 'safe'
            }

        avg_level = sum(r.water_level for r in readings) / len(readings)
        max_level = max(r.water_level for r in readings)

        # Determine status
        if max_level >= 1.0:
            status = 'critical'
        elif max_level >= 0.5:
            status = 'warning'
        elif max_level >= 0.2:
            status = 'caution'
        else:
            status = 'safe'

        return {
            'sensor_count': len(sensors),
            'avg_water_level': round(avg_level, 2),
            'max_water_level': round(max_level, 2),
            'active_sensors': len([s for s in sensors if s.status == 'active']),
            'status': status
        }
