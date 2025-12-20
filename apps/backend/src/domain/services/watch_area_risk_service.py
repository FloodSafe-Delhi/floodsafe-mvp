"""
Watch Area Risk Assessment based on nearby hotspots.

Calculates flood risk for user watch areas by analyzing nearby waterlogging
hotspots and their current FHI (Flood Hazard Index) scores.

Part of FloodSafe - Nonprofit flood monitoring platform.
"""

from typing import List, Optional
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from geopy.distance import geodesic
import logging

from .hotspot_routing import fetch_hotspots_with_fhi
from ...infrastructure import models

logger = logging.getLogger(__name__)


@dataclass
class HotspotInWatchArea:
    """Represents a hotspot within a watch area's radius."""
    id: int
    name: str
    fhi_score: float
    fhi_level: str
    fhi_color: str
    distance_meters: float


@dataclass
class WatchAreaRiskAssessment:
    """Complete risk assessment for a watch area."""
    watch_area_id: UUID
    watch_area_name: str
    latitude: float
    longitude: float
    radius: float
    nearby_hotspots: List[HotspotInWatchArea]
    nearby_hotspots_count: int
    critical_hotspots_count: int
    average_fhi: float
    max_fhi: float
    max_fhi_level: str
    is_at_risk: bool
    risk_flag_reason: Optional[str]
    last_calculated: datetime


class WatchAreaRiskService:
    """Service for calculating flood risk in watch areas."""

    def __init__(self, db: Session):
        self.db = db

    async def calculate_risk_for_watch_area(
        self, watch_area: models.WatchArea, hotspots: List[dict]
    ) -> WatchAreaRiskAssessment:
        """
        Calculate flood risk for a single watch area.

        Args:
            watch_area: WatchArea model instance
            hotspots: List of GeoJSON features from fetch_hotspots_with_fhi()

        Returns:
            WatchAreaRiskAssessment with risk metrics and nearby hotspots
        """
        nearby_hotspots = []
        fhi_scores = []

        wa_coords = (watch_area.latitude, watch_area.longitude)

        # Find all hotspots within watch area radius
        for hotspot in hotspots:
            props = hotspot.get('properties', {})
            coords = hotspot.get('geometry', {}).get('coordinates', [])

            if len(coords) < 2:
                continue

            # GeoJSON coordinates are [lng, lat], convert to (lat, lng) for geodesic
            h_coords = (coords[1], coords[0])
            distance_m = geodesic(wa_coords, h_coords).meters

            if distance_m <= watch_area.radius:
                fhi_score = props.get('fhi_score', 0.25)
                fhi_level = props.get('fhi_level', 'moderate')

                fhi_scores.append(fhi_score)
                nearby_hotspots.append(HotspotInWatchArea(
                    id=props.get('id', 0),
                    name=props.get('name', 'Unknown'),
                    fhi_score=fhi_score,
                    fhi_level=fhi_level,
                    fhi_color=props.get('fhi_color', '#9ca3af'),
                    distance_meters=round(distance_m, 1)
                ))

        # Calculate risk metrics
        avg_fhi = sum(fhi_scores) / len(fhi_scores) if fhi_scores else 0.0
        max_fhi = max(fhi_scores) if fhi_scores else 0.0

        # Find highest risk level
        max_hotspot = max(nearby_hotspots, key=lambda h: h.fhi_score, default=None)
        max_fhi_level = max_hotspot.fhi_level if max_hotspot else 'low'

        # Count critical hotspots (HIGH or EXTREME)
        critical_hotspots = [h for h in nearby_hotspots if h.fhi_level in ['high', 'extreme']]
        critical_count = len(critical_hotspots)

        # Determine if area is at risk
        # Flag as at-risk if:
        # 1. Average FHI > 0.5 (high risk threshold)
        # 2. ANY hotspot is HIGH or EXTREME
        is_at_risk = avg_fhi > 0.5 or critical_count > 0

        # Generate risk flag reason
        risk_flag_reason = None
        if is_at_risk:
            if critical_count > 0:
                # List up to 2 critical hotspots
                critical_names = [h.name for h in critical_hotspots[:2]]
                if critical_count == 1:
                    risk_flag_reason = f"HIGH/EXTREME risk at: {critical_names[0]}"
                else:
                    risk_flag_reason = f"HIGH/EXTREME risk at: {', '.join(critical_names)}"
                    if critical_count > 2:
                        risk_flag_reason += f" (+{critical_count - 2} more)"
            else:
                risk_flag_reason = f"Average FHI ({avg_fhi:.2f}) exceeds threshold"

        return WatchAreaRiskAssessment(
            watch_area_id=watch_area.id,
            watch_area_name=watch_area.name,
            latitude=watch_area.latitude,
            longitude=watch_area.longitude,
            radius=watch_area.radius,
            nearby_hotspots=nearby_hotspots[:10],  # Limit to 10 for response size
            nearby_hotspots_count=len(nearby_hotspots),
            critical_hotspots_count=critical_count,
            average_fhi=round(avg_fhi, 3),
            max_fhi=round(max_fhi, 3),
            max_fhi_level=max_fhi_level,
            is_at_risk=is_at_risk,
            risk_flag_reason=risk_flag_reason,
            last_calculated=datetime.utcnow()
        )

    async def calculate_risk_for_user_watch_areas(
        self, user_id: UUID
    ) -> List[WatchAreaRiskAssessment]:
        """
        Calculate risk for all of a user's watch areas.

        Args:
            user_id: UUID of the user

        Returns:
            List of WatchAreaRiskAssessment for each watch area
        """
        # Fetch user's watch areas
        watch_areas = self.db.query(models.WatchArea).filter(
            models.WatchArea.user_id == user_id
        ).all()

        if not watch_areas:
            logger.info(f"No watch areas found for user {user_id}")
            return []

        # Fetch hotspots with current FHI scores
        hotspots = await fetch_hotspots_with_fhi(include_fhi=True)

        if not hotspots:
            logger.warning("No hotspot data available for risk calculation")
            # Return assessments with zero risk
            return [
                WatchAreaRiskAssessment(
                    watch_area_id=wa.id,
                    watch_area_name=wa.name,
                    latitude=wa.latitude,
                    longitude=wa.longitude,
                    radius=wa.radius,
                    nearby_hotspots=[],
                    nearby_hotspots_count=0,
                    critical_hotspots_count=0,
                    average_fhi=0.0,
                    max_fhi=0.0,
                    max_fhi_level='low',
                    is_at_risk=False,
                    risk_flag_reason=None,
                    last_calculated=datetime.utcnow()
                )
                for wa in watch_areas
            ]

        # Calculate risk for each watch area
        assessments = []
        for wa in watch_areas:
            assessment = await self.calculate_risk_for_watch_area(wa, hotspots)
            assessments.append(assessment)

        logger.info(f"Calculated risk for {len(assessments)} watch areas (user {user_id})")
        return assessments
