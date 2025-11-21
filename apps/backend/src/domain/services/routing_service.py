"""
Safe Route Navigation Service
Calculates flood-aware routes using pgRouting
"""

from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..models import RouteOption
import uuid
import json


class RoutingService:
    def __init__(self, db: Session):
        self.db = db

    async def calculate_safe_routes(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        city_code: str = 'BLR',
        mode: str = 'driving',
        max_routes: int = 3
    ) -> List[dict]:
        """
        Calculate multiple route options avoiding flood zones.
        Returns routes sorted by safety score.
        """
        routes = []

        # Safe route (heavily penalizes flooded areas - 1000x cost)
        safe_route = self._query_safe_route(origin, destination, city_code, penalty=1000)
        if safe_route:
            routes.append(self._format_route(safe_route, "safe", city_code))

        # Fast route (moderate penalty - 10x cost)
        fast_route = self._query_safe_route(origin, destination, city_code, penalty=10)
        if fast_route and len(routes) == 0 or self._is_different_route(fast_route, safe_route):
            routes.append(self._format_route(fast_route, "fast", city_code))

        # Balanced route (light penalty - 3x cost)
        balanced_route = self._query_safe_route(origin, destination, city_code, penalty=3)
        if balanced_route and len(routes) < 2:
            routes.append(self._format_route(balanced_route, "balanced", city_code))

        return routes

    def _query_safe_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        city_code: str,
        penalty: int
    ):
        query = text("""
            SELECT * FROM calculate_safe_route(
                :start_lon, :start_lat, :end_lon, :end_lat, :city_code, :penalty
            )
        """)

        try:
            result = self.db.execute(query, {
                "start_lon": origin[0],
                "start_lat": origin[1],
                "end_lon": destination[0],
                "end_lat": destination[1],
                "city_code": city_code,
                "penalty": penalty
            }).fetchall()
            return result
        except Exception as e:
            print(f"Routing error: {e}")
            return None

    def _format_route(self, route_data, route_type: str, city_code: str) -> dict:
        if not route_data:
            return None

        # Extract coordinates from geometry
        coordinates = []
        total_distance = 0
        flood_intersections = 0
        max_severity = 0

        for row in route_data:
            if row.geometry:
                # Extract LineString coordinates
                geom_text = self.db.scalar(text("SELECT ST_AsGeoJSON(:geom)"), {"geom": row.geometry})
                geom_json = json.loads(geom_text) if geom_text else None
                if geom_json and geom_json.get("coordinates"):
                    coordinates.extend(geom_json["coordinates"])

            total_distance += row.cost
            if row.intersects_flood:
                flood_intersections += 1
                max_severity = max(max_severity, row.flood_severity or 0)

        # Compute safety score (0-100)
        safety_score = self._compute_safety_score(
            flood_intersections,
            max_severity,
            total_distance
        )

        return {
            "id": str(uuid.uuid4()),
            "type": route_type,
            "city_code": city_code,
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "distance_meters": total_distance,
            "flood_intersections": flood_intersections,
            "safety_score": safety_score,
            "risk_level": "low" if safety_score > 70 else "medium" if safety_score > 40 else "high",
            "instructions": None
        }

    def _compute_safety_score(
        self,
        flood_count: int,
        max_severity: int,
        distance: float
    ) -> int:
        base_score = 100
        base_score -= flood_count * 15
        base_score -= max_severity * 0.5
        return max(0, min(100, int(base_score)))

    def _is_different_route(self, route1, route2) -> bool:
        """Check if two routes are significantly different"""
        if not route1 or not route2:
            return True

        edges1 = {row.edge for row in route1 if row.edge != -1}
        edges2 = {row.edge for row in route2 if row.edge != -1}

        if not edges1 or not edges2:
            return True

        # If less than 30% overlap, consider them different
        overlap = len(edges1 & edges2) / max(len(edges1), len(edges2))
        return overlap < 0.7
