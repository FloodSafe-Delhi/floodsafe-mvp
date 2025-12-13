"""
Safe Route Navigation Service
Calculates flood-aware routes using pgRouting with Mapbox fallback
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..models import RouteOption
import uuid
import json
import logging
import httpx
from geopy.distance import geodesic

from src.core.config import settings
from src.infrastructure.models import Report, Sensor
from .hotspot_routing import (
    fetch_hotspots_with_fhi,
    analyze_route_hotspots,
    calculate_hotspots_avoided,
    RouteHotspotAnalysis,
)

logger = logging.getLogger(__name__)

# Flood delay penalties (seconds) - added time when traveling through flooded areas
FLOOD_DELAY_PENALTIES = {
    "ankle": 30,        # Minor slowdown
    "knee": 120,        # Significant slowdown
    "waist": 300,       # Very slow
    "impassable": 600,  # May need detour
    "warning": 60,      # Sensor warning
    "critical": 180,    # Sensor critical
}

# Stuck time estimates (minutes) - how long you could be stuck if you try to pass
STUCK_TIME_ESTIMATES = {
    "ankle": {
        "min_wait": 5,
        "avg_wait": 15,
        "worst_case": 30,
    },
    "knee": {
        "min_wait": 15,
        "avg_wait": 45,
        "worst_case": 120,
    },
    "waist": {
        "min_wait": 30,
        "avg_wait": 90,
        "worst_case": 240,
    },
    "impassable": {
        "min_wait": 60,
        "avg_wait": 180,
        "worst_case": 480,
    },
    "warning": {
        "min_wait": 10,
        "avg_wait": 30,
        "worst_case": 60,
    },
    "critical": {
        "min_wait": 30,
        "avg_wait": 90,
        "worst_case": 180,
    },
}


class RoutingService:
    def __init__(self, db: Session):
        self.db = db
        self.mapbox_token = getattr(settings, 'MAPBOX_ACCESS_TOKEN', None)
        self.mapbox_base_url = "https://api.mapbox.com/directions/v5/mapbox"
        self.use_pgrouting = True  # Try pgRouting first, fallback to Mapbox

    async def calculate_safe_routes(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        city_code: str = 'BLR',
        mode: str = 'driving',
        max_routes: int = 3,
        avoid_ml_risk: bool = False
    ) -> Dict:
        """
        Calculate multiple route options avoiding flood zones.
        Returns routes sorted by safety score with flood zones GeoJSON.

        Tries pgRouting first, falls back to Mapbox if unavailable.
        """
        routes = []

        # Try pgRouting first
        if self.use_pgrouting:
            try:
                routes = await self._calculate_with_pgrouting(origin, destination, city_code, max_routes)
            except Exception as e:
                logger.warning(f"pgRouting failed, falling back to Mapbox: {str(e)}")
                self.use_pgrouting = False  # Disable for future requests
                self.db.rollback()  # Rollback failed transaction

        # Fallback to Mapbox if pgRouting unavailable or failed
        if not routes and self.mapbox_token:
            logger.info(f"Using Mapbox for routing (token present: {bool(self.mapbox_token)})")
            routes = await self._calculate_with_mapbox(origin, destination, mode)
        elif not routes:
            logger.warning("No routing method available (pgRouting failed, no Mapbox token)")

        # Get active flood zones for visualization
        flood_zones = await self._get_active_flood_zones()
        flood_zones_geojson = await self._flood_zones_to_geojson(flood_zones)

        return {
            "routes": routes,
            "flood_zones": flood_zones_geojson,
        }

    async def _calculate_with_pgrouting(
        self, origin: Tuple[float, float], destination: Tuple[float, float],
        city_code: str, max_routes: int
    ) -> List[dict]:
        """Calculate routes using pgRouting stored procedure."""
        routes = []

        # Safe route (heavily penalizes flooded areas - 1000x cost)
        safe_route = self._query_safe_route(origin, destination, city_code, penalty=1000)
        if safe_route:
            routes.append(self._format_route(safe_route, "safe", city_code))

        # Fast route (moderate penalty - 10x cost)
        fast_route = self._query_safe_route(origin, destination, city_code, penalty=10)
        if fast_route and (len(routes) == 0 or self._is_different_route(fast_route, safe_route)):
            routes.append(self._format_route(fast_route, "fast", city_code))

        # Balanced route (light penalty - 3x cost)
        balanced_route = self._query_safe_route(origin, destination, city_code, penalty=3)
        if balanced_route and len(routes) < 2:
            routes.append(self._format_route(balanced_route, "balanced", city_code))

        return routes

    async def _calculate_with_mapbox(
        self, origin: Tuple[float, float], destination: Tuple[float, float], mode: str
    ) -> List[dict]:
        """Calculate routes using Mapbox Directions API."""
        try:
            # Map mode to Mapbox profile
            profile_map = {
                "driving": "driving-traffic",
                "walking": "walking",
                "cycling": "cycling",
            }
            profile = profile_map.get(mode, "driving-traffic")

            # Get 3 alternate routes from Mapbox
            routes_data = await self._fetch_mapbox_routes(
                origin, destination, profile, alternatives=True, max_alternatives=2
            )

            if not routes_data or "routes" not in routes_data:
                return []

            # Get active flood zones for safety calculation
            flood_zones = await self._get_active_flood_zones()

            # Process each route and calculate safety
            processed_routes = []
            for idx, route in enumerate(routes_data["routes"][:3]):
                route_option = await self._process_mapbox_route(route, idx, flood_zones, mode)
                processed_routes.append(route_option)

            # Sort routes: safe first, then balanced, then fast
            processed_routes.sort(key=lambda r: (r["safety_score"], -r["distance_meters"]), reverse=True)

            # Assign types based on characteristics
            if len(processed_routes) >= 3:
                processed_routes[0]["type"] = "safe"
                processed_routes[1]["type"] = "balanced"
                processed_routes[2]["type"] = "fast"
            elif len(processed_routes) == 2:
                processed_routes[0]["type"] = "safe"
                processed_routes[1]["type"] = "fast"
            elif len(processed_routes) == 1:
                processed_routes[0]["type"] = "balanced"

            return processed_routes

        except Exception as e:
            logger.error(f"Error calculating routes with Mapbox: {str(e)}")
            return []

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

    # ==================== Mapbox Integration Methods ====================

    async def _fetch_mapbox_routes(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        profile: str,
        alternatives: bool = True,
        max_alternatives: int = 2,
    ) -> Dict:
        """Fetch routes from Mapbox Directions API."""
        coords = f"{origin[0]},{origin[1]};{destination[0]},{destination[1]}"
        url = f"{self.mapbox_base_url}/{profile}/{coords}"

        params = {
            "access_token": self.mapbox_token,
            "geometries": "geojson",
            "overview": "full",
        }

        if alternatives:
            params["alternatives"] = "true"
            # Note: Mapbox automatically returns up to 3 alternatives, no max_alternatives param needed

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                if response.status_code != 200:
                    logger.error(f"Mapbox API error {response.status_code}: {response.text}")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Mapbox HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Mapbox request failed: {str(e)}")
            raise

    async def _process_mapbox_route(
        self, route: Dict, index: int, flood_zones: List[Dict], mode: str
    ) -> Dict:
        """Process a single Mapbox route and calculate safety score."""
        geometry = route["geometry"]
        distance = route["distance"]  # meters
        duration = route["duration"]  # seconds

        # Calculate safety score based on flood proximity
        safety_score, flood_intersections = await self._calculate_route_safety(
            geometry["coordinates"], flood_zones
        )

        return {
            "id": str(uuid.uuid4()),
            "type": "balanced",  # Will be reassigned later
            "geometry": geometry,
            "distance_meters": distance,
            "duration_seconds": duration,
            "safety_score": safety_score,
            "flood_intersections": flood_intersections,
            "instructions": [],
        }

    async def _get_active_flood_zones(self) -> List[Dict]:
        """Get active flood locations from reports and sensors."""
        flood_zones = []

        # Get recent flood reports (last 24 hours)
        try:
            reports = (
                self.db.query(Report)
                .filter(
                    Report.water_depth.in_(["knee", "waist", "impassable"]),
                    text("created_at > NOW() - INTERVAL '24 hours'"),
                )
                .all()
            )

            for report in reports:
                flood_zones.append({
                    "type": "report",
                    "lat": report.latitude,
                    "lng": report.longitude,
                    "severity": report.water_depth,
                })
        except Exception as e:
            logger.warning(f"Error fetching reports: {str(e)}")
            self.db.rollback()  # Rollback failed transaction

        # Get active sensor warnings
        try:
            sensors = (
                self.db.query(Sensor)
                .filter(Sensor.status.in_(["warning", "critical"]))
                .all()
            )

            for sensor in sensors:
                flood_zones.append({
                    "type": "sensor",
                    "lat": sensor.latitude,
                    "lng": sensor.longitude,
                    "severity": sensor.status,
                })
        except Exception as e:
            logger.warning(f"Error fetching sensors: {str(e)}")
            self.db.rollback()  # Rollback failed transaction

        return flood_zones

    async def _get_hotspots_for_routing(
        self,
        city_code: str,
        test_fhi_override: str = None
    ) -> List[Dict]:
        """
        Fetch Delhi hotspots with live FHI scores for routing.

        Hotspots are only available for Delhi (DEL) - returns empty list for other cities.
        Used by compare_routes() to implement HARD AVOID strategy.

        Args:
            city_code: City code ('DEL' for Delhi, 'BLR' for Bangalore, etc.)
            test_fhi_override: Override FHI for testing: 'high', 'extreme', or 'mixed'

        Returns:
            List of GeoJSON features with hotspot properties including FHI
        """
        # Hotspots only available for Delhi
        if city_code.upper() not in ['DEL', 'DELHI']:
            logger.debug(f"Hotspots not available for city: {city_code}")
            return []

        try:
            hotspots = await fetch_hotspots_with_fhi(
                include_fhi=True,
                test_fhi_override=test_fhi_override
            )
            logger.info(f"Fetched {len(hotspots)} hotspots for routing in Delhi" +
                       (f" (TEST MODE: {test_fhi_override})" if test_fhi_override else ""))
            return hotspots
        except Exception as e:
            logger.warning(f"Failed to fetch hotspots for routing: {e}")
            return []

    async def _calculate_route_safety(
        self, coordinates: List[List[float]], flood_zones: List[Dict]
    ) -> Tuple[int, int]:
        """
        Calculate safety score (0-100) and count flood intersections.

        Logic:
        - 100 = No floods within 1km of route
        - 75 = Floods 500m-1km away
        - 50 = Floods 200m-500m away
        - 25 = Floods within 200m (warning)
        - 0 = Route passes through flood zone
        """
        if not flood_zones:
            return 100, 0

        min_distance_to_flood = float("inf")
        flood_intersections = 0

        # Check distance from each route point to nearest flood zone
        for coord in coordinates[::10]:  # Sample every 10th point for performance
            route_point = (coord[1], coord[0])  # (lat, lng) for geodesic

            for zone in flood_zones:
                flood_point = (zone["lat"], zone["lng"])
                distance_km = geodesic(route_point, flood_point).kilometers

                min_distance_to_flood = min(min_distance_to_flood, distance_km)

                # Count intersections (within 200m = 0.2km)
                if distance_km < 0.2:
                    flood_intersections += 1

        # Calculate safety score
        if min_distance_to_flood >= 1.0:
            safety_score = 100
        elif min_distance_to_flood >= 0.5:
            safety_score = 75
        elif min_distance_to_flood >= 0.2:
            safety_score = 50
        elif min_distance_to_flood >= 0.1:
            safety_score = 25
        else:
            safety_score = 0

        return safety_score, flood_intersections

    async def _calculate_flood_adjusted_duration(
        self, coordinates: List[List[float]], duration_seconds: float, flood_zones: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate flood-adjusted duration with time penalties for passing through floods.

        Returns:
            Tuple of (adjusted_duration_seconds, list of flood impacts)
        """
        if not flood_zones:
            return duration_seconds, []

        total_penalty = 0
        flood_impacts = []

        # Sample route at regular intervals
        for coord in coordinates[::10]:  # Sample every 10th point
            route_point = (coord[1], coord[0])  # (lat, lng) for geodesic

            for zone in flood_zones:
                flood_point = (zone["lat"], zone["lng"])
                distance_km = geodesic(route_point, flood_point).kilometers

                # If within 200m of flood zone, add penalty
                if distance_km < 0.2:
                    severity = zone.get("severity", "ankle")
                    penalty = FLOOD_DELAY_PENALTIES.get(severity, 30)
                    total_penalty += penalty
                    flood_impacts.append({
                        "lat": zone["lat"],
                        "lng": zone["lng"],
                        "severity": severity,
                        "type": zone.get("type", "report"),
                        "penalty_seconds": penalty,
                    })
                    break  # Don't double-count same zone

        adjusted_duration = duration_seconds + total_penalty
        return adjusted_duration, flood_impacts

    async def _calculate_stuck_time_risk(
        self, flood_impacts: List[Dict]
    ) -> Dict:
        """
        Calculate estimated time if user gets stuck on normal route.

        Returns risk estimates based on flood severity along route.
        """
        if not flood_impacts:
            return {
                "min_stuck_minutes": 0,
                "avg_stuck_minutes": 0,
                "worst_case_minutes": 0,
                "risk_factors": [],
                "severity_level": "none",
            }

        # Find the worst severity on the route
        severity_order = ["ankle", "knee", "waist", "impassable", "warning", "critical"]
        max_severity = "ankle"
        risk_factors = []

        for impact in flood_impacts:
            severity = impact.get("severity", "ankle")
            if severity in severity_order:
                if severity_order.index(severity) > severity_order.index(max_severity):
                    max_severity = severity

            # Track risk factors for display
            zone_type = impact.get("type", "report")
            risk_factors.append(f"{severity} ({zone_type})")

        estimates = STUCK_TIME_ESTIMATES.get(max_severity, STUCK_TIME_ESTIMATES["ankle"])

        # Multiply by number of intersections (capped at 3x)
        multiplier = min(len(flood_impacts), 3)

        return {
            "min_stuck_minutes": estimates["min_wait"] * multiplier,
            "avg_stuck_minutes": estimates["avg_wait"] * multiplier,
            "worst_case_minutes": estimates["worst_case"] * multiplier,
            "risk_factors": risk_factors[:5],  # Limit to 5 for display
            "severity_level": max_severity,
        }

    async def _get_ml_flood_zones_along_route(
        self,
        coordinates: List[List[float]],
        horizon_hours: int = 24
    ) -> List[Dict]:
        """
        Fetch ML-predicted flood zones along the route.

        This method is designed for easy integration with the ML service
        when it's fully deployed and reliable.

        Args:
            coordinates: Route coordinates [[lng, lat], ...]
            horizon_hours: Prediction horizon (default 24h)

        Returns:
            List of ML-predicted flood zones
        """
        # Check if ML routing is enabled
        if not getattr(settings, 'ML_ROUTING_ENABLED', False):
            return []

        ml_zones = []

        try:
            # Sample route at regular intervals (every ~500m)
            sample_points = self._sample_route_points(coordinates, interval_meters=500)

            async with httpx.AsyncClient(timeout=15.0) as client:
                min_confidence = getattr(settings, 'ML_MIN_CONFIDENCE', 0.7)

                # Batch request to ML service (limit for performance)
                for point in sample_points[:20]:
                    response = await client.get(
                        f"{settings.ML_SERVICE_URL}/api/v1/predictions/point",
                        params={
                            "lat": point[1],
                            "lng": point[0],
                            "horizon_days": horizon_hours // 24
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        predictions = data.get("predictions", [])

                        for pred in predictions:
                            prob = pred.get("flood_probability", 0)
                            confidence = pred.get("confidence", 0.8)

                            # Only use predictions above confidence threshold and >= 0.5 probability
                            if prob >= 0.5 and confidence >= min_confidence:
                                ml_zones.append({
                                    "type": "ml_prediction",
                                    "source": "lstm_model",
                                    "lat": point[1],
                                    "lng": point[0],
                                    "severity": self._ml_prob_to_severity(prob),
                                    "ml_probability": prob,
                                    "ml_confidence": confidence,
                                    "prediction_horizon_hours": horizon_hours,
                                })

            return ml_zones

        except Exception as e:
            logger.warning(f"ML flood zone fetch failed (graceful degradation): {e}")
            return []  # Graceful degradation - don't break routing

    def _sample_route_points(
        self, coordinates: List[List[float]], interval_meters: int = 500
    ) -> List[List[float]]:
        """Sample route points at regular intervals."""
        if not coordinates:
            return []

        sample_points = [coordinates[0]]
        accumulated_distance = 0

        for i in range(1, len(coordinates)):
            prev_point = (coordinates[i - 1][1], coordinates[i - 1][0])
            curr_point = (coordinates[i][1], coordinates[i][0])
            segment_distance = geodesic(prev_point, curr_point).meters
            accumulated_distance += segment_distance

            if accumulated_distance >= interval_meters:
                sample_points.append(coordinates[i])
                accumulated_distance = 0

        return sample_points

    def _ml_prob_to_severity(self, probability: float) -> str:
        """Map ML probability to severity level for consistent UX."""
        if probability >= 0.75:
            return "impassable"  # Extreme risk
        elif probability >= 0.5:
            return "waist"       # High risk
        elif probability >= 0.25:
            return "knee"        # Moderate risk
        else:
            return "ankle"       # Low risk

    async def _flood_zones_to_geojson(self, flood_zones: List[Dict]) -> Dict:
        """Convert flood zones to GeoJSON FeatureCollection."""
        features = []

        for zone in flood_zones:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [zone["lng"], zone["lat"]],
                },
                "properties": {
                    "type": zone["type"],
                    "severity": zone["severity"],
                },
            })

        return {"type": "FeatureCollection", "features": features}

    # ==================== Metro Station Methods ====================

    async def get_nearby_metros(
        self, lat: float, lng: float, city: str = "BLR", radius_km: float = 2.0
    ) -> List[Dict]:
        """
        Find nearby metro stations within radius.

        Args:
            lat: Latitude
            lng: Longitude
            city: City code (BLR/DEL)
            radius_km: Search radius in kilometers

        Returns:
            List of metro stations with distance and walking time
        """
        try:
            # Load metro stations GeoJSON for city
            metro_file = self._get_metro_file_path(city)

            if not metro_file.exists():
                logger.warning(f"Metro file not found: {metro_file}")
                return []

            with open(metro_file, "r", encoding="utf-8") as f:
                metro_data = json.load(f)

            user_location = (lat, lng)
            nearby_stations = []

            # Find stations within radius
            for feature in metro_data.get("features", []):
                properties = feature.get("properties", {})
                geometry = feature.get("geometry", {})

                if geometry.get("type") != "Point":
                    continue

                station_coords = geometry["coordinates"]
                station_location = (station_coords[1], station_coords[0])  # (lat, lng)

                distance_km = geodesic(user_location, station_location).kilometers

                if distance_km <= radius_km:
                    # Estimate walking time (avg 5 km/h)
                    walking_minutes = int((distance_km / 5.0) * 60)

                    nearby_stations.append({
                        "id": properties.get("id", properties.get("name", "unknown")),
                        "name": properties.get("name", "Unknown Station"),
                        "line": properties.get("line", "Unknown Line"),
                        "color": properties.get("color", "#888888"),
                        "lat": station_coords[1],
                        "lng": station_coords[0],
                        "distance_meters": int(distance_km * 1000),
                        "walking_minutes": walking_minutes,
                    })

            # Sort by distance
            nearby_stations.sort(key=lambda s: s["distance_meters"])

            return nearby_stations

        except Exception as e:
            logger.error(f"Error finding nearby metros: {str(e)}")
            return []

    def _get_metro_file_path(self, city: str) -> Path:
        """Get path to metro stations GeoJSON file."""
        metro_files = {
            "BLR": "metro-stations.geojson",
            "DEL": "delhi-metro-stations.geojson",
        }

        filename = metro_files.get(city.upper(), metro_files["BLR"])

        # Check if running in Docker (volume mounted at /frontend-public)
        docker_path = Path("/frontend-public") / filename
        if docker_path.exists():
            return docker_path

        # Fall back to relative path for local development
        base_path = Path(__file__).parent.parent.parent.parent.parent / "frontend" / "public"
        return base_path / filename

    async def _fetch_normal_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        mode: str,
        flood_zones: List[Dict],
    ) -> Optional[Dict]:
        """
        Fetch the normal (fastest) route without flood avoidance.

        Returns route with safety analysis against current flood zones.
        """
        if not self.mapbox_token:
            logger.warning("Cannot fetch normal route: Mapbox token not configured")
            return None

        try:
            profile_map = {
                "driving": "driving-traffic",
                "walking": "walking",
                "cycling": "cycling",
            }
            profile = profile_map.get(mode, "driving-traffic")

            # Get single fastest route (no alternatives)
            routes_data = await self._fetch_mapbox_routes(
                origin, destination, profile, alternatives=False
            )

            if not routes_data or "routes" not in routes_data or not routes_data["routes"]:
                return None

            route = routes_data["routes"][0]
            geometry = route["geometry"]
            coordinates = geometry["coordinates"]
            duration = route["duration"]
            distance = route["distance"]

            # Calculate safety score against flood zones
            safety_score, flood_intersections = await self._calculate_route_safety(
                coordinates, flood_zones
            )

            # Calculate flood-adjusted duration
            adjusted_duration, flood_impacts = await self._calculate_flood_adjusted_duration(
                coordinates, duration, flood_zones
            )

            return {
                "id": str(uuid.uuid4()),
                "type": "normal",
                "geometry": geometry,
                "distance_meters": distance,
                "duration_seconds": duration,
                "adjusted_duration_seconds": adjusted_duration,
                "safety_score": safety_score,
                "flood_intersections": flood_intersections,
                "flood_impacts": flood_impacts,
                "instructions": [],
            }

        except Exception as e:
            logger.error(f"Error fetching normal route: {str(e)}")
            return None

    async def compare_routes(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        city_code: str = 'BLR',
        mode: str = 'driving',
        test_fhi_override: str = None,
    ) -> Dict:
        """
        Compare normal route vs FloodSafe route.

        Returns both routes with comparison metrics including:
        - Time penalty
        - Flood zones avoided
        - Stuck time estimates
        - Risk breakdown
        - Hotspot analysis (Delhi only)

        HARD AVOID: Routes with HIGH/EXTREME FHI hotspots are flagged for rerouting.
        """
        # Get active flood zones
        flood_zones = await self._get_active_flood_zones()
        flood_zones_geojson = await self._flood_zones_to_geojson(flood_zones)

        # Get hotspots for routing (Delhi only)
        hotspots = await self._get_hotspots_for_routing(city_code, test_fhi_override)

        # Fetch normal route (fastest, may pass through floods)
        normal_route = await self._fetch_normal_route(origin, destination, mode, flood_zones)

        # Fetch FloodSafe routes (existing logic)
        safe_result = await self.calculate_safe_routes(
            origin, destination, city_code, mode, max_routes=1
        )
        floodsafe_route = safe_result["routes"][0] if safe_result.get("routes") else None

        # Build risk breakdown
        report_count = sum(1 for z in flood_zones if z.get("type") == "report")
        sensor_count = sum(1 for z in flood_zones if z.get("type") == "sensor")

        risk_breakdown = {
            "active_reports": report_count,
            "sensor_warnings": sensor_count,
            "ml_high_risk_zones": 0,  # Placeholder for ML integration
            "ml_extreme_risk_zones": 0,
            "ml_max_probability": 0.0,
            "ml_avg_probability": 0.0,
            "historical_flood_frequency": 0,
            "current_rain_intensity_mm": 0.0,
            "forecast_rain_24h_mm": 0.0,
            "total_flood_zones_avoided": normal_route.get("flood_intersections", 0) if normal_route else 0,
            "overall_risk_score": 0,
        }

        # Calculate comparison metrics
        if normal_route and floodsafe_route:
            normal_duration = normal_route.get("duration_seconds", 0)
            safe_duration = floodsafe_route.get("duration_seconds", 0) or 0
            time_penalty_seconds = safe_duration - normal_duration

            normal_distance = normal_route.get("distance_meters", 0)
            safe_distance = floodsafe_route.get("distance_meters", 0)
            distance_diff = safe_distance - normal_distance

            flood_zones_avoided = normal_route.get("flood_intersections", 0)

            # Calculate stuck time estimates
            flood_impacts = normal_route.get("flood_impacts", [])
            stuck_estimate = await self._calculate_stuck_time_risk(flood_impacts)

            # Calculate net time saved
            penalty_minutes = time_penalty_seconds / 60
            net_time_saved = {
                "vs_average_stuck": max(0, stuck_estimate["avg_stuck_minutes"] - penalty_minutes),
                "vs_worst_case": max(0, stuck_estimate["worst_case_minutes"] - penalty_minutes),
            }

            # Generate recommendation
            recommendation = self._generate_recommendation(
                penalty_minutes, stuck_estimate, flood_zones_avoided
            )
        else:
            time_penalty_seconds = 0
            distance_diff = 0
            flood_zones_avoided = 0
            stuck_estimate = {
                "min_stuck_minutes": 0,
                "avg_stuck_minutes": 0,
                "worst_case_minutes": 0,
                "risk_factors": [],
                "severity_level": "none",
            }
            net_time_saved = {"vs_average_stuck": 0, "vs_worst_case": 0}
            recommendation = "Unable to calculate route comparison"

        # Analyze hotspots for both routes (Delhi only)
        hotspot_analysis = None
        if hotspots:
            try:
                # Analyze normal route hotspots
                normal_coords = []
                if normal_route:
                    geometry = normal_route.get("geometry", {})
                    normal_coords = geometry.get("coordinates", [])

                # Analyze FloodSafe route hotspots
                safe_coords = []
                if floodsafe_route:
                    geometry = floodsafe_route.get("geometry", {})
                    safe_coords = geometry.get("coordinates", [])

                # Get analysis for normal route
                normal_hotspot_analysis = analyze_route_hotspots(normal_coords, hotspots)

                # Get analysis for safe route
                safe_hotspot_analysis = analyze_route_hotspots(safe_coords, hotspots)

                # Calculate hotspots avoided by taking FloodSafe route
                avoided = calculate_hotspots_avoided(normal_hotspot_analysis, safe_hotspot_analysis)

                # Build response structure with nearby hotspots from normal route
                hotspot_analysis = {
                    "total_hotspots_on_normal": normal_hotspot_analysis.total_hotspots_nearby,
                    "total_hotspots_on_safe": safe_hotspot_analysis.total_hotspots_nearby,
                    "hotspots_avoided": avoided["total_hotspots_avoided"],
                    "critical_hotspots_avoided": avoided["critical_hotspots_avoided"],
                    "highest_fhi_normal": normal_hotspot_analysis.highest_fhi_score,
                    "highest_fhi_safe": safe_hotspot_analysis.highest_fhi_score,
                    "normal_route_safe": normal_hotspot_analysis.route_is_safe,
                    "safe_route_safe": safe_hotspot_analysis.route_is_safe,
                    "must_reroute": normal_hotspot_analysis.must_reroute,
                    "warning_message": normal_hotspot_analysis.warning_message,
                    "nearby_hotspots": [
                        {
                            "id": h.id,
                            "name": h.name,
                            "fhi_level": h.fhi_level,
                            "fhi_color": h.fhi_color,
                            "fhi_score": h.fhi_score,
                            "distance_to_route_m": h.distance_to_route_m,
                            "estimated_delay_seconds": h.estimated_delay_seconds,
                            "must_avoid": h.must_avoid,
                        }
                        for h in normal_hotspot_analysis.nearby_hotspots[:5]  # Top 5
                    ],
                }

                # Update recommendation if hotspots require rerouting
                if normal_hotspot_analysis.must_reroute and not safe_hotspot_analysis.must_reroute:
                    recommendation = f"MUST REROUTE: {normal_hotspot_analysis.warning_message}. FloodSafe route avoids {avoided['critical_hotspots_avoided']} critical hotspot(s)."

                logger.info(f"Hotspot analysis: {normal_hotspot_analysis.total_hotspots_nearby} on normal, {safe_hotspot_analysis.total_hotspots_nearby} on safe")

            except Exception as e:
                logger.warning(f"Error analyzing hotspots: {e}")
                hotspot_analysis = None

        return {
            "normal_route": normal_route,
            "floodsafe_route": floodsafe_route,
            "time_penalty_seconds": time_penalty_seconds,
            "distance_difference_meters": distance_diff,
            "flood_zones_avoided": flood_zones_avoided,
            "risk_breakdown": risk_breakdown,
            "stuck_time_estimate": stuck_estimate,
            "net_time_saved": net_time_saved,
            "recommendation": recommendation,
            "hotspot_analysis": hotspot_analysis,
            "flood_zones": flood_zones_geojson,
        }

    def _generate_recommendation(
        self, floodsafe_penalty_min: float, stuck_estimate: Dict, flood_zones_avoided: int
    ) -> str:
        """Generate user-friendly recommendation based on comparison metrics."""
        avg_stuck = stuck_estimate["avg_stuck_minutes"]
        worst_stuck = stuck_estimate["worst_case_minutes"]
        severity = stuck_estimate["severity_level"]

        if avg_stuck == 0:
            return "Normal route appears safe - no flood zones detected"

        net_saved = avg_stuck - floodsafe_penalty_min

        if severity in ["waist", "impassable"]:
            hours = worst_stuck // 60
            return f"STRONGLY RECOMMENDED: FloodSafe route. Normal route has {severity}-level flooding - dangerous and could strand you for {hours}+ hours"

        if net_saved > 30:
            return f"FloodSafe recommended: Adds {int(floodsafe_penalty_min)} min but could save {int(net_saved)}-{int(worst_stuck - floodsafe_penalty_min)} min if stuck"

        if net_saved > 0:
            return f"FloodSafe advised: Small {int(floodsafe_penalty_min)} min penalty vs potential {int(avg_stuck)} min delay"

        if flood_zones_avoided > 0:
            return f"FloodSafe adds {int(floodsafe_penalty_min)} min to avoid {flood_zones_avoided} flood zone(s). Low flood risk, but safer option available"

        return "Both routes appear safe"

    async def calculate_walking_route(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> Optional[Dict]:
        """
        Calculate walking route between two points.
        Used for walking to metro stations.
        """
        if not self.mapbox_token:
            logger.warning("Cannot calculate walking route: Mapbox token not configured")
            return None

        try:
            routes_data = await self._fetch_mapbox_routes(
                origin, destination, "walking", alternatives=False
            )

            if not routes_data or "routes" not in routes_data or not routes_data["routes"]:
                return None

            route = routes_data["routes"][0]
            return {
                "id": str(uuid.uuid4()),
                "type": "walking",
                "geometry": route["geometry"],
                "distance_meters": route["distance"],
                "duration_seconds": route["duration"],
                "safety_score": 100,  # Walking routes default to safe
                "flood_intersections": 0,
                "instructions": [],
            }

        except Exception as e:
            logger.error(f"Error calculating walking route: {str(e)}")
            return None
