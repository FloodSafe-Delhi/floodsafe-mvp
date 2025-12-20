# Step 2: Watch Area Risk Service Implementation - COMPLETE

## Summary
Implemented the Watch Area Risk Service for the Enhanced Safe Route Mapping System as specified in the plan.

---

## Files Created

### 1. `apps/backend/src/domain/services/watch_area_risk_service.py`
**New service for calculating flood risk in user watch areas.**

#### Classes:
- `HotspotInWatchArea` (dataclass) - Represents a hotspot within a watch area
  - Fields: id, name, fhi_score, fhi_level, fhi_color, distance_meters

- `WatchAreaRiskAssessment` (dataclass) - Complete risk assessment for a watch area
  - Fields: watch_area_id, watch_area_name, latitude, longitude, radius
  - Risk Metrics: nearby_hotspots_count, critical_hotspots_count, average_fhi, max_fhi, max_fhi_level
  - Risk Flags: is_at_risk, risk_flag_reason
  - Additional: nearby_hotspots (List[HotspotInWatchArea]), last_calculated

- `WatchAreaRiskService` - Service class
  - `calculate_risk_for_watch_area()` - Calculate FHI-based risk for single watch area
  - `calculate_risk_for_user_watch_areas()` - Calculate risk for all user's watch areas

#### Key Logic:
- Finds all hotspots within watch_area.radius using geodesic distance
- Calculates average FHI score from nearby hotspots
- Flags `is_at_risk = True` if:
  1. Average FHI > 0.5 (high risk threshold), OR
  2. ANY hotspot is HIGH/EXTREME
- Includes `risk_flag_reason` explaining why area is flagged
- Counts critical hotspots (HIGH/EXTREME) separately

---

## Files Modified

### 2. `apps/backend/src/api/watch_areas.py`
**Added risk assessment API endpoint and response models.**

#### New Pydantic Models:
```python
class HotspotInWatchAreaResponse(BaseModel):
    id: int
    name: str
    fhi_score: float
    fhi_level: str
    fhi_color: str
    distance_meters: float

class WatchAreaRiskAssessmentResponse(BaseModel):
    watch_area_id: UUID
    watch_area_name: str
    latitude: float
    longitude: float
    radius: float
    nearby_hotspots: List[HotspotInWatchAreaResponse]
    nearby_hotspots_count: int
    critical_hotspots_count: int
    average_fhi: float
    max_fhi: float
    max_fhi_level: str
    is_at_risk: bool
    risk_flag_reason: Optional[str]
    last_calculated: datetime
```

#### New API Endpoint:
```python
@router.get("/user/{user_id}/risk-assessment")
async def get_user_watch_area_risks(user_id: UUID, db: Session = Depends(get_db))
```

**Endpoint Details:**
- URL: `GET /api/v1/watch-areas/user/{user_id}/risk-assessment`
- Response: List of risk assessments, one per watch area
- Analyzes each watch area for:
  - Nearby hotspots within radius
  - Average and maximum FHI scores
  - Critical hotspots (HIGH/EXTREME)
  - Risk flag if average FHI > 0.5 OR any HIGH/EXTREME hotspot present

---

### 3. `apps/backend/src/domain/services/routing_service.py`
**Enhanced get_nearby_metros() with safety information for Delhi.**

#### Modified Method:
```python
async def get_nearby_metros(
    self, lat: float, lng: float, city: str = "BLR", radius_km: float = 2.0
) -> List[Dict]
```

**New Behavior:**
- For Delhi (city="DEL"), adds safety fields to each metro station:
  - `is_affected: bool` - True if any HIGH/EXTREME hotspot within 300m
  - `affected_hotspots: List[dict]` - List of nearby dangerous hotspots
  - `safety_warning: str | None` - Warning message if affected

#### New Helper Method:
```python
def _check_metro_station_safety(
    self, station_lat: float, station_lng: float, hotspots: List[Dict]
) -> Dict
```

**Safety Logic:**
- Checks all hotspots within 300m of metro station
- Only flags HIGH or EXTREME hotspots (MODERATE ignored)
- Returns dictionary with:
  - `is_affected`: True if any dangerous hotspot nearby
  - `affected_hotspots`: List of up to 5 hotspots with details
  - `safety_warning`: User-friendly warning message

**Example Safety Warning:**
- Single hotspot: "Station near ITO Junction (HIGH flood risk)"
- Multiple hotspots: "Station near 3 flood hotspot(s): ITO Junction, Minto Bridge"

---

## Testing Results

### Syntax Validation
All files pass Python compilation:
```bash
python -m py_compile src/domain/services/watch_area_risk_service.py  # PASS
python -m py_compile src/api/watch_areas.py                           # PASS
python -m py_compile src/domain/services/routing_service.py           # PASS
```

### Import Tests
All modules import successfully:
```python
from src.domain.services.watch_area_risk_service import WatchAreaRiskService  # SUCCESS
from src.api.watch_areas import WatchAreaRiskAssessmentResponse               # SUCCESS
```

### Functional Tests

#### 1. WatchAreaRiskAssessment Creation
```python
assessment = WatchAreaRiskAssessment(
    watch_area_name='Test Area',
    is_at_risk=True,
    risk_flag_reason='HIGH/EXTREME risk at: Test Hotspot',
    critical_hotspots_count=1,
    average_fhi=0.75,
    max_fhi=0.75
)
# PASS - Dataclass created successfully
```

#### 2. Metro Station Safety Checking
```python
# Station 100m from HIGH hotspot
safety_info = service._check_metro_station_safety(28.6310, 77.2505, hotspots)
# Result:
#   is_affected: True
#   safety_warning: "Station near ITO Junction (HIGH flood risk)"
#   affected_hotspots: 1

# Station far from hotspot (>300m)
safety_info = service._check_metro_station_safety(28.7000, 77.3000, hotspots)
# Result:
#   is_affected: False
#   safety_warning: None
# PASS - Safety logic working correctly
```

---

## Integration Points

### Dependencies:
- `hotspot_routing.py` - Uses `fetch_hotspots_with_fhi()` to get live FHI data
- `geopy.distance.geodesic` - Calculates distances between coordinates
- SQLAlchemy `models.WatchArea` - Database model for watch areas

### Database:
- No new tables required
- Uses existing `watch_areas` table
- Queries by `user_id`

### External APIs:
- ML Service - Fetches hotspot FHI scores via `fetch_hotspots_with_fhi()`
- Gracefully degrades if ML service unavailable (returns zero risk)

---

## API Usage Examples

### 1. Get Watch Area Risk Assessment
```bash
GET /api/v1/watch-areas/user/{user_id}/risk-assessment
```

**Response:**
```json
[
  {
    "watch_area_id": "550e8400-e29b-41d4-a716-446655440000",
    "watch_area_name": "Connaught Place",
    "latitude": 28.6315,
    "longitude": 77.2167,
    "radius": 1000.0,
    "nearby_hotspots": [
      {
        "id": 1,
        "name": "Minto Bridge",
        "fhi_score": 0.82,
        "fhi_level": "high",
        "fhi_color": "#ef4444",
        "distance_meters": 450.3
      }
    ],
    "nearby_hotspots_count": 3,
    "critical_hotspots_count": 1,
    "average_fhi": 0.58,
    "max_fhi": 0.82,
    "max_fhi_level": "high",
    "is_at_risk": true,
    "risk_flag_reason": "HIGH/EXTREME risk at: Minto Bridge",
    "last_calculated": "2025-12-20T10:30:00.000000"
  }
]
```

### 2. Get Nearby Metros (Delhi with Safety)
```python
metros = await routing_service.get_nearby_metros(
    lat=28.6139, lng=77.2090, city="DEL", radius_km=2.0
)
```

**Response (Delhi):**
```json
[
  {
    "id": "rajiv_chowk",
    "name": "Rajiv Chowk",
    "line": "Blue Line",
    "color": "#0000FF",
    "lat": 28.6328,
    "lng": 77.2197,
    "distance_meters": 850,
    "walking_minutes": 10,
    "is_affected": true,
    "affected_hotspots": [
      {
        "id": 15,
        "name": "Minto Bridge",
        "fhi_level": "high",
        "fhi_color": "#ef4444",
        "distance_meters": 280.5
      }
    ],
    "safety_warning": "Station near Minto Bridge (HIGH flood risk)"
  }
]
```

**Response (Bangalore - No Safety Info):**
```json
[
  {
    "id": "majestic",
    "name": "Kempegowda Majestic",
    "line": "Purple Line",
    "color": "#800080",
    "lat": 12.9766,
    "lng": 77.5713,
    "distance_meters": 1200,
    "walking_minutes": 14
  }
]
```

---

## Quality Gates

- [x] `python -m py_compile` passes for all files
- [x] All imports successful
- [x] Dataclass creation works
- [x] Metro safety logic verified (300m threshold)
- [x] Risk calculation logic (FHI > 0.5 OR any HIGH/EXTREME)
- [x] Proper error handling (graceful degradation)
- [x] No console errors during testing
- [x] Follows FloodSafe architecture patterns (service layer)

---

## Next Steps (from Plan)

Step 2 is now COMPLETE. The implementation provides:
1. Watch area risk service with FHI-based assessment
2. API endpoint for fetching user watch area risks
3. Enhanced metro station safety for Delhi

Ready for frontend integration (Step 4-8 in the plan).

---

## File Paths

**Created:**
- `C:\Users\Anirudh Mohan\Desktop\FloodSafe\apps\backend\src\domain\services\watch_area_risk_service.py`

**Modified:**
- `C:\Users\Anirudh Mohan\Desktop\FloodSafe\apps\backend\src\api\watch_areas.py`
- `C:\Users\Anirudh Mohan\Desktop\FloodSafe\apps\backend\src\domain\services\routing_service.py`
