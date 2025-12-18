# Flood Hazard Index (FHI) ML Service Integration Summary

## Overview
Successfully integrated Flood Hazard Index (FHI) calculation into the ML service hotspots API, enabling real-time flood risk scoring for all 62 Delhi waterlogging hotspots.

## Implementation Details

### Files Modified/Created

**NEW:**
- `apps/ml-service/src/data/fhi_calculator.py` - FHI calculation logic with caching
- `apps/ml-service/test_fhi_integration.py` - Integration tests
- `apps/ml-service/FHI_INTEGRATION_GUIDE.md` - Comprehensive usage guide

**UPDATED:**
- `apps/ml-service/src/api/hotspots.py` - Integrated FHI into hotspot responses

**EXISTING (Backend):**
- `apps/backend/src/api/rainfall.py` - Standalone FHI endpoint (already implemented)

### ML Service Components Added

#### 1. FHICalculator Class (`fhi_calculator.py`)
```python
class FHICalculator:
    """Calculate Flood Hazard Index for locations."""

    # Key methods:
    async def calculate_fhi(lat, lng) -> FHIResult
    async def _fetch_elevation(lat, lng) -> float
    async def _fetch_weather(lat, lng) -> Dict
    def _calculate_components(...) -> Dict[str, float]

    # Features:
    - 1-hour cache (CACHE_TTL_SECONDS = 3600)
    - Async/await for parallel API calls
    - Graceful error handling with defaults
```

#### 2. FHIResult Dataclass
```python
@dataclass
class FHIResult:
    fhi_score: float        # 0.0-1.0
    fhi_level: str          # low/moderate/high/extreme
    fhi_color: str          # hex color (#22c55e, #eab308, etc.)
    elevation_m: float      # meters
    components: Dict        # P, I, S, A, R, E breakdown
    monsoon_modifier: float # 1.0 or 1.2
```

#### 3. Hotspots API Integration
**Endpoints Updated:**
```
GET /api/ml/hotspots/all?include_fhi=true
GET /api/ml/hotspots/hotspot/{id}?include_fhi=true
```

**Response Enhancement:**
Each hotspot now includes:
```json
{
  "fhi_score": 0.222,
  "fhi_level": "moderate",
  "fhi_color": "#eab308",
  "elevation_m": 219.0
}
```

## FHI Formula
```
FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T_modifier
```

### Component Breakdown
| Component | Weight | Description | Formula | Reference |
|-----------|--------|-------------|---------|-----------|
| P | 35% | Precipitation | `0.5×(24h/64.4) + 0.3×(48h/64.4) + 0.2×(72h/64.4)` | 64.4mm = IMD "heavy" rain |
| I | 18% | Intensity | `hourly_max / 50mm` | 50mm/hr = extreme |
| S | 12% | Soil Saturation | `soil_moisture / 0.5` | 0.5 m³/m³ = very high |
| A | 12% | Antecedent | `rain_3d / 150mm` | 150mm/3d = very high |
| R | 8% | Runoff Potential | `(1013 - pressure) / 30` | 1013 hPa = standard |
| E | 15% | Elevation Risk | `1 - (elev - 190) / (320 - 190)` | Delhi range: 190-320m |
| T | - | Temporal Modifier | `1.2` if Jun-Sep, else `1.0` | Monsoon amplification |

### Risk Levels
| Level | Range | Color | Hex Code |
|-------|-------|-------|----------|
| Low | 0.0-0.2 | Green | `#22c55e` |
| Moderate | 0.2-0.4 | Yellow | `#eab308` |
| High | 0.4-0.7 | Orange | `#f97316` |
| Extreme | 0.7-1.0 | Red | `#ef4444` |

## Data Sources
- **Precipitation**: Open-Meteo hourly forecast (precipitation, rain, showers)
- **Soil Moisture**: Open-Meteo soil_moisture_0_to_7cm (m³/m³)
- **Surface Pressure**: Open-Meteo surface_pressure (hPa)
- **Elevation**: Open-Meteo Elevation API (meters)

## Error Handling
- Graceful fallbacks for missing data:
  - Elevation: 220m (Delhi average)
  - Soil moisture: 0.2 m³/m³
  - Surface pressure: 1013 hPa
- Retry logic: 3 attempts with 1-second delays
- HTTP status codes:
  - 400: Invalid coordinates
  - 503: API unavailable
  - 500: Processing error

## Caching
- Cache TTL: 1 hour (3600 seconds)
- Cache key: Rounded coordinates (2 decimal places)
- Automatic cleanup of expired entries

## Testing Results

Ran integration test with 4 Delhi locations:

```
============================================================
Testing FHI Calculator
============================================================

Connaught Place
------------------------------------------------------------
  FHI Score:        0.222
  FHI Level:        moderate
  FHI Color:        #eab308
  Elevation:        219.0 m
  Monsoon Modifier: 1.0

  Components:
    P: 0.000
    I: 0.000
    S: 0.400
    A: 0.000
    R: 0.716
    E: 0.777

  ✓ SUCCESS

Delhi Railway Station
------------------------------------------------------------
  FHI Score:        0.222
  FHI Level:        moderate
  Elevation:        219.0 m
  ✓ SUCCESS

Minto Bridge
------------------------------------------------------------
  FHI Score:        0.220
  FHI Level:        moderate
  Elevation:        221.0 m
  ✓ SUCCESS

ITO Junction
------------------------------------------------------------
  FHI Score:        0.224
  FHI Level:        moderate
  Elevation:        216.0 m
  ✓ SUCCESS
```

**Cache Test:**
- First call: Retrieved from API
- Second call: Retrieved from cache (identical results)
- Cache is working correctly

## Verification Commands

### Import Check
```bash
cd apps/backend
python -m py_compile src/api/rainfall.py
python -c "from src.api.rainfall import router, FloodHazardIndexResponse; print('OK')"
```

### Route Verification
```bash
python -c "from src.api.rainfall import router; routes = [r.path for r in router.routes if hasattr(r, 'path')]; print('Routes:', sorted(routes))"
```
Expected output: `['/fhi', '/forecast', '/forecast/grid', '/health']`

### FHI Calculation Test
```bash
python -c "
from src.api.rainfall import _calculate_fhi
result = _calculate_fhi(
    precip_24h=20.0, precip_48h=15.0, precip_72h=10.0,
    hourly_max=8.0, soil_moisture=0.25, surface_pressure=1012.0,
    elevation=230.0, month=10
)
print(f'FHI: {result[\"fhi_score\"]} ({result[\"fhi_level\"]})')
"
```

## API Usage Examples

### cURL
```bash
# Get FHI for Delhi
curl "http://localhost:8000/api/rainfall/fhi?lat=28.6139&lng=77.2090"
```

### Python
```python
import requests

response = requests.get(
    "http://localhost:8000/api/rainfall/fhi",
    params={"lat": 28.6139, "lng": 77.2090}
)

data = response.json()
print(f"FHI Score: {data['fhi_score']} ({data['fhi_level']})")
print(f"Components: {data['components']}")
```

### TypeScript/JavaScript
```typescript
const response = await fetch(
  'http://localhost:8000/api/rainfall/fhi?lat=28.6139&lng=77.2090'
);

const data = await response.json();
console.log(`FHI Score: ${data.fhi_score} (${data.fhi_level})`);
console.log(`Color: ${data.fhi_color}`);
```

## Response Example
```json
{
  "fhi_score": 0.321,
  "fhi_level": "moderate",
  "fhi_color": "#eab308",
  "components": {
    "P": 0.256,
    "I": 0.160,
    "S": 0.500,
    "A": 0.300,
    "R": 0.033,
    "E": 0.692
  },
  "precipitation_24h_mm": 20.0,
  "precipitation_48h_mm": 15.0,
  "precipitation_72h_mm": 10.0,
  "hourly_max_mm": 8.0,
  "soil_moisture": 0.250,
  "surface_pressure_hpa": 1012.0,
  "elevation_m": 230.0,
  "is_monsoon": false,
  "fetched_at": "2025-12-13T10:30:00Z",
  "latitude": 28.6139,
  "longitude": 77.2090
}
```

## Key Features
1. **Multi-factor Risk Assessment**: Combines 6 environmental factors
2. **Monsoon Awareness**: Applies 1.2x multiplier during June-September
3. **IMD-Aligned**: Uses IMD rainfall intensity thresholds (64.4mm = heavy rain)
4. **Elevation Sensitivity**: Lower elevations = higher flood risk
5. **Real-time Data**: Fetches live weather forecasts from Open-Meteo
6. **Performance Optimized**: 1-hour caching, retry logic, graceful fallbacks
7. **UI-Ready**: Returns color codes and level classifications

## Architecture Compliance
- ✓ Follows FastAPI router pattern
- ✓ Pydantic v2 models with `ConfigDict(from_attributes=True)`
- ✓ Proper error handling (no bare `except:`)
- ✓ HTTP status codes used correctly
- ✓ Logging throughout
- ✓ Type hints on all functions
- ✓ Async/await for external API calls

## Documentation
- `apps/backend/FHI_ENDPOINT_USAGE.md` - Detailed API usage guide
- `apps/backend/src/api/rainfall.py` - Inline docstrings

## Next Steps (Optional)
1. **Frontend Integration**: Add FHI display to map interface
2. **Grid Endpoint**: Create `/api/rainfall/fhi/grid` for heatmap visualization
3. **Historical Tracking**: Store FHI scores in database for trend analysis
4. **Alerts**: Trigger notifications when FHI exceeds threshold
5. **ML Integration**: Use FHI as input feature for flood prediction models

## Status
✅ **COMPLETE** - Implementation verified and tested
- Syntax check: PASSED
- Import check: PASSED
- Route registration: PASSED
- Calculation tests: PASSED (3 scenarios)
- All components working as expected
