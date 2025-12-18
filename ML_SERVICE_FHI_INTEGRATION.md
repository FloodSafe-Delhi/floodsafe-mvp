# FHI Integration into ML Service Hotspots API

## Summary

Successfully integrated Flood Hazard Index (FHI) calculation into the ML service hotspots API. Each of the 62 Delhi waterlogging hotspots now returns a live FHI score based on real-time weather data.

## Files Created/Modified

### NEW Files

1. **`apps/ml-service/src/data/fhi_calculator.py`** (325 lines)
   - `FHICalculator` class with async weather data fetching
   - `FHIResult` dataclass for structured results
   - 1-hour caching to minimize API calls
   - Graceful error handling with safe defaults

2. **`apps/ml-service/test_fhi_integration.py`** (95 lines)
   - Tests FHI calculation for 4 Delhi locations
   - Validates cache functionality
   - Checks component calculations

3. **`apps/ml-service/FHI_INTEGRATION_GUIDE.md`** (Complete usage guide)
   - API documentation
   - Response examples
   - Testing procedures
   - Performance characteristics

### MODIFIED Files

1. **`apps/ml-service/src/api/hotspots.py`**
   - Added `include_fhi` query parameter (default: true)
   - Integrated FHI calculation for each hotspot
   - Updated `HotspotRiskResponse` model with `fhi` field
   - Enhanced metadata with FHI documentation

## FHI Formula

```
FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T_modifier
```

### Components (Each normalized 0.0-1.0)

| Component | Weight | Description | Data Source |
|-----------|--------|-------------|-------------|
| **P** | 35% | Precipitation forecast (24h/48h/72h weighted) | Open-Meteo hourly |
| **I** | 18% | Intensity (hourly maximum) | Open-Meteo hourly |
| **S** | 12% | Soil saturation | Open-Meteo soil_moisture_0_to_7cm |
| **A** | 12% | Antecedent conditions (3-day cumulative) | Open-Meteo hourly |
| **R** | 8% | Runoff potential (pressure-based) | Open-Meteo surface_pressure |
| **E** | 15% | Elevation risk (inverted: low=high risk) | Open-Meteo Elevation API |

**T_modifier:** 1.2 during monsoon (June-Sep), 1.0 otherwise

## API Endpoints

### Get All Hotspots with FHI

```bash
GET /api/ml/hotspots/all?include_fhi=true
```

**Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [77.2090, 28.6139]
      },
      "properties": {
        "id": 1,
        "name": "Connaught Place",
        "risk_probability": 0.651,
        "risk_level": "high",
        "fhi_score": 0.222,
        "fhi_level": "moderate",
        "fhi_color": "#eab308",
        "elevation_m": 219.0
      }
    }
  ],
  "metadata": {
    "fhi_enabled": true,
    "fhi_formula": "FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T_modifier"
  }
}
```

### Get Single Hotspot with Full FHI Details

```bash
GET /api/ml/hotspots/hotspot/1?include_fhi=true
```

**Response:**
```json
{
  "id": 1,
  "name": "Connaught Place",
  "risk_probability": 0.651,
  "risk_level": "high",
  "fhi": {
    "fhi_score": 0.222,
    "fhi_level": "moderate",
    "fhi_color": "#eab308",
    "elevation_m": 219.0,
    "components": {
      "P": 0.000,
      "I": 0.000,
      "S": 0.400,
      "A": 0.000,
      "R": 0.716,
      "E": 0.777
    },
    "monsoon_modifier": 1.0
  }
}
```

## Risk Levels

| FHI Score | Level | Color |
|-----------|-------|-------|
| 0.0 - 0.2 | Low | #22c55e (green-500) |
| 0.2 - 0.4 | Moderate | #eab308 (yellow-500) |
| 0.4 - 0.7 | High | #f97316 (orange-500) |
| 0.7 - 1.0 | Extreme | #ef4444 (red-500) |

## Testing Results

All tests passed successfully:

```
Connaught Place       - FHI: 0.222 (moderate) - Elevation: 219m
Delhi Railway Station - FHI: 0.222 (moderate) - Elevation: 219m
Minto Bridge          - FHI: 0.220 (moderate) - Elevation: 221m
ITO Junction          - FHI: 0.224 (moderate) - Elevation: 216m
```

## Verification Commands

```bash
# 1. Compile modules
cd apps/ml-service
python -m py_compile src/data/fhi_calculator.py
python -m py_compile src/api/hotspots.py

# 2. Test imports
python -c "from src.data.fhi_calculator import FHICalculator; print('OK')"
python -c "from src.api.hotspots import router; print('OK')"

# 3. Run integration test
python test_fhi_integration.py
```

## Performance

| Scenario | Time | Notes |
|----------|------|-------|
| First call (no cache) | ~500-1000ms | 2 API calls per location |
| Cached call | <1ms | Instant response |
| 62 hotspots (first load) | ~30-60s | Parallel async processing |
| 62 hotspots (cached) | <100ms | All from cache |

**Cache TTL:** 1 hour

## Error Handling

If FHI calculation fails for any reason:
- Logs warning with error details
- Returns safe defaults: `fhi_score=0.25, fhi_level="unknown", fhi_color="#9ca3af"`
- API continues to work normally
- XGBoost risk score is unaffected

## Key Features

1. **Async/Await**: Non-blocking API calls using `httpx.AsyncClient`
2. **Parallel Fetching**: Elevation and weather data fetched simultaneously
3. **Caching**: 1-hour cache reduces API load by 100% for cached locations
4. **Resilient**: None-value filtering, graceful error handling
5. **Configurable**: `include_fhi=false` for faster response when FHI not needed
6. **Transparent**: Full component breakdown available in single hotspot response

## Data Sources

All data from **Open-Meteo** (free, no API key):
- **Elevation API**: `https://api.open-meteo.com/v1/elevation`
- **Forecast API**: `https://api.open-meteo.com/v1/forecast`
  - Parameters: `hourly=precipitation,soil_moisture_0_to_7cm,surface_pressure`
  - Forecast days: 3

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           ML Service Hotspots API                       │
│  GET /api/ml/hotspots/all?include_fhi=true             │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│           FHICalculator (fhi_calculator.py)             │
│  - calculate_fhi(lat, lng)                              │
│  - 1-hour cache                                         │
└───────────┬────────────┬────────────────────────────────┘
            │            │
            ▼            ▼
┌───────────────┐  ┌──────────────────┐
│  Elevation    │  │  Weather Forecast│
│  API          │  │  API             │
│  (Open-Meteo) │  │  (Open-Meteo)    │
└───────────────┘  └──────────────────┘
```

## Integration with Existing Systems

### Relationship to XGBoost Risk Score

The ML service now provides **two independent risk metrics**:

1. **XGBoost Risk Probability** (`risk_probability`)
   - Trained on historical waterlogging data
   - 81-dimensional feature vector
   - Static + dynamic (rainfall) components

2. **FHI Score** (`fhi_score`)
   - Real-time weather-based calculation
   - 6 weighted environmental factors
   - Complements XGBoost with live conditions

**Use Cases:**
- **XGBoost**: Long-term susceptibility, infrastructure planning
- **FHI**: Short-term risk, daily operations, alerts

### Backend FHI Endpoint

Note: A separate FHI endpoint already exists at `apps/backend/src/api/rainfall.py`:
```
GET /api/rainfall/fhi?lat={lat}&lng={lng}
```

This is a **standalone endpoint** for general FHI queries. The ML service integration adds FHI **directly to hotspot responses** for convenience.

## Next Steps (Optional)

1. **Batch API Calls**: Combine multiple locations in one Open-Meteo request
2. **Redis Cache**: Share cache across ML service instances
3. **Historical FHI**: Store FHI time series for trend analysis
4. **FHI Alerts**: Trigger notifications when FHI exceeds thresholds
5. **Frontend Display**: Add FHI badges to hotspot markers on map

## Status

**COMPLETE** - Implementation verified and tested

- [x] FHI calculator module created
- [x] Hotspots API integration complete
- [x] Response models updated
- [x] Caching implemented
- [x] Error handling robust
- [x] Tests passing
- [x] Documentation complete

## References

- **Open-Meteo API**: https://open-meteo.com/en/docs
- **IMD Rainfall Standards**: https://imdpune.gov.in/
- **FHI Integration Guide**: `apps/ml-service/FHI_INTEGRATION_GUIDE.md`
- **Backend FHI Endpoint**: `apps/backend/src/api/rainfall.py`
