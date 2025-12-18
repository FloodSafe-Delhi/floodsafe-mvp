# Flood Hazard Index (FHI) Endpoint

## Overview
The FHI endpoint calculates a real-time Flood Hazard Index score (0-1) using multiple environmental factors and Open-Meteo weather data.

## Endpoint
```
GET /api/rainfall/fhi?lat={latitude}&lng={longitude}
```

## Query Parameters
- `lat` (required): Latitude (-90 to 90)
- `lng` (required): Longitude (-180 to 180)

## Response Format
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

## FHI Formula
```
FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T_modifier
```

### Components (all normalized to 0-1):
- **P (35%)**: Precipitation - `0.5×(24h/64.4) + 0.3×(48h/64.4) + 0.2×(72h/64.4)`
- **I (18%)**: Intensity - `hourly_max / 50mm`
- **S (12%)**: Soil Saturation - `soil_moisture / 0.5`
- **A (12%)**: Antecedent - `rain_3d / 150mm`
- **R (8%)**: Runoff Potential - `(1013 - pressure) / 30`
- **E (15%)**: Elevation Risk - `1 - (elev - 190) / (320 - 190)` (inverted)
- **T_modifier**: 1.2 if month is June-September (monsoon), else 1.0

### Risk Levels
- **low** (0.0-0.2): Green `#22c55e`
- **moderate** (0.2-0.4): Yellow `#eab308`
- **high** (0.4-0.7): Orange `#f97316`
- **extreme** (0.7-1.0): Red `#ef4444`

## Example Usage

### Using curl
```bash
# Get FHI for Delhi (Connaught Place)
curl "http://localhost:8000/api/rainfall/fhi?lat=28.6139&lng=77.2090"
```

### Using Python
```python
import requests

response = requests.get(
    "http://localhost:8000/api/rainfall/fhi",
    params={"lat": 28.6139, "lng": 77.2090}
)

data = response.json()
print(f"FHI Score: {data['fhi_score']} ({data['fhi_level']})")
print(f"24h Rainfall: {data['precipitation_24h_mm']}mm")
print(f"Components: {data['components']}")
```

### Using JavaScript/TypeScript
```typescript
const response = await fetch(
  'http://localhost:8000/api/rainfall/fhi?lat=28.6139&lng=77.2090'
);

const data = await response.json();
console.log(`FHI Score: ${data.fhi_score} (${data.fhi_level})`);
console.log(`Color: ${data.fhi_color}`);
```

## Data Sources
- **Precipitation**: Open-Meteo hourly forecast (3-day)
- **Soil Moisture**: Open-Meteo soil_moisture_0_to_7cm (m³/m³)
- **Surface Pressure**: Open-Meteo surface_pressure (hPa)
- **Elevation**: Open-Meteo Elevation API (meters)

## Caching
- Results are cached for **1 hour** (3600 seconds)
- Cache key is based on rounded coordinates (2 decimal places)
- Nearby points (within ~1km) may share cached results

## Error Handling
- **400 Bad Request**: Invalid coordinates
- **503 Service Unavailable**: Open-Meteo API unavailable
- **500 Internal Server Error**: Data processing failure

## Notes
- Default elevation: 220m (Delhi average) if API fails
- Default soil moisture: 0.2 m³/m³ if data missing
- Default surface pressure: 1013 hPa if data missing
- Monsoon season: June-September (applies 1.2x multiplier)

## Implementation Details
- File: `apps/backend/src/api/rainfall.py`
- Response Model: `FloodHazardIndexResponse`
- Calculation Function: `_calculate_fhi()`
- Elevation Helper: `_fetch_elevation()`
- Extended Forecast: `_fetch_open_meteo_extended()`

## Testing
```bash
# Verify import
cd apps/backend
python -c "from src.api.rainfall import router, FloodHazardIndexResponse; print('OK')"

# Test FHI calculation
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
