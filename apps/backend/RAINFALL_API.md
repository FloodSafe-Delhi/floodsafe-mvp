# Rainfall Forecast API

RESTful API endpoints for 3-day rainfall forecasts using Open-Meteo weather data.

## Endpoints

### GET /api/rainfall/forecast
Get rainfall forecast for a single point.

**Query Parameters:**
- `lat` (float, required): Latitude (-90 to 90)
- `lng` (float, required): Longitude (-180 to 180)

**Example:**
```bash
curl "http://localhost:8000/api/rainfall/forecast?lat=28.6&lng=77.2"
```

**Response:**
```json
{
  "latitude": 28.6,
  "longitude": 77.2,
  "forecast_24h_mm": 12.5,
  "forecast_48h_mm": 8.2,
  "forecast_72h_mm": 3.1,
  "forecast_total_3d_mm": 23.8,
  "probability_max_pct": 85,
  "intensity_category": "moderate",
  "hourly_max_mm": 5.2,
  "fetched_at": "2025-12-13T10:00:00Z",
  "source": "open-meteo"
}
```

### GET /api/rainfall/forecast/grid
Get rainfall forecast grid for visualization.

**Query Parameters:**
- `lat_min` (float, required): Minimum latitude
- `lng_min` (float, required): Minimum longitude
- `lat_max` (float, required): Maximum latitude
- `lng_max` (float, required): Maximum longitude
- `resolution` (float, optional): Grid spacing in degrees (default: 0.05, range: 0.01-0.5)

**Example:**
```bash
curl "http://localhost:8000/api/rainfall/forecast/grid?lat_min=28.5&lng_min=77.1&lat_max=28.7&lng_max=77.3&resolution=0.1"
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
        "coordinates": [77.1, 28.5]
      },
      "properties": {
        "forecast_24h_mm": 12.5,
        "intensity_category": "moderate",
        "latitude": 28.5,
        "longitude": 77.1
      }
    }
  ],
  "metadata": {
    "bbox": [77.1, 28.5, 77.3, 28.7],
    "resolution": 0.1,
    "total_points": 9,
    "fetched_at": "2025-12-13T10:00:00Z",
    "source": "open-meteo"
  }
}
```

### GET /api/rainfall/health
Check service health.

**Example:**
```bash
curl "http://localhost:8000/api/rainfall/health"
```

**Response:**
```json
{
  "status": "healthy",
  "service": "open-meteo",
  "cache_entries": 42
}
```

## IMD Intensity Classification

Rainfall intensity follows India Meteorological Department (IMD) standards:

| Category | 24-hour Rainfall |
|----------|------------------|
| Light | < 7.5 mm |
| Moderate | 7.5 - 35.5 mm |
| Heavy | 35.5 - 64.4 mm |
| Very Heavy | 64.4 - 124.4 mm |
| Extremely Heavy | ≥ 124.4 mm |

## Features

### Caching
- Forecasts cached for **1 hour** to reduce API calls
- Cache keys rounded to 2 decimal places (increases hit rate)
- Automatic cleanup of expired entries

### Retry Logic
- **3 attempts** with 1-second delay between retries
- Handles timeouts and transient errors
- Returns HTTP 503 if all retries fail

### Error Handling
- **400 Bad Request**: Invalid coordinates or parameters
- **503 Service Unavailable**: Open-Meteo API unavailable
- Proper error messages in all failure cases

### Grid Size Limits
- Maximum **400 points** per grid request
- Prevents abuse and API overload
- Suggest increasing resolution or reducing area if limit exceeded

## Data Source

**Open-Meteo API**: https://open-meteo.com/
- Free, no API key required
- 3-day forecast horizon
- Hourly precipitation data
- Daily precipitation probabilities

## Implementation Details

**File**: `apps/backend/src/api/rainfall.py`

**Architecture**:
- Router pattern (FastAPI)
- Pydantic v2 response models with validation
- Async HTTP client (httpx)
- In-memory cache (use Redis in production)

**Key Functions**:
- `_classify_intensity()`: IMD classification
- `_fetch_open_meteo_forecast()`: API call with retry logic
- `_process_forecast_data()`: Parse and calculate forecasts
- `_get_cache_key()`, `_is_cache_valid()`, `_cleanup_cache()`: Cache management

## Testing

Run the test suite:
```bash
# Start backend server first
docker-compose up -d db
cd apps/backend
python -m uvicorn src.main:app --reload

# In another terminal
cd apps/backend
python test_rainfall_api.py
```

Manual testing:
```bash
# Delhi forecast
curl "http://localhost:8000/api/rainfall/forecast?lat=28.6&lng=77.2"

# Bangalore forecast
curl "http://localhost:8000/api/rainfall/forecast?lat=12.97&lng=77.59"

# Small grid (3x3 around Delhi)
curl "http://localhost:8000/api/rainfall/forecast/grid?lat_min=28.5&lng_min=77.1&lat_max=28.7&lng_max=77.3&resolution=0.1"
```

## Future Enhancements

- [ ] Redis caching for production
- [ ] Webhook notifications for high rainfall alerts
- [ ] Historical rainfall data integration
- [ ] Machine learning forecast corrections
- [ ] Multi-model ensemble forecasting
