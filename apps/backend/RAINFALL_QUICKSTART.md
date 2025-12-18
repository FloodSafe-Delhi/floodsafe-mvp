# Rainfall API - Quick Start Guide

## Start the Server

```bash
# Option 1: Docker (recommended)
docker-compose up

# Option 2: Local development
docker-compose up -d db
cd apps/backend
python -m uvicorn src.main:app --reload
```

Server runs at: `http://localhost:8000`

## Test the API

### 1. Interactive API Docs
Open in browser: **http://localhost:8000/docs**

FastAPI provides interactive Swagger UI where you can:
- See all rainfall endpoints
- Try requests directly in browser
- View response schemas

### 2. Quick curl Tests

```bash
# Health check
curl http://localhost:8000/api/rainfall/health

# Delhi forecast
curl "http://localhost:8000/api/rainfall/forecast?lat=28.6&lng=77.2"

# Bangalore forecast
curl "http://localhost:8000/api/rainfall/forecast?lat=12.97&lng=77.59"

# Small grid (3x3 around Delhi)
curl "http://localhost:8000/api/rainfall/forecast/grid?lat_min=28.5&lng_min=77.1&lat_max=28.7&lng_max=77.3&resolution=0.1"
```

### 3. Automated Test Suite

```bash
cd apps/backend
python test_rainfall_api.py
```

Expected: `4/4 tests passed`

## Example Responses

### Single Point Forecast
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

### IMD Categories
- `light`: < 7.5 mm/24h
- `moderate`: 7.5 - 35.5 mm/24h
- `heavy`: 35.5 - 64.4 mm/24h
- `very_heavy`: 64.4 - 124.4 mm/24h
- `extremely_heavy`: ≥ 124.4 mm/24h

## Next Steps

See full documentation: `RAINFALL_API.md`
