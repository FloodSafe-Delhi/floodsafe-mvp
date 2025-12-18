# @ml Domain Context

Load the ML Service domain and work on: $ARGUMENTS

## Files Location
`apps/ml-service/**` (to be created)

## GCP Project
`gen-lang-client-0669818939`

## Primary Data Sources (GEE)
```python
datasets = {
    'alphaearth': 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',  # 64-dim, 10m
    'dem': 'USGS/SRTMGL1_003',
    'surface_water': 'JRC/GSW1_4/GlobalSurfaceWater',
    'precipitation': 'UCSB-CHG/CHIRPS/DAILY',
    'era5_land': 'ECMWF/ERA5_LAND/DAILY_AGGR',
    'landcover': 'ESA/WorldCover/v200',
}
```

## Model Progression
1. ARIMA (baseline)
2. Prophet (seasonality)
3. LSTM (temporal patterns)
4. Ensemble (production)

## Rules
- NO synthetic data - real sources only
- Always compare against baseline
- Use ml_flood architecture as reference

## Now proceed to work on the task specified.
