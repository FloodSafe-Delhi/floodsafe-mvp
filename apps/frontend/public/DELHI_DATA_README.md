# Delhi Flood Atlas Data Files - MISSING

⚠️ **IMPORTANT**: The following files are required for Delhi flood atlas to work properly but are currently missing:

## Required Files

### 1. `delhi-basemap.pmtiles` (MISSING)
**Size**: ~10-15 MB
**Source**: Generated from OpenStreetMap data
**Generation**: Run `apps/frontend/data/delhi/basemap/tiles.sh`

```bash
cd apps/frontend/data/delhi/basemap
./tiles.sh
# Output: delhi.pmtiles
# Move to: apps/frontend/public/delhi-basemap.pmtiles
```

### 2. `delhi-tiles.pmtiles` (MISSING)
**Size**: ~50-100 MB
**Source**: Generated from Copernicus DEM flood processing
**Generation**: See instructions in `apps/frontend/data/delhi/README.md`

```bash
cd apps/frontend/data/delhi/dem
python process_dem_to_flood_vector.py
tippecanoe -o delhi-tiles.pmtiles \
  -Z12 -z15 \
  --layer=stream_influence_water_difference \
  tiles/stream_influence_reclass.shp
# Move to: apps/frontend/public/delhi-tiles.pmtiles
```

### 3. Metro Data (PLACEHOLDER)
**Status**: Minimal placeholder GeoJSON created
**Files**:
- `delhi-metro-lines.geojson` ✅ (placeholder only)
- `delhi-metro-stations.geojson` ✅ (placeholder only)

**Real Data Needed**: Extract from OpenStreetMap or Delhi Metro GTFS/official sources

## Current Workaround

The Delhi city toggle will work but the map will show errors for missing tiles. The application gracefully handles missing resources and will continue to function with the Bangalore data.

## How to Get Real Delhi Data

See the complete data generation guide in:
`apps/frontend/data/delhi/README.md`

This will be created in the next step of the implementation.
