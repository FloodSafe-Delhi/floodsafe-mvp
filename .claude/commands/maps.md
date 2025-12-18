# @maps Domain Context

Load the Maps & Geospatial domain files and work on: $ARGUMENTS

## Files to Read First
- `apps/frontend/src/components/MapComponent.tsx`
- `apps/frontend/src/lib/map/useMap.ts`
- `apps/frontend/src/lib/map/cityConfigs.ts`

## Data Locations
- `apps/frontend/data/delhi/dem/` - PMTiles, DEM data

## Patterns
- MapLibre GL JS for rendering
- PMTiles for efficient tile serving
- PostGIS with SRID 4326 (WGS84)
- ST_DWithin for radius queries

## Key Coordinates
- Delhi: [77.1, 28.6] center
- Bangalore: [77.59, 12.97] center

## Now proceed to work on the task specified.
