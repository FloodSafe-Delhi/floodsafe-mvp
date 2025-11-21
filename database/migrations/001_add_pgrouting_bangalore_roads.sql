-- Migration: Add pgRouting and Bangalore Road Network
-- Phase 1 of Safe Route Navigation System
-- Date: 2025-11-20

-- Step 1: Enable PostGIS and pgRouting extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgrouting;

-- Step 2: Road network tables are created and populated by osm2pgrouting
-- Tables: road_network_bangalore, road_network_bangalore_vertices_pgr
-- Import command used:
--   osm2pgrouting --file bangalore.osm --conf mapconfig_for_cars.xml \
--     --dbname floodsafe --username user --password password \
--     --host localhost --port 5432 --schema public --prefix blr_ --clean

-- Step 3: Schema enhancements
ALTER TABLE road_network_bangalore 
  ADD COLUMN IF NOT EXISTS city_code VARCHAR(10) DEFAULT 'BLR',
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();

-- Step 4: Update cost values
UPDATE road_network_bangalore 
SET cost = length_meters 
WHERE cost IS NULL OR cost = 0;

UPDATE road_network_bangalore 
SET reverse_cost = length_meters 
WHERE reverse_cost IS NULL OR reverse_cost = 0;

-- Step 5: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_road_network_blr_source ON road_network_bangalore(source);
CREATE INDEX IF NOT EXISTS idx_road_network_blr_target ON road_network_bangalore(target);
CREATE INDEX IF NOT EXISTS idx_road_network_blr_city ON road_network_bangalore(city_code);
CREATE INDEX IF NOT EXISTS idx_road_network_blr_geom ON road_network_bangalore USING GIST(geometry);

-- Step 6: Validate topology
SELECT pgr_analyzeGraph('road_network_bangalore', 0.001, 'geometry', 'id', 'source', 'target');

-- Statistics:
-- - Total roads: 321,563 segments
-- - Total vertices: 257,808 intersections
-- - Isolated segments: 119 (0.04%)
-- - Coverage: Central Bangalore (12.8-13.1°N, 77.5-77.8°E)
