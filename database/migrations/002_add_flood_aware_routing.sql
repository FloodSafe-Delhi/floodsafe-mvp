-- Migration: Add Flood-Aware Routing Logic
-- Phase 2 of Safe Route Navigation System
-- Date: 2025-11-20

-- Step 1: Create all application tables
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    role VARCHAR DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW(),
    points INTEGER DEFAULT 0,
    level INTEGER DEFAULT 1,
    reports_count INTEGER DEFAULT 0,
    verified_reports_count INTEGER DEFAULT 0,
    reputation_score INTEGER DEFAULT 0,
    -- ... other fields as in infrastructure/models.py
);

CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    description VARCHAR,
    location GEOMETRY(POINT, 4326),
    timestamp TIMESTAMP DEFAULT NOW(),
    verified BOOLEAN DEFAULT FALSE,
    verification_score INTEGER DEFAULT 0,
    water_depth VARCHAR(20),
    risk_polygon GEOMETRY(POLYGON, 4326),
    risk_radius_meters INTEGER DEFAULT 100
);

CREATE INDEX IF NOT EXISTS idx_reports_location ON reports USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_reports_risk_polygon ON reports USING GIST(risk_polygon);

-- Step 2: Create risk polygon trigger function
CREATE OR REPLACE FUNCTION trigger_compute_risk_polygon_before()
RETURNS TRIGGER AS $$
DECLARE
    buffer_distance INTEGER;
BEGIN
    buffer_distance := CASE
        WHEN NEW.water_depth = 'impassable' THEN 200
        WHEN NEW.water_depth = 'waist' THEN 150
        WHEN NEW.water_depth = 'knee' THEN 100
        WHEN NEW.water_depth = 'ankle' THEN 50
        WHEN NEW.description ILIKE '%impassable%' THEN 200
        WHEN NEW.description ILIKE '%waist%' THEN 150
        WHEN NEW.description ILIKE '%knee%' THEN 100
        WHEN NEW.description ILIKE '%ankle%' THEN 50
        ELSE 75
    END;

    NEW.risk_polygon := ST_Buffer(NEW.location::geography, buffer_distance)::geometry;
    NEW.risk_radius_meters := buffer_distance;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS reports_compute_risk_polygon_before ON reports;
CREATE TRIGGER reports_compute_risk_polygon_before
BEFORE INSERT OR UPDATE ON reports
FOR EACH ROW
EXECUTE FUNCTION trigger_compute_risk_polygon_before();

-- Step 3: Create calculate_safe_route function
CREATE OR REPLACE FUNCTION calculate_safe_route(
    start_lon DOUBLE PRECISION,
    start_lat DOUBLE PRECISION,
    end_lon DOUBLE PRECISION,
    end_lat DOUBLE PRECISION,
    city_code VARCHAR(10) DEFAULT 'BLR',
    flood_penalty_multiplier INTEGER DEFAULT 1000
)
RETURNS TABLE(
    seq INTEGER,
    path_seq INTEGER,
    node BIGINT,
    edge BIGINT,
    cost DOUBLE PRECISION,
    agg_cost DOUBLE PRECISION,
    geometry GEOMETRY,
    road_name TEXT,
    intersects_flood BOOLEAN,
    flood_severity INTEGER
) AS $$
DECLARE
    road_table TEXT;
    vertices_table TEXT;
    start_node BIGINT;
    end_node BIGINT;
BEGIN
    IF city_code = 'BLR' THEN
        road_table := 'road_network_bangalore';
        vertices_table := 'road_network_bangalore_vertices_pgr';
    ELSIF city_code = 'DEL' THEN
        road_table := 'road_network_delhi';
        vertices_table := 'road_network_delhi_vertices_pgr';
    ELSE
        RAISE EXCEPTION 'Unsupported city code: %', city_code;
    END IF;

    EXECUTE format('SELECT id FROM %I ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint($1, $2), 4326) LIMIT 1', vertices_table)
    INTO start_node USING start_lon, start_lat;

    EXECUTE format('SELECT id FROM %I ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint($1, $2), 4326) LIMIT 1', vertices_table)
    INTO end_node USING end_lon, end_lat;

    IF start_node IS NULL OR end_node IS NULL THEN
        RAISE EXCEPTION 'Could not find nearest nodes';
    END IF;

    RETURN QUERY EXECUTE format('
        SELECT
            r.seq::INTEGER, r.path_seq::INTEGER, r.node, r.edge, r.cost, r.agg_cost,
            rn.geometry, rn.name::TEXT,
            EXISTS(SELECT 1 FROM reports rep WHERE rep.verified = true AND rep.risk_polygon IS NOT NULL AND ST_Intersects(rep.risk_polygon, rn.geometry))::BOOLEAN,
            COALESCE((SELECT MAX(rep.verification_score)::INTEGER FROM reports rep WHERE rep.verified = true AND rep.risk_polygon IS NOT NULL AND ST_Intersects(rep.risk_polygon, rn.geometry)), 0)
        FROM pgr_dijkstra(
            ''SELECT id, source, target,
                CASE WHEN EXISTS (SELECT 1 FROM reports r WHERE r.verified = true AND r.risk_polygon IS NOT NULL AND ST_Intersects(r.risk_polygon, rn.geometry)) THEN cost * %s ELSE cost END as cost,
                CASE WHEN oneway = ''''F'''' THEN -1 WHEN EXISTS (SELECT 1 FROM reports r WHERE r.verified = true AND r.risk_polygon IS NOT NULL AND ST_Intersects(r.risk_polygon, rn.geometry)) THEN reverse_cost * %s ELSE reverse_cost END as reverse_cost
            FROM %I rn'',
            %s, %s, directed := true
        ) r
        LEFT JOIN %I rn ON r.edge = rn.id
        WHERE r.edge != -1 ORDER BY r.seq
    ', flood_penalty_multiplier, flood_penalty_multiplier, road_table, start_node, end_node, road_table);
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION calculate_safe_route TO "user";

-- Testing:
-- SELECT * FROM calculate_safe_route(77.5946, 12.9716, 77.5971, 12.9781, 'BLR', 1000) LIMIT 10;
