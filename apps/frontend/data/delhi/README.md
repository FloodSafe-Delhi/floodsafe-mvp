# Delhi Flood Atlas Data Generation Guide

This guide provides step-by-step instructions for generating all required data files for the Delhi flood atlas.

## Prerequisites

Before starting, ensure you have the following tools installed:

### Required Software

1. **Docker Desktop** (for basemap generation)
   - Download: https://www.docker.com/products/docker-desktop/
   - Required for running Tilemaker to generate basemap PMTiles

2. **AWS CLI** (for DEM data download)
   ```bash
   pip install awscli
   ```
   - Used to download Copernicus DEM tiles from AWS S3

3. **GDAL/OGR Tools** (for geospatial data processing)
   ```bash
   # Windows (using Conda)
   conda install -c conda-forge gdal

   # macOS
   brew install gdal

   # Linux
   sudo apt-get install gdal-bin python3-gdal
   ```

4. **Python 3.8+** with required packages
   ```bash
   pip install whitebox
   pip install gdal
   ```

5. **Tippecanoe** (for vector tile generation)
   - Download from: https://github.com/felt/tippecanoe/releases
   - Extract and add to PATH

### Verify Installation

```bash
docker --version
aws --version
gdalinfo --version
python --version
tippecanoe --version
```

## Data Generation Pipeline

### Step 1: Generate Delhi Basemap PMTiles

**Estimated Time**: 15-20 minutes
**Output**: `delhi.pmtiles` (~13 MB)

#### Process:

```bash
cd apps/frontend/data/delhi/basemap

# Run the tile generation script
./tiles.sh

# This will:
# 1. Pull the Tilemaker Docker image
# 2. Process NewDelhi.osm.pbf (34 MB OpenStreetMap data)
# 3. Generate delhi.pmtiles using OpenMapTiles schema
```

#### Expected Output:

```
Delhi Basemap Generation
========================
Input: NewDelhi.osm.pbf (34 MB)
Output: delhi.pmtiles
...
✅ Basemap tiles generated successfully
```

#### Move to Public Directory:

```bash
# Copy the generated file to the frontend public directory
cp delhi.pmtiles ../../public/delhi-basemap.pmtiles
```

### Step 2: Download and Process DEM Data

**Estimated Time**: 30-45 minutes
**Output**: `delhi_dem.tif` (~50 MB)

#### Process:

```bash
cd ../dem

# Option 1: Using shell script
./fetch_and_process_dem.sh

# Option 2: Using Python script (recommended)
python fetch_and_process_dem.py
```

#### What This Does:

1. Downloads two Copernicus DEM tiles from AWS S3:
   - N28_00_E076_00_DEM (covers 28-29°N, 76-77°E)
   - N28_00_E077_00_DEM (covers 28-29°N, 77-78°E)

2. Combines tiles into a virtual raster (VRT)

3. Clips to Delhi bounding box:
   - Min Longitude: 76.94°E
   - Max Longitude: 77.46°E
   - Min Latitude: 28.42°N
   - Max Latitude: 28.88°N

4. Outputs: `delhi_dem.tif`

#### Troubleshooting:

**Error**: "AWS CLI not found"
```bash
# Install AWS CLI
pip install awscli

# Configure (no credentials needed for public data)
aws configure set region eu-central-1
```

**Error**: "GDAL not found"
```bash
# Verify GDAL is installed and in PATH
gdalinfo --version

# If not, install using conda or system package manager
conda install -c conda-forge gdal
```

### Step 3: Generate Flood Influence Vectors

**Estimated Time**: 1-2 hours (computationally intensive)
**Output**: `tiles/stream_influence_reclass.shp`

#### Process:

```bash
# Still in apps/frontend/data/delhi/dem
python process_dem_to_flood_vector.py
```

#### What This Does:

1. Uses WhiteboxTools to process the DEM:
   - Breach depressions (hydrological conditioning)
   - Calculate D8 flow accumulation
   - Extract stream network
   - Compute distance to streams (flood influence)
   - Reclassify into flood zones

2. Converts raster to vector polygons

3. Outputs to `tiles/stream_influence_reclass.shp`

#### Expected Processing Steps:

```
Starting flood data processing for Delhi DEM...
1. Breach Depressions...
   ✓ Complete (5-10 minutes)

2. Flow Accumulation...
   ✓ Complete (10-15 minutes)

3. Extract Streams...
   ✓ Complete (5 minutes)

4. Distance to Streams...
   ✓ Complete (15-20 minutes)

5. Reclassify...
   ✓ Complete (5 minutes)

6. Polygonize...
   ✓ Complete (10-15 minutes)

✅ Flood vector processing complete
```

#### Output Files:

- `tiles/stream_influence_reclass.shp` - Main shapefile
- `tiles/stream_influence_reclass.shx` - Shape index
- `tiles/stream_influence_reclass.dbf` - Attribute database
- `tiles/stream_influence_reclass.prj` - Projection file

### Step 4: Convert to PMTiles

**Estimated Time**: 5-10 minutes
**Output**: `delhi-tiles.pmtiles` (~50-100 MB)

#### Process:

```bash
# Still in apps/frontend/data/delhi/dem
tippecanoe -o delhi-tiles.pmtiles \
  -Z12 \
  -z15 \
  --layer=stream_influence_water_difference \
  --force \
  tiles/stream_influence_reclass.shp
```

#### Parameters Explained:

- `-o delhi-tiles.pmtiles`: Output file name
- `-Z12`: Minimum zoom level
- `-z15`: Maximum zoom level
- `--layer=stream_influence_water_difference`: Layer name (MUST match Bangalore)
- `--force`: Overwrite existing file

#### Expected Output:

```
For layer 0, using name "stream_influence_water_difference"
...
Sorting 1234567 features
...
✅ Tile generation complete
```

#### Move to Public Directory:

```bash
cp delhi-tiles.pmtiles ../../public/delhi-tiles.pmtiles
```

### Step 5: Obtain Delhi Metro Data

**Estimated Time**: 30-60 minutes
**Output**: `delhi-metro-lines.geojson`, `delhi-metro-stations.geojson`

#### Option A: Extract from OpenStreetMap (Recommended)

Use Overpass Turbo to query OSM data:

**Website**: https://overpass-turbo.eu/

**Query for Metro Lines**:
```overpass
[bbox:28.42,76.94,28.88,77.46];
(
  way["railway"="subway"]["network"="Delhi Metro"];
  relation["route"="subway"]["network"="Delhi Metro"];
);
out geom;
```

**Query for Metro Stations**:
```overpass
[bbox:28.42,76.94,28.88,77.46];
(
  node["railway"="station"]["network"="Delhi Metro"];
  node["railway"="subway_entrance"]["network"="Delhi Metro"];
);
out;
```

**Steps**:
1. Go to https://overpass-turbo.eu/
2. Paste the query
3. Click "Run"
4. Click "Export" → "GeoJSON"
5. Save as `delhi-metro-lines.geojson` or `delhi-metro-stations.geojson`

#### Option B: Manual GeoJSON Creation

Create GeoJSON files manually using Delhi Metro official route maps:

**Delhi Metro Lines** (12 lines):
- Red Line (#e41e26)
- Yellow Line (#ffcc00)
- Blue Line (#0047ba)
- Green Line (#76b947)
- Violet Line (#9966cc)
- Orange Line (#f7941d)
- Pink Line (#ea4c89)
- Magenta Line (#da487e)
- Grey Line (#808080)
- Airport Express (#f8b93c)
- Aqua Line (#00bfff)
- Rapid Metro (#c41e3a)

**GeoJSON Structure**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Red Line",
        "ref": "Red",
        "colour": "#e41e26",
        "network": "Delhi Metro"
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [[lon1, lat1], [lon2, lat2], ...]
      }
    }
  ]
}
```

#### Place Files:

```bash
cp delhi-metro-lines.geojson ../../public/
cp delhi-metro-stations.geojson ../../public/
```

## Verification

After completing all steps, verify you have the following files:

```
apps/frontend/public/
├── delhi-basemap.pmtiles        ✅ (~13 MB)
├── delhi-tiles.pmtiles           ✅ (~50-100 MB)
├── delhi-metro-lines.geojson     ✅ (~200 KB)
└── delhi-metro-stations.geojson  ✅ (~100 KB)
```

## Testing

1. **Start the development server**:
   ```bash
   cd apps/frontend
   npm run dev
   ```

2. **Open the app**: http://localhost:5173

3. **Navigate to Flood Atlas**

4. **Select Delhi from city dropdown**

5. **Verify**:
   - Map loads Delhi area (centered around 77.209°E, 28.614°N)
   - Basemap tiles render correctly
   - Flood layer displays
   - Metro lines and stations appear
   - Zoom controls work
   - Geolocation works for Delhi bounds

## Troubleshooting

### Basemap Generation Fails

**Issue**: Docker container crashes or fails to generate tiles

**Solutions**:
1. Ensure Docker has sufficient memory allocated (4GB minimum)
2. Check Docker is running: `docker ps`
3. Pull Tilemaker image manually: `docker pull ghcr.io/systemed/tilemaker`
4. Check OSM file is not corrupted: `osmconvert NewDelhi.osm.pbf --out-statistics`

### DEM Processing Takes Too Long

**Issue**: Flood vector processing exceeds expected time

**Solutions**:
1. This is normal for DEM processing - can take 1-2 hours
2. Monitor progress - script outputs status for each step
3. Ensure sufficient disk space (~2GB free)
4. Check CPU usage - process is compute-intensive

### PMTiles File Too Large

**Issue**: Generated PMTiles exceed 200 MB

**Solutions**:
1. Reduce max zoom: use `-z14` instead of `-z15`
2. Simplify geometries: add `--simplification=10`
3. Drop attributes: add `--drop-densest-as-needed`

### Metro Data Not Displaying

**Issue**: Metro lines/stations don't show on map

**Solutions**:
1. Verify GeoJSON files are valid: https://geojson.io/
2. Check file names match exactly (case-sensitive)
3. Ensure `colour` property exists for lines
4. Check coordinates are in [longitude, latitude] order

## File Size Reference

| File | Size | Compression |
|------|------|-------------|
| delhi-basemap.pmtiles | ~13 MB | Gzip compressed tiles |
| delhi-tiles.pmtiles | 50-100 MB | Varies by detail level |
| delhi-metro-lines.geojson | ~200 KB | Text, compresses well |
| delhi-metro-stations.geojson | ~100 KB | Text, compresses well |

## Next Steps

Once all data files are generated and placed correctly:

1. Test city switching between Bangalore and Delhi
2. Verify geolocation bounds work correctly for both cities
3. Test all layer toggles (flood, metro, sensors, reports)
4. Check map performance with Delhi tiles loaded
5. Validate flood data accuracy against known flood-prone areas

## Support

For issues with data generation:
- Check WhiteboxTools documentation: https://www.whiteboxgeo.com/
- Tippecanoe documentation: https://github.com/felt/tippecanoe
- OpenStreetMap wiki: https://wiki.openstreetmap.org/

## Future Enhancements

- Add more detailed DEM processing (stream order, watershed delineation)
- Include historical flood data overlay
- Add real-time rainfall data integration
- Create multi-resolution pyramid for faster loading
- Implement tile caching strategy
