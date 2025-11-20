# Alternative approach: Shapefile -> GeoJSON -> PMTiles
# Uses ogr2ogr (Docker) + pmtiles CLI (npm)

$SCRIPT_DIR = $PSScriptRoot

Write-Host "Phase 4: Converting shapefile to PMTiles (Alternative Method)" -ForegroundColor Green

# Check if shapefile exists
if (-not (Test-Path "$SCRIPT_DIR\tiles\stream_influence_reclass.shp")) {
    Write-Host "Error: stream_influence_reclass.shp not found" -ForegroundColor Red
    exit 1
}

# Step 1: Convert shapefile to GeoJSON using ogr2ogr (Docker)
Write-Host "`n[Step 1/2] Converting shapefile to GeoJSON..." -ForegroundColor Yellow

docker run --rm `
  -v "${SCRIPT_DIR}:/data" `
  ghcr.io/osgeo/gdal:alpine-small-latest `
  ogr2ogr `
  -f GeoJSON `
  /data/tiles/stream_influence_reclass.geojson `
  /data/tiles/stream_influence_reclass.shp

if ($LASTEXITCODE -ne 0) {
    Write-Host "GeoJSON conversion failed" -ForegroundColor Red
    exit 1
}

Write-Host "GeoJSON conversion complete" -ForegroundColor Green

# Step 2: Convert GeoJSON to PMTiles using pmtiles CLI
Write-Host "`n[Step 2/2] Converting GeoJSON to PMTiles..." -ForegroundColor Yellow

# pmtiles doesn't support direct GeoJSON to PMTiles, so we need tippecanoe-like tool
# Let's use ogr2ogr to convert to MBTiles, then pmtiles to convert MBTiles to PMTiles

Write-Host "Converting to MBTiles first..." -ForegroundColor Gray

docker run --rm `
  -v "${SCRIPT_DIR}:/data" `
  ghcr.io/osgeo/gdal:alpine-small-latest `
  ogr2ogr `
  -f MBTiles `
  /data/delhi-tiles.mbtiles `
  /data/tiles/stream_influence_reclass.geojson `
  -dsco MINZOOM=12 `
  -dsco MAXZOOM=15

if ($LASTEXITCODE -ne 0) {
    Write-Host "MBTiles conversion failed" -ForegroundColor Red
    exit 1
}

Write-Host "MBTiles conversion complete" -ForegroundColor Green

# Convert MBTiles to PMTiles
Write-Host "`nConverting MBTiles to PMTiles..." -ForegroundColor Gray

Set-Location $SCRIPT_DIR
pmtiles convert delhi-tiles.mbtiles delhi-tiles.pmtiles

if ($LASTEXITCODE -ne 0) {
    Write-Host "PMTiles conversion failed" -ForegroundColor Red
    exit 1
}

$outputSize = (Get-Item "$SCRIPT_DIR\delhi-tiles.pmtiles").Length / 1MB
Write-Host "`nPhase 4 complete!" -ForegroundColor Green
Write-Host "Output: delhi-tiles.pmtiles ($([math]::Round($outputSize, 2)) MB)" -ForegroundColor Cyan

Write-Host "`nNext: Copy to public directory" -ForegroundColor Yellow
Write-Host "cp delhi-tiles.pmtiles ../../public/delhi-tiles.pmtiles" -ForegroundColor Gray
