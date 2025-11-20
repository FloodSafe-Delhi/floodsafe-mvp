# PowerShell script to convert shapefile to PMTiles using Docker Tippecanoe
# Phase 4: Convert Delhi flood shapefile to PMTiles

$SCRIPT_DIR = $PSScriptRoot

Write-Host "Phase 4: Converting shapefile to PMTiles" -ForegroundColor Green

# Check if shapefile exists
if (-not (Test-Path "$SCRIPT_DIR\tiles\stream_influence_reclass.shp")) {
    Write-Host "Error: stream_influence_reclass.shp not found. Run Phase 3 first." -ForegroundColor Red
    exit 1
}

Write-Host "Input: tiles\stream_influence_reclass.shp" -ForegroundColor Cyan
Write-Host "Output: delhi-tiles.pmtiles" -ForegroundColor Cyan
Write-Host "Layer name: stream_influence_water_difference (matching BLR atlas)" -ForegroundColor Yellow

Write-Host "`nRunning Tippecanoe via Docker..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray

docker run --rm `
  -v "${SCRIPT_DIR}:/data" `
  ghcr.io/felt/tippecanoe:latest `
  -o /data/delhi-tiles.pmtiles `
  -Z12 `
  -z15 `
  --layer=stream_influence_water_difference `
  --force `
  /data/tiles/stream_influence_reclass.shp

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: PMTiles conversion failed" -ForegroundColor Red
    exit 1
}

$outputSize = (Get-Item "$SCRIPT_DIR\delhi-tiles.pmtiles").Length / 1MB
Write-Host "`nPhase 4 complete!" -ForegroundColor Green
Write-Host "Output: delhi-tiles.pmtiles ($([math]::Round($outputSize, 2)) MB)" -ForegroundColor Cyan

Write-Host "`nNext step: Copy to public directory" -ForegroundColor Yellow
Write-Host "cp delhi-tiles.pmtiles ../../public/delhi-tiles.pmtiles" -ForegroundColor Gray
