# Convert GeoJSON to PMTiles using Tippecanoe Docker image

$SCRIPT_DIR = $PSScriptRoot

Write-Host "Converting GeoJSON to PMTiles using Tippecanoe..." -ForegroundColor Green

if (-not (Test-Path "$SCRIPT_DIR\tiles\stream_influence_reclass.geojson")) {
    Write-Host "Error: GeoJSON file not found" -ForegroundColor Red
    exit 1
}

Write-Host "Input: tiles\stream_influence_reclass.geojson (176 MB)" -ForegroundColor Cyan
Write-Host "Using Docker image: mapsam/tippecanoe" -ForegroundColor Yellow

docker run --rm `
  -v "${SCRIPT_DIR}:/data" `
  mapsam/tippecanoe:latest `
  -o /data/delhi-tiles.pmtiles `
  -Z12 `
  -z15 `
  --layer=stream_influence_water_difference `
  --force `
  /data/tiles/stream_influence_reclass.geojson

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nConversion failed" -ForegroundColor Red
    exit 1
}

$outputSize = (Get-Item "$SCRIPT_DIR\delhi-tiles.pmtiles").Length / 1MB
Write-Host "`nPhase 4 complete!" -ForegroundColor Green
Write-Host "Output: delhi-tiles.pmtiles ($([math]::Round($outputSize, 2)) MB)" -ForegroundColor Cyan
