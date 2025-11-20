# Use Ubuntu Docker container to install and run Tippecanoe

$SCRIPT_DIR = $PSScriptRoot

Write-Host "Converting GeoJSON to PMTiles using Ubuntu + Tippecanoe" -ForegroundColor Green

if (-not (Test-Path "$SCRIPT_DIR\tiles\stream_influence_reclass.geojson")) {
    Write-Host "Error: GeoJSON not found" -ForegroundColor Red
    exit 1
}

Write-Host "`nThis will:" -ForegroundColor Yellow
Write-Host "1. Start Ubuntu container" -ForegroundColor Gray
Write-Host "2. Install Tippecanoe (~2-3 minutes)" -ForegroundColor Gray
Write-Host "3. Convert GeoJSON to PMTiles (~5-10 minutes)" -ForegroundColor Gray
Write-Host "`nTotal time: ~10-15 minutes`n" -ForegroundColor Cyan

docker run --rm `
  -v "${SCRIPT_DIR}:/work" `
  -w /work `
  ubuntu:22.04 `
  bash -c "
    echo 'Installing dependencies...' &&
    apt-get update -qq &&
    apt-get install -y -qq tippecanoe > /dev/null 2>&1 &&
    echo 'Running Tippecanoe conversion...' &&
    tippecanoe -o /work/delhi-tiles.pmtiles \
      -Z12 -z15 \
      --layer=stream_influence_water_difference \
      --force \
      /work/tiles/stream_influence_reclass.geojson &&
    echo 'Conversion complete!'
  "

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nConversion failed" -ForegroundColor Red
    exit 1
}

$outputSize = (Get-Item "$SCRIPT_DIR\delhi-tiles.pmtiles").Length / 1MB
Write-Host "`n✓ Phase 4 complete!" -ForegroundColor Green
Write-Host "delhi-tiles.pmtiles: $([math]::Round($outputSize, 2)) MB" -ForegroundColor Cyan

Write-Host "`nNext: Copy to public directory:" -ForegroundColor Yellow
Write-Host "cp delhi-tiles.pmtiles ..\..\public" -ForegroundColor Gray
