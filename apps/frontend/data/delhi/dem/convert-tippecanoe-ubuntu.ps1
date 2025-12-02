# Use Ubuntu Docker container to install and run Tippecanoe

$SCRIPT_DIR = $PSScriptRoot

Write-Host "Converting GeoJSON to PMTiles using Ubuntu + Tippecanoe" -ForegroundColor Green

docker run --rm `
  -v "${SCRIPT_DIR}:/work" `
  -w /work `
  ubuntu:22.04 `
  bash -c "set -x && apt-get update && apt-get install -y tippecanoe && tippecanoe -o /work/delhi-tiles.pmtiles -Z12 -z15 --layer=stream_influence_water_difference --force /work/tiles/stream_influence_reclass.geojson"

if ($LASTEXITCODE -ne 0) {
  Write-Host "Conversion failed with exit code $LASTEXITCODE" -ForegroundColor Red
  exit 1
}

$outputSize = (Get-Item "$SCRIPT_DIR\delhi-tiles.pmtiles").Length / 1MB
Write-Host "Phase 4 complete!" -ForegroundColor Green
Write-Host "delhi-tiles.pmtiles: $([math]::Round($outputSize, 2)) MB" -ForegroundColor Cyan
