# Convert GeoJSON to PMTiles using Ubuntu + Tippecanoe

$SCRIPT_DIR = $PSScriptRoot

Write-Host "Converting GeoJSON to PMTiles..." -ForegroundColor Green
Write-Host "This will take 10-15 minutes" -ForegroundColor Yellow

docker run --rm `
  -v "${SCRIPT_DIR}:/work" `
  -w /work `
  ubuntu:22.04 `
  bash -c "apt-get update -qq && apt-get install -y -qq tippecanoe && tippecanoe -o /work/delhi-tiles.pmtiles -Z12 -z15 --layer=stream_influence_water_difference --force /work/tiles/stream_influence_reclass.geojson"

if ($LASTEXITCODE -eq 0) {
    $size = (Get-Item "$SCRIPT_DIR\delhi-tiles.pmtiles").Length / 1MB
    Write-Host "`nSUCCESS! Created delhi-tiles.pmtiles ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "`nFAILED" -ForegroundColor Red
}
