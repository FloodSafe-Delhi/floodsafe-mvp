# PowerShell script to generate Delhi basemap PMTiles
# This avoids Git Bash path translation issues on Windows

$BBOX = "76.94,28.42,77.46,28.88"
$PBF_FILE = "NewDelhi.osm.pbf"
$OUTPUT_PMTILES = "delhi.pmtiles"
$SCRIPT_DIR = $PSScriptRoot

Write-Host "Generating basemap PMTiles for Delhi..." -ForegroundColor Green
Write-Host "Working directory: $SCRIPT_DIR" -ForegroundColor Yellow

# Run Docker with Windows-native paths
docker run --rm `
  -v "${SCRIPT_DIR}:/data" `
  ghcr.io/systemed/tilemaker:master `
  "/data/$PBF_FILE" `
  --output "/data/$OUTPUT_PMTILES" `
  --bbox "$BBOX" `
  --process /data/process-openmaptiles.lua `
  --config /data/config-openmaptiles.json `
  --skip-integrity

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Delhi basemap PMTiles generated successfully!" -ForegroundColor Green
    Write-Host "Output file: $SCRIPT_DIR\$OUTPUT_PMTILES" -ForegroundColor Cyan
    Write-Host "`nNext step: Copy to public directory" -ForegroundColor Yellow
    Write-Host "cp $SCRIPT_DIR\$OUTPUT_PMTILES ..\..\public\delhi-basemap.pmtiles" -ForegroundColor Gray
} else {
    Write-Host "`n❌ Basemap generation failed" -ForegroundColor Red
    exit 1
}
