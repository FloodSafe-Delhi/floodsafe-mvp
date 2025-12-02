# Generate PMTiles using Python Docker image + pip install tippecanoe
# This works because PyPI has manylinux wheels for Tippecanoe!

$SCRIPT_DIR = $PSScriptRoot
$INPUT_FILE = "tiles/stream_influence_reclass.geojson"
$OUTPUT_FILE = "delhi-tiles.pmtiles"

Write-Host "Generating PMTiles using Python + pip install tippecanoe..." -ForegroundColor Green

if (-not (Test-Path "$SCRIPT_DIR\$INPUT_FILE")) {
    Write-Host "Error: Input GeoJSON not found at $SCRIPT_DIR\$INPUT_FILE" -ForegroundColor Red
    exit 1
}

# Command to run inside Docker:
# 1. Install tippecanoe via pip
# 2. Run tippecanoe
$DOCKER_CMD = "pip install tippecanoe && " +
"tippecanoe -o /data/$OUTPUT_FILE -Z12 -z15 --layer=stream_influence_water_difference --force /data/$INPUT_FILE"

Write-Host "Starting Docker container..." -ForegroundColor Yellow

docker run --rm `
    -v "${SCRIPT_DIR}:/data" `
    python:3.11-slim `
    bash -c "$DOCKER_CMD"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Tippecanoe conversion failed" -ForegroundColor Red
    exit 1
}

$outputSize = (Get-Item "$SCRIPT_DIR\$OUTPUT_FILE").Length / 1MB
Write-Host "Success! Generated $OUTPUT_FILE" -ForegroundColor Green
Write-Host "Size: $([math]::Round($outputSize, 2)) MB" -ForegroundColor Cyan

# Copy to public folder
$PUBLIC_DIR = "$SCRIPT_DIR\..\..\public"
Write-Host "Copying to $PUBLIC_DIR..." -ForegroundColor Yellow
Copy-Item "$SCRIPT_DIR\$OUTPUT_FILE" -Destination "$PUBLIC_DIR" -Force
Write-Host "Done." -ForegroundColor Green
