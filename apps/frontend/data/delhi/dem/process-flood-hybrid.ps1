# Hybrid PowerShell script to process DEM to flood vectors
# Uses local WhiteboxTools + Docker GDAL to avoid GDAL installation issues

$SCRIPT_DIR = $PSScriptRoot

Write-Host "Starting Phase 3: DEM to Flood Vectors Processing" -ForegroundColor Green
Write-Host "This will take 1-2 hours..." -ForegroundColor Yellow

# Check if delhi_dem.tif exists
if (-not (Test-Path "$SCRIPT_DIR\delhi_dem.tif")) {
    Write-Host "Error: delhi_dem.tif not found. Run Phase 2 first." -ForegroundColor Red
    exit 1
}

# Create tiles directory
New-Item -ItemType Directory -Force -Path "$SCRIPT_DIR\tiles" | Out-Null

Write-Host "`nRunning WhiteboxTools hydrological processing..." -ForegroundColor Cyan
Write-Host "This Python script will take 1-2 hours to complete." -ForegroundColor Yellow
Write-Host "You can monitor progress in the terminal..." -ForegroundColor Gray

# Run the Python script that uses WhiteboxTools
# We'll modify it to skip GDAL parts
python "$SCRIPT_DIR\process_dem_to_flood_vector.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: WhiteboxTools processing failed" -ForegroundColor Red
    exit 1
}

Write-Host "`nPhase 3 processing complete!" -ForegroundColor Green
Write-Host "Output: tiles\stream_influence_reclass.shp" -ForegroundColor Cyan
