# PowerShell script to process DEM tiles using Docker with GDAL
# This avoids needing to install GDAL locally on Windows

$SCRIPT_DIR = $PSScriptRoot
$BBOX = "76.94 28.42 77.46 28.88"

Write-Host "Processing DEM tiles using Docker + GDAL..." -ForegroundColor Green
Write-Host "Working directory: $SCRIPT_DIR" -ForegroundColor Yellow

# Check if both DEM tiles exist
if (-not (Test-Path "$SCRIPT_DIR\dem_n28e076.tif")) {
    Write-Host "Error: dem_n28e076.tif not found" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$SCRIPT_DIR\dem_n28e077.tif")) {
    Write-Host "Error: dem_n28e077.tif not found" -ForegroundColor Red
    exit 1
}

Write-Host "`nBoth DEM tiles found" -ForegroundColor Green
Write-Host "- dem_n28e076.tif: $((Get-Item "$SCRIPT_DIR\dem_n28e076.tif").Length / 1MB) MB" -ForegroundColor Cyan
Write-Host "- dem_n28e077.tif: $((Get-Item "$SCRIPT_DIR\dem_n28e077.tif").Length / 1MB) MB" -ForegroundColor Cyan

# Step 1: Create VRT (Virtual Raster) to combine tiles
Write-Host "`n[Step 1/3] Creating virtual raster (VRT)..." -ForegroundColor Yellow

docker run --rm `
  -v "${SCRIPT_DIR}:/data" `
  ghcr.io/osgeo/gdal:alpine-small-latest `
  gdalbuildvrt `
  /data/delhi_combined.vrt `
  /data/dem_n28e076.tif `
  /data/dem_n28e077.tif

if ($LASTEXITCODE -ne 0) {
    Write-Host "VRT creation failed" -ForegroundColor Red
    exit 1
}

Write-Host "VRT created successfully" -ForegroundColor Green

# Step 2: Clip to Delhi bounds
Write-Host "`n[Step 2/3] Clipping to Delhi bounding box ($BBOX)..." -ForegroundColor Yellow

docker run --rm `
  -v "${SCRIPT_DIR}:/data" `
  ghcr.io/osgeo/gdal:alpine-small-latest `
  gdal_translate `
  -projwin 76.94 28.88 77.46 28.42 `
  -co COMPRESS=LZW `
  -co TILED=YES `
  /data/delhi_combined.vrt `
  /data/delhi_dem.tif

if ($LASTEXITCODE -ne 0) {
    Write-Host "Clipping failed" -ForegroundColor Red
    exit 1
}

Write-Host "DEM clipped successfully" -ForegroundColor Green

# Step 3: Clean up VRT file
Write-Host "`n[Step 3/3] Cleaning up temporary files..." -ForegroundColor Yellow
Remove-Item "$SCRIPT_DIR\delhi_combined.vrt" -ErrorAction SilentlyContinue

# Display final output info
$outputSize = (Get-Item "$SCRIPT_DIR\delhi_dem.tif").Length / 1MB
Write-Host "`nDEM processing complete!" -ForegroundColor Green
Write-Host "Output file: delhi_dem.tif ($([math]::Round($outputSize, 2)) MB)" -ForegroundColor Cyan
Write-Host "Bounds: $BBOX" -ForegroundColor Gray

Write-Host "`nNext step: Process DEM to flood vectors (Phase 3)" -ForegroundColor Yellow
Write-Host "This will require WhiteboxTools for hydrological analysis" -ForegroundColor Gray
