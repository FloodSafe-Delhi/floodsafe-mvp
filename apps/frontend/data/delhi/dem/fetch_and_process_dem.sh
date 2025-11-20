#!/bin/bash

# Working directory is assumed to be 'c:/Users/Anirudh Mohan/Desktop/FloodSafe' when executed.
# We will execute commands relative to 'apps/frontend/data/delhi/dem'

OUTPUT_DIR="apps/frontend/data/delhi/dem"
FINAL_OUTPUT_TIF="${OUTPUT_DIR}/delhi_dem.tif"
TEMP_VRT="${OUTPUT_DIR}/temp_delhi.vrt"

# Delhi Bounds: MinLng: 76.94, MaxLat: 28.88, MaxLng: 77.46, MinLat: 28.42 (min_lon max_lat max_lon min_lat)
BBOX_ARG="76.94 28.88 77.46 28.42" 

# Tile names covering Delhi (28.42-28.88N, 76.94-77.46E). We use tiles N28E076 and N28E077 to cover the area.
# N28E076: 28N-29N, 76E-77E
# N28E077: 28N-29N, 77E-78E
TILE_FILE_1="dem_n28e076.tif"
TILE_FILE_2="dem_n28e077.tif"
S3_BASE="s3://copernicus-dem-30m/Copernicus_DSM_COG_10_"

echo "Starting DEM data acquisition and processing for Delhi..."

# 1. Fetch tiles using AWS CLI (assuming aws is available and configured)
echo "1. Fetching DEM tiles..."
AWS_DEFAULT_REGION=eu-central-1 aws s3 cp --no-sign-request "${S3_BASE}N28_00_E076_00_DEM/${S3_BASE}N28_00_E076_00_DEM.tif" "${OUTPUT_DIR}/${TILE_FILE_1}"
AWS_DEFAULT_REGION=eu-central-1 aws s3 cp --no-sign-request "${S3_BASE}N28_00_E077_00_DEM/${S3_BASE}N28_00_E077_00_DEM.tif" "${OUTPUT_DIR}/${TILE_FILE_2}"

# 2. Combine tiles into VRT
echo "2. Combining tiles with gdalbuildvrt..."
gdalbuildvrt -overwrite "${TEMP_VRT}" "${OUTPUT_DIR}/${TILE_FILE_1}" "${OUTPUT_DIR}/${TILE_FILE_2}"

# 3. Clip to Delhi bounds
echo "3. Clipping to Delhi bounds..."
# gdal_translate -projwin min_lon max_lat max_lon min_lat temp.vrt output.tif
gdal_translate -of GTiff -projwin ${BBOX_ARG} "${TEMP_VRT}" "${FINAL_OUTPUT_TIF}"

echo "✅ Delhi DEM processing complete. Output saved to: ${FINAL_OUTPUT_TIF}"

# 4. Clean up
echo "4. Cleaning up temporary files..."
rm -f "${OUTPUT_DIR}/${TILE_FILE_1}" "${OUTPUT_DIR}/${TILE_FILE_2}" "${TEMP_VRT}"

echo "Cleanup finished."