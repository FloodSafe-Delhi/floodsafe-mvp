#!/bin/bash

# Working directory will be relative to the script location, which we will execute from
OUTPUT_DIR=$(dirname "$0")
cd "${OUTPUT_DIR}" || exit 1

echo "Fetching DEM tiles for Delhi region..."

# 1. Fetch necessary DEM tiles from Copernicus (similar to Bangalore script)
# Delhi is around 28.65N, 77.2E. We need tiles covering 28N-29N and 76E-77E.
# Tile N28E076 covers 28N to 29N, 76E to 77E.
# Tile N29E076 covers 29N to 30N, 76E to 77E (Might be too far north, let's check N28E077 if it exists or stick to one).
# Given Bangalore used two tiles for a small area, let's check the 28N, 77E tile first.

# Check if aws CLI is available. If not, this step will fail, but we proceed with the script.
# We will attempt to fetch tiles that cover the Delhi area (approx. 28.42-28.88N, 76.94-77.46E).
# N28E077 covers Lat 28-29, Lon 77-78.
# N28E076 covers Lat 28-29, Lon 76-77.
# N27E077 covers Lat 27-28, Lon 77-78.

# Based on the need for Lon 76.94 to 77.46 and Lat 28.42 to 28.88, the N28E076 tile seems most relevant for the western edge,
# and N28E077 for the eastern edge. Let's try to download one tile that spans the longitude well, e.g., N28E076 or N28E077.
# Given the structure of the Copernicus tiles, let's use the tile that spans the core area: N28E077 (28N to 29N, 77E to 78E) might be slightly off for the western edge.
# The Bangalore one used N12E077 and N13E077.
# For Delhi (28.65N, 77.2E), we look for N28E077 (28N-29N, 77E-78E) and N28E076 (28N-29N, 76E-77E).

# Let's stick to the pattern and use the tile covering the latitude band, focusing on longitude.
# We will download the tile that contains the center (28.65N, 77.2E). N28E077 looks like a better starting point if we must pick one or two.
# Let's use the two tiles that bracket the longitude range 76.94 to 77.46: N28E076 and N28E077.

AWS_DEFAULT_REGION=eu-central-1 aws s3 cp --no-sign-request s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N28_00_E076_00_DEM/Copernicus_DSM_COG_10_N28_00_E076_00_DEM.tif ./dem_n28e076.tif
AWS_DEFAULT_REGION=eu-central-1 aws s3 cp --no-sign-request s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N28_00_E077_00_DEM/Copernicus_DSM_COG_10_N28_00_E077_00_DEM.tif ./dem_n28e077.tif

echo "Combining and clipping DEM tiles..."

# 2. Combine tiles
gdalbuildvrt temp_delhi.vrt dem_n28e076.tif dem_n28e077.tif

# 3. Clip to Delhi bounds (MinLng: 76.94, MaxLat: 28.88, MaxLng: 77.46, MinLat: 28.42)
# gdal_translate -projwin min_lon max_lat max_lon min_lat temp_vrt output.tif
gdal_translate -projwin 76.94 28.88 77.46 28.42 temp_delhi.vrt delhi_dem.tif

# Clean up temporary files
rm temp_delhi.vrt dem_n28e076.tif dem_n28e077.tif

echo "Delhi DEM processing complete: delhi_dem.tif"