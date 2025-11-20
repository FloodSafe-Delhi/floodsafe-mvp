#!/bin/bash

# Define the bounding box for Delhi (min_lon, min_lat, max_lon, max_lat)
# Using the bounds identified earlier: [[76.94, 28.42], [77.46, 28.88]]
BBOX="76.94,28.42,77.46,28.88"

# Path to the downloaded PBF file
PBF_FILE="NewDelhi.osm.pbf"
OUTPUT_PMTILES="delhi.pmtiles"

echo "Generating basemap PMTiles for Delhi..."

docker run -it --rm --pull always -v "$(pwd)/apps/frontend/data/delhi/basemap:/data" \
  ghcr.io/systemed/tilemaker:master \
  "/data/${PBF_FILE}" \
  --output "/data/${OUTPUT_PMTILES}" \
  --bbox "${BBOX}" \
  --process /data/process-openmaptiles.lua \
  --config /data/config-openmaptiles.json \
  --skip-integrity

echo "Delhi basemap PMTiles generated: apps/frontend/data/delhi/basemap/${OUTPUT_PMTILES}"