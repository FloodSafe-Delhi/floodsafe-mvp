#!/usr/bin/env python3
"""
Simplified DEM processing script that avoids problematic imports
Uses WhiteboxTools for hydrological processing
Adapted for Windows to use Docker for GDAL/OGR operations
"""
import os
import sys
import subprocess
import whitebox

# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
wbt.work_dir = script_dir
wbt.verbose = True

# Define paths (relative to script location)
dem_path = os.path.join(script_dir, "delhi_dem.tif")
output_dir = os.path.join(script_dir, "tiles")

os.makedirs(output_dir, exist_ok=True)

# Output files
filled_dem = os.path.join(output_dir, "filled_dem.tif")
flow_dir = os.path.join(output_dir, "flow_dir.tif")
flow_accum = os.path.join(output_dir, "flow_accum.tif")
streams = os.path.join(output_dir, "streams.tif")
influence = os.path.join(output_dir, "stream_influence.tif")
influence_reclass = os.path.join(output_dir, "stream_influence_reclass.tif")
vector_shp = os.path.join(output_dir, "stream_influence_reclass.shp")
vector_geojson = os.path.join(output_dir, "stream_influence_reclass.geojson")

print("\n" + "="*60)
print("DELHI DEM TO FLOOD VECTORS - SIMPLIFIED (DOCKER ADAPTED)")
print("="*60 + "\n")

# Check if DEM exists
if not os.path.exists(dem_path):
    print(f"ERROR: DEM file not found at {dem_path}")
    print("Please ensure delhi_dem.tif exists in the same directory as this script")
    sys.exit(1)

# Step 1: Fill depressions
print("[1/11] Filling depressions...")
wbt.fill_depressions(dem_path, filled_dem)
print("COMPLETE\n")

# Step 2: Flow direction
print("[2/11] Calculating flow direction (D8 pointer)...")
wbt.d8_pointer(filled_dem, flow_dir)
print("COMPLETE\n")

# Step 3: Flow accumulation
print("[3/11] Calculating flow accumulation...")
wbt.d8_flow_accumulation(filled_dem, flow_accum)
print("COMPLETE\n")

# Step 4: Extract streams
print("[4/11] Extracting streams (threshold=1000)...")
wbt.extract_streams(flow_accum, streams, threshold=1000)
print("COMPLETE\n")

# Step 5: Gaussian filter for stream influence
print("[5/11] Calculating stream influence (Gaussian filter)...")
wbt.gaussian_filter(flow_accum, influence, sigma=0.25)
print("COMPLETE\n")

# Step 6: Natural log
print("[6/11] Calculating natural log...")
wbt.ln(influence, influence)
print("COMPLETE\n")

# Step 7: Contrast stretch
print("[7/11] Applying contrast stretch...")
wbt.standard_deviation_contrast_stretch(influence, influence, stdev=2, num_tones=3)
print("COMPLETE\n")

# Step 8: Rescale to 1-4
print("[8/11] Rescaling to range 1-4...")
wbt.rescale_value_range(influence, influence, out_min_val=1, out_max_val=4)
print("COMPLETE\n")

# Step 9: Reclass
print("[9/11] Reclassifying (using Docker GDAL)...")
# Docker command to run gdal_translate
# We mount script_dir to /data
# Input: /data/tiles/stream_influence.tif
# Output: /data/tiles/stream_influence_reclass.tif

docker_cmd_reclass = [
    "docker", "run", "--rm",
    "-v", f"{script_dir}:/data",
    "ghcr.io/osgeo/gdal:alpine-small-latest",
    "gdal_translate",
    "-ot", "Byte",
    "-a_nodata", "0",
    "-co", "COMPRESS=LZW",
    "/data/tiles/stream_influence.tif",
    "/data/tiles/stream_influence_reclass.tif"
]

result = subprocess.run(docker_cmd_reclass, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Warning: gdal_translate failed: {result.stderr}")
    print("Trying to copy file instead...")
    # Fallback to copy if docker fails (unlikely but safe)
    import shutil
    shutil.copy2(influence, influence_reclass)
else:
    print("Docker GDAL reclass successful")

print("COMPLETE\n")

# Step 10: Convert to vector shapefile
print("[10/11] Converting to vector polygons (shapefile)...")
wbt.raster_to_vector_polygons(influence_reclass, vector_shp)
print("COMPLETE\n")

# Step 11: Convert shapefile to GeoJSON
print("[11/11] Converting shapefile to GeoJSON (using Docker OGR)...")

docker_cmd_ogr = [
    "docker", "run", "--rm",
    "-v", f"{script_dir}:/data",
    "ghcr.io/osgeo/gdal:alpine-small-latest",
    "ogr2ogr",
    "-f", "GeoJSON",
    "/data/tiles/stream_influence_reclass.geojson",
    "/data/tiles/stream_influence_reclass.shp"
]

result = subprocess.run(docker_cmd_ogr, capture_output=True, text=True)

if result.returncode != 0:
    print(f"ERROR: ogr2ogr failed: {result.stderr}")
    sys.exit(1)

print("COMPLETE\n")

print("="*60)
print(f"SUCCESS!")
print(f"Output shapefile: {vector_shp}")
print(f"Output GeoJSON: {vector_geojson}")
print("="*60 + "\n")
