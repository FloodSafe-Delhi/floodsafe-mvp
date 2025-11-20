import os
import subprocess
import whitebox
import numpy as np

# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()
# Ensure whitebox is configured to use the local data directory
wbt.work_dir = os.path.dirname(os.path.abspath(__file__))

def reclassify_influence_raster_docker(input_file, output_file, num_classes=4):
    """
    Reclassify using numpy (we already have it installed) instead of Docker GDAL
    The input values are already scaled 1-4, so we just need to discretize them
    """
    print(f"Reclassifying {input_file} using numpy...")

    # Since we have numpy, let's just do simple file copy
    # The values are already in range 1-4 from the rescale operation
    # We can use gdal_translate via Docker to convert type to Byte

    work_dir = os.path.dirname(os.path.abspath(input_file))
    input_filename = os.path.basename(input_file)
    output_filename = os.path.basename(output_file)

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{work_dir}:/data",
        "ghcr.io/osgeo/gdal:alpine-small-latest",
        "gdal_translate",
        "-ot", "Byte",
        "-a_nodata", "0",
        "-co", "COMPRESS=LZW",
        f"/data/{input_filename}",
        f"/data/{output_filename}"
    ]

    result = subprocess.run(docker_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in reclassification: {result.stderr}")
        return None

    print(f"Reclassification complete: {output_file}")
    return output_file

def convert_raster_to_vector(input_raster, output_vector):
    """
    Convert a raster file to vector format using WhiteboxTools
    """
    print(f"Converting raster to vector: {input_raster} -> {output_vector}")

    # WhiteboxTools raster_to_vector_polygons
    wbt.raster_to_vector_polygons(input_raster, output_vector)

    print(f"Vector conversion complete: {output_vector}")
    return output_vector

def process_dem_for_flood_atlas(dem_path, output_dir, flow_accum_threshold=1000, max_influence_distance=1):
    """
    Generate flood influence vector data from a clipped DEM.
    """
    print(f"\n{'='*60}")
    print(f"Starting flood influence processing on {dem_path}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    flow_accum = os.path.join(output_dir, "flow_accum.tif")
    flow_dir = os.path.join(output_dir, "flow_dir.tif")
    streams = os.path.join(output_dir, "streams.tif")
    influence = os.path.join(output_dir, "stream_influence.tif")

    # WhiteboxTools workflow (mirrored from BLR script)
    print("[1/9] Filling depressions...")
    wbt.fill_depressions(dem_path, filled_dem)
    print("COMPLETE\n")

    print("[2/9] Calculating flow direction (D8 pointer)...")
    wbt.d8_pointer(filled_dem, flow_dir)
    print("COMPLETE\n")

    print("[3/9] Calculating flow accumulation (D8)...")
    wbt.d8_flow_accumulation(filled_dem, flow_accum)
    print("COMPLETE\n")

    print(f"[4/9] Extracting streams (threshold={flow_accum_threshold})...")
    wbt.extract_streams(flow_accum, streams, threshold=flow_accum_threshold)
    print("COMPLETE\n")

    print(f"[5/9] Calculating stream influence (Gaussian filter, sigma={max_influence_distance/4})...")
    wbt.gaussian_filter(flow_accum, influence, sigma=max_influence_distance/4)
    print("COMPLETE\n")

    print("[6/9] Calculating natural log of stream influence...")
    wbt.ln(influence, influence)
    print("COMPLETE\n")

    print("[7/9] Applying standard deviation contrast stretch (stdev=2, tones=3)...")
    wbt.standard_deviation_contrast_stretch(influence, influence, stdev=2, num_tones=3)
    print("COMPLETE\n")

    print("[8/9] Rescaling influence to range 1-4...")
    wbt.rescale_value_range(influence, influence, out_min_val=1, out_max_val=4)
    print("COMPLETE\n")

    # Reclassify influence using Docker GDAL
    print("[9/9] Reclassifying influence (using Docker GDAL)...")
    input_file = os.path.join(output_dir, "stream_influence.tif")
    output_file = os.path.join(output_dir, "stream_influence_reclass.tif")
    reclassify_influence_raster_docker(input_file, output_file, num_classes=4)
    print("COMPLETE\n")

    # Convert to vector
    print("[10/10] Converting to vector polygons...")
    vector_file = os.path.join(output_dir, "stream_influence_reclass.shp")
    convert_raster_to_vector(output_file, vector_file)
    print("COMPLETE\n")

    print(f"\n{'='*60}")
    print(f"SUCCESS: Flood influence vector data complete!")
    print(f"Output: {vector_file}")
    print(f"{'='*60}\n")

    return vector_file

if __name__ == "__main__":
    pwd = os.path.dirname(os.path.abspath(__file__))
    dem_path = os.path.join(pwd, "delhi_dem.tif")
    output_dir = os.path.join(pwd, "tiles")

    if not os.path.exists(dem_path):
        print(f"Error: Required DEM file not found at {dem_path}. Please run Phase 2 first.")
    else:
        print("\n" + "="*60)
        print("PHASE 3: DEM TO FLOOD VECTORS")
        print("Estimated time: 1-2 hours")
        print("="*60 + "\n")

        process_dem_for_flood_atlas(dem_path, output_dir, flow_accum_threshold=1000, max_influence_distance=1)
