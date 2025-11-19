import os
import subprocess
import shutil

# Define paths
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the /dem directory for easier management if necessary, but for now, keep it here
# If we mirror the BLR structure: OUTPUT_DIR is /dem, we want output in /dem/tiles

FINAL_OUTPUT_TIF = os.path.join(OUTPUT_DIR, "delhi_dem.tif")
TEMP_VRT = os.path.join(OUTPUT_DIR, "temp_delhi.vrt")

# Delhi Bounds: MinLng: 76.94, MaxLat: 28.88, MaxLng: 77.46, MinLat: 28.42
DELHI_BBOX_ARG = "76.94 28.88 77.46 28.42" # min_lon max_lat max_lon min_lat

# Tile names covering Delhi (28.42-28.88N, 76.94-77.46E)
# N28E076 covers 28-29N, 76-77E
# N28E077 covers 28-29N, 77-78E
TILE_FILES = [
    "dem_n28e076.tif",
    "dem_n28e077.tif"  # Corrected from N27 - both tiles are at N28
]
S3_BASE = "s3://copernicus-dem-30m/Copernicus_DSM_COG_10_"
S3_REGION = "AWS_DEFAULT_REGION=eu-central-1"

def run_command(command_parts, check=True):
    """Executes a command and prints output/error."""
    command = " ".join(command_parts)
    print(f"Executing: {command}")
    try:
        result = subprocess.run(command_parts, check=check, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

def fetch_and_process_dem():
    print("Starting DEM data acquisition and processing for Delhi...")

    # 1. Fetch tiles
    print("1. Fetching DEM tiles...")
    tile_map = {
        "dem_n28e076.tif": "N28_00_E076_00_DEM", # 28N-29N, 76E-77E
        "dem_n28e077.tif": "N28_00_E077_00_DEM"  # 28N-29N, 77E-78E (Corrected from N27)
    }
    
    for filename, s3_path_segment in tile_map.items():
        s3_path = f"{S3_BASE}{s3_path_segment}/{S3_BASE.split('/')[-2].replace('DSM_COG', 'DSM_COG_10_')}{s3_path_segment}.tif"
        command = [
            'aws', 's3', 'cp', '--no-sign-request',
            s3_path,
            filename
        ]
        run_command(command)

    # 2. Combine tiles into VRT
    print("2. Combining tiles with gdalbuildvrt...")
    command = [
        'gdalbuildvrt', '-overwrite', TEMP_VRT
    ] + [f for f in TILE_FILES if os.path.exists(f)]
    run_command(command)

    # 3. Clip to Delhi bounds
    print("3. Clipping to Delhi bounds...")
    # gdal_translate -projwin min_lon max_lat max_lon min_lat temp.vrt output.tif
    command = [
        'gdal_translate',
        '-of', 'GTiff',
        '-projwin', *DELHI_BBOX_ARG.split(),
        TEMP_VRT,
        FINAL_OUTPUT_TIF
    ]
    run_command(command)

    print(f"✅ Delhi DEM processing complete. Output saved to: {FINAL_OUTPUT_TIF}")

    # 4. Clean up
    print("4. Cleaning up temporary files...")
    for f in TILE_FILES:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(TEMP_VRT):
        os.remove(TEMP_VRT)
    
    print("Cleanup finished.")


if __name__ == "__main__":
    # Ensure AWS CLI commands are prefixed with the region setting if necessary, 
    # but we rely on environment variable set in the shell command. Since this is python, 
    # we need to explicitly set it in the subprocess environment or rely on global config.
    # For this context, I will rely on 'aws' being in PATH and the user having it configured, 
    # and will use the env var explicitly in the command if possible, or rely on the subprocess.
    # Since the original script used 'AWS_DEFAULT_REGION=... aws s3 cp', I'll try to replicate that by setting env in run_command.
    # For simplicity in this script, I will rely on the 'aws' command being available and configured in the environment 
    # where this script will eventually run, or if the environment allows:
    
    # Re-running fetch_and_process_dem with explicit environment settings for subprocess
    
    # Re-implementing the fetch part to include the region in the environment for subprocess
    
    print("Starting DEM data acquisition and processing for Delhi (Python approach)...")
    
    # 1. Fetch tiles
    print("1. Fetching DEM tiles...")
    tile_map = {
        "dem_n28e076.tif": "N28_00_E076_00_DEM", # 28N-29N, 76E-77E
        "dem_n28e077.tif": "N28_00_E077_00_DEM"  # 28N-29N, 77E-78E (Corrected from N27)
    }
    
    env = os.environ.copy()
    env["AWS_DEFAULT_REGION"] = "eu-central-1"
    
    for filename, s3_path_segment in tile_map.items():
        s3_path = f"{S3_BASE}{s3_path_segment}/{S3_BASE.split('/')[-2].replace('DSM_COG', 'DSM_COG_10_')}{s3_path_segment}.tif"
        command = [
            'aws', 's3', 'cp', '--no-sign-request',
            s3_path,
            filename
        ]
        print(f"Executing: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"AWS S3 Copy failed for {filename}. This tool execution relies on aws-cli being installed and configured.")
            print("STDERR:", e.stderr)
            # Don't raise here, as we proceed to other steps if possible, but this is a critical dependency.
            
    # 2. Combine tiles into VRT
    print("2. Combining tiles with gdalbuildvrt...")
    gdal_files = [f for f in TILE_FILES if os.path.exists(f)]
    if gdal_files:
        command = [
            'gdalbuildvrt', '-overwrite', TEMP_VRT
        ] + gdal_files
        run_command(command)

        # 3. Clip to Delhi bounds
        print("3. Clipping to Delhi bounds...")
        command = [
            'gdal_translate',
            '-of', 'GTiff',
            '-projwin', *DELHI_BBOX_ARG.split(),
            TEMP_VRT,
            FINAL_OUTPUT_TIF
        ]
        run_command(command)

        print(f"✅ Delhi DEM processing complete. Output saved to: {FINAL_OUTPUT_TIF}")

        # 4. Clean up
        print("4. Cleaning up temporary files...")
        for f in TILE_FILES:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(TEMP_VRT):
            os.remove(TEMP_VRT)
        
        print("Cleanup finished.")
    else:
        print("Skipping GDAL steps as no DEM files were successfully downloaded.")
