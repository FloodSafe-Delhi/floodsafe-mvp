import os
import whitebox
from osgeo import gdal
import numpy as np

# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()
# Ensure whitebox is configured to use the local data directory
wbt.work_dir = os.path.dirname(os.path.abspath(__file__))

def reclassify_influence_raster(input_file, output_file, num_classes=4):
    """
    Reclassify the influence raster using GDAL with a specified number of classes
    
    Parameters:
    input_file (str): Path to input raster file
    output_file (str): Path to output raster file
    num_classes (int): Number of desired classes (default=4)
    """
    # Open input dataset
    src_ds = gdal.Open(input_file)
    if not src_ds:
        print(f"Error: Could not open {input_file}")
        return
    src_band = src_ds.GetRasterBand(1)
    src_data = src_band.ReadAsArray()
    
    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_file, 
                          src_ds.RasterXSize, 
                          src_ds.RasterYSize, 
                          1, 
                          gdal.GDT_Byte)
    
    # Copy projection and geotransform
    dst_ds.SetProjection(src_ds.GetProjection())
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    
    # Calculate breaks for N classes (assuming a 1-4 range for flood influence as per BLR script)
    min_val = 1
    max_val = 4
    interval = (max_val - min_val) / num_classes
    
    # Perform reclassification
    dst_data = np.zeros_like(src_data, dtype=np.uint8)
    
    for i in range(num_classes):
        lower = min_val + (i * interval)
        upper = min_val + ((i + 1) * interval)
        
        if i == 0:
            # Include the lower bound for first class
            mask = (src_data >= lower) & (src_data <= upper)
        else:
            # Exclude lower bound for other classes
            mask = (src_data > lower) & (src_data <= upper)
            
        dst_data[mask] = i + 1  # Classes start from 1
    
    # Write output and set nodata value
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.WriteArray(dst_data)
    dst_band.SetNoDataValue(0)
    
    # Clean up
    src_ds = None
    dst_ds = None
 
    return output_file

def convert_raster_to_vector(input_raster, output_vector):
    """
    Convert a raster file to vector format using WhiteboxTools
    """
    # Set nodata value for WhiteboxTools processing
    # The original script does this, so we'll keep it.
    wbt.set_nodata_value(
        input_raster, 
        input_raster, 
        back_value=1
    )

    wbt.raster_to_vector_polygons(input_raster, output_vector)
    return output_vector

def process_dem_for_flood_atlas(dem_path, output_dir, flow_accum_threshold=1000, max_influence_distance=1):
    """
    Generate flood influence vector data from a clipped DEM.
    """
    print(f"Starting flood influence processing on {dem_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    flow_accum = os.path.join(output_dir, "flow_accum.tif")
    flow_dir = os.path.join(output_dir, "flow_dir.tif")
    streams = os.path.join(output_dir, "streams.tif")
    influence = os.path.join(output_dir, "stream_influence.tif")
    
    # WhiteboxTools workflow (mirrored from BLR script)
    print("Filling depressions...")
    wbt.fill_depressions(dem_path, filled_dem)
    
    print("Calculating flow direction...")
    wbt.d8_pointer(filled_dem, flow_dir)
    
    print("Calculating flow accumulation...")
    wbt.d8_flow_accumulation(filled_dem, flow_accum)
    
    print("Extracting streams...")
    wbt.extract_streams(flow_accum, streams, threshold=flow_accum_threshold)
    
    print("Calculating stream influence...")
    wbt.gaussian_filter(flow_accum, influence, sigma=max_influence_distance/4)

    print("Calculating natural log of stream influence...")
    wbt.ln(influence, influence)

    print("Calculating standard deviation contrast stretch...")
    wbt.standard_deviation_contrast_stretch(influence, influence, stdev=2, num_tones=3)

    print("Rescaling influence...")
    wbt.rescale_value_range(influence, influence, out_min_val=1, out_max_val=4)

    # Reclassify influence
    print("Reclassifying influence...")
    input_file = os.path.join(output_dir, "stream_influence.tif")
    output_file = os.path.join(output_dir, "stream_influence_reclass.tif")
    reclassify_influence_raster(input_file, output_file, num_classes=4)
    
    # Convert to vector
    vector_file = os.path.join(output_dir, "stream_influence_reclass.shp")
    convert_raster_to_vector(output_file, vector_file)
    
    print(f"Flood influence vector data complete. Results saved to: {vector_file}")
    
    return vector_file

if __name__ == "__main__":
    pwd = os.path.dirname(os.path.abspath(__file__))
    dem_path = os.path.join(pwd, "delhi_dem.tif")
    output_dir = os.path.join(pwd, "tiles")
    
    if not os.path.exists(dem_path):
        print(f"Error: Required DEM file not found at {dem_path}. Please run fetch_and_process_dem.sh first.")
    else:
        process_dem_for_flood_atlas(dem_path, output_dir, flow_accum_threshold=1000, max_influence_distance=1)