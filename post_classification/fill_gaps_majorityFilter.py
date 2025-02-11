import os
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from scipy.ndimage import generic_filter
import timeit

def majority_filter_func(data, nodata_value=0, window_size=3, max_iterations=100):
    """
    Fill gaps in a raster using an iterative majority filter, excluding real NoData pixels.
    Continues until all gaps are filled or max_iterations is reached.

    Parameters:
        data (numpy.ndarray): 2D array with classified raster data.
        nodata_value (int or float): Value representing NoData (e.g., 0).
        window_size (int): Size of the moving window for majority filter (odd number).
        max_iterations (int): Maximum number of iterations to perform.

    Returns:
        numpy.ndarray: 2D array with gaps filled.
    """
    def mode_filter(values):
        # Exclude nodata values from mode calculation
        valid_values = values[values != nodata_value]
        if len(valid_values) == 0:
            return nodata_value
        # Find the most common value (mode)
        return np.bincount(valid_values.astype(int)).argmax()

    output_data = data.copy()
    iteration = 0
    
    while iteration < max_iterations:
        # Count initial number of NoData pixels
        initial_nodata_count = np.sum(output_data == nodata_value)
        
        if initial_nodata_count == 0:
            print(f"All gaps filled after {iteration} iterations")
            break
            
        # Apply mode filter
        filled_data = generic_filter(
            output_data,
            function=mode_filter,
            size=window_size,
            mode='constant',
            cval=nodata_value
        )
        
        # Update only the NoData pixels
        nodata_mask = (output_data == nodata_value)
        output_data[nodata_mask] = filled_data[nodata_mask]
        
        # Check if any pixels were filled in this iteration
        final_nodata_count = np.sum(output_data == nodata_value)
        pixels_filled = initial_nodata_count - final_nodata_count
        
        print(f"Iteration {iteration + 1}: Filled {pixels_filled} pixels, {final_nodata_count} remaining")
        
        # If no pixels were filled in this iteration, increase window size
        if pixels_filled == 0:
            window_size += 2
            print(f"Increasing window size to {window_size}")
        
        iteration += 1
        
        # Break if no more NoData pixels or if window size gets too large
        if final_nodata_count == 0 or window_size > 21:  # You can adjust the maximum window size
            break
    
    if iteration == max_iterations:
        print(f"Reached maximum iterations ({max_iterations}). Some gaps may remain.")
    
    return output_data

def fill_gaps(input_raster, output_raster, aoi, nodata_value=0, window_size=3):
    """
    Main function to fill gaps in a GeoTIFF raster only within a polygon from a shapefile.

    Parameters:
        input_raster (str): Path to the input GeoTIFF file.
        output_raster (str): Path to the output GeoTIFF file.
        aoi (geodataframe): Geodataframe containing the Area of interest (mask).
        nodata_value (int or float): Value representing NoData in the input raster.
        window_size (int): Initial size of the moving window for majority filter (odd number).
    """
    # Open the input GeoTIFF
    print(f'Reading input raster: {os.path.basename(input_raster)}')
    with rasterio.open(input_raster) as src:
        # Read the data and metadata
        data = src.read(1)
        profile = src.profile  # Get metadata (e.g., CRS, transform, etc.)
        transform = src.transform
        
        # Handle the nodata value from the metadata
        if src.nodata is not None:
            nodata_value = src.nodata

        # Rasterize the polygon mask
        print('Handling the geometry mask')
        polygon_mask = geometry_mask(
            geometries=aoi.geometry,
            out_shape=data.shape,
            transform=transform,
            invert=True  # invert to mask outside the polygon
        )

        # Apply the mask to exclude areas outside the polygon
        masked_data = np.where(polygon_mask, data, nodata_value)

        # Fill gaps (0 values) only within the polygon
        print('Filling the gaps using iterative majority filter')
        filled_data = majority_filter_func(masked_data, nodata_value=nodata_value, window_size=window_size)

        # Merge filled data back into the original raster
        output_data = np.where(polygon_mask, filled_data, data)

        # Cast the data to uint8
        output_data = output_data.astype(np.uint8)

        # Update the metadata for 8-bit format
        profile.update(dtype=rasterio.uint8)

    # Save the filled raster to a new GeoTIFF
    print(f'Saving the output raster: {os.path.basename(output_raster)}')
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(output_data, 1)


if __name__ == "__main__":
    start_t = timeit.default_timer()
    
    #inputs
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    aoi_shp = os.path.join(wks, 'data' ,'AOIs', 'bc_tiles_200km_modified.shp')
    gdf = gpd.read_file(aoi_shp)
    input_raster = os.path.join(wks, 'classification', 'v10' ,'tile45.tif')
    output_raster = os.path.join(wks, 'classification', 'gap_filled' ,'tile45_gapFilled_maj.tif')

    # Run the script
    gdf = gdf[gdf['tile_id'] == 45]
    fill_gaps(input_raster, output_raster, gdf, nodata_value=0, window_size=3)
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')