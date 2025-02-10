"""
Fill gaps of unclassified pixels using a Majority Filter
"""

import os
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import mode

import timeit


def majority_mode_filter_func(data, nodata_value=0, size=3):
    """
    Fill gaps in a raster using a Majority/Mode Filter.
    
    This function slides a window (default 3×3) over the data array. For any pixel that is unclassified 
    (i.e. equals nodata_value), it computes the mode of its neighboring pixels that are valid. 
    Pixels that are already classified remain unchanged.
    
    Parameters:
        data (numpy.ndarray): 2D array with classified raster data.
        nodata_value (int or float): Value representing NoData (e.g., 0).
        size (int): Window size (default 3, corresponding to a 3×3 window).
    
    Returns:
        numpy.ndarray: 2D array with gaps filled using the mode of neighboring pixels.
    """
    
    def mode_filter(window):
        center = window[len(window) // 2]  # Center pixel of the window
        # If the center pixel is valid, leave it unchanged
        if center != nodata_value:
            return center
        # For a gap pixel, collect all valid neighbors
        valid = window[window != nodata_value]
        if valid.size > 0:
            m = mode(valid, nan_policy='omit')
            return m.mode[0]
        else:
            return nodata_value

    return generic_filter(data, mode_filter, size=size, mode='mirror')

def fill_gaps(input_raster, output_raster, aoi, nodata_value=0):
    """
    Main function to fill gaps in a GeoTIFF raster within a polygon from a shapefile.

    Parameters:
        input_raster (str): Path to the input GeoTIFF file.
        output_raster (str): Path to the output GeoTIFF file.
        aoi (GeoDataFrame): Geodataframe containing the area of interest (mask).
        nodata_value (int or float): Value representing NoData in the input raster.
    """
    print(f'Reading input raster: {os.path.basename(input_raster)}')
    with rasterio.open(input_raster) as src:
        data = src.read(1)
        profile = src.profile
        transform = src.transform
        
        # Use nodata from the metadata if provided
        if src.nodata is not None:
            nodata_value = src.nodata

        print('Handling the geometry mask')
        polygon_mask = geometry_mask(
            geometries=aoi.geometry,
            out_shape=data.shape,
            transform=transform,
            invert=True  # Invert to mask outside the polygon
        )

        # Create a masked version of the data: outside the AOI is set to nodata_value
        masked_data = np.where(polygon_mask, data, nodata_value)

        # Fill gaps (pixels equal to nodata_value) only within the polygon
        print('Filling the gaps using Majority/Mode Filter')
        filled_data = majority_mode_filter_func(masked_data, nodata_value=nodata_value, size=3)

        # Merge the filled data back into the original raster (outside the polygon, retain original)
        output_data = np.where(polygon_mask, filled_data, data)

        # Cast data to uint8 for output
        output_data = output_data.astype(np.uint8)
        profile.update(dtype=rasterio.uint8)

    print(f'Saving the output raster: {os.path.basename(output_raster)}')
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(output_data, 1)

if __name__ == "__main__":
    start_t = timeit.default_timer()
    
    # Define inputs
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    aoi_shp = os.path.join(wks, 'data', 'AOIs', 'bc_tiles_200km_modified.shp')
    gdf = gpd.read_file(aoi_shp)
    input_raster = os.path.join(wks, 'classification', 'v10', 'tile45.tif')
    output_raster = os.path.join(wks, 'classification', 'gap_filled', 'tile45_gapFilled_maj.tif')

    # Filter AOI based on tile_id (if needed)
    gdf = gdf[gdf['tile_id'] == 45]
    fill_gaps(input_raster, output_raster, gdf, nodata_value=0)
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
