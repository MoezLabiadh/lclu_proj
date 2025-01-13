"""
Fill gaps of unclassified pixels using a Nearest Neighbor Interpolation
"""

import os
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from scipy.ndimage import distance_transform_edt
import timeit

def nearest_neighbor_func(data, nodata_value=0):
    """
    Fill gaps in a raster using Nearest Neighbor Interpolation, excluding real NoData pixels.

    Parameters:
        data (numpy.ndarray): 2D array with classified raster data.
        nodata_value (int or float): Value representing NoData (e.g., 0).

    Returns:
        numpy.ndarray: 2D array with gaps filled.
    """
    # Mask for NoData pixels
    nodata_mask = (data == nodata_value)
    
    # Compute the distance to the nearest valid pixel for each NoData pixel
    distances, nearest_indices = distance_transform_edt(
        nodata_mask,  # Areas to interpolate
        return_indices=True
    )
    
    # Map the nearest valid pixel's value to each NoData pixel
    filled_data = data.copy()
    filled_data[nodata_mask] = data[nearest_indices[0][nodata_mask], nearest_indices[1][nodata_mask]]
    
    return filled_data


def fill_gaps(input_raster, output_raster, aoi, nodata_value=0):
    """
    Main function to fill gaps in a GeoTIFF raster only within a polygon from a shapefile.

    Parameters:
        input_raster (str): Path to the input GeoTIFF file.
        output_raster (str): Path to the output GeoTIFF file.
        aoi (geodataframe): Geodataframe containing the Area of interest (mask).
        nodata_value (int or float): Value representing NoData in the input raster.
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
        print('Filling the gaps')
        filled_data = nearest_neighbor_func(masked_data, nodata_value=nodata_value)

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
    input_raster = os.path.join(wks, 'classification', 'Tile19_v5.tif')
    output_raster = os.path.join(wks, 'classification', 'Tile19_v5_gapFilled.tif')

    # Run the script
    gdf = gdf[gdf['tile_id'] == 19]
    fill_gaps(input_raster, output_raster, gdf, nodata_value=0)
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
