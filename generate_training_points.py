"""
Generate training data for Land Cover classification.

"""
import os
import random
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import binary_erosion

def generate_training_points(raster_path, n_points=1000, crs=3005, dist_from_edge=2, mask=None):
    """
    Generates random points covering each class/value of a spatial raster, with an optional exclusion mask.

    Parameters
    ----------
    raster_path : str
        Path to the input raster file.
    n_points : int
        Number of points to be generated for each class.
    dist_from_edge : int
        Buffer distance from the edge pixels in pixels.
    crs : int
        EPSG code of the input raster CRS. Default is BC Albers (EPSG:3005).
    mask : str, optional
        Path to a raster mask where pixel values of 1 indicate areas to exclude.
    """
    # Open the main raster file
    print ('Reading the Training Raster')
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform

        # Identify unique categories/values, ignoring no-data values
        unique_values = np.unique(data[data != src.nodata])
        
        # Load the mask if provided
        
        if mask:
            print ('Reading the Mask Raster')
            with rasterio.open(mask) as mask_src:
                mask = mask_src.read(1)
        else:
            mask = None

        points = []
        
        for value in unique_values:
            print(f"Generating points for category {value} of {len(unique_values)}")
            # Create a mask for the current category/value
            category_mask = data == value
            
            # Erode the mask to avoid edge pixels based on the buffer distance
            eroded_mask = binary_erosion(
                category_mask,
                structure=np.ones((dist_from_edge * 2 + 1, dist_from_edge * 2 + 1))
            )
            
            # If a mask is provided, exclude areas where mask == 1
            if mask is not None:
                eroded_mask = eroded_mask & (mask != 1)
            
            # Get row, col indices of valid pixels
            valid_indices = np.argwhere(eroded_mask)
            
            # Check if there are enough valid pixels
            if len(valid_indices) < n_points:
                print(f"..not enough pixels for category {value}, reducing points to {len(valid_indices)}.")
                n_points_for_category = len(valid_indices)
            else:
                n_points_for_category = n_points
            
            # Randomly sample valid pixels
            sampled_indices = random.sample(valid_indices.tolist(), n_points_for_category)
            
            for row, col in sampled_indices:
                # Convert pixel coordinates to geographic/map coordinates
                x, y = rasterio.transform.xy(transform, row, col, offset="center")
                points.append({"value": value, "geometry": Point(x, y)})
    
    # Create a GeoDataFrame for the points
    gdf = gpd.GeoDataFrame(points)
    gdf.set_crs(epsg=crs, inplace=True)
    
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    
    return gdf



if __name__ == '__main__':
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    raster = os.path.join(wks, 'data', 'training_data', 'training_raster.tif')
    mask = os.path.join(wks, 'data', 'masks', 'ocean_mask_binary.tif')
    
    gdf= generate_training_points(
        raster, 
        n_points=5000, 
        crs=3005,
    )
    
    gdf.to_file(os.path.join(wks, 'data', 'training_data', 'training_points.shp'))