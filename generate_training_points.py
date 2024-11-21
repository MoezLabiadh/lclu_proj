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

def generate_training_points(raster_path, n_points=1000):
    # Open the raster file
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs
        
        # Identify unique categories/values, ignoring no-data values
        unique_values = np.unique(data[data != src.nodata])
        
        points = []
        
        for value in unique_values:
            print (f'Generating points for category {value} of {len(unique_values)}')
            # Create a mask for the current category/value
            category_mask = data == value
            
            # Erode the mask to avoid edge pixels
            eroded_mask = binary_erosion(category_mask, structure=np.ones((3, 3)))
            
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
                # Convert pixel coordinates to geographic coordinates
                x, y = rasterio.transform.xy(transform, row, col, offset="center")
                points.append({"ID": value, "latitude": y, "longitude": x, "geometry": Point(x, y)})
    
    # Create a GeoDataFrame from the points
    gdf = gpd.GeoDataFrame(points, crs=crs)
    
    
    return gdf



wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
raster= os.path.join(wks, 'data', 'vri_bclcs_l4.tif')

gdf= generate_training_points(raster, n_points=1000)

gdf.to_file(os.path.join(wks, 'data', 'training_points.shp'))