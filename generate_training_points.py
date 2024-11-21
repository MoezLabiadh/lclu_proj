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

def generate_training_points(raster_path, n_points=1000, crs=3005, dist_from_border=100):
    """
    Generates random points covering each class/value of a spatial raster

    Parameters
    ----------
    n_points : int
        Number of points to be generated for each class.
    dist_from_border : float
        Buffer distance from the edge pixels in meters.
    crs : int
        EPSG code of the input raster CRS. Default is BC Albers (EPSG:3005).
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        pixel_size = (transform[0], -transform[4])  # (width, height) of a pixel in meters

        # Calculate buffer in pixels
        buffer_pixels_x = int(dist_from_border / pixel_size[0])
        buffer_pixels_y = int(dist_from_border / pixel_size[1])

        # Identify unique categories/values, ignoring no-data values
        unique_values = np.unique(data[data != src.nodata])
        
        points = []
        
        for value in unique_values:
            print(f"Generating points for category {value} of {len(unique_values)}")
            # Create a mask for the current category/value
            category_mask = data == value
            
            # Erode the mask to avoid edge pixels based on the buffer distance
            eroded_mask = binary_erosion(
                category_mask,
                structure=np.ones((buffer_pixels_y * 2 + 1, buffer_pixels_x * 2 + 1))
            )
            
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
    

    gdf = gpd.GeoDataFrame(points)
    gdf.set_crs(epsg=crs, inplace=True)
    
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    

    return gdf



if __name__ == '__main__':
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    raster= os.path.join(wks, 'data', 'vri_bclcs_l4.tif')
    
    gdf= generate_training_points(raster, n_points=1000, crs=3005)
    
    gdf.to_file(os.path.join(wks, 'data', 'training_points_vri.shp'))