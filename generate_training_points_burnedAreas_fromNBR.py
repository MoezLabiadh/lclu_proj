"""
Generate traning data for burned areas from NBR (Normalized Burned Area) rasters
"""

import os
import random
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def generate_random_points(input_folder, min_value, max_value):
    """
    Generate random points within raster bounds where pixel values fall within a specified range.

    Parameters:
    input_folder (str): Path to the folder containing GeoTIFF rasters.
    min_value (float): Minimum pixel value.
    max_value (float): Maximum pixel value.

    Returns:
    GeoDataFrame: A GeoDataFrame containing aggregated random points from all rasters.
    """
    all_points = []
    all_pixel_values = []

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".tif"):
            file_path = os.path.join(input_folder, file_name)

            print(f"Working on {file_name}")

            with rasterio.open(file_path) as src:
                if src.crs.to_string() != "EPSG:4326":
                    print(f"Skipping {file_name}: Not in EPSG:4326.")
                    continue

                # Get raster bounds and transformation
                bounds = src.bounds

                # Generate 500 random points within the raster bounds
                points = []
                pixel_values = []
                while len(points) < 500:
                    x = random.uniform(bounds.left, bounds.right)
                    y = random.uniform(bounds.bottom, bounds.top)

                    # Get pixel row and column
                    row, col = src.index(x, y)

                    # Read the pixel value
                    try:
                        window = Window(col, row, 1, 1)
                        data = src.read(1, window=window)
                        value = data[0, 0]

                        # Check if the pixel value is in the desired range
                        if min_value <= value <= max_value:
                            points.append(Point(x, y))
                            pixel_values.append(value)
                    except IndexError:
                        # Skip if the point is outside the raster bounds
                        continue

                # Append results from the current raster to the aggregated list
                all_points.extend(points)
                all_pixel_values.extend(pixel_values)

    # Create a single aggregated GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'class_id': [10] * len(all_points),
        'pixel_value': all_pixel_values,
        'geometry': all_points  
    }, crs="EPSG:4326")

    return gdf



if __name__ == '__main__':
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    
    nbr_folder= os.path.join(wks, "data", "burned_areas")
    gdf= generate_random_points(nbr_folder, -1, -0.4)
    
    # Save the GeoDataFrame to a shapefile
    output_file = os.path.join(wks, "data", "burned_areas", "nbr_extracted_points.shp")
    gdf.to_file(output_file)