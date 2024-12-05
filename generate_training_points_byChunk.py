"""
Generate training data for Land Cover classification.
    An n-number of points is generated for each land cover class
    based on a training raster. 
    
    The raster is process in chunks due to its large size (memory allocation issues)
    
Author: Moez Labiadh

"""

import os
import random
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from rasterio.windows import Window
from scipy.ndimage import binary_erosion
import timeit


def process_chunk(data_chunk, transform, dist_from_edge, n_points, unique_values):
    """
    Processes a single chunk of raster data to generate training points.

    Parameters
    ----------
    data_chunk : ndarray
        A chunk of the raster data.
    transform : Affine
        Affine transformation for the current chunk.
    dist_from_edge : int
        Buffer distance in pixels.
    n_points : int
        Number of points to sample per class.
    unique_values : list
        List of unique values (categories) in the raster.

    Returns
    -------
    list
        A list of points (as dictionaries) with class values and geometries.
    """
    points = []
    for value in unique_values:
        print(f"Processing value {value} in the current chunk...")
        category_mask = data_chunk == value
        eroded_mask = binary_erosion(
            category_mask,
            structure=np.ones((dist_from_edge * 2 + 1, dist_from_edge * 2 + 1))
        )

        valid_indices = np.argwhere(eroded_mask)
        print(f"Found {len(valid_indices)} valid pixels for value {value}.")

        if len(valid_indices) < n_points:
            print(f"Not enough pixels for value {value}, reducing points to {len(valid_indices)}.")
            n_points_for_category = len(valid_indices)
        else:
            n_points_for_category = n_points

        sampled_indices = random.sample(valid_indices.tolist(), n_points_for_category)
        for row, col in sampled_indices:
            x, y = rasterio.transform.xy(transform, row, col, offset="center")
            points.append({"value": value, "geometry": Point(x, y)})
    return points


def generate_training_points(raster_path, n_points=1000, crs=3005, dist_from_edge=5, chunk_size_pixels=10240):
    """
    Generates random points covering each class/value of a spatial raster in chunks.

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
    chunk_size_pixels : int
        Number of pixels in each chunk.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the generated points.
    """
    points = []
    print("\nOpening raster file...")
    with rasterio.open(raster_path) as src:
        unique_values = np.unique(src.read(1)[src.read(1) != src.nodata])
        print(f"Identified {len(unique_values)} unique values in the raster.")

        width, height = src.width, src.height
        chunk_cols = min(chunk_size_pixels, width)
        chunk_rows = min(chunk_size_pixels, height)
        print(f"Processing raster in chunks of size {chunk_cols} x {chunk_rows} pixels.")

        for col_off in range(0, width, chunk_cols):
            for row_off in range(0, height, chunk_rows):
                print(f"\nProcessing chunk at col: {col_off}, row: {row_off}...")
                window = Window(col_off, row_off, chunk_cols, chunk_rows)
                data_chunk = src.read(1, window=window)
                chunk_transform = src.window_transform(window)
                chunk_points = process_chunk(
                    data_chunk, chunk_transform, dist_from_edge, n_points, unique_values
                )
                points += chunk_points
                print(f"Generated {len(chunk_points)} points from the current chunk.")

    # Create GeoDataFrame from collected points
    print("Creating GeoDataFrame from generated points...")
    gdf = gpd.GeoDataFrame(points)
    gdf.set_crs(epsg=crs, inplace=True)
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    print("GeoDataFrame created successfully.")

    return gdf


def process_vector(gdf):
    """
    CLeansup and reproject the gdf

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing the generated points.

    Returns
    -------
    cleaned-up and ready-to-use GeoDataFrame 
        
    """
    gdf = gdf.rename(columns={'value': 'class_id'})

    gdf = gdf.to_crs(epsg=4326)
    
    gdf['latitude'] = gdf.geometry.y
    gdf['longitude'] = gdf.geometry.x
    
    return gdf
        
    


if __name__ == '__main__':
    start_t = timeit.default_timer()  # Start time

    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    raster = os.path.join(wks, 'data', 'training_data', 'training_raster.tif')

    # Generate training points with chunk processing
    gdf = generate_training_points(
        raster, 
        n_points=2000, 
        crs=3005, 
        dist_from_edge=5, 
        chunk_size_pixels=10240)

    # Save the resulting GeoDataFrame to a shapefile
    print("Saving the training points vector file...")
    gdf =  process_vector(gdf)
   
    output_file = os.path.join(wks, 'data', 'training_data', 'training_points.shp')
    gdf.to_file(output_file)

    
    finish_t = timeit.default_timer()  # Finish time
    t_sec = round(finish_t - start_t)
    mins = int(t_sec / 60)
    secs = int(t_sec % 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
