import os
import random
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from rasterio.windows import Window
from scipy.ndimage import binary_erosion


def process_chunk(data_chunk, mask_chunk, transform, dist_from_edge, n_points, unique_values):
    """
    Processes a single chunk of raster data to generate training points.

    Parameters
    ----------
    data_chunk : ndarray
        A chunk of the raster data.
    mask_chunk : ndarray
        Corresponding chunk of the mask raster (if provided).
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

        # Handle mask alignment
        if mask_chunk is not None:
            if eroded_mask.shape != mask_chunk.shape:
                print("Resizing mask to match chunk dimensions...")
                mask_chunk = mask_chunk[:eroded_mask.shape[0], :eroded_mask.shape[1]]
            eroded_mask &= (mask_chunk != 1)

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


def generate_training_points(raster_path, n_points=1000, crs=3005, dist_from_edge=5, mask_path=None, chunk_size_pixels=10240):
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
    mask_path : str, optional
        Path to a raster mask where pixel values of 1 indicate areas to exclude.
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

        mask_data = None
        if mask_path:
            print("\nOpening mask raster file...")
            with rasterio.open(mask_path) as mask_src:
                mask_data = mask_src.read(1)
            print("Mask raster loaded.")

        width, height = src.width, src.height
        chunk_cols = min(chunk_size_pixels, width)
        chunk_rows = min(chunk_size_pixels, height)
        print(f"Processing raster in chunks of size {chunk_cols} x {chunk_rows} pixels.")

        for col_off in range(0, width, chunk_cols):
            for row_off in range(0, height, chunk_rows):
                print(f"\nProcessing chunk at col: {col_off}, row: {row_off}...")
                window = Window(col_off, row_off, chunk_cols, chunk_rows)
                data_chunk = src.read(1, window=window)
                mask_chunk = None
                if mask_data is not None:
                    mask_chunk = mask_data[window.toslices()]

                chunk_transform = src.window_transform(window)
                chunk_points = process_chunk(
                    data_chunk, mask_chunk, chunk_transform, dist_from_edge, n_points, unique_values
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


if __name__ == '__main__':
    # Define file paths and parameters
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    raster = os.path.join(wks, 'data', 'training_data', 'training_raster.tif')
    mask = os.path.join(wks, 'data', 'masks', 'ocean_mask_binary.tif')

    # Generate training points with chunk processing
    gdf = generate_training_points(
        raster, 
        n_points=5000, 
        crs=3005, 
        dist_from_edge=2, 
        mask_path=mask,
        chunk_size_pixels=10240)

    # Save the resulting GeoDataFrame to a shapefile
    print("Saving the gdf to shapefile")
    output_file = os.path.join(wks, 'data', 'training_points_vri_with_mask_chunks.shp')
    gdf.to_file(os.path.join(wks, 'data', 'training_data', 'training_points.shp'))