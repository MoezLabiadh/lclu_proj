import os
import rasterio
from rasterio.windows import Window
from scipy.ndimage import binary_erosion
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import timeit


def generate_random_points(geo_transform, mask, num_points, crs):
    """
    Generate random points within areas where the mask is True.

    Args:
    - geo_transform: Geotransform of the raster (rasterio transform object)
    - mask: 2D boolean mask indicating valid areas
    - num_points: Number of points to generate
    - crs: Coordinate reference system

    Returns:
    - GeoDataFrame of randomly generated points
    """
    rows, cols = np.where(mask)
    available_points = len(rows)
    num_points_generated = min(num_points, available_points)  # Adjust if not enough valid pixels
    selected_indices = np.random.choice(available_points, num_points_generated, replace=False)
    selected_rows = rows[selected_indices]
    selected_cols = cols[selected_indices]

    # Convert row/col to geographic coordinates
    points = [
        Point(rasterio.transform.xy(geo_transform, row, col))
        for row, col in zip(selected_rows, selected_cols)
    ]
    return gpd.GeoDataFrame(geometry=points, crs=crs), num_points_generated


def process_land_cover_class(class_row, raster_paths, num_points, crs, window_size=1024):
    """
    Generate random points for a single class based on pixel values from multiple rasters.

    Args:
    - class_row: Row from the class-matching DataFrame
    - raster_paths: Dictionary of raster paths
    - num_points: Number of points to generate for each class
    - crs: Coordinate reference system
    - window_size: Size of chunks to process the raster (default is 1024x1024)

    Returns:
    - Tuple containing:
        - GeoDataFrame with points for this class
        - Number of points actually generated
    """
    masks = []

    for raster_key, value in class_row.items():
        if raster_key in raster_paths and pd.notna(value):
            print(f"Processing {raster_key} raster for class {class_row['class_id']} ({class_row['class_name']})")
            with rasterio.open(raster_paths[raster_key]) as src:
                height, width = src.height, src.width
                geo_transform = src.transform
                combined_mask = np.zeros((height, width), dtype=bool)

                for row_start in range(0, height, window_size):
                    for col_start in range(0, width, window_size):
                        row_end = min(row_start + window_size, height)
                        col_end = min(col_start + window_size, width)

                        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                        data = src.read(1, window=window)

                        # Debugging: Check the value range and data type of the chunk
                        print(f"Chunk ({row_start}:{row_end}, {col_start}:{col_end}) stats: "
                              f"Min={data.min()}, Max={data.max()}, dtype={data.dtype}")

                        mask_chunk = (data == value)
                        if mask_chunk.any():  # Check if any pixels match the value
                            print(f"Valid pixels found in chunk ({row_start}:{row_end}, {col_start}:{col_end}) for value {value}")

                        eroded_chunk = binary_erosion(mask_chunk, structure=np.ones((3, 3)))
                        combined_mask[row_start:row_end, col_start:col_end] = eroded_chunk

                if not combined_mask.any():
                    print(f"No valid pixels found for {raster_key} raster and class {class_row['class_id']} ({class_row['class_name']}).")
                masks.append(combined_mask)

    if masks:
        final_mask = np.logical_and.reduce(masks)
        return generate_random_points(geo_transform, final_mask, num_points, crs)
    else:
        raise ValueError("No valid masks for this class.")


def process_all_classes(class_matching_df, raster_paths, num_points, crs, window_size=1024):
    """
    Process all classes and generate random points using chunk-based processing.

    Args:
    - class_matching_df: DataFrame containing the class matching table
    - raster_paths: Dictionary of raster paths with keys 'esa', 'nrcan', 'esri', 'wfs'
    - num_points: Number of points to generate for each class
    - crs: Coordinate reference system
    - window_size: Size of chunks to process rasters (default is 1024x1024)

    Returns:
    - GeoDataFrame containing points for all classes
    """
    all_points = []

    for i, class_row in class_matching_df.iterrows():
        print(f"\n...processing class {class_row['class_id']} ({class_row['class_name']}) [{i+1}/{len(class_matching_df)}]")
        try:
            points_gdf, num_points_generated = process_land_cover_class(
                class_row, raster_paths, num_points, crs, window_size
            )
            points_gdf["class_id"] = class_row["class_id"]
            points_gdf["class_name"] = class_row["class_name"]
            all_points.append(points_gdf)
            print(f"...generated {num_points_generated} points for class {class_row['class_id']} ({class_row['class_name']})")
        except ValueError as e:
            print(f"...warning: {e} for class {class_row['class_id']} ({class_row['class_name']}). Skipping.")

    if all_points:
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True))
        print("Finished processing all classes.")
        return combined_gdf
    else:
        print("No points generated for any class.")
        return gpd.GeoDataFrame(geometry=[])


if __name__ == '__main__':
    start_t = timeit.default_timer()  # Start time

    # Define working directory and file paths
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    xlsx_path = os.path.join(wks, 'documents', 'classification_schema.xlsx')  # xlsx file containing training rules

    raster_paths = {
        "esa": os.path.join(wks, 'data', 'existing_data', 'esa', 'esa_lc_10m_mosaic_bc.tif'),
        "nrcan": os.path.join(wks, 'data', 'existing_data', 'nrcan', 'nrcan_lc_30m_bc.tif'),
        "esri": os.path.join(wks, 'data', 'existing_data', 'esri', 'esri_lc_10m_mosaic_bc.tif'),
        "wfs": os.path.join(wks, 'data', 'existing_data', 'wsf', 'wfs_10m_mosaic_bc.tif'),
    }

    print('\nReading the training rules')
    class_matching_df = pd.read_excel(xlsx_path, 'training_pixel_values')

    print('\nRunning the process')
    gdf = process_all_classes(
        class_matching_df,
        raster_paths,
        num_points=5000,
        crs="EPSG:3005",  # Specify CRS as EPSG:3005
        window_size=1024  # Chunk size
    )

    # Save the final GeoDataFrame
    print('\nSaving the output file')
    output_path = os.path.join(wks, 'data', 'training_points.shp')
    gdf.to_file(output_path)

    finish_t = timeit.default_timer()  # Finish time
    t_sec = round(finish_t - start_t)
    mins = int(t_sec / 60)
    secs = int(t_sec % 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
