"""
Generate training data for Land Cover classification:

    A specified number of points (n) is generated for each land cover 
    class based on a training raster.

    The raster is processed in chunks due to its large size 
    to avoid memory allocation issues.

    The n_points parameter defines the number of points generated per chunk. 
    This number may be reduced if there are not enough available pixels.

    Edge pixels are excluded to prevent potential misalignment and mixed-pixels.
    
    The generated points are split 80/20 to create training and test datasets. 
    
Author: Moez Labiadh
"""

import os
import random
import logging
from typing import Generator

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import binary_erosion

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

import timeit

class TrainingPointGenerator:
    def __init__(
        self, 
        raster_path: str, 
        n_points: int = 1000, 
        crs: int = 3005, 
        dist_from_edge: int = 5, 
        chunk_size_pixels: int = 10240
    ):
        """
        Initialize the training point generator.

        Parameters
        ----------
        raster_path : str
            Path to the input raster file.
        n_points : int, optional
            Number of points to be generated for each class. Default is 1000.
        crs : int, optional
            EPSG code of the input raster CRS. Default is BC Albers (EPSG:3005).
        dist_from_edge : int, optional
            Buffer distance from the edge pixels in pixels. Default is 5.
        chunk_size_pixels : int, optional
            Number of pixels in each chunk. Default is 10240.
        """
        # Validate inputs
        self._validate_inputs(raster_path, n_points, dist_from_edge, chunk_size_pixels)
        
        self.raster_path = raster_path
        self.n_points = n_points
        self.crs = crs
        self.dist_from_edge = dist_from_edge
        self.chunk_size_pixels = chunk_size_pixels
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _validate_inputs(
        self, 
        raster_path: str, 
        n_points: int, 
        dist_from_edge: int, 
        chunk_size_pixels: int
    ):
        """
        Validate input parameters.

        Raises
        ------
        ValueError
            If inputs are invalid.
        FileNotFoundError
            If raster file does not exist.
        """
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        if n_points <= 0:
            raise ValueError("Number of points must be a positive integer.")
        
        if dist_from_edge < 0:
            raise ValueError("Distance from edge must be a non-negative integer.")
        
        if chunk_size_pixels <= 0:
            raise ValueError("Chunk size must be a positive integer.")

    def _generate_chunk_points(
        self, 
        data_chunk: np.ma.MaskedArray, 
        chunk_transform: rasterio.Affine
    ) -> Generator[dict, None, None]:
        """
        Generate points for a single raster chunk.

        Parameters
        ----------
        data_chunk : np.ma.MaskedArray
            A chunk of the raster data.
        chunk_transform : rasterio.Affine
            Affine transformation for the current chunk.

        Yields
        ------
        dict
            Dictionary containing point value and geometry.
        """
        # Identify unique values, excluding nodata
        unique_values = [val for val in np.unique(data_chunk.compressed()) if val != data_chunk.fill_value]
        
        for value in unique_values:
            self.logger.info(f"Processing value {value} in current chunk...")
            
            # Create category mask and erode edges
            category_mask = data_chunk == value
            eroded_mask = binary_erosion(
                category_mask,
                structure=np.ones((self.dist_from_edge * 2 + 1, self.dist_from_edge * 2 + 1))
            )

            # Find valid pixel indices
            valid_indices = np.argwhere(eroded_mask)
            
            if len(valid_indices) == 0:
                self.logger.warning(f"No valid pixels found for value {value}")
                continue
            
            # Determine number of points to sample
            n_points_for_category = min(self.n_points, len(valid_indices))
            
            # Randomly sample points
            sampled_indices = random.sample(valid_indices.tolist(), n_points_for_category)
            
            for row, col in sampled_indices:
                x, y = rasterio.transform.xy(chunk_transform, row, col, offset="center")
                yield {"value": value, "geometry": Point(x, y)}

    def generate_points(self) -> gpd.GeoDataFrame:
        """
        Generate random training points from the raster.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with generated training points.
        """
        points = []
        
        try:
            with rasterio.open(self.raster_path) as src:
                width, height = src.width, src.height
                
                # Determine chunk dimensions
                chunk_cols = min(self.chunk_size_pixels, width)
                chunk_rows = min(self.chunk_size_pixels, height)

                # Process raster in chunks
                for col_off in range(0, width, chunk_cols):
                    for row_off in range(0, height, chunk_rows):
                        self.logger.info(f"\nProcessing chunk at col: {col_off}, row: {row_off}")
                        
                        # Read raster chunk
                        window = Window(col_off, row_off, chunk_cols, chunk_rows)
                        data_chunk = src.read(1, window=window, masked=True)
                        chunk_transform = src.window_transform(window)
                        
                        # Generate and collect points for this chunk
                        points.extend(self._generate_chunk_points(data_chunk, chunk_transform))
                        
                        self.logger.info(f"Generated {len(points)} points so far...")

        except Exception as e:
            self.logger.error(f"Error processing raster: {e}")
            raise

        # Create GeoDataFrame
        self.logger.info("Creating GeoDataFrame from generated points...")
        gdf = gpd.GeoDataFrame(points)
        gdf.set_crs(epsg=self.crs, inplace=True)
        
        # Add latitude and longitude columns
        gdf['latitude'] = gdf.geometry.y
        gdf['longitude'] = gdf.geometry.x
        
        self.logger.info("Point generation complete.")
        
        return gdf

def process_train_test(
    gdf: gpd.GeoDataFrame, 
    output_path: str, 
    target_crs: int = 4326
) -> gpd.GeoDataFrame:
    """
    Split the generated points into train and test (80/20)
    and save to seperate files.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame of generated points.
    output_path : str
        Path to save the output files.
    target_crs : int, optional
        Target coordinate reference system. Default is WGS84 (EPSG:4326).

    Returns
    -------
    gpd.GeoDataFrame
        Processed GeoDataFrame.
    """
    # Rename value column
    gdf = gdf.rename(columns={'value': 'class_id'})
    
    # Print the number of rows per class_id
    class_counts = gdf['class_id'].value_counts()
    print("Number of rows per class_id:")
    print(class_counts)

    # Reproject to target CRS
    gdf = gdf.to_crs(epsg=target_crs)
    
    # Ensure lat/long columns are updated
    gdf['latitude'] = gdf.geometry.y
    gdf['longitude'] = gdf.geometry.x
    
    # Split 80/20 based on class_id
    train_list = []
    test_list = []

    # Split by unique class_id
    for class_id in gdf['class_id'].unique():
        gdf_class = gdf[gdf['class_id'] == class_id]
        
        # Perform train-test split for each class
        train, validate = train_test_split(gdf_class, test_size=0.2, random_state=42)
        
        # Collect the splits
        train_list.append(train)
        test_list.append(validate)

    # Concatenate into final GeoDataFrames
    gdf_train = gpd.GeoDataFrame(pd.concat(train_list, ignore_index=True), crs=gdf.crs)
    gdf_test= gpd.GeoDataFrame(pd.concat(test_list, ignore_index=True), crs=gdf.crs)
    
    # Export to file
    gdf_train.to_file(os.path.join(output_path, 'points_train.shp'))
    gdf_test.to_file(os.path.join(output_path, 'points_test.shp'))
    
    return gdf_train, gdf_test



if __name__ == '__main__':
    start_t = timeit.default_timer()
    
    try:
        wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
        raster = os.path.join(wks, 'data', 'training_data', 'training_raster_v6.tif')
        
        # Generate training points
        generator = TrainingPointGenerator(
            raster_path=raster, 
            n_points=1000, 
            crs=3005, 
            dist_from_edge=4, 
            chunk_size_pixels=20480
        )
        gdf = generator.generate_points()
        
        # Save processed points
        output_path = os.path.join(wks, 'data', 'training_data')
        gdf_train, gdf_test = process_train_test(gdf, output_path)
        
        # Calculate processing time
        finish_t = timeit.default_timer()
        t_sec = round(finish_t - start_t)
        mins, secs = divmod(t_sec, 60)
        print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise