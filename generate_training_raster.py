"""
Create a training raster by combining existing land cover rasters:
    ESA WorldCover, ESRI Lnad Cover, Natural Resource Canada Land Cover, 
    DLR World Settelement Footprint (WSF).
    
    The rasters were resamples and aligned beforehand using the ESA raster as refrenece.
    The rasters are processed in chunks due to their large size (memory allocation issues).
    
Author: Moez Labiadh
"""

import os
import pandas as pd
import numpy as np
import rasterio
import timeit


def create_training_raster(raster_paths, output_path, px_values, crs, chunk_size=1024):
    """
    Classify raster pixels in chunks based on provided rules.
    
    Parameters:
    raster_paths (dict): Dictionary containing paths to input rasters with keys 'esa', 'nrcan', 'esri', and 'wfs'.
    output_path (str): Path to save the classified raster.
    px_values (pd.DataFrame): DataFrame containing classification rules with columns 'class_id', 'esa_value',
                              'nrcan_value', 'esri_value', and 'wfs_value'.
    crs (str): Coordinate Reference System for the output raster.
    chunk_size (int): Size of chunks (in pixels) for processing.
    """
    print('\nReading input raster metadata')
    with rasterio.open(raster_paths['esa']) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        transform = src.transform

    profile.update(
        dtype=rasterio.uint8, 
        count=1, 
        nodata=0, 
        compress='lzw'
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        print('\nProcessing in chunks...')
        for row_start in range(0, height, chunk_size):
            for col_start in range(0, width, chunk_size):
                # Calculate the actual chunk size
                chunk_height = min(chunk_size, height - row_start)
                chunk_width = min(chunk_size, width - col_start)

                # Define the window precisely
                window = rasterio.windows.Window(
                    col_start, row_start,
                    chunk_width,
                    chunk_height
                )

                # Read data for the current window
                print(f'Processing window: {window}')
                esa_data = rasterio.open(raster_paths['esa']).read(1, window=window)
                nrcan_data = rasterio.open(raster_paths['nrcan']).read(1, window=window)
                esri_data = rasterio.open(raster_paths['esri']).read(1, window=window)
                wfs_data = rasterio.open(raster_paths['wfs']).read(1, window=window)

  
                # Debug: Print shapes of input arrays
                print("Input array shapes:")
                print(f"ESA: {esa_data.shape}")
                print(f"NRCAN: {nrcan_data.shape}")
                print(f"ESRI: {esri_data.shape}")
                print(f"WSF: {wfs_data.shape}")
 
                # Ensure all arrays have the same shape by trimming to the smallest
                min_height = min(esa_data.shape[0], nrcan_data.shape[0], 
                                 esri_data.shape[0], wfs_data.shape[0])
                min_width = min(esa_data.shape[1], nrcan_data.shape[1], 
                                esri_data.shape[1], wfs_data.shape[1])

                esa_data = esa_data[:min_height, :min_width]
                nrcan_data = nrcan_data[:min_height, :min_width]
                esri_data = esri_data[:min_height, :min_width]
                wfs_data = wfs_data[:min_height, :min_width]


                # Initialize output chunk
                output_chunk = np.zeros_like(esa_data, dtype=np.uint8)

                # Apply classification rules
                for _, rule in px_values.iterrows():
                    # Create condition mask for this rule
                    condition = np.ones_like(esa_data, dtype=bool)
                    
                    # Apply each non-NaN value condition
                    if not np.isnan(rule['esa_value']):
                        condition &= (esa_data == rule['esa_value'])
                    if not np.isnan(rule['nrcan_value']):
                        condition &= (nrcan_data == rule['nrcan_value'])
                    if not np.isnan(rule['esri_value']):
                        condition &= (esri_data == rule['esri_value'])
                    if not np.isnan(rule['wfs_value']):
                        condition &= (wfs_data == rule['wfs_value'])
                    
                    # Assign class ID where condition is met
                    output_chunk[condition] = np.uint8(rule['class_id'])

                # Write the output chunk with its precise window
                dst.write(output_chunk, 1, window=window)


if __name__ == '__main__':
    start_t = timeit.default_timer()  # Start time
    
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    xlsx_path = os.path.join(wks, 'documents', 'classification_schema.xlsx')  # xlsx file containing pixel values
    values_df = pd.read_excel(xlsx_path, 'training_pixel_values')

    input_rasters_path= os.path.join(wks, 'data', 'training_data', 'input_rasters')
    raster_paths = {
        "esa": os.path.join(wks, 'data', 'existing_data', 'esa', 'esa_lc_10m_mosaic_bc.tif'),
        "nrcan": os.path.join(input_rasters_path, 'aligned_nrcan_lc_10m_bc.tif'),
        "esri": os.path.join(input_rasters_path, 'aligned_esri_lc_10m_mosaic_bc.tif'),
        "wfs": os.path.join(input_rasters_path, 'aligned_wfs_10m_mosaic_bc_4.tif'),
    }
    
    output_path = os.path.join(wks, 'data', 'training_data', 'training_raster_v5.tif')
    
    create_training_raster(
        raster_paths, 
        output_path, 
        values_df, 
        crs='EPSG:3005', 
        chunk_size=20480
    )

    finish_t = timeit.default_timer()  # Finish time
    t_sec = round(finish_t - start_t)
    mins = int(t_sec / 60)
    secs = int(t_sec % 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')