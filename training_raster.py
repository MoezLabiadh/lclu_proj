"""
Create a training raster by combining existing land cover rasters:
    ESA WorldCover, ESRI Lnad Cover, Natural Resource Canada Land Cover, 
    DLR World Settelement Footprint (WSF).
"""

import os
import pandas as pd
import numpy as np
import rasterio
import timeit


def create_training_raster(raster_paths, output_path, px_values, crs, chunk_size=1024):
    """
    Classify raster pixels in chunks based on provided rules
    
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
        profile = src.profile  # Save profile to use for output raster
        height, width = src.height, src.width
        transform = src.transform

    # Update the profile for the output raster
    profile.update(dtype=rasterio.uint8, count=1, nodata=0, crs=crs, compress='lzw')

    # Initialize the output raster
    print(f'\nCreating output raster at {output_path}')
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Process in chunks
        print('\nProcessing in chunks...')
        for row_start in range(0, height, chunk_size):
            for col_start in range(0, width, chunk_size):
                # Define the window
                window = rasterio.windows.Window(
                    col_start, row_start,
                    min(chunk_size, width - col_start),
                    min(chunk_size, height - row_start)
                )

                # Create a consistent transform for the window
                window_transform = rasterio.windows.transform(window, transform)

                # Read the data for the current window
                print(f'Processing window: {window}')
                esa_data = rasterio.open(raster_paths['esa']).read(1, window=window, boundless=True, fill_value=0)
                nrcan_data = rasterio.open(raster_paths['nrcan']).read(1, window=window, boundless=True, fill_value=0)
                esri_data = rasterio.open(raster_paths['esri']).read(1, window=window, boundless=True, fill_value=0)
                wfs_data = rasterio.open(raster_paths['wfs']).read(1, window=window, boundless=True, fill_value=0)

                # Ensure all arrays have the same shape
                esa_data, nrcan_data, esri_data, wfs_data = [
                    np.pad(
                        data, 
                        ((0, chunk_size - data.shape[0]), (0, chunk_size - data.shape[1])), 
                        constant_values=0
                    ) 
                    if data.shape != (chunk_size, chunk_size) else data
                    for data in [esa_data, nrcan_data, esri_data, wfs_data]
                ]

                # Initialize output chunk
                output_chunk = np.zeros_like(esa_data, dtype=np.uint8)

                # Apply classification rules
                for _, rule in px_values.iterrows():
                    condition = np.ones_like(esa_data, dtype=bool)
                    if not np.isnan(rule['esa_value']):
                        condition &= (esa_data == rule['esa_value'])
                    if not np.isnan(rule['nrcan_value']):
                        condition &= (nrcan_data == rule['nrcan_value'])
                    if not np.isnan(rule['esri_value']):
                        condition &= (esri_data == rule['esri_value'])
                    if not np.isnan(rule['wfs_value']):
                        condition &= (wfs_data == rule['wfs_value'])
                    
                    output_chunk[condition] = np.uint8(rule['class_id'])

                # Write the output chunk to the output raster
                dst.write(output_chunk, 1, window=window)

    print('\nClassification complete.')



if __name__ == '__main__':
    start_t = timeit.default_timer()  # Start time
    
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    xlsx_path = os.path.join(wks, 'documents', 'classification_schema.xlsx')  # xlsx file containing pixel values
    values_df = pd.read_excel(xlsx_path, 'training_pixel_values')

    raster_paths = {
        "esa": os.path.join(wks, 'data', 'existing_data', 'esa', 'esa_lc_10m_mosaic_bc.tif'),
        "nrcan": os.path.join(wks, 'data', 'existing_data', 'nrcan', 'nrcan_lc_10m_bc.tif'),
        "esri": os.path.join(wks, 'data', 'existing_data', 'esri', 'esri_lc_10m_mosaic_bc.tif'),
        "wfs": os.path.join(wks, 'data', 'existing_data', 'wsf', 'wfs_10m_mosaic_bc_4.tif'),
    }
    
    output_path = os.path.join(wks, 'data', 'training_data', 'training_raster_v2.tif')
    
    create_training_raster(
        raster_paths, 
        output_path, 
        values_df, 
        crs='EPSG:3005', 
        chunk_size=10240
    )

    finish_t = timeit.default_timer()  # Finish time
    t_sec = round(finish_t - start_t)
    mins = int(t_sec / 60)
    secs = int(t_sec % 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
