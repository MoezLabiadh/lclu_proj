"""
Merge geotiff files into a single mosaic file.
"""

import os
import glob
import rasterio
from rasterio.merge import merge
import numpy as np

def mosaic_rasters(folder_path, output_filename, bit_depth, chunk_size):
    """
    Mosaic GeoTIFF files.

    Args:
        folder_path (str): Path to the folder containing GeoTIFF files.
        output_filename (str): Name of the output file.
        bit_depth (str): Pixel depth ('uint8', 'uint16', 'int16', 'float32', 'float64').
        chunk_size (int): Size of chunks (in pixels).
    """
    
    tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))
    if not tiff_files:
        print("No GeoTIFF files found in the specified folder.")
        return
    
    print("Reading GeoTIFF files...")
    datasets = [rasterio.open(f) for f in tiff_files]
    
    print(f"Merging {len(datasets)} GeoTIFF files...")
    mosaic, mosaic_transform = merge(datasets)
    
    # Define the output data type based on the specified bit depth
    dtype_mapping = {
        'uint8': 'uint8',
        'uint16': 'uint16',
        'int16': 'int16',
        'float32': 'float32',
        'float64': 'float64'
    }
    
    if bit_depth not in dtype_mapping:
        print(f"Invalid bit depth specified: {bit_depth}")
        return
    
    
    # Convert the mosaic to the specified data type
    print("Writing mosaic file...")
    mosaic = mosaic.astype(dtype_mapping[bit_depth])
    
    # Define the output file path
    output_file_path = os.path.join(folder_path, output_filename)
    
    # Save the mosaic to a new GeoTIFF file
    with rasterio.open(
        output_file_path,
        'w',
        driver='GTiff',
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype=dtype_mapping[bit_depth],
        crs=datasets[0].crs,
        transform=mosaic_transform,
        compress='LZW'
    ) as dst:
        dst.write(mosaic)
    
    print(f"Mosaic saved to {output_file_path}")





if __name__ == '__main__':
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    
    folder_path = os.path.join(wks, 'data', 'existing_data', 'wsf', 'test')
    
    mosaic_rasters(
        folder_path=folder_path, 
        output_filename="wfs_30m_mosaic.tif",
        bit_depth="uint8", 
        chunk_size=1024)
