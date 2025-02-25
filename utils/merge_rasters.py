"""
Merge geotiff files into a single mosaic file.
"""



import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
import numpy as np

def mosaic_rasters(folder_path, output_filename, bit_depth, chunk_size):
    """
    Mosaic GeoTIFF files in chunks to handle large files without memory overflow.

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
    # Merge datasets and let rasterio handle alignment and transformation
    mosaic, mosaic_transform = merge(datasets)
    mosaic_crs = datasets[0].crs

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
    
    output_file_path = os.path.join(folder_path, output_filename)
    mosaic_height, mosaic_width = mosaic.shape[1], mosaic.shape[2]

    print("Creating mosaic output file...")
    with rasterio.open(
        output_file_path,
        'w',
        driver='GTiff',
        height=mosaic_height,
        width=mosaic_width,
        count=1,
        dtype=dtype_mapping[bit_depth],
        crs=mosaic_crs,
        transform=mosaic_transform,
        compress='LZW'
    ) as dst:
        print("Writing mosaic in chunks...")
        for row_start in range(0, mosaic_height, chunk_size):
            for col_start in range(0, mosaic_width, chunk_size):
                row_end = min(row_start + chunk_size, mosaic_height)
                col_end = min(col_start + chunk_size, mosaic_width)
                
                # Define the window for this chunk
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                
                # Extract the chunk from the mosaic
                chunk = mosaic[:, row_start:row_end, col_start:col_end]
                
                # Convert the chunk to the specified data type
                chunk = chunk.astype(dtype_mapping[bit_depth])
                
                # Write the chunk to the output file
                dst.write(chunk, window=window)
    
    print(f"Mosaic saved to {output_file_path}")



if __name__ == '__main__':
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    
    folder_path = os.path.join(wks, 'classification', 'mosaic')
    
    mosaic_rasters(
        folder_path=folder_path, 
        output_filename="mosaic_south.tif",
        bit_depth="uint8", 
        chunk_size=1024)