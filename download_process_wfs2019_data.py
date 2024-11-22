"""
Download and mosaic the DLR World Settelement Footprint 2019 data
https://download.geoservice.dlr.de/WSF2019/#download
"""

import os
import requests
import rasterio
from rasterio.merge import merge

def download_tifs(url_file, download_dir):
    """
    Download .tif files from a text file containing URLs.
    
    Args:
        url_file (str): Path to the text file containing URLs.
        download_dir (str): Directory to save downloaded .tif files.
    """
    os.makedirs(download_dir, exist_ok=True)  # Ensure the download directory exists

    with open(url_file, "r") as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()  # Remove any whitespace or newlines
        if url:  # Skip empty lines
            file_name = os.path.basename(url)  # Extract the file name from the URL
            download_path = os.path.join(download_dir, file_name)
            try:
                print(f"\nDownloading {file_name}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an error for bad status codes
                with open(download_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"..downloaded: {file_name}")
            except requests.exceptions.RequestException as e:
                print(f"..failed to download {url}: {e}")



def mosaic_tifs(input_dir, output_file):
    """
    Mosaic all .tif files in a directory into a single .tif file.
    
    Args:
        input_dir (str): Directory containing .tif files to mosaic.
        output_file (str): Path to the output .tif file.
    """
    tif_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".tif")]

    if not tif_files:
        print("No .tif files found in the directory.")
        return

    print("Reading .tif files for mosaicking...")
    src_files_to_mosaic = []
    for tif in tif_files:
        src = rasterio.open(tif)
        src_files_to_mosaic.append(src)

    print("Merging files...")
    mosaic, transform = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform
    })

    print(f"Saving mosaic to {output_file}...")
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print("Mosaicking completed.")




if __name__ == '__main__':
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    url_file = os.path.join(wks, 'data', 'built-up', 'WFS2019', 'WSF2019-url-list.txt')
    download_dir = os.path.join(wks, 'data', 'built-up', 'WFS2019', 'raw_geotiffs')
    output_mosaic = os.path.join(wks, 'data', 'built-up', 'WFS2019', 'mosaic')
    
    # Step 1: Download all .tif files
    download_tifs(url_file, download_dir)
    
    # Step 2: Mosaic the downloaded .tif files
    #mosaic_tifs(download_dir, output_mosaic)
