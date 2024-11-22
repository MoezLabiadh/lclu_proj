"""
Download DLR World Settelement Footprint 2019 data
https://download.geoservice.dlr.de/WSF2019/#download
"""

import os
import requests
import time


def download_tifs(url_file, download_dir):
    """
    Download .tif files from a text file containing URLs
    
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
                
                # Wait 15 seconds before next download to avoid being blocked by the server
                time.sleep(15)
            
            except requests.exceptions.RequestException as e:
                print(f"..failed to download {url}: {e}")






if __name__ == '__main__':
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    url_file = os.path.join(wks, 'data', 'existing_data', 'wsf', 'WSF2019-url-list.txt')
    download_dir = os.path.join(wks, 'data', 'existing_data', 'wsf', 'raw_geotiffs')
    output_mosaic = os.path.join(wks, 'data', 'built-up', 'WFS2019', 'mosaic')
    
    download_tifs(url_file, download_dir)
