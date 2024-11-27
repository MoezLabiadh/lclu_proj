"""
Upload an asset (geojson) as FeatureCollection to EE
    e.g. asset_id= "users/your_username/training_data"
"""

import os
import subprocess

def upload_geojson_to_gee(geojson_path, asset_id= ''):
    """
    Uploads a GeoJSON file to Google Earth Engine as a FeatureCollection.

    Parameters:
    -----------
    geojson_path : str
        Path to the GeoJSON file to upload.
    asset_id : str
        Full GEE asset ID where the file will be uploaded 
        (e.g., "users/your_username/training_data").
    
    Returns:
    --------
    None
    """
    # Check if the file exists
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    
    # Construct the Earth Engine CLI command
    command = [
        "earthengine", "upload", "table", geojson_path,
        "--asset_id=" + asset_id
    ]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        print(f"GeoJSON successfully uploaded to: {asset_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error during upload: {e}")