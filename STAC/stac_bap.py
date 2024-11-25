"""
An attempt to apply the Best-Available-Pixel (BAP) compositing approach
using Sentinel-2 aquired via STAC (Planetary Computer).

Based on White et al. (2014), Pixel-Based Image Compositing for Large-Area Dense 
 Time Series Applications and Science -
 https://www.tandfonline.com/doi/full/10.1080/07038992.2014.945827 
  
                           
The scoring functions rank each S2 pixel for: 
    
    (i) proximity to the target date, 
            Quantifies how close the image's date is to a specified target_date, 
            with higher scores given to images taken nearer the target date.
            A Gaussian-like decay function assigns the DateScore.
            
 
    (ii) cloud cover in the scene, 
            is calculated as a measure of how cloud-free the image is over the 
            specified area of interest (aoi).
            
    (iii) distance to clouds
            If the distance from cloudy pixels is greater than 150 meters, the score is set to 1, 
            indicating the pixel is sufficiently far from clouds.
            If the distance is less than or equal to 150 meters, 
            a Gaussian-like decay function assigns a score based on proximity.
    
    
Author: Moez Labiadh    
"""

import os

import numpy as np
import xarray as xr
import rasterio

import stackstac
from pystac_client import Client
from planetary_computer import sign


import geopandas as gpd
from shapely.geometry import box

from datetime import datetime, timedelta


def query_stac_sentinel2(aoi, target_date, cloud_pct=30, buffer_days=45):
    start_date = (target_date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    end_date = (target_date + timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=aoi.total_bounds,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_pct}}
    )
    
    items = [sign(item) for item in search.items()]
    
    # Verify SCL availability in assets
    for item in items:
        if "SCL" not in item.assets:
            raise KeyError("SCL asset is missing in one or more items.")
    
    ds = stackstac.stack(
        items,
        assets=["B02", "B03", "B04", "B8A", "SCL"],
        resolution=10,
        bounds_latlon=aoi.total_bounds,
        chunksize=2048,
    )
    
    return ds



def compute_date_score(dates, target_date):
    days_diff = (dates - np.datetime64(target_date)) / np.timedelta64(1, 'D')
    
    return np.exp(-0.5 * (days_diff / 15) ** 2)



def compute_distance_score(scl_data):
    scl_array = scl_data.values
    cloud_mask = (scl_array == 8) | (scl_array == 9)
    distance = np.where(cloud_mask, 0, np.inf)
    
    for i in range(1, distance.shape[0] - 1):
        for j in range(1, distance.shape[1] - 1):
            if cloud_mask[i, j]:
                continue
            dist = min(distance[i - 1:i + 2, j - 1:j + 2].ravel()) + 10
            distance[i, j] = min(distance[i, j], dist)
    
    distance_score = np.exp(-0.5 * (distance / 50) ** 2)
    
    return xr.DataArray(
        distance_score,
        dims=scl_data.dims,
        coords=scl_data.coords
    )


def compute_coverage_score(ds):
    """
    Compute the coverage score based on the cloud cover percentage for each time step.
    """
    # Retrieve the SCL band
    scl_data = ds.sel(band="SCL")
    
    # Cloud mask: classify cloudy pixels (values 8 and 9 in SCL are clouds)
    cloud_mask = (scl_data == 8) | (scl_data == 9)
    
    # Compute the cloud cover percentage for each time step
    cloud_fraction = cloud_mask.mean(dim=["x", "y"])
    
    # Convert cloud fraction to coverage score (inverse exponential decay)
    coverage_score = xr.DataArray(
        1 - cloud_fraction,  # Inverse cloud fraction
        dims=["time"],
        coords={"time": ds["time"]},
    ).clip(0.1, 1.0)  # Ensure minimum score is 0.1
    
    return coverage_score


def compute_quality_score(ds, target_date):
    """
    Compute the overall quality score for BAP compositing
    """
    # Compute date scores
    date_scores = compute_date_score(ds["time"].values, target_date)
    date_scores_da = xr.DataArray(
        date_scores,
        dims=["time"],
        coords={"time": ds["time"]},
    )
    date_scores_da = date_scores_da.expand_dims({"x": ds["x"], "y": ds["y"]}).transpose("time", "y", "x")
    
    # Compute distance scores
    scl_data = ds.sel(band="SCL")
    dist_scores = scl_data.map_blocks(compute_distance_score)
    
    # Compute coverage scores
    coverage_scores = compute_coverage_score(ds).expand_dims({"x": ds["x"], "y": ds["y"]}).transpose("time", "y", "x")
    
    # Adjust weights: penalize clouds more heavily
    quality_score = (date_scores_da * dist_scores**2 * coverage_scores**2)
    
    return quality_score



def create_composite(ds, quality_score):
    """
    Create a BAP composite using the quality_score to select the best pixel for each location.
    """
    best_quality_time = quality_score.argmax(dim="time").compute()  # Convert to NumPy
    
    selected_bands = ds.sel(band=["B02", "B03", "B04", "B8A"])
    composites = []
    for band in selected_bands["band"].values:
        band_data = selected_bands.sel(band=band)
        band_composite = band_data.isel(time=best_quality_time)
        composites.append(band_composite)
    
    composite = xr.concat(composites, dim="band")
    composite["band"] = ["B02", "B03", "B04", "B8A"]  # Assign band names
    
    return composite



def reproject_to_bc_albers(composite, target_crs="EPSG:3005"):
    """
    Reproject the composite dataset to BC Albers (EPSG:3005), handling dimension conflicts.
    """
    # Automatically detect and set CRS if not already defined
    if not composite.rio.crs:
        detected_crs = composite.attrs.get("crs", None)
        if detected_crs:
            print(f"..detected CRS from attributes: {detected_crs}")
            composite = composite.rio.write_crs(detected_crs, inplace=True)
        else:
            raise ValueError("..no CRS detected. Please check the dataset or provide a default CRS.")
    
    # Drop any conflicting or unnecessary attributes
    print("..cleaning dataset to resolve dimension conflicts...")
    composite = composite.reset_coords(drop=True)  # Drops extra metadata/coordinates that conflict with dimensions
    
    # Debugging input CRS and bounds
    print("..input CRS:", composite.rio.crs)
    print("..spatial bounds before reprojection:", composite.rio.bounds())
    
    # Reproject to the target CRS
    composite_reprojected = composite.rio.reproject(target_crs)
    
    # Debugging output CRS and bounds
    print("..output CRS:", composite_reprojected.rio.crs)
    print("..spatial bounds after reprojection:", composite_reprojected.rio.bounds())
    
    return composite_reprojected



def export_to_geotiff(composite, outpath):
    """
    Export the composite dataset to a GeoTIFF file.
    """
    with rasterio.open(
        outpath,
        "w",
        driver="GTiff",
        height=composite.sizes["y"],
        width=composite.sizes["x"],
        count=composite.shape[0],
        dtype=composite.dtype,
        crs=composite.rio.crs.to_string(),
        transform=composite.rio.transform(),
    ) as dst:
        for i, band in enumerate(composite.values, start=1):
            dst.write(band, i)
    print(f"Exported GeoTIFF to {outpath}")
    
    
    

if __name__ == "__main__":
    wks = r"Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification"
    aoi = gpd.GeoDataFrame(
        {"geometry": [box(-123.9852, 49.1399, -123.8652, 49.2199)]},
        crs="EPSG:4326"
    )
    target_date = datetime(2024, 8, 15)
    
    print("Querying Sentinel-2 data...")
    ds = query_stac_sentinel2(aoi, target_date)
    
    print("Computing quality scores...")
    quality_score = compute_quality_score(ds, target_date)
    
    print("Creating BAP composite...")
    composite = create_composite(ds, quality_score)
    
    print("Reprojecting composite to BC Albers (EPSG:3005)...")
    composite_bc_albers = reproject_to_bc_albers(composite)
    
    print("Exporting BAP composite...")
    out_path = os.path.join(wks, "work", "test_bap_composite_stac.tif")
    export_to_geotiff(composite_bc_albers, out_path)
    
    print("BAP processing completed.")
