import os
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from shapely.geometry import box
import geopandas as gpd
import stackstac
from pystac_client import Client
from planetary_computer import sign
import rasterio
from rasterio.transform import from_bounds

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

def compute_quality_score(ds, target_date):
    """
    Compute the quality score for BAP compositing.
    """
    # Compute date scores
    date_scores = compute_date_score(ds["time"].values, target_date)
    
    # Expand dimensions of date_scores to match dist_scores
    date_scores_da = xr.DataArray(
        date_scores,
        dims=["time"],
        coords={"time": ds["time"]},
    )
    date_scores_da = date_scores_da.expand_dims({"x": ds["x"], "y": ds["y"]}).transpose("time", "y", "x")
    
    # Retrieve the SCL band
    scl_data = ds.sel(band="SCL")
    
    # Compute distance scores
    dist_scores = scl_data.map_blocks(compute_distance_score)
    
    # Combine scores
    return date_scores_da * dist_scores

def create_composite(ds, quality_score):
    """
    Create a BAP composite using the quality_score to select the best pixel for each location.
    """
    # Find the time index with the maximum quality score for each pixel
    best_quality_time = quality_score.argmax(dim="time").compute()  # Convert to NumPy
    
    # Select the bands of interest
    selected_bands = ds.sel(band=["B02", "B03", "B04", "B8A"])
    
    # For each band, select the pixel corresponding to the best quality score
    composites = []
    for band in selected_bands["band"].values:
        band_data = selected_bands.sel(band=band)
        band_composite = band_data.isel(time=best_quality_time)
        composites.append(band_composite)
    
    # Combine all band composites into a single dataset
    composite = xr.concat(composites, dim="band")
    composite["band"] = ["B02", "B03", "B04", "B8A"]  # Assign band names
    
    return composite

def export_to_geotiff(composite, aoi, outpath):
    
    transform = from_bounds(
        aoi.total_bounds[0],
        aoi.total_bounds[1],
        aoi.total_bounds[2],
        aoi.total_bounds[3],
        composite.sizes["x"],
        composite.sizes["y"],
    )
    with rasterio.open(
        outpath,
        "w",
        driver="GTiff",
        height=composite.sizes["y"],
        width=composite.sizes["x"],
        count=composite.shape[0],
        dtype=composite.dtype,
        crs="EPSG:4326",
        transform=transform,
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
    print("Exporting BAP composite...")
    out_path = os.path.join(wks, "work", "test_bap_composite_stac.tif")
    export_to_geotiff(composite, aoi, out_path)
    print("BAP processing completed.")