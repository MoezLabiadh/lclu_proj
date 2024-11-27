"""
An attempt to apply the Best-Available-Pixel (BAP) compositing approach
using Sentinel-2 imagery in Earth Engine (EE).

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
            
    (iii) distance to clouds and cloud shadows
            If the distance from cloudy pixels is greater than 150 meters, the score is set to 1, 
            indicating the pixel is sufficiently far from clouds.
            If the distance is less than or equal to 150 meters, 
            a Gaussian-like decay function assigns a score based on proximity.
    
Author: Moez Labiadh    
"""

import os
import ee
import geemap
import numpy as np
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import mapping
import timeit


def gdf_to_ee_geometry(gdf):
    """
    Converts a gdf to ee.geometry
    """
    return ee.Geometry.Polygon(mapping(gdf.unary_union)['coordinates'])


def s2_cloud_mask (image):
    """
    Computes a cloud mask for Sentinel-2 image in EE
    """
    cloud_mask = image.select('SCL').neq(3).And( # Cloud Shadows
        image.select('SCL').neq(7)).And(         # Clouds Low Probability
        image.select('SCL').neq(8)).And(         # Clouds Medium Probability
        image.select('SCL').neq(9)).And(         # Clouds High Probability
        image.select('SCL').neq(10))             # Cirrus
    
    return cloud_mask
                                        

def add_distance_score(image):
    """
    Adds the distance to clouds score to an s2 image

    """
    cloud_mask= s2_cloud_mask (image)                              
    distance = cloud_mask.fastDistanceTransform(256, 'manhattan').sqrt().multiply(20)
    distance_score = distance.expression(
        'distance > 150 ? 1 : exp(-0.5 * pow((distance / 50), 2))', {'distance': distance}
    )
    
    return image.addBands(distance_score.rename('DistanceScore').toFloat())



def add_coverage_score(image, aoi):
    """
    Adds the cloud coverage score to an s2 image
    """
    cloud_mask = image.select('MSK_CLDPRB').gt(50)
    cloud_percentage = cloud_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        maxPixels=1e10,
        bestEffort=True
    ).get('MSK_CLDPRB')

    # Inverse cloud percentage with exponential decay
    coverage_score = ee.Number(1).subtract(ee.Number(cloud_percentage)).max(0.1)  # Minimum score = 0.1
    coverage_band = ee.Image.constant(coverage_score).rename('CoverageScore').toFloat()

    return image.addBands(coverage_band)


def add_date_score(image, target_date):
    """
    Adds the proximity to target date score to an s2 image
    """
    target = ee.Date(target_date)
    date_diff = image.date().difference(target, 'day').abs()
    date_score = date_diff.expression(
        'exp(-0.5 * pow((days / 15), 2))', {'days': date_diff}
    )
    date_band = ee.Image.constant(date_score).rename('DateScore').toFloat()
    
    return image.addBands(date_band)



def process_collection(aoi, target_date, time_step, cloud_pct):
    """
    Adds and filters an s2 collection. Applies the BAP scores
    """
    # Convert target_date to an Earth Engine date
    target = ee.Date(target_date)
    
    # Calculate start_date and end_date as n-days before and after the target_date
    start_date = target.advance(-time_step, 'day')
    end_date = target.advance(time_step, 'day')
    
    def apply_cloud_mask(image):
        cloud_mask= s2_cloud_mask (image)  
        return image.updateMask(cloud_mask)

    # Load Sentinel-2 SR imagery
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct)) \
        .map(apply_cloud_mask) \
        .map(lambda img: add_distance_score(img)) \
        .map(lambda img: add_coverage_score(img, aoi)) \
        .map(lambda img: add_date_score(img, target_date))\
        
    
        
    # Add a total qualityscore to the S2 collection 
    def add_quality_score(image):
        distance_score = image.select('DistanceScore')
        coverage_score = image.select('CoverageScore')
        date_score = image.select('DateScore')
        #quality_score = distance_score.multiply(coverage_score).multiply(date_score)
        quality_score = distance_score.multiply(2).add(
            coverage_score.multiply(2)).add(
            date_score)
        
        return image.addBands(quality_score.rename('QualityScore').toFloat())

    collection = collection.map(add_quality_score)
    
    return collection



def create_bap_composite(collection):
    """
    # Use quality mosaic to create the BAP composite
    """
    composite = collection.qualityMosaic('QualityScore')
    
    return composite



def export_composite_to_drive(composite, region, description, folder):
    """
    Export an ee.image to Google Drive
    """
    task = ee.batch.Export.image.toDrive(
        image=composite.clip(region),
        description=description,
        folder=folder,
        scale=20,
        region=region.getInfo()['coordinates'],
        maxPixels=1e13
    )
    task.start()
    


def export_geotiff(image, aoi, outpath, bands=None, res=20):
    """
    Exports the selected bands of an ee.image to a geotiff file.
    """
    
    # Select specific bands if provided
    if bands:
        image = image.select(bands)
    
    # Get bounds of the AOI
    bounds = aoi.bounds()
    coords = bounds.coordinates().getInfo()[0]  # Extract the coordinates

    # Get the image as a numpy array
    image_array = geemap.ee_to_numpy(
        image, 
        region=aoi, 
        scale=res # spatial resolution in meters
    )

    # Get the projection
    projection = image.projection()
    crs = projection.crs().getInfo()  # Retrieve CRS as EPSG
    crs_wkt = CRS.from_string(crs).to_wkt()  # Convert to WKT

    # Create transform using the bounds
    transform = rasterio.transform.from_bounds(
        west= min(coord[0] for coord in coords), 
        south= min(coord[1] for coord in coords), 
        east= max(coord[0] for coord in coords), 
        north= max(coord[1] for coord in coords), 
        width=image_array.shape[1],
        height=image_array.shape[0]
    )

    # Write to GeoTIFF
    with rasterio.open(
        outpath, 
        'w', 
        driver='GTiff',
        height=image_array.shape[0],
        width=image_array.shape[1],
        count=image_array.shape[2],
        dtype=image_array.dtype,
        crs=crs_wkt,  # Use the WKT CRS
        transform=transform
    ) as dst:
        for i in range(image_array.shape[2]):
            dst.write(image_array[:, :, i], i+1)    



if __name__ == '__main__':
    start_t = timeit.default_timer() #start time
    
    #workspace
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    
    #initialize ee with a service account key
    print ('\nConnecting to Earth Engine')
    try:
        service_account = 'lclu-942@ee-lclu-bc.iam.gserviceaccount.com'
        pkey= os.path.join(wks, 'work', 'ee-lclu-bc-b2fb2131d77b.json')
        credentials = ee.ServiceAccountCredentials(service_account, pkey)
    
        ee.Initialize(credentials)
        print('..EE initialized successfully!')
        
    except ee.EEException as e:
        print("..Unexpected error:", e)
 
    
    # Define area of interest as a GeoPandas GeoDataFrame
    print ('\nReading the Area of Interest')
    # Sample AOI

    shp_aoi= os.path.join(wks, "data", "aoi.shp")
    gdf= gpd.read_file(shp_aoi)
    aoi = gdf_to_ee_geometry(gdf)


    print ('\nComputing the BAP Composite')
    target_date = '2024-08-15' #target date
    time_step = 45 #time step (nbr of days)
    cloud_pct= 10 # max Cloud %

    # Create the BAP composite
    collection = process_collection(
        aoi, 
        target_date,
        time_step, 
        cloud_pct
    )
    
    bap_composite= create_bap_composite(collection)
    
    
    '''
    print ('\nExporting the BAP Composite to geotiff file')
    #Export the BAP composite
    bands= ['B2', 'B3', 'B4', 'B8A']
    filename= 'test_composite_vis_nir.tif'
    export_geotiff(
        image= bap_composite, 
        aoi= aoi, 
        bands= bands,
        outpath= os.path.join(wks, 'work', filename)
    )
    '''
    '''
    # Visualize the composite. Works in Notebook only!!
    Map = geemap.Map()
    Map.addLayer(bap_composite, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, 'BAP Composite')
    Map.centerObject(aoi)
    Map
    '''
    
    finish_t = timeit.default_timer() #finish time
    t_sec = round(finish_t-start_t)
    mins = int (t_sec/60)
    secs = int (t_sec%60)
    print('\nProcessing Completed in {} minutes and {} seconds'.format (mins,secs))