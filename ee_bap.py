"""
An attempt to apply the Best-Available-Pixel (BAP) compositing approach
using Sentinel-2 imagery in Earth Engine (EE).

Based on White et al. (2014), Pixel-Based Image Compositing for Large-Area Dense 
 Time Series Applications and Science -
 https://www.tandfonline.com/doi/full/10.1080/07038992.2014.945827 
                              
The scoring functions rank each S2 pixel observation for: 
    (i) proximity to the target date, 
    (ii) cloud cover in the scene, 
    (iii) distance to clouds
"""

import os
import ee
import geemap
import numpy as np
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import timeit


def gdf_to_ee_geometry(gdf):
    """
    Converts a gdf to ee.geometry
    """
    return ee.Geometry.Polygon(mapping(gdf.unary_union)['coordinates'])


def add_distance_score(image):
    """
    Caluclates and adds a Distance-to-Cloud Score
    """
    cloud_mask = image.select('MSK_CLDPRB').gt(60)  # Cloud probability > 60%
    distance = cloud_mask.fastDistanceTransform(256, 'manhattan').sqrt().multiply(20)  # Convert to meters
    distance_score = distance.expression(
        'distance > 150 ? 1 : exp(-0.5 * pow((distance / 50), 2))', {'distance': distance}
    )
    return image.addBands(distance_score.rename('DistanceScore'))



def add_coverage_score(image, aoi):
    """
    Caluclates and adds a Coverage Score
    """
    # Create a cloud mask using MSK_CLDPRB
    cloud_mask = image.select('MSK_CLDPRB').gt(60)  # Cloud probability > 60%
    
    # Reduce region with maxPixels and bestEffort
    cloud_percentage = cloud_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        maxPixels=1e10,  # Increased maxPixels
        bestEffort=True  # Allow scale adjustment
    ).get('MSK_CLDPRB')
    
    # Compute the Coverage Score
    coverage_score = ee.Number(1).subtract(ee.Number(cloud_percentage))
    return image.set('CoverageScore', coverage_score)




def add_date_score(image, target_date):
    """
    Caluclates and adds a Date Score based on proximity to the target date
    """

    target = ee.Date(target_date)
    date_diff = image.date().difference(target, 'day').abs()
    date_score = date_diff.expression(
        'exp(-0.5 * pow((days / 15), 2))', {'days': date_diff}
    )
    return image.set('DateScore', date_score)



def create_bap_composite(aoi, start_date, end_date):
    """
    Create a BAP composite image
    """
    # Set the target date to the median of start_date and end_date
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    target_date = start.advance(end.difference(start, 'day').divide(2), 'day')

    # Load Sentinel-2 SR imagery
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .map(lambda img: add_distance_score(img)) \
        .map(lambda img: add_coverage_score(img, aoi)) \
        .map(lambda img: add_date_score(img, target_date))
        
        
    # Add a total qualityscore to the S2 collection 
    def add_quality_score(img):

        '''
        quality_score = img.expression(
            'DistanceScore * CoverageScore * DateScore',
            {
                'DistanceScore': img.select('DistanceScore'),
                'CoverageScore': img.get('CoverageScore'),
                'DateScore': img.get('DateScore')
            }
        )
        '''
        
        distance_score = img.select('DistanceScore')
        coverage_score = ee.Number(img.get('CoverageScore'))
        date_score = ee.Number(img.get('DateScore'))
        
        quality_score = distance_score.multiply(coverage_score).multiply(date_score)
        
        return img.addBands(quality_score.rename('QualityScore'))

    collection = collection.map(add_quality_score)

    # Use quality mosaic to create the BAP composite
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
    


def export_geotiff(image, aoi, outpath, bands=None):
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
        scale=20  # 20m resolution
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
    try:
        service_account = 'lclu-942@ee-lclu-bc.iam.gserviceaccount.com'
        pkey= os.path.join(wks, 'work', 'ee-lclu-bc-b2fb2131d77b.json')
        credentials = ee.ServiceAccountCredentials(service_account, pkey)
    
        ee.Initialize(credentials)
        print('EE initialized successfully!')
        
    except ee.EEException as e:
        print("Unexpected error:", e)
 
    
    # Define area of interest as a GeoPandas GeoDataFrame
    # Sample AOI
    coordinates = [
        [-123.9852, 49.1399],
        [-123.9852, 49.2199],  
        [-123.8652, 49.2199], 
        [-123.8652, 49.1399],  
        [-123.9852, 49.1399]   
    ]

    # Create a shapely Polygon
    polygon = Polygon(coordinates)

    # Create GeoDataFrame
    data = {'geometry': [polygon]}
    gdf = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")
    aoi = gdf_to_ee_geometry(gdf)

    # Define the time range
    start_date = '2024-07-01'
    end_date = '2024-09-30'

    # Create the BAP composite
    bap_composite = create_bap_composite(aoi, start_date, end_date)
    
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
    # Visualize the composite
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