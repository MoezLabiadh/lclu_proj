"""
Computes a cloudless surface reflectance sentinel-2 composite image in Earth Engine.

Based on the s2cloudless cloud masking approach:
    https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

and the modified cloud filtering params by SashaNasonova:
    https://github.com/SashaNasonova/geeMosaics/blob/main/S2_SR_8bit_quarterly_mosaics.ipynb 
    

Next Steps:    
    Exporting the image to Geotiff is currently not working for large AOIs.
    Ideas: convert to 8-bit, doanload data by chunk
"""

import os
import ee
import geemap
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import timeit    


def gdf_to_ee_geometry(gdf):
    """
    Converts a gdf to ee.geometry
    """
    return ee.Geometry.Polygon(mapping(gdf.unary_union)['coordinates'])


def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


# Get cloud bands
def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

# Create cloud shadow bands
# NIR_DRK_THRESH is defined in the main body of the script
def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

# Merge cloud and cloud shadow mask, clean up
# BUFFER is defined in the main body of the script, default 50m, I use 10m. Again, too aggressive
# There is confusion between snow and cloud

def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

# Apply all masks
def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)



def add_indices(img):
    """
    Adds spectral indices to the S2 mosaic
    """
    # NDVI (Vegetation Index)
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # EVI (Enhanced Vegetation Index)
    evi = img.expression(
        '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
        {'B8': img.select('B8'), 'B4': img.select('B4'), 'B2': img.select('B2')}
    ).rename('EVI')
    
    # MNDWI (Water Index)
    mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    
    # BSI (Bare Soil Index)
    bsi = img.expression(
        '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
        {'B11': img.select('B11'), 'B4': img.select('B4'), 
         'B8': img.select('B8'), 'B2': img.select('B2')}
    ).rename('BSI')
    
    # SAVI (Soil-Adjusted Vegetation Index)
    savi = img.expression(
        '((B8 - B4) / (B8 + B4 + L)) * (1 + L)',
        {'B8': img.select('B8'), 'B4': img.select('B4'), 'L': 0.5}  # Typical L=0.5
    ).rename('SAVI')
    
    # NDMI (Normalized Difference Moisture Index)
    ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    # NBR (Normalized Burn Ratio)
    nbr = img.normalizedDifference(['B8', 'B12']).rename('NBR')
    
    # Add all indices as bands
    return img.addBands([ndvi, evi, mndwi, bsi, savi, ndmi, nbr])


def add_terrain(img, aoi):
    """
    Adds terrain features to the S2 mosaic.
    """
    # Load the SRTM DEM
    dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)
    
    # Elevation
    elevation = dem.select('elevation').rename('Elevation')
    
    # Slope
    slope = ee.Terrain.slope(dem).rename('Slope')
    
    # Aspect
    aspect = ee.Terrain.aspect(dem).rename('Aspect')
    
    # Terrain Roughness Index (TRI)
    tri = dem.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(3)
    ).rename('TRI')
    
    # Add all terrain bands
    return img.addBands([elevation, slope, aspect, tri])



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


    target_date = '2024-08-15' #center date
    target = ee.Date(target_date)
    time_step = 45 #time step (nbr of days)

    START_DATE = target.advance(-time_step, 'day')
    END_DATE = target.advance(time_step, 'day')

    CLOUD_FILTER = 80
    CLD_PRB_THRESH = 50
    NIR_DRK_THRESH = 0.15 
    CLD_PRJ_DIST = 1
    BUFFER = 10

    
    print ('\nComputing a cloudless s2 mosaic')
    col = get_s2_sr_cld_col(aoi, START_DATE, END_DATE)
    col_wmsks = col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
    s2_mosaic = col_wmsks.median()


    print ('\nAdding spectral indices to the s2 mosaic')
    s2_mosaic = add_indices(s2_mosaic)

    print ('\nAdding Trraing bands to the s2 mosaic')
    s2_mosaic = add_terrain(s2_mosaic, aoi)



    finish_t = timeit.default_timer() #finish time
    t_sec = round(finish_t-start_t)
    mins = int (t_sec/60)
    secs = int (t_sec%60)
    print('\nProcessing Completed in {} minutes and {} seconds'.format (mins,secs))