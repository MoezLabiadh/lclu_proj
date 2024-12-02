"""
Computes a mosaic of sentinel-1 composite image in Earth Engine.
Band added are VV and VH in addition to some polarimetric indices.

Note: a significant portion of BC's eastern area lacks S1 coverage
        for 2024. 

"""

import os
import ee
import geemap
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping

def gdf_to_ee_geometry(gdf):
    """
    Converts a gdf to ee.geometry
    """
    return ee.Geometry.Polygon(mapping(gdf.unary_union)['coordinates'])

def get_s1_mosaic(aoi, start_date, end_date):



    # Filter S1 collection
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                .sort('system:time_start', False))
    
    s1_mosaic = s1_col.select(['VV', 'VH']).mean()

    return s1_mosaic



def add_pol_indices(image):
    # RVI
    rvi = image.select('VH').multiply(4).divide(
        image.select('VV').add(image.select('VH'))
    ).rename('RVI')
    
    # VH/VV Ratio
    vh_vv_ratio = image.select('VH').divide(
        image.select('VV')
    ).rename('Ratio_VH_VV')
    
    # NDPI
    ndpi = image.select('VV').subtract(image.select('VH')).divide(
        image.select('VV').add(image.select('VH'))
    ).rename('NDPI')
    
    # DPD
    dpd = image.select('VV').subtract(image.select('VH')).rename('DPD')
    
    # PDI
    pdi = image.select('VV').subtract(image.select('VH')).divide(
        image.select('VV')
    ).rename('PDI')
    
    # WRI
    wri = image.select('VV').add(image.select('VH')).divide(
        image.select('VV').subtract(image.select('VH'))
    ).rename('WRI')
    
    # Add bands to the image
    return image.addBands([vh_vv_ratio, rvi, ndpi, dpd, pdi, wri])



if __name__ == '__main__':

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
    
    print ('Create a S1 mosaic with VV and VH bands')
    s1_mosaic = get_s1_mosaic(aoi, START_DATE, END_DATE)
    
    print ('calculate polarimetric indices')
    s1_mosaic= add_pol_indices(s1_mosaic)
    

    
    # vissulation. Notebook only!
    Map = geemap.Map()
    viz_s1= {'min': -30, 'max': 0}
    Map.addLayer(s1_mosaic.select('VH').clip(aoi), viz_s1 , 'S1-VH')
    Map.centerObject(aoi)
    Map