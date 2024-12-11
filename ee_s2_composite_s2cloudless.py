"""
Computes a cloudless surface reflectance sentinel-2 composite image in Earth Engine
and adds spectral indices and terrain variables for classification.

Based on the s2cloudless cloud masking approach:
    https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

and the modified cloud filtering params by SashaNasonova:
    https://github.com/SashaNasonova/geeMosaics/blob/main/S2_SR_8bit_quarterly_mosaics.ipynb 

"""

import os
import ee
#import geemap
#import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import timeit

class EEAuthenticator:
    def __init__(self, service_account, pkey_path):
        self.service_account = service_account
        self.pkey_path = pkey_path
        self.initialize_ee()

    def initialize_ee(self):
        print('\nConnecting to Earth Engine')
        try:
            credentials = ee.ServiceAccountCredentials(self.service_account, self.pkey_path)
            ee.Initialize(credentials)
            print('..EE initialized successfully!')
        except ee.EEException as e:
            print("..Unexpected error:", e)

class AOIHandler:
    def __init__(self, shp_aoi_path):
        self.shp_aoi_path = shp_aoi_path
        self.aoi = self.read_aoi()

    def read_aoi(self):
        print('\nReading the Area of Interest')
        gdf = gpd.read_file(self.shp_aoi_path)
        return ee.Geometry.Polygon(mapping(gdf.unary_union)['coordinates'])


class S2Processor:
    def __init__(self, workspace: str, 
                 aoi_handler: AOIHandler, 
                 ee_authenticator: EEAuthenticator, 
             target_date: str, time_step: int, cloud_filter: float, 
             cld_prb_thresh: float, nir_drk_thresh: float, 
             cld_prj_dist: float, buffer: int
    ):
        self.workspace = workspace
        self.aoi_handler = aoi_handler
        self.ee_authenticator = ee_authenticator
        self.aoi = aoi_handler.aoi
        self.target_date = target_date
        self.time_step = time_step
        self.start_date = ee.Date(self.target_date).advance(-self.time_step, 'day')
        self.end_date = ee.Date(self.target_date).advance(self.time_step, 'day')
        self.cloud_filter = cloud_filter
        self.cld_prb_thresh = cld_prb_thresh
        self.nir_drk_thresh = nir_drk_thresh
        self.cld_prj_dist = cld_prj_dist
        self.buffer = buffer

    def get_s2_sr_cld_col(self):
        s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(self.aoi)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_filter)))

        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(self.aoi)
            .filterDate(self.start_date, self.end_date))

        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        }))

    @staticmethod
    def add_cloud_bands(img, cld_prb_thresh):
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(cld_prb_thresh).rename('clouds')
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    @staticmethod
    def add_shadow_bands(img, nir_drk_thresh, cld_prj_dist):
        not_water = img.select('SCL').neq(6)
        dark_pixels = img.select('B8').lt(nir_drk_thresh * 1e4).multiply(not_water).rename('dark_pixels')
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, cld_prj_dist * 10)
                    .reproject(crs=img.select(0).projection(), scale=100)
                    .select('distance').mask().rename('cloud_transform'))
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(self, img):
        img = self.add_cloud_bands(img, self.cld_prb_thresh)
        img = self.add_shadow_bands(img, self.nir_drk_thresh, self.cld_prj_dist)
        is_cld_shdw = img.select('clouds').add(img.select('shadows')).gt(0)
        is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(self.buffer * 2 / 20)
            .reproject(crs=img.select([0]).projection(), scale=20).rename('cloudmask'))
        return img.addBands(is_cld_shdw)

    @staticmethod
    def apply_cld_shdw_mask(img):
        not_cld_shdw = img.select('cloudmask').Not()
        return img.select('B.*').updateMask(not_cld_shdw)

    @staticmethod
    def add_indices(img):
        """
        Adds spectral indices to the S2 mosaic
        """
        indices = {
            'NDVI': img.normalizedDifference(['B8', 'B4']),
            'EVI': img.expression(
                '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
                {'B8': img.select('B8'), 'B4': img.select('B4'), 'B2': img.select('B2')}
            ),
            'MNDWI': img.normalizedDifference(['B3', 'B11']),
            'BSI': img.expression(
                '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
                {'B11': img.select('B11'), 'B4': img.select('B4'), 'B8': img.select('B8'), 'B2': img.select('B2')}
            ),
            'SAVI': img.expression(
                '((B8 - B4) / (B8 + B4 + L)) * (1 + L)',
                {'B8': img.select('B8'), 'B4': img.select('B4'), 'L': 0.5}
            ),
            'NDMI': img.normalizedDifference(['B8', 'B11']),
            'NBR': img.normalizedDifference(['B8', 'B12'])
        }
        return img.addBands([indices[key].rename(key) for key in indices])

    @staticmethod
    def add_terrain(img, aoi):
        """
        Adds terrain features to the S2 mosaic.
        """
        dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)
        elevation = dem.select('elevation').rename('Elevation')
        slope = ee.Terrain.slope(dem).rename('Slope')
        aspect = ee.Terrain.aspect(dem).rename('Aspect')
        tri = dem.reduceNeighborhood(reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.square(3)).rename('TRI')
        return img.addBands([elevation, slope, aspect, tri])
    
    @staticmethod
    def add_temporal_metrics(img, col, aoi):
        """
        Adds descriptive statistics to the S2 mosaic.
        Metrics: Mean, Median, Min, Max, StdDev, Percentiles (10th, 50th, 90th), and IQR (75th - 25th).
        """
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8', 'B8A', 'B11', 'B12']
    
        # Core statistics
        mean_img = col.select(bands).mean().clip(aoi).rename([f'{band}_Mean' for band in bands])
        median_img = col.select(bands).median().clip(aoi).rename([f'{band}_Median' for band in bands])
        min_img = col.select(bands).min().clip(aoi).rename([f'{band}_Min' for band in bands])
        max_img = col.select(bands).max().clip(aoi).rename([f'{band}_Max' for band in bands])
        std_img = col.select(bands).reduce(ee.Reducer.stdDev()).clip(aoi).rename([f'{band}_StdDev' for band in bands])
    
        # Percentiles
        percentiles = col.select(bands).reduce(
            ee.Reducer.percentile([10, 50, 90])
        ).clip(aoi).rename(
            [f'{band}_P10' for band in bands] +
            [f'{band}_P50' for band in bands] +
            [f'{band}_P90' for band in bands]
        )
        
        # Interquartile Range (IQR = P75 - P25)
        p75_img = col.select(bands).reduce(
            ee.Reducer.percentile([75])
        ).clip(aoi).rename([f'{band}_P75' for band in bands])
        
        p25_img = col.select(bands).reduce(
            ee.Reducer.percentile([25])
        ).clip(aoi).rename([f'{band}_P25' for band in bands])
        
        iqr_img = p75_img.subtract(p25_img).rename([f'{band}_IQR' for band in bands])
    
        return img.addBands([mean_img, median_img, min_img, max_img, std_img, percentiles, iqr_img])

if __name__ == '__main__':
    start_t = timeit.default_timer()

    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    service_account = 'lclu-942@ee-lclu-bc.iam.gserviceaccount.com'
    pkey = os.path.join(wks, 'work', 'ee-lclu-bc-b2fb2131d77b.json')
    EE = EEAuthenticator(service_account, pkey)

    shp_aoi = os.path.join(wks, "data", "AOIs" ,"aoi.shp")

    AOI = AOIHandler(shp_aoi)

    print('\nProcessing the S2 time series')
    S2 = S2Processor(
        wks, 
        AOI, 
        EE, 
        target_date = '2024-08-15',
        time_step = 45,
        cloud_filter = 80,
        cld_prb_thresh = 50,
        nir_drk_thresh = 0.15,
        cld_prj_dist = 1,
        buffer = 10
    )

    col = S2.get_s2_sr_cld_col()
    col_wmsks = col.map(S2.add_cld_shdw_mask).map(S2.apply_cld_shdw_mask)

    print ('\nComputing a cloudless S2 mosaic')
    s2_mosaic = col_wmsks.median()

    print ('\nAdding spectral indices to the s2 mosaic')
    s2_mosaic = S2.add_indices(s2_mosaic)

    print ('\nAdding Terrain bands to the s2 mosaic')
    s2_mosaic = S2.add_terrain(s2_mosaic, AOI.aoi)
    
    print ('\nAdding descriptive stats to the s2 mosaic')
    #s2_mosaic = S2.add_temporal_metrics(s2_mosaic, col_wmsks, AOI.aoi)

    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')