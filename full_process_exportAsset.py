import os
import ee
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

  

if __name__ == '__main__':
    start_t = timeit.default_timer()

    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    service_account = 'lclu-942@ee-lclu-bc.iam.gserviceaccount.com'
    pkey = os.path.join(wks, 'work', 'ee-lclu-bc-b2fb2131d77b.json')
    EE = EEAuthenticator(service_account, pkey)

    shp_aoi = os.path.join(wks, "data", "AOIs" ,"bc.shp")

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

    print ('\nRunning the classification')

    # Load the training points from your asset
    training_points = ee.FeatureCollection('projects/ee-lclu-bc/assets/training_points')

    # Get all band names from the Sentinel-2 mosaic
    #bands = s2_mosaic.bandNames().getInfo()
    bands= ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    print ('\nFeature extraction..')
    # Sample the input image at the training points
    training = s2_mosaic.select(bands).sampleRegions(
        collection=training_points,
        properties=['class_id'],
        scale=10  # Adjust based on image resolution
    )

    # Train a classifier
    print ('\nTrain a classifier..')
    classifier = ee.Classifier.smileRandomForest(50).train(
        features=training,
        classProperty='class_id',
        inputProperties=bands
    )

    # Classify the image
    print ('\nClassify the image..')
    classified = s2_mosaic.select(bands).classify(classifier)
    
    # Export the classified image to Google Drive
    print('\nExporting the classified image to Asset.')
    task = ee.batch.Export.image.toAsset(
        image=classified,
        description='Land_Cover_Classification_AOI',
        assetId='projects/ee-lclu-bc/assets/LandCover_test_BC',
        scale=10,
        region=AOI.aoi.getInfo()['coordinates'],
        maxPixels=1e13
    )
    

    task.start()
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')