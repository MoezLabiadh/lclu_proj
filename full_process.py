import os
import ee
import geemap
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import timeit

import rasterio
from rasterio.transform import from_bounds

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
    def __init__(self, workspace, aoi_handler, ee_authenticator, target_date, time_step, cloud_filter, cld_prb_thresh, nir_drk_thresh, cld_prj_dist, buffer):
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



def export_image_to_local(
    classified, 
    output_dir, 
    file_name, 
    aoi, 
    scale=10, 
    chunk_size=1024
):
    """
    Export a Google Earth Engine image to local GeoTIFF files in chunks.
    
    Args:
        classified (ee.Image): The image to export
        output_dir (str): Directory to save output files
        file_name (str): Base name for output files
        aoi (ee.Geometry): Area of interest
        scale (int, optional): Resolution of export. Defaults to 10.
        chunk_size (int, optional): Size of chunks to export. Defaults to 5000.
    """

    crs = 'EPSG:4326'
    
    # Get bounding box of the AOI with error handling
    try:
        coords = aoi.bounds().getInfo()['coordinates'][0]
        min_x = min(c[0] for c in coords)
        max_x = max(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        max_y = max(c[1] for c in coords)
    except Exception as e:
        print(f"Error getting AOI bounds: {e}")
        return
    
    # Compute steps more efficiently
    x_steps = np.arange(min_x, max_x, chunk_size * scale)
    y_steps = np.arange(min_y, max_y, chunk_size * scale)
    
    # Use list comprehension to track export progress
    exported_files = []
    
    for i, x_start in enumerate(x_steps):
        for j, y_start in enumerate(y_steps):
            # Compute chunk boundaries
            x_end = min(x_start + chunk_size * scale, max_x)
            y_end = min(y_start + chunk_size * scale, max_y)
            
            print(f'Exporting chunk ({i}, {j})...')
            
            # Create region geometry once
            region = ee.Geometry.Rectangle([x_start, y_start, x_end, y_end])
            
            try:
                # Convert image to numpy array 
                # Removed CRS parameter to avoid potential issues
                image_array = geemap.ee_to_numpy(
                    classified, 
                    region=region, 
                    scale=scale
                )
                
                # Construct output filename
                output_file = os.path.join(
                    output_dir, 
                    f'{file_name}_chunk_{i}_{j}.tif'
                )
                
                # Create transform
                transform = from_bounds(
                    x_start, y_start, x_end, y_end, 
                    image_array.shape[1], image_array.shape[0]
                )
                
                # Write GeoTIFF with more robust opening
                with rasterio.open(
                    output_file, 
                    'w', 
                    driver='GTiff',
                    height=image_array.shape[0], 
                    width=image_array.shape[1], 
                    count=1,
                    dtype=image_array.dtype, 
                    crs=crs, 
                    transform=transform
                ) as dst:
                    dst.write(image_array, 1)
                
                exported_files.append(output_file)
                print(f'Chunk ({i}, {j}) saved at {output_file}')
            
            except Exception as e:
                print(f"Error exporting chunk ({i}, {j}): {e}")
    
    return exported_files 
            

if __name__ == '__main__':
    start_t = timeit.default_timer()

    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    service_account = 'lclu-942@ee-lclu-bc.iam.gserviceaccount.com'
    pkey = os.path.join(wks, 'work', 'ee-lclu-bc-b2fb2131d77b.json')
    EE = EEAuthenticator(service_account, pkey)

    shp_aoi = os.path.join(wks, "data", "AOIs" ,"aoi.shp")

    AOI = AOIHandler(shp_aoi)

    target_date = '2024-08-15'
    time_step = 45

    print('\nProcessing the S2 time series')
    S2 = S2Processor(
        wks, 
        AOI, 
        EE, 
        target_date = target_date,
        time_step = time_step,
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



    ######################## RADAR ##############################################
    print ('\nAdding Radar bands')
    start_date = ee.Date(target_date).advance(-time_step, 'day')
    end_date = ee.Date(target_date).advance(time_step, 'day')
    
    s1_mosaic= get_s1_mosaic(AOI.aoi, start_date, end_date)
    s1_mosaic= add_pol_indices(s1_mosaic)

    s1_s2_mosaic = s2_mosaic.addBands(s1_mosaic)

    ####################CLASSIFICATION###########################################
    print ('\nRunning the classification')

    # Load the training points from your asset
    training_points = ee.FeatureCollection('projects/ee-lclu-bc/assets/training_points_test_aoi')

    # Get all band names from the Sentinel-2 mosaic
    bands = s1_s2_mosaic.bandNames().getInfo()

    print ('\nFeature extraction..')
    # Sample the input image at the training points
    training = s1_s2_mosaic.select(bands).sampleRegions(
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
    classified = s1_s2_mosaic.select(bands).classify(classifier)
    
    
    print ('\nExport images to geotiff..')
    output_dir= os.path.join(wks, "data", "outputs")
    file_name= 'test_lanCover_aoi'
    
    exported= export_image_to_local(
        classified, 
        output_dir, 
        file_name, 
       AOI.aoi, 
        scale=10, 
        chunk_size=1024
    )


    '''
    print ('\nExport images to assets..')
    projection = ee.Projection('EPSG:4326')
    grid = AOI.aoi.coveringGrid(proj= projection, scale=10000)
    count= grid.toList(grid.size()).getInfo()
    
    for i, feature in enumerate(grid.toList(grid.size()).getInfo()):
        task = ee.batch.Export.image.toAsset(
            image=classified,
            description=f'Land_Cover_Classification_Asset_tile_{i}',
            assetId=f'projects/ee-lclu-bc/assets/LandCover_tile_{i}',
            scale=10,
            region=ee.Feature(feature).geometry(),
            maxPixels=1e13
        )
        #task.start()
    '''
    
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')