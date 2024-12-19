// Function to get Sentinel-2 Surface Reflectance and Cloud Probability Collections
function getS2SrCldCol(aoi, startDate, endDate, cloudFilter) {
  var s2SrCol = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloudFilter));

  var s2CloudlessCol = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterBounds(aoi)
    .filterDate(startDate, endDate);

  return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply({
    primary: s2SrCol,
    secondary: s2CloudlessCol,
    condition: ee.Filter.equals({
      leftField: 'system:index',
      rightField: 'system:index'
    })
  }));
}

// Function to add cloud bands
function addCloudBands(img, cldPrbThresh) {
  var cldPrb = ee.Image(img.get('s2cloudless')).select('probability');
  var isCloud = cldPrb.gt(cldPrbThresh).rename('clouds');
  return img.addBands([cldPrb, isCloud]);
}

// Function to add shadow bands
function addShadowBands(img, nirDrkThresh, cldPrjDist) {
  var notWater = img.select('SCL').neq(6);
  var darkPixels = img.select('B8').lt(nirDrkThresh * 1e4).multiply(notWater).rename('dark_pixels');
  var shadowAzimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));
  var cldProj = img.select('clouds').directionalDistanceTransform(shadowAzimuth, cldPrjDist * 10)
    .reproject({crs: img.select(0).projection(), scale: 100})
    .select('distance').mask().rename('cloud_transform');
  var shadows = cldProj.multiply(darkPixels).rename('shadows');
  return img.addBands([darkPixels, cldProj, shadows]);
}

// Function to add cloud and shadow mask
function addCldShdwMask(img, cldPrbThresh, nirDrkThresh, cldPrjDist, buffer) {
  img = addCloudBands(img, cldPrbThresh);
  img = addShadowBands(img, nirDrkThresh, cldPrjDist);
  var isCldShdw = img.select('clouds').add(img.select('shadows')).gt(0);
  isCldShdw = isCldShdw.focalMin(2).focalMax(buffer * 2 / 20)
    .reproject({crs: img.select(0).projection(), scale: 20}).rename('cloudmask');
  return img.addBands(isCldShdw);
}

// Function to apply the cloud and shadow mask
function applyCldShdwMask(img) {
  var notCldShdw = img.select('cloudmask').not();
  return img.select('B.*').updateMask(notCldShdw);
}

// Function to add spectral indices
function addIndices(img) {
  var indices = {
    'NDVI': img.normalizedDifference(['B8', 'B4']),
    'EVI': img.expression(
      '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
      {B8: img.select('B8'), B4: img.select('B4'), B2: img.select('B2')}
    ),
    'MNDWI': img.normalizedDifference(['B3', 'B11']),
    'BSI': img.expression(
      '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
      {B11: img.select('B11'), B4: img.select('B4'), B8: img.select('B8'), B2: img.select('B2')}
    ),
    'SAVI': img.expression(
      '((B8 - B4) / (B8 + B4 + L)) * (1 + L)',
      {B8: img.select('B8'), B4: img.select('B4'), L: 0.5}
    ),
    'NDMI': img.normalizedDifference(['B8', 'B11']),
    'NBR': img.normalizedDifference(['B8', 'B12']),
    'NDRE': img.normalizedDifference(['B8', 'B5']),
    'BAEI': img.expression(
      '(B4 + 0.3) / (B3 + B11)',
      {B4: img.select('B4'), B3: img.select('B3'), B11: img.select('B11')}
    ),
    'NBAI': img.expression(
      '(B11 - B12) / B3 / ((B11 + B12) / B3)',
      {B11: img.select('B11'), B12: img.select('B12'), B3: img.select('B3')}
    ),
    'NDSI': img.normalizedDifference(['B3', 'B11'])
  };

  return img.addBands(Object.keys(indices).map(function(key) { 
    return indices[key].rename(key); 
  }));
}



// Function to get Sentinel-1 mosaic and descriptive stats
function getS1Stats(aoi, startDate, endDate) {
  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    .sort('system:time_start', false);

  var meanBands = s1Col.select(['VV', 'VH']).mean();
  
  var percentiles = s1Col.reduce(ee.Reducer.percentile([10, 50, 90]));
  var p75 = s1Col.reduce(ee.Reducer.percentile([75]));
  var p25 = s1Col.reduce(ee.Reducer.percentile([25]));
  var iqr = p75.subtract(p25).rename(p75.bandNames()
                             .map(function(name) { return ee.String(name).cat('_IQR'); }));

  return meanBands.rename(meanBands.bandNames()).addBands(percentiles).addBands(iqr);
}

// Function to add polarimetric indices
function addPolIndices(image) {
  var rvi = image.select('VH').multiply(4).divide(
    image.select('VV').add(image.select('VH'))).rename('RVI');

  var vhVvRatio = image.select('VH').divide(
    image.select('VV')).rename('Ratio_VH_VV');

  var ndpi = image.select('VV').subtract(image.select('VH')).divide(
    image.select('VV').add(image.select('VH'))).rename('NDPI');

  var dpd = image.select('VV').subtract(image.select('VH')).rename('DPD');

  var pdi = image.select('VV').subtract(image.select('VH')).divide(
    image.select('VV')).rename('PDI');

  var wri = image.select('VV').add(image.select('VH')).divide(
    image.select('VV').subtract(image.select('VH'))).rename('WRI');

  return image.addBands([vhVvRatio, rvi, ndpi, dpd, pdi, wri]);
}


// Function to add terrain bands
function addTerrain(img, aoi) {
  var dem = ee.Image('USGS/SRTMGL1_003').clip(aoi);
  var elevation = dem.select('elevation').rename('Elevation');
  var slope = ee.Terrain.slope(dem).rename('Slope');
  var aspect = ee.Terrain.aspect(dem).rename('Aspect');
  var tri = dem.reduceNeighborhood({
    reducer: ee.Reducer.stdDev(),
    kernel: ee.Kernel.square(3)
  }).rename('TRI');
  return img.addBands([elevation, slope, aspect, tri]);
}


// Function to get a VIIRS radiance image
function getVIIRSradiance(img, startDate, endDate, aoi) {

  var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
                  .filter(ee.Filter.date(startDate, endDate));
  
  var nighttime = dataset.select('avg_rad').min().clip(aoi);
  
  return img.addBands([nighttime]);
}


// Main Process
var targetDate_summer = '2024-07-15';
var targetDate_fall = '2024-09-15';
var timeStep = 30;
var cloudFilter = 80;
var cldPrbThresh = 50;
var nirDrkThresh = 0.15;
var cldPrjDist = 1;
var buffer = 10;

// Function to process Sentinel-2 data
function processMosaic(targetDate) {
  var col = getS2SrCldCol(AOI_BC, 
                          ee.Date(targetDate).advance(-timeStep, 'day'), 
                          ee.Date(targetDate).advance(timeStep, 'day'), 
                          cloudFilter);
                          
  var colWmsks = col.map(function(img) {
    return applyCldShdwMask(addCldShdwMask(img, cldPrbThresh, nirDrkThresh, 
                                           cldPrjDist, buffer));
  });

  var mosaic = colWmsks.reduce(ee.Reducer.percentile([35]));
  var originalBandNames = colWmsks.first().bandNames();
  mosaic = mosaic.rename(originalBandNames);
  return addIndices(mosaic);
}

// Create S2 summer and fall mosaics
var summerBands = processMosaic(targetDate_summer);
var s2Mosaic_summer = summerBands.rename(ee.List(summerBands.bandNames())
                                 .map(function(name) { return ee.String(name).cat('_summer'); }));
var fallBands = processMosaic(targetDate_fall);
var s2Mosaic_fall = fallBands.rename(ee.List(fallBands.bandNames())
                             .map(function(name) { return ee.String(name).cat('_fall'); }));

// Merge S2 summer and fall mosaics
var s2Mosaic_sumFall = s2Mosaic_summer.addBands(s2Mosaic_fall);

print('S2 Mosaic:', s2Mosaic_sumFall);



// Create S1 summer and fall mosaics
var s1Stats_summer = getS1Stats(AOI_BC, 
                                ee.Date(targetDate_summer).advance(-timeStep, 'day'), 
                                ee.Date(targetDate_summer).advance(timeStep, 'day'));
var s1Mosaic_summer = addPolIndices(s1Stats_summer).rename(ee.List(addPolIndices(s1Stats_summer).bandNames())
                                                    .map(function(name) { return ee.String(name).cat('_summer'); }));

var s1Stats_fall = getS1Stats(AOI_BC, 
                              ee.Date(targetDate_fall).advance(-timeStep, 'day'), 
                              ee.Date(targetDate_fall).advance(timeStep, 'day'));
var s1Mosaic_fall = addPolIndices(s1Stats_fall).rename(ee.List(addPolIndices(s1Stats_fall).bandNames())
                                               .map(function(name) { return ee.String(name).cat('_fall'); }));

// Merge S1 summer and fall mosaics
var s1Mosaic_sumFall = s1Mosaic_summer.addBands(s1Mosaic_fall);
print('S1 Mosaic:', s1Mosaic_sumFall);


// Merge all input features: S1, S2, terrain and radiance
var allMosaic = s2Mosaic_sumFall.addBands(s1Mosaic_sumFall);
var allMosaic = addTerrain(allMosaic, AOI_BC);
var allMosaic = getVIIRSradiance(allMosaic, 
                                ee.Date(targetDate_summer).advance(-timeStep, 'day'), 
                                ee.Date(targetDate_summer).advance(timeStep, 'day'), 
                                AOI_BC);
                                
print('All features Mosaic:', allMosaic);


print('Running the classification');
var trainingPoints = ee.FeatureCollection('projects/ee-lclu-bc/assets/training_points_bc_reduced')
                        .filterBounds(AOI_med);
                        
//var bands = allMosaic.bandNames();


var bands = [
  'B1_summer', 'B2_summer', 'B3_summer', 'B4_summer', 'B5_summer', 'B6_summer', 'B7_summer', 
  'B8_summer', 'B8A_summer', 'B9_summer', 'B11_summer', 'B12_summer', 'NDVI_summer', 
  'EVI_summer', 'NDRE_summer', 'SAVI_summer', 'MNDWI_summer', 'NDMI_summer', 'NBR_summer', 
  'BSI_summer', 'BAEI_summer', 'NBAI_summer', 'NDSI_summer', 
  
  'VV_summer', 'VH_summer', 'Ratio_VH_VV_summer', 
  'RVI_summer', 'NDPI_summer', 'DPD_summer', 'PDI_summer', 'WRI_summer', 
  'VV_p10_summer', 'VV_p50_summer', 'VV_p90_summer', 'VV_p75_IQR_summer',
  'VH_p10_summer', 'VH_p50_summer', 'VH_p90_summer', 'VH_p75_IQR_summer',
  
  'B1_fall', 'B2_fall', 'B3_fall', 'B4_fall', 'B5_fall', 'B6_fall', 'B7_fall', 
  'B8_fall', 'B8A_fall', 'B9_fall', 'B11_fall', 'B12_fall', 'NDVI_fall', 
  'EVI_fall', 'NDRE_fall', 'SAVI_fall', 'MNDWI_fall', 'NDMI_fall', 'NBR_fall', 
  'BSI_fall', 'BAEI_fall', 'NBAI_fall',  'NDSI_fall', 
  
  'VV_fall', 'VH_fall', 'Ratio_VH_VV_fall', 
  'RVI_fall', 'NDPI_fall', 'DPD_fall', 'PDI_fall', 'WRI_fall', 
  'VV_p10_fall', 'VV_p50_fall', 'VV_p90_fall', 'VV_p75_IQR_fall',
  'VH_p10_fall', 'VH_p50_fall', 'VH_p90_fall', 'VH_p75_IQR_fall',
  
  'Elevation', 'Slope', 'Aspect', 'TRI', 'avg_rad'];

  
// Feature extraction
var training = allMosaic.select(bands).sampleRegions({
  collection: trainingPoints,
  properties: ['class_id'],
  scale: 10
});



// Train classifier: Random Forest
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 1000
}).train({
  features: training,
  classProperty: 'class_id',
  inputProperties: bands
});


/*
// Train classifier: Gradient boost
var classifier = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 500, 
  shrinkage: 0.01, 
  samplingRate: 0.6, 
  maxNodes: 30, 
  loss: 'Huber', 
  seed: 42
}).train({
  features: training,
  classProperty: 'class_id',
  inputProperties: bands
});
*/


// Classify the image
var classified = allMosaic.clip(AOI_med).select(bands).classify(classifier).toUint8();

print('Exporting the classified image to Asset.');

Export.image.toAsset({
  image: classified,
  description: 'LandCover_AOImed_trainingBC_RF1000',
  assetId: 'projects/ee-lclu-bc/assets/LandCover_AOImed_trainingBC_RF1000',
  scale: 10,
  region: AOI_med,
  maxPixels: 1e13
});

/*
Map.centerObject(AOI_BC, 8);
Map.addLayer(s2Mosaic_sumFall.clip(AOI_BC), {bands: ['B8_summer', 'B4_summer', 'B3_summer'], min: 0, max: 3000}, 'S2 Mosaic Summer');
Map.addLayer(s2Mosaic_sumFall.clip(AOI_BC), {bands: ['B8_fall', 'B4_fall', 'B3_fall'], min: 0, max: 3000}, 'S2 Mosaic Fall');

Map.addLayer(s1Mosaic_sumFall.select('VH_summer').clip(AOI_BC), {min: -25, max: 5}, 'S1 Mosaic Summer');
Map.addLayer(s1Mosaic_sumFall.select('VH_fall').clip(AOI_BC), {min: -25, max: 5}, 'S1 Mosaic Fall');

*/