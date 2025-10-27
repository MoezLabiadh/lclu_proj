
//************************************************** SENTINEL-2 ***********************************************************************//

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
   return img.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
                      'B8', 'B8A', 'B11', 'B12']).updateMask(notCldShdw);
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


//************************************************** SENTINEL-1 ***********************************************************************//

// Function to get Sentinel-1 features and descriptive stats
function getS1bands(aoi, startDate, endDate) {
  // Load Sentinel-1 ImageCollection
  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    .sort('system:time_start', false);
 
  // Compute Mean for VV and VH
  var vvMean = s1Col.select('VV').mean().rename('VV_Mean');
  var vhMean = s1Col.select('VH').mean().rename('VH_Mean');

  // Compute Variance for VV and VH
  var vvVariance = s1Col.select('VV').reduce(ee.Reducer.variance()).rename('VV_Variance');
  var vhVariance = s1Col.select('VH').reduce(ee.Reducer.variance()).rename('VH_Variance');

  // Compute Standard Deviation for VV and VH
  var vvStdDev = s1Col.select('VV').reduce(ee.Reducer.stdDev()).rename('VV_StdDev');
  var vhStdDev = s1Col.select('VH').reduce(ee.Reducer.stdDev()).rename('VH_StdDev');

  // Compute Percentiles for VV and VH
  var vvPercentiles = s1Col.select('VV').reduce(ee.Reducer.percentile([10, 25, 75, 90]));
  var vhPercentiles = s1Col.select('VH').reduce(ee.Reducer.percentile([10, 25, 75, 90]));

  // Extract specific percentiles
  var vvPerc10 = vvPercentiles.select('VV_p10').rename('VV_Perc10');
  var vvPerc25 = vvPercentiles.select('VV_p25');
  var vvPerc75 = vvPercentiles.select('VV_p75');
  var vvPerc90 = vvPercentiles.select('VV_p90').rename('VV_Perc90');
  var vvIQR = vvPerc75.subtract(vvPerc25).rename('VV_IQR');

  var vhPerc10 = vhPercentiles.select('VH_p10').rename('VH_Perc10');
  var vhPerc25 = vhPercentiles.select('VH_p25');
  var vhPerc75 = vhPercentiles.select('VH_p75');
  var vhPerc90 = vhPercentiles.select('VH_p90').rename('VH_Perc90');
  var vhIQR = vhPerc75.subtract(vhPerc25).rename('VH_IQR');

  // Compute additional bands using VV_Mean and VH_Mean
  var rvi = vhMean.multiply(4).divide(vvMean.add(vhMean)).rename('RVI');
  var vhVvRatio = vhMean.divide(vvMean).rename('Ratio_VH_VV');
  var ndpi = vvMean.subtract(vhMean).divide(vvMean.add(vhMean)).rename('NDPI');
  var dpd = vvMean.subtract(vhMean).rename('DPD');
  var pdi = vvMean.subtract(vhMean).divide(vvMean).rename('PDI');
  var wri = vvMean.add(vhMean).divide(vvMean.subtract(vhMean)).rename('WRI');

  // Compute GLCM Texture Metrics for VV and VH
  var glcmVV = vvMean.unitScale(0, 1).multiply(255).toByte().glcmTexture({
    size: 3
  });
  var glcmVH = vhMean.unitScale(0, 1).multiply(255).toByte().glcmTexture({
    size: 3
  });
  
  // Select relevant GLCM bands for land cover mapping
  var glcmBandsVV = glcmVV.select(
    ['VV_Mean_contrast', 'VV_Mean_corr', 'VV_Mean_ent', 'VV_Mean_var', 'VV_Mean_diss'],
    ['GLCM_VV_Contrast', 'GLCM_VV_Correlation', 'GLCM_VV_Entropy', 'GLCM_VV_Variance', 'GLCM_VV_Dissimilarity']
  );
  var glcmBandsVH = glcmVH.select(
    ['VH_Mean_contrast', 'VH_Mean_corr', 'VH_Mean_ent', 'VH_Mean_var', 'VH_Mean_diss'],
    ['GLCM_VH_Contrast', 'GLCM_VH_Correlation', 'GLCM_VH_Entropy', 'GLCM_VH_Variance', 'GLCM_VH_Dissimilarity']
  );

  // Combine metrics into a single image
  var s1Bands = vvMean
    .addBands(vhMean)
    .addBands(vvVariance)
    .addBands(vhVariance)
    .addBands(vvStdDev)
    .addBands(vhStdDev)
    .addBands(vvPerc10)
    .addBands(vvPerc90)
    .addBands(vvIQR)
    .addBands(vhPerc10)
    .addBands(vhPerc90)
    .addBands(vhIQR)
    .addBands(rvi)
    .addBands(vhVvRatio)
    .addBands(ndpi)
    .addBands(dpd)
    .addBands(pdi)
    .addBands(wri)
    .addBands(glcmBandsVV)
    .addBands(glcmBandsVH);

  return s1Bands;
}

//************************************************** OTHER LAYERS ***********************************************************************//

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


// Function to get MODIS burned areas (2022 - 2024)
function getMODISburned(img, aoi) {

  // Load MODIS burned area dataset
  var dataset = ee.ImageCollection('MODIS/061/MCD64A1')
                  .filter(ee.Filter.date('2022-01-01', '2024-12-31'))
                  .filterBounds(aoi);

  // Calculate mean burned area
  var burnedArea = dataset.select('BurnDate').mean();
  
  // Set burned pixels to 1 and unmask the rest with 0
  var burnedBinary = burnedArea.gt(0).rename('burned').unmask(0).clip(AOI_BC);

  // Add the burned band to the input image
  return img.addBands([burnedBinary]);
}

function getEsriLC(img, aoi) {
  function remapper(image) {
    var remapped = image.remap([1, 2, 4, 5, 7, 8, 9, 10, 11],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    return remapped;
  }

  var esri_lulc10 = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS");

  var esri2023 = esri_lulc10.filterDate('2023-01-01', '2023-12-31')
                            .map(remapper)
                            .mosaic()
                            .clip(aoi)
                            .rename('esri2023lc');

  return img.addBands([esri2023]);
}


function getDwLC(img, aoi) {
  var collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                  .filterDate('2024-05-15', '2024-09-15');
  
  // Function to get the most common land cover for each pixel
  var mostCommon = collection.map(function(image) {
    return image.select('label').clip(aoi);
  }).reduce(ee.Reducer.mode());
  
  var dw = mostCommon.rename('dw');

  return img.addBands([dw]);
}




//******************************************************* DATA PREP***********************************************************************//


var targetDate_summer = '2024-07-15';
var targetDate_fall = '2024-09-15';
var targetDate_s1nogap = '2024-06-01';
var timeStep_s1 = 180;
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




// Get Sentinel-1 Features
var s1Features = getS1bands(AOI_BC, 
                         ee.Date(targetDate_s1nogap).advance(-timeStep_s1, 'day'), 
                         ee.Date(targetDate_s1nogap).advance(timeStep_s1, 'day'));

print('S1 Mosaic:', s1Features);




// Merge all input features: S1, S2, terrain and radiance
var allMosaic = s2Mosaic_sumFall.addBands(s1Features);

var allMosaic = addTerrain(allMosaic, AOI_BC);


var allMosaic = getVIIRSradiance(allMosaic, 
                                ee.Date(targetDate_summer).advance(-timeStep, 'day'), 
                                ee.Date(targetDate_summer).advance(timeStep, 'day'), 
                                AOI_BC);

var allMosaic = getMODISburned (allMosaic, AOI_BC);

//var allMosaic = getDwLC(allMosaic, AOI_BC);  
//var allMosaic =  getEsriLC(allMosaic, AOI_BC);

print('All features Mosaic:', allMosaic);



//******************************************************* CLASSIFICATION***********************************************************************//

var trainingPoints = ee.FeatureCollection('projects/ee-lclu-bc/assets/points_train_reduced_v10_noBR')
                              .filterBounds(AOI_training_v3);

var trainingPointsSize = trainingPoints.size();
print('Number of training points:', trainingPointsSize);
                        
var bands = allMosaic.bandNames();

print ('Bands used in classification:', bands);


// Define tile size in meters (approximately 100 km)
var tileSize = 100000;  // 100 km in meters

// Create grid of 100km tiles
var tiles = trainingPoints.geometry().coveringGrid('EPSG:4326', tileSize);

// Function to sample points within each tile
var sampleTile = function(tile) {
  // Filter points within the current tile
  var pointsInTile = trainingPoints.filterBounds(tile.geometry());

  // Sample points within the tile
  var sampledPoints = allMosaic.select(bands).sampleRegions({
    collection: pointsInTile,
    properties: ['class_id'],
    scale: 10
  });
  return sampledPoints;
};

// Map over all tiles and flatten results
var training = tiles.map(sampleTile).flatten();


// Train classifier: Random Forest
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 500
}).train({
  features: training,
  classProperty: 'class_id',
  inputProperties: bands
});


// Classify the image 
var tiles = ee.FeatureCollection("projects/ee-lclu-bc/assets/bc_tiles_200km_modified");
var tile24 = tiles.filter(ee.Filter.eq('tile_id', 24));

var classified = allMosaic.clip(tile24).select(bands).classify(classifier).toUint8();

Export.image.toAsset({
  image: classified,
  description: 'LandCover_tile24_trainNoBR_RF500_v10',
  assetId: 'projects/ee-lclu-bc/assets/LandCover_tile24_trainNoBR_RF500_v10',
  scale: 10,
  region: tile24,
  maxPixels: 1e13
});


/*
// Export classifier as asset
print('Exporting the classifier as Asset');
Export.classifier.toAsset({
  classifier: classifier,
  description: 'classifier_RF_v10 ',
  assetId: 'projects/ee-lclu-bc/assets/classifier_RF_v10 ',
  priority: 100
});


Map.centerObject(AOI_BC, 8);
Map.addLayer(s2Mosaic_sumFall.clip(AOI_BC), {bands: ['B8_summer', 'B4_summer', 'B3_summer'], min: 0, max: 3000}, 'S2 Mosaic Summer');
Map.addLayer(s2Mosaic_sumFall.clip(AOI_BC), {bands: ['B8_fall', 'B4_fall', 'B3_fall'], min: 0, max: 3000}, 'S2 Mosaic Fall');

Map.addLayer(s1Mosaic_sumFall.select('VH_summer').clip(AOI_BC), {min: -25, max: 5}, 'S1 Mosaic Summer');
Map.addLayer(s1Mosaic_sumFall.select('VH_fall').clip(AOI_BC), {min: -25, max: 5}, 'S1 Mosaic Fall');

*/
