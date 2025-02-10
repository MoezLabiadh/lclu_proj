// Define visualization parameters
var visLc = {
  min: 1,
  max: 10,
  palette: [
    '00841f', // Tree cover
    '58f315', // Shrubland
    'fdf635', // Grassland
    'f4bb00', // Cropland
    '5c9cb9', // Wetland
    '0016e0', // Water
    'd20c1d', // Urban
    'b5761d', // Bare ground
    '7f13de',  // Snow and ice
    '000000'  // Burned Area
  ]
};

// Load the images as assets
var tile19 = ee.Image('projects/ee-lclu-bc/assets/LandCover_tile19_training70k_RF500_v8');
var tile26 = ee.Image('projects/ee-lclu-bc/assets/LandCover_tile26_training70k_RF500_v8');
var tile24 = ee.Image('projects/ee-lclu-bc/assets/LandCover_tile24_training70k_RF500_v8');
var tile29 = ee.Image('projects/ee-lclu-bc/assets/LandCover_tile29_training70k_RF500_v8');
var tile8 = ee.Image('projects/ee-lclu-bc/assets/LandCover_tile8_training70k_RF500_v8');
var tile10 = ee.Image('projects/ee-lclu-bc/assets/LandCover_tile10_training70k_RF500_v8');
var tile13 = ee.Image('projects/ee-lclu-bc/assets/LandCover_tile13_training70k_RF500_v8');

// Create a map and set options
Map.setOptions('SATELLITE');
Map.setCenter(-125.0, 54.0, 5); // Centered on British Columbia, Canada, with a zoom level of 5

// Add images to the map and create layers
var layers = [
  {label: 'tile19', image: tile19},
  {label: 'tile26', image: tile26},
  {label: 'tile24', image: tile24},
  {label: 'tile29', image: tile29},
  {label: 'tile08', image: tile8},
  {label: 'tile10', image: tile10},
  {label: 'tile13', image: tile13},
];

// Add layers to the map by default
layers.forEach(function(layerObj) {
  Map.addLayer(layerObj.image, visLc, layerObj.label, true);
});

// Create a unified panel
var panel = ui.Panel({
  layout: ui.Panel.Layout.Flow('vertical'),
  style: {position: 'top-right', padding: '8px', width: '300px'}
});

// Add checkboxes and "Zoom to" links for each layer
layers.forEach(function(layerObj) {
  var checkbox = ui.Checkbox(layerObj.label, true, function(checked) {
    if (checked) {
      Map.addLayer(layerObj.image, visLc, layerObj.label);
    } else {
      Map.layers().forEach(function(layer) {
        if (layer.getName() === layerObj.label) {
          Map.layers().remove(layer);
        }
      });
    }
  });
  checkbox.setValue(true); // Ensure checkbox is checked by default

  // Create a clickable "Zoom to" link using a button
  var zoomButton = ui.Button({
    label: 'Zoom to',
    style: {margin: '0 0 0 6px', padding: '0 4px'},
    onClick: function() {
      Map.centerObject(layerObj.image, 8); // Zoom to the layer 
    }
  });

  // Add checkbox and button to the panel
  var row = ui.Panel({
    widgets: [checkbox, zoomButton],
    layout: ui.Panel.Layout.Flow('horizontal'),
    style: {margin: '0 0 4px 0'}
  });
  panel.add(row);
});

// Add the legend to the same panel
var legendHeader = ui.Label('Legend', {fontWeight: 'bold', margin: '10px 0 0 0'});
panel.add(legendHeader);

var palette = visLc.palette;
var labels = [
  'Tree cover', 'Shrubland', 'Grassland', 'Cropland',
  'Wetland', 'Water', 'Built-up', 'Bare ground',
  'Snow and ice', 'Burned Area'
];

for (var i = 0; i < palette.length; i++) {
  var colorBox = ui.Label('', {
    backgroundColor: palette[i],
    padding: '8px',
    margin: '0 8px 4px 0',
    border: '1px solid black'
  });
  var label = ui.Label(labels[i]);
  var legendItem = ui.Panel({
    widgets: [colorBox, label],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
  panel.add(legendItem);
}

// Add the unified panel to the map
Map.add(panel);
