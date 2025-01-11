// ===================================
// Step 1: Define Area of Interest (AOI)
// ===================================
var aoi = table;
Map.centerObject(aoi, 8);

// ===================================
// Step 2: Define Timeframe and Visualization Parameters
// ===================================
var year = 2015;
var startDate = year + '-08-01';
var endDate = year + '-11-30';

var visParams = {
  min: -1,
  max: 1,
  palette: ['red', 'yellow', 'green']
};

// ===================================
// Step 3: Load and Preprocess Landsat 8 Data
// ===================================
function maskL8sr(image) {
  var cloudShadowBitMask = (1 << 3); 
  var cloudsBitMask = (1 << 5); 
  var qa = image.select('QA_PIXEL'); 
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
    .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask); 
}

// Load and preprocess Landsat 8 data
var landsat8Image = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUD_COVER', 1))
  .map(maskL8sr)
  .median()
  .clip(aoi);

var scaledLandsat8 = landsat8Image
  .select(['SR_B.*'])
  .multiply(0.0000275)
  .add(-0.2)
  .addBands(
    landsat8Image.select('ST_B10').multiply(0.00341802).add(149.0)
  );

// ===================================
// Step 4: Calculate Ecological Indices
// ===================================
// NDVI Calculation
var ndvi = scaledLandsat8.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
var ndviStats = ndvi.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: aoi,
  scale: 30,
  maxPixels: 1e13
});
var ndviMin = ee.Number(ndviStats.get('NDVI_min'));
var ndviMax = ee.Number(ndviStats.get('NDVI_max'));

var fv = ndvi.subtract(ndviMin)
  .divide(ndviMax.subtract(ndviMin))
  .pow(2)
  .rename('FV');

var em = fv.multiply(0.004).add(0.986).rename('EM');

// Land Surface Temperature (LST)
var thermal = scaledLandsat8.select('ST_B10').rename('thermal');
var lst = thermal.expression(
  '(TB / (1 + (0.00115 * (TB / 1.438)) * log(em))) - 273.15', {
    'TB': thermal.select('thermal'),
    'em': em.max(0.001)
  }).rename('LST');

// Wetness Index
var wetness = scaledLandsat8.expression(
  '0.1511 * B2 + 0.1973 * B3 + 0.3283 * B4 + 0.3407 * B5 - 0.7117 * B6 - 0.4559 * B7', {
    'B2': scaledLandsat8.select('SR_B2'),
    'B3': scaledLandsat8.select('SR_B3'),
    'B4': scaledLandsat8.select('SR_B4'),
    'B5': scaledLandsat8.select('SR_B5'),
    'B6': scaledLandsat8.select('SR_B6'),
    'B7': scaledLandsat8.select('SR_B7')
  }).rename('Wetness');

// NDBSI Index
var ndbsi = scaledLandsat8.expression(
  '((RED + SWIR1) - (NIR + BLUE)) / ((RED + SWIR1) + (NIR + BLUE))', {
    'RED': scaledLandsat8.select('SR_B4'),
    'SWIR1': scaledLandsat8.select('SR_B6'),
    'NIR': scaledLandsat8.select('SR_B5'),
    'BLUE': scaledLandsat8.select('SR_B2')
  }).rename('NDBSI');

// ===================================
// Step 5: Standardize Components
// ===================================
function standardize(band, bandName) {
  var stats = band.reduceRegion({
    reducer: ee.Reducer.mean().combine({
      reducer2: ee.Reducer.stdDev(),
      sharedInputs: true
    }),
    geometry: aoi,
    scale: 30,
    maxPixels: 1e13
  });
  
  var mean = ee.Number(stats.get(bandName + '_mean'));
  var stdDev = ee.Number(stats.get(bandName + '_stdDev'));
  
  return band.subtract(mean).divide(stdDev).rename(bandName + '_std');
}

// Standardize Bands
var standardizedComponents = ndvi
  .addBands(standardize(wetness, 'Wetness'))
  .addBands(standardize(lst, 'LST'))
  .addBands(standardize(ndbsi, 'NDBSI'))
  .clip(aoi);

// ===================================
// Step 6: Perform PCA
// ===================================
var arrayImage = standardizedComponents.toArray();
var covariance = arrayImage.reduceRegion({
  reducer: ee.Reducer.centeredCovariance(),
  geometry: aoi,
  scale: 30,
  maxPixels: 1e9
});

var covarianceArray = ee.Array(covariance.get('array'));
var eigens = covarianceArray.eigen();
var eigenValues = eigens.slice(1, 0, 1);
var eigenVectors = eigens.slice(1, 1);
print('Eigenvalues:', eigenValues);

var principalComponents = ee.Image(eigenVectors)
  .matrixMultiply(arrayImage.toArray(1))
  .arrayProject([0])
  .arrayFlatten([getNewBandNames('pc')]);

var principalComponents1 = principalComponents.select('pc1');

var principalComponents1 = principalComponents1 
                               .clamp(-4, 4)
// Map.addLayer(principalComponents1, {palette: ['blue', 'white', 'red']}, 'PC1');

// =====================================================
// Step 7: Normalize PC1 to [-1, +1] and declare it RSEI
// =====================================================
var pc1Stats = principalComponents1.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: aoi,
  scale: 30,
  maxPixels: 1e9
});

var pc1Min = ee.Number(pc1Stats.get('pc1_min'));
var pc1Max = ee.Number(pc1Stats.get('pc1_max'));

print(pc1Min)
print(pc1Max)
var RSEI = principalComponents1
  .subtract(pc1Min)
  .divide(pc1Max.subtract(pc1Min))
  .rename('RSEI');

Map.addLayer(RSEI, {palette: ['red', 'yellow', 'green']}, 'RSEI');

// ===================================
// Step 7: Classify RSEI into 5 Classes
// ===================================
var thresholds = [0.2, 0.4, 0.6, 0.8];
var LSES = RSEI.expression(
  '(b(0) <= thresholds[0]) ? 1 : (b(0) <= thresholds[1]) ? 2 : ' +
  '(b(0) <= thresholds[2]) ? 3 : (b(0) <= thresholds[3]) ? 4 : 5', {
    'thresholds': thresholds
  }).clip(aoi)
  .rename('LSES_Class');

Map.addLayer(LSES, {
  min: 1,
  max: 5,
  palette: ['#B2182B', '#EF6548', '#FEE08B', '#66BD63', '#006837']
}, 'LSES Classes');

// // ===================================
// // Step 8: Visualization and Histogram
// // ===================================
// var RSEIHistogram = RSEI.reduceRegion({
//   reducer: ee.Reducer.histogram(255),
//   geometry: aoi,
//   scale: 30,
//   maxPixels: 1e9
// });

// var histogram = ee.Dictionary(RSEIHistogram.get('RSEI'));
// var bucketMeans = ee.List(histogram.get('bucketMeans'));
// var histogramValues = ee.List(histogram.get('histogram'));

// var chart = ui.Chart.array.values({
//   array: ee.Array(histogramValues),
//   axis: 0,
//   xLabels: bucketMeans
// })
// .setChartType('ColumnChart')
// .setOptions({
//   title: 'RSEI distribution',
//   hAxis: {title: 'RSEI'},
//   vAxis: {title: 'Frequency'},
//   legend: {position: 'none'},
//   colors: ['#1f77b4']
// });
// print(chart);

// ===================================
// Step 9: Add Map Title and Legend
// ===================================

// // Add Map Title
// Map.addLayer(ee.Image(0), {}, year + ' LSES Map');

// // Add Legend for LSES Classes
// var legend = ui.Panel({
//   style: {
//     position: 'bottom-left',
//     padding: '8px 15px'
//   }
// });

// // Legend Title
// legend.add(ui.Label({
//   value: 'LSES Class Legend ',
//   style: {
//     fontWeight: 'bold',
//     fontSize: '14px',
//     margin: '0 0 4px 0',
//     textAlign: 'center'
//   }
// }));

// // Class Labels and Colors
// var classNames = ['Very Low', 'Low', 'Moderate', 'High', 'Very High'];
// var classColors = ['#B2182B', '#EF6548', '#FEE08B', '#66BD63', '#006837'];

// for (var i = 0; i < classNames.length; i++) {
//   legend.add(ui.Panel({
//     widgets: [
//       ui.Label({
//         style: {
//           backgroundColor: classColors[i],
//           padding: '8px',
//           margin: '0 8px 4px 0'
//         }
//       }),
//       ui.Label({
//         value: classNames[i],
//         style: {
//           margin: '0 0 4px 0'
//         }
//       })
//     ],
//     layout: ui.Panel.Layout.Flow('horizontal')
//   }));
// }

// // Add Legend to the Map
// Map.add(legend);



// ===================================
// Supporting Function for Band Names
// ===================================
function getNewBandNames(prefix) {
  return ee.List.sequence(1, 4).map(function(i) {
    return ee.String(prefix).cat(ee.Number(i).int());
  });
}
