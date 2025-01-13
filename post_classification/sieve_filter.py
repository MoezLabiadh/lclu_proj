"""
Apply a Sieve filter to cleanup classification results
    The default Minimum Mapping Unit (MMU) - Threshold is 25 pixels (0.25 ha)
    The default connectedness uses the Von Neumann neighborhood (connectedness=4).    
"""

import os
from osgeo import gdal
import timeit


def apply_sieve_filter(input_raster, output_raster, threshold=25, connectedness=4):
    """
    Applies a sieve filter to the input raster.

    Parameters:
        input_raster (str): Path to the input raster file.
        output_raster (str): Path to the output raster file.
        threshold_size (int): Minimum size (in pixels) for sieve operation.
        threshold_size (int): Nbr of connecting pixels, Von Neumann neighborhood (4) 
                              or Moore neighborhood (8).
    """
    # Register all drivers
    gdal.AllRegister()

    # Open the input dataset
    print (f'Reading input raster: {os.path.basename(input_raster)}')
    dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"..unable to open: {os.path.basename(input_raster)}")

    # Get the input raster band
    band = dataset.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = 0

    # Read the raster data into memory
    raster_array = band.ReadAsArray()

    # Create an in-memory raster to apply the sieve filter
    print ('Applying the GDAL Sieve filter')
    driver = gdal.GetDriverByName('MEM')
    mem_ds = driver.Create('', dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(dataset.GetGeoTransform())
    mem_ds.SetProjection(dataset.GetProjection())

    mem_band = mem_ds.GetRasterBand(1)
    mem_band.WriteArray(raster_array)
    mem_band.SetNoDataValue(nodata_value)

    # Apply sieve filter
    gdal.SieveFilter(srcBand=mem_band, 
                     maskBand=None, 
                     dstBand=mem_band, 
                     threshold=threshold, 
                     connectedness=connectedness)

    # Create the output raster with LZW compression
    print ('Saving the output raster: {os.path.basename(output_raster)}')
    output_driver = gdal.GetDriverByName('GTiff')
    output_ds = output_driver.Create(
        output_raster,
        dataset.RasterXSize,
        dataset.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=['COMPRESS=LZW']
    )
    output_ds.SetGeoTransform(dataset.GetGeoTransform())
    output_ds.SetProjection(dataset.GetProjection())

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(mem_band.ReadAsArray())
    output_band.SetNoDataValue(nodata_value)

    # Cleanup
    band = None
    dataset = None
    mem_band = None
    mem_ds = None
    output_band = None
    output_ds = None


if __name__ == "__main__":
    start_t = timeit.default_timer()
    
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    
    input_raster = os.path.join(wks, 'classification', 'Tile19_MODIS.tif')
    output_raster = os.path.join(wks, 'classification', 'Tile19_sieve_thresh50_connect4.tif')

    apply_sieve_filter(input_raster, output_raster, threshold=50, connectedness=4)
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
