"""
Apply a Majority filter to smooth classification results
The default kernel size is 5*5
"""
import os
import rasterio
#from rasterio.windows import Window
import numpy as np
from scipy.ndimage import generic_filter
import timeit


def majority_filter_func(values):
    values = values[values > 0]  # Exclude 0 values (treat as NoData)
    if len(values) == 0:
        return 0  # Keep NoData (0) if neighborhood is all NoData
    return np.bincount(values.astype(int)).argmax()


def apply_majority_filter(input_raster, output_raster, kernel_size=5):
    with rasterio.open(input_raster) as src:
        profile = src.profile
        
        profile.update(dtype='uint8', compress='lzw')

        print (f'Reading input raster: {os.basename(input_raster)}')
        data = src.read(1, masked=False)
        
        print ('Applying the Majority Filter function')
        filtered_data = generic_filter(
            data,
            function=majority_filter_func,
            size=kernel_size,
            mode='constant',
            cval=0  # Use 0 for padding (NoData)
        )

        print ('Saving the output raster')
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(filtered_data.astype('uint8'), 1)



if __name__ == "__main__":
    start_t = timeit.default_timer()
    
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    
    input_raster = os.path.join(wks, 'classification', 'Tile19_MODIS.tif')
    output_raster = os.path.join(wks, 'classification', 'Tile19_majFilter_kernel5.tif')
    apply_majority_filter(input_raster, output_raster, kernel_size=5)
