import os
import rasterio
import numpy as np
import geopandas as gpd
import timeit

def change_raster_values_from_polygons(
        input_raster, 
        polygon_path, 
        output_raster, 
        old_value_col='old_value', 
        new_value_col='new_value'
):
    """
    Change raster pixel values within multiple polygons based on their attributes
    Output is 8-bit with LZW compression
    
    Parameters:
    input_raster (str): Path to input raster file
    polygon_path (str): Path to polygon shapefile
    output_raster (str): Path to output raster file
    old_value_col (str): Column name containing values to replace
    new_value_col (str): Column name containing new values
    
    Returns:
    str: Path to output raster
    """
    
    # Read the raster
    print('Reading input raster')
    with rasterio.open(input_raster) as src:
        # Read the raster data
        raster_data = src.read(1)  # Read first band
        output_data = raster_data.copy()
        
        # Read the polygon file
        gdf = gpd.read_file(polygon_path)
        
        # Verify columns exist
        if old_value_col not in gdf.columns or new_value_col not in gdf.columns:
            raise ValueError(f"Columns {old_value_col} and/or {new_value_col} not found in shapefile")
        
        # Reproject polygon to raster's CRS if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # Process each polygon
        for idx, row in gdf.iterrows():
            old_val = row[old_value_col]
            new_val = row[new_value_col]
            
            # Validate values are within 8-bit range
            if not (0 <= new_val <= 255):
                raise ValueError(f"New value {new_val} is outside 8-bit range (0-255)")
            
            # Create mask for current polygon
            print (f'Coverting pixel value {old_val} to {new_val}')
            mask = rasterio.features.geometry_mask(
                geometries=[row.geometry],
                out_shape=raster_data.shape,
                transform=src.transform,
                invert=True  # True = mask inside polygon
            )
            
            # Update values for this polygon
            output_data[(mask) & (raster_data == old_val)] = new_val
        
        kwargs = src.meta.copy()
        # Force 8-bit unsigned integer
        kwargs.update({
            'dtype': 'uint8',
            'compress': 'lzw',
            'driver': 'GTiff'
        })
        
        # Convert output data to 8-bit if needed
        print('Writing output raster')
        output_data = output_data.astype('uint8')
        
        with rasterio.open(output_raster, 'w', **kwargs) as dst:
            dst.write(output_data, 1)
        


if __name__ == "__main__":
    start_t = timeit.default_timer()
    
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification\work\wetlands'
    input_raster = os.path.join(wks, 'mosaic_clipped_sieved100px_fraserLowland.tif')
    polygon_path = os.path.join(wks, 'change_values_polys.shp')
    output_raster= os.path.join(wks, 'mosaic_clipped_sieved100px_fraserLowland_EDITED.tif')
    
    try:
        change_raster_values_from_polygons(
            input_raster,
            polygon_path,
            output_raster,
            old_value_col='old_value',
            new_value_col='new_value'
        )

    except ValueError as e:
        print(f"Error: {e}")
    
    finally:
        finish_t = timeit.default_timer()
        t_sec = round(finish_t - start_t)
        mins, secs = divmod(t_sec, 60)
        print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')