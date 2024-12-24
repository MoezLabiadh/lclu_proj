
import os
import geopandas as gpd
from shapely.geometry import box
import numpy as np


def create_grid(gdf, tile_size):
    # Ensure the CRS uses meters
    gdf = gdf.to_crs("EPSG:3005")
    
    minx, miny, maxx, maxy = gdf.total_bounds
    
    x_coords = np.arange(minx, maxx, tile_size)
    y_coords = np.arange(miny, maxy, tile_size)
    
    tiles = []
    for x in x_coords:
        for y in y_coords:
            tiles.append(box(x, y, x + tile_size, y + tile_size))
            
    grid= gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)
    
    tiles_gdf = gpd.overlay(gdf, grid, how='intersection')
    tiles_gdf['tile_id'] = np.arange(len(tiles_gdf))
    
    return tiles_gdf[['tile_id', 'geometry']]
    

wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'

shp_aoi = os.path.join(wks, "data", "AOIs" ,"bc.shp")
    
    
gdf = gpd.read_file(shp_aoi)
tile_size = 200000  # 200 km2
tiles_gdf = create_grid(gdf, tile_size)
tiles_gdf.to_file(os.path.join(wks, "data", "AOIs" ,"bc_tiles_200km.shp"))


