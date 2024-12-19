import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import timeit



# Function to calculate nearest neighbors and filter points
def filter_points(gdf, class_id, threshold=50):
    class_gdf = gdf[gdf['class_id'] == class_id]
    if len(class_gdf) <= 15000:
        return gdf

    # Calculate pairwise distances
    print ('..calculate pairwise distances')
    points = class_gdf.geometry
    distances = points.apply(lambda p: points.distance(p))
    
    # Find points with distances below the threshold
    print ('..identify points to remove')
    close_points = distances.apply(lambda d: d[d < threshold].index.tolist())
    close_points_flat = set([idx for sublist in close_points for idx in sublist])

    # Determine number of points to remove
    points_to_remove = int(len(class_gdf) * 0.3)

    print ('..remove points points')
    # Prioritize removing close points
    remove_indices = list(close_points_flat)[:points_to_remove]
    if len(remove_indices) < points_to_remove:
        remaining_needed = points_to_remove - len(remove_indices)
        remaining_indices = class_gdf.index.difference(remove_indices)
        random_indices = np.random.choice(remaining_indices, remaining_needed, replace=False)
        remove_indices.extend(random_indices)

    # Drop points from original GeoDataFrame
    reduced_gdf = gdf.drop(remove_indices)
    return reduced_gdf


if __name__ == '__main__':
    start_t = timeit.default_timer()
    
    print ('\nReading the input gdf')
    # Load your geodataframe (example)
    gdf = gpd.read_file(r"Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification\data\training_data\points_train.shp")
    gdf = gdf.to_crs(epsg=3005)
    
    # Apply the filter to each unique class_id
    for class_id in gdf['class_id'].unique():
        print (f'\nWorking on class {class_id}')
        gdf = filter_points(gdf, class_id)
    
    
    print ('\nSaving the output gdf')
    gdf = gdf.to_crs(epsg=4326)
    
    # Save the reduced GeoDataFrame
    gdf.to_file(r"Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification\data\training_data\points_train_reduced_v2.shp")
    
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
