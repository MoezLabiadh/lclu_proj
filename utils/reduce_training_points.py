import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import timeit

# Function to calculate nearest neighbors and filter points
def filter_points(gdf, class_id, threshold=50):
    class_gdf = gdf[gdf['class_id'] == class_id]
    if len(class_gdf) <= 10000:
        return gdf

    # Create BallTree for efficient spatial querying
    print('..calculating nearest neighbors')
    coords = np.array([(geom.x, geom.y) for geom in class_gdf.geometry])
    tree = BallTree(coords, metric='euclidean')

    # Query distances for neighbors within the threshold
    indices = tree.query_radius(coords, r=threshold)
    avg_distances = np.array([np.mean(tree.query([coords[i]], k=len(neighbors))[0][0]) 
                              for i, neighbors in enumerate(indices)])

    # Sort points by their average distance to prioritize removal
    print('..prioritizing points to remove')
    sorted_indices = np.argsort(avg_distances)  # Smaller distances first
    points_to_remove = int(len(class_gdf) * 0.3)

    # Select indices of points to remove
    remove_indices = class_gdf.index[sorted_indices[:points_to_remove]]

    # Drop points from the original GeoDataFrame
    print('..removing points')
    reduced_gdf = gdf.drop(remove_indices)
    return reduced_gdf


if __name__ == '__main__':
    start_t = timeit.default_timer()
    
    print('\nReading the input gdf')
    # Load your geodataframe (example)
    gdf = gpd.read_file(r"Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification\data\training_data\points_train_reduced_v3.shp")
    gdf = gdf.to_crs(epsg=3005)
    
    # Apply the filter to each unique class_id
    for class_id in gdf['class_id'].unique():
        print(f'\nWorking on class {class_id}')
        gdf = filter_points(gdf, class_id)
    
    print('\nSaving the output gdf')
    gdf = gdf.to_crs(epsg=4326)
    
    # Save the reduced GeoDataFrame
    gdf.to_file(r"Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification\data\training_data\points_train_reduced_v4.shp")
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
