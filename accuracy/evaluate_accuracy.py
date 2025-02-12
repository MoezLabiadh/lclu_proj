import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             cohen_kappa_score, classification_report)

def load_evaluation_points(shapefile_fp):
    """
    Reads evaluation points from a shapefile and returns ground truth class IDs and their coordinates.
    """
    points_gdf = gpd.read_file(shapefile_fp)
    ground_truth = points_gdf['class_id'].values
    coords = [(pt.x, pt.y) for pt in points_gdf.geometry]
    return ground_truth, coords

def sample_raster_values(raster_fp, coords):
    """
    Samples raster values at the given coordinates.
    Assumes the raster has a single band.
    """
    with rasterio.open(raster_fp) as src:
        sampled_vals = list(src.sample(coords))
    predicted = [val[0] for val in sampled_vals]
    return predicted

def create_confusion_matrix(ground_truth, predicted, labels):
    """
    Computes the confusion matrix and returns a DataFrame that includes row and column totals.
    """
    cmatrix = confusion_matrix(ground_truth, predicted, labels=labels)
    cm_df = pd.DataFrame(cmatrix,
                         index=[f'Actual_{i}' for i in labels],
                         columns=[f'Predicted_{i}' for i in labels])
    # Add totals as an extra column
    cm_df['Total'] = cm_df.sum(axis=1)
    # Add totals as an extra row using pd.concat instead of append
    total_row = cm_df.sum(axis=0)
    total_row_df = pd.DataFrame(total_row).T.rename(index={0: 'Total'})
    cm_df = pd.concat([cm_df, total_row_df])
    
    return cm_df

def print_accuracy_metrics(ground_truth, predicted, labels):
    """
    Computes and prints overall accuracy, Cohen's Kappa, and a detailed classification report.
    """
    overall_accuracy = accuracy_score(ground_truth, predicted)
    kappa = cohen_kappa_score(ground_truth, predicted)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print("\nClassification Report:")
    print(classification_report(ground_truth, predicted, labels=labels))

def plot_confusion_heatmap(cm_df, output_image_path):
    """
    Displays the confusion matrix as a heatmap with totals and saves the image.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap="Blues", cbar=False)
    plt.title('Confusion Matrix with Totals')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    start_t = timeit.default_timer()

    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification\validation'
    points_fp = os.path.join(wks, 'validation_dataset', 'validation_points_tempo_bcAlbers.shp')
    raster_fp = os.path.join(wks, 'tiles19_26.tif')

    # Define class labels (1 to 9)
    labels = list(range(1, 10))
    
    # Load evaluation points and extract ground truth and coordinates
    ground_truth, coords = load_evaluation_points(points_fp)
    
    # Sample raster values at the point coordinates
    predicted = sample_raster_values(raster_fp, coords)
    
    # Create confusion matrix with totals
    cm_df = create_confusion_matrix(ground_truth, predicted, labels)
    print("Confusion Matrix with Totals:")
    print(cm_df)
    
    # Print accuracy metrics
    print_accuracy_metrics(ground_truth, predicted, labels)
    
    # Plot and export the heatmap
    output_image_path = os.path.join(wks, 'confusion_matix_heatmap.png')
    plot_confusion_heatmap(cm_df, output_image_path)
    
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
