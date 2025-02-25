"""
Generates Accuracy Assesement metrics and heatmaps 
for the land cover classification

Author: moez
"""

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
    Filters out predictions not between 1 and 9.
    Returns the filtered predictions and the corresponding valid indices.
    """
    with rasterio.open(raster_fp) as src:
        sampled_vals = list(src.sample(coords))
    predictions = [val[0] for val in sampled_vals]
    valid_indices = [i for i, p in enumerate(predictions) if 1 <= p <= 9]
    filtered_predictions = [predictions[i] for i in valid_indices]
    
    return filtered_predictions, valid_indices


def create_confusion_matrix(ground_truth, predicted, labels):
    """
    Computes the confusion matrix and returns a DataFrame that includes row and column totals.
    """
    cmatrix = confusion_matrix(ground_truth, predicted, labels=labels)
    cm_df = pd.DataFrame(cmatrix,
                         index=[f'{i}' for i in labels],
                         columns=[f'{i}' for i in labels])
    cm_df['Total'] = cm_df.sum(axis=1)
    total_row = cm_df.sum(axis=0)
    total_row_df = pd.DataFrame(total_row).T.rename(index={0: 'Total'})
    cm_df = pd.concat([cm_df, total_row_df])
    
    return cm_df


def print_accuracy_metrics(ground_truth, predicted, labels):
    """
    Computes and prints overall accuracy, Cohen's Kappa, and a detailed classification report.
    Returns overall accuracy and kappa coefficient.
    """
    overall_accuracy = accuracy_score(ground_truth, predicted)
    kappa = cohen_kappa_score(ground_truth, predicted)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print("\nClassification Report:")
    print(classification_report(ground_truth, predicted, labels=labels))
    
    return overall_accuracy, kappa


def plot_confusion_heatmap(cm_df, output_image_path):
    """
    Displays the confusion matrix as a heatmap and saves the image.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.show()
    

def plot_classification_report_heatmap(report_df, classes, output_image_path):
    """
    Creates and saves a heatmap for the classification report.
    X-axis: metrics (precision, recall, f1-score in that order)
    Y-axis: classes.
    """
    # Create a DataFrame with classes as rows and metrics as columns.
    metrics_df = report_df.loc[classes, ["precision", "recall", "f1-score"]]
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="RdYlBu", cbar=True)
    plt.title("Classification Report Metrics Heatmap")
    plt.xlabel("Metrics", fontweight='bold')
    plt.ylabel("Classes", fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.show()
    

def export_accuracy_summary(samples_df, cm_df, classification_report_df, excel_file_path, overall_accuracy, kappa):
    """
    Exports the sample counts, confusion matrix, and classification report.
    """
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # Write samples per class as the first sheet.
        samples_df.to_excel(writer, sheet_name='Samples Per Class', index=False)
        
        # Write confusion matrix.
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')
        
        # Write classification report.
        classification_report_df.to_excel(writer, sheet_name='Classification Report', startrow=0)
        
        # Access the worksheet for Classification Report.
        ws = writer.sheets['Classification Report']
        # Calculate the starting row for the metrics.
        # Note: to_excel writes the header as well, so the total rows = df.shape[0] + 1.
        start_row_metrics = classification_report_df.shape[0] + 1 + 3  # 3 blank rows
        ws.cell(row=start_row_metrics, column=1, value=f"Overall Accuracy: {overall_accuracy:.4f}")
        ws.cell(row=start_row_metrics + 1, column=1, value=f"Kappa Coefficient: {kappa:.4f}")
        
        

if __name__ == '__main__':
    start_t = timeit.default_timer()

    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    points_fp = os.path.join(wks, 'validation', 'validation_dataset', 'validation_points_test.shp')
    raster_fp = os.path.join(wks, 'classification', 'mosaic', 'mosaic_south_sieve_thresh25_connect4.tif')

    # Mapping from numeric IDs to descriptive class labels.
    class_label_mapping = {
        1: 'Tree cover',
        2: 'Shrubland',
        3: 'Grassland',
        4: 'Cropland',
        5: 'Wetland',
        6: 'Permanent Water',
        7: 'Built-up',
        8: 'Bare ground',
        9: 'Snow and ice'
    }
    
    numeric_labels = list(range(1, 10))
    string_labels = [class_label_mapping[i] for i in numeric_labels]

    # Load evaluation points.
    ground_truth_numeric, coords = load_evaluation_points(points_fp)
    
    # Count samples per class.
    unique, counts = np.unique(ground_truth_numeric, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print("Number of samples per class:")
    for num in numeric_labels:
        print(f"{class_label_mapping[num]} ({num}): {count_dict.get(num, 0)}")
    print("\n" + "="*60 + "\n")
    
    # Create a DataFrame for sample counts.
    samples_df = pd.DataFrame({
        "Class ID": numeric_labels,
        "Class Label": [class_label_mapping[num] for num in numeric_labels],
        "Number of Samples": [count_dict.get(num, 0) for num in numeric_labels]
    })

    # Sample raster values and get valid indices.
    predicted_numeric, valid_indices = sample_raster_values(raster_fp, coords)
    
    # Filter ground truth and coordinates to only include valid points.
    ground_truth_numeric = [ground_truth_numeric[i] for i in valid_indices]
    coords = [coords[i] for i in valid_indices]

    # Convert numeric labels to descriptive text labels.
    ground_truth = [class_label_mapping[x] for x in ground_truth_numeric]
    predicted = [class_label_mapping[x] for x in predicted_numeric]

    # Create and display the confusion matrix.
    cm_df = create_confusion_matrix(ground_truth, predicted, string_labels)
    print("Confusion Matrix:")
    print(cm_df)

    # Print overall accuracy metrics and capture them.
    overall_accuracy, kappa = print_accuracy_metrics(ground_truth, predicted, string_labels)

    # Compute the classification report and convert it to a DataFrame.
    report_dict = classification_report(ground_truth, predicted, labels=string_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Plot and save the confusion matrix heatmap.
    heatmap_path = os.path.join(wks, 'validation', 'heatmap_confusion_matix.png')
    plot_confusion_heatmap(cm_df, heatmap_path)

    # Plot and save the classification report heatmap.
    classification_heatmap_path = os.path.join(wks, 'validation', 'heatmap_classification_report.png')
    plot_classification_report_heatmap(report_df, string_labels, classification_heatmap_path)

    # Define the path for the Excel report.
    excel_file_path = os.path.join(os.path.dirname(heatmap_path), 'accuracy_assesement_summary.xlsx')
    export_accuracy_summary(samples_df, cm_df, report_df, excel_file_path, overall_accuracy, kappa)
    
    

    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')
