"""
Create a training raster by combining existing land cover rasters:
    ESA WorldCover, ESRI Lnad Cover, Natural Resource Canada Land Cover, 
    DLR World Settelement Footprint (WSF).
    
    The rasters were resampled and aligned beforehand using the ESA raster as refrenece.
    
    The rasters are processed in chunks due to their large size (memory allocation issues).
    
Author: Moez Labiadh
"""

import os
import logging
import pandas as pd
import numpy as np
import rasterio
import timeit
from typing import Dict, Union

class TrainingRasterCreator:
    def __init__(
        self, 
        raster_paths: Dict[str, str], 
        output_path: str, 
        classification_rules: pd.DataFrame,
        chunk_size: int = 1024,
        logger: Union[logging.Logger, None] = None
    ):
        """
        Initialize the training raster creator.

        Parameters:
        -----------
        raster_paths : Dict[str, str]
            Dictionary of input raster paths
        output_path : str
            Path to save the output raster
        classification_rules : pd.DataFrame
            DataFrame containing classification rules
        chunk_size : int, optional
            Size of chunks for processing (default: 1024)
        logger : logging.Logger, optional
            Custom logger (default: None)
        """
        self.raster_paths = raster_paths
        self.output_path = output_path
        self.classification_rules = classification_rules
        self.chunk_size = chunk_size

        # Setup logging
        self.logger = logger or self._setup_logger()

        # Validate inputs
        self._validate_inputs()

    def _setup_logger(self) -> logging.Logger:
        """
        Setup default logger.

        Returns:
        --------
        logging.Logger
            Configured logger instance
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        return logging.getLogger(__name__)

    def _validate_inputs(self):
        """
        Validate input raster paths and classification rules.

        Raises:
        -------
        FileNotFoundError
            If any input raster is not found
        ValueError
            If classification rules are invalid
        """
        # Check raster paths
        for source, path in self.raster_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input raster not found: {path}")

        # Validate classification rules
        required_columns = ['class_id', 'esa_value', 'nrcan_value', 'esri_value', 'wfs_value']
        if not all(col in self.classification_rules.columns for col in required_columns):
            raise ValueError("Classification rules missing required columns")

    def _validate_raster_alignment(self, rasters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Check and align input rasters.

        Parameters:
        -----------
        rasters : Dict[str, np.ndarray]
            Dictionary of input raster arrays

        Returns:
        --------
        Dict[str, np.ndarray]
            Aligned (trimmed) raster arrays
        """
        # Get shapes of all rasters
        shapes = [arr.shape for arr in rasters.values()]
        
        # Find minimum dimensions
        min_height = min(shape[0] for shape in shapes)
        min_width = min(shape[1] for shape in shapes)
        
        # If shapes are different, log and trim
        if len(set(shapes)) > 1:
            self.logger.warning(f"Misaligned raster sizes detected. Trimming to {min_height}x{min_width}")
            
            # Trim all rasters to the smallest dimensions
            aligned_rasters = {
                source: arr[:min_height, :min_width] 
                for source, arr in rasters.items()
            }
            return aligned_rasters
        
        # If all rasters are already aligned, return as-is
        return rasters

    def create_training_raster(self):
        """
        Create training raster by processing input rasters in chunks.
        """
        # Open reference raster to get metadata
        with rasterio.open(self.raster_paths['esa']) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width

        # Update profile for output
        profile.update(
            dtype=rasterio.uint8, 
            count=1, 
            nodata=0, 
            compress='lzw'
        )

        with rasterio.open(self.output_path, 'w', **profile) as dst:
            # Process in chunks
            for row_start in range(0, height, self.chunk_size):
                for col_start in range(0, width, self.chunk_size):
                    # Calculate chunk dimensions
                    chunk_height = min(self.chunk_size, height - row_start)
                    chunk_width = min(self.chunk_size, width - col_start)

                    
                    # Define window
                    window = rasterio.windows.Window(
                        col_start, row_start,
                        chunk_width,
                        chunk_height
                    )
                    
                    self.logger.info(f"Processing chunk: {window}")
                    # Read chunk data
                    raster_chunks = {
                        source: rasterio.open(path).read(1, window=window)
                        for source, path in self.raster_paths.items()
                    }

                    # Validate and align raster chunks
                    raster_chunks = self._validate_raster_alignment(raster_chunks)

                    # Initialize output chunk
                    output_chunk = np.zeros_like(raster_chunks['esa'], dtype=np.uint8)

                    # Apply classification rules
                    for _, rule in self.classification_rules.iterrows():
                        # Create condition mask
                        condition = self._create_condition_mask(raster_chunks, rule)
                        
                        # Assign class ID
                        output_chunk[condition] = np.uint8(rule['class_id'])

                    # Write chunk
                    dst.write(output_chunk, 1, window=window)

                    

    def _create_condition_mask(
        self, 
        raster_chunks: Dict[str, np.ndarray], 
        rule: pd.Series
    ) -> np.ndarray:
        """
        Create a condition mask for a specific classification rule.

        Parameters:
        -----------
        raster_chunks : Dict[str, np.ndarray]
            Dictionary of input raster chunks
        rule : pd.Series
            Single row from classification rules

        Returns:
        --------
        np.ndarray
            Boolean mask for the classification rule
        """
        # Sources mapping
        sources = {
            'esa': 'esa_value',
            'nrcan': 'nrcan_value',
            'esri': 'esri_value',
            'wfs': 'wfs_value'
        }

        # Initialize condition as all True
        condition = np.ones_like(raster_chunks['esa'], dtype=bool)

        # Apply each non-NaN condition
        for source, column in sources.items():
            if not np.isnan(rule[column]):
                condition &= (raster_chunks[source] == rule[column])

        return condition
    
if __name__ == '__main__':
    start_t = timeit.default_timer()
    
    # Setup paths
    wks = r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    xlsx_path = os.path.join(wks, 'documents', 'classification_schema.xlsx')
    values_df = pd.read_excel(xlsx_path, 'training_pixel_values')

    input_rasters_path = os.path.join(wks, 'data', 'training_data', 'input_rasters')
    raster_paths = {
        "esa": os.path.join(wks, 'data', 'existing_data', 'esa', 'esa_lc_10m_mosaic_bc.tif'),
        "nrcan": os.path.join(input_rasters_path, 'aligned_nrcan_lc_10m_bc.tif'),
        "esri": os.path.join(input_rasters_path, 'aligned_esri_lc_10m_mosaic_bc.tif'),
        "wfs": os.path.join(input_rasters_path, 'aligned_wfs_10m_mosaic_bc_4.tif'),
    }
    
    output_path = os.path.join(wks, 'data', 'training_data', 'training_raster_v6.tif')
    
    # Create and run training raster creator
    creator = TrainingRasterCreator(
        raster_paths, 
        output_path, 
        values_df, 
        chunk_size=20480
    )
    creator.create_training_raster()

    # Timing end
    finish_t = timeit.default_timer()
    t_sec = round(finish_t - start_t)
    mins, secs = divmod(t_sec, 60)
    print(f'\nProcessing Completed in {mins} minutes and {secs} seconds')  