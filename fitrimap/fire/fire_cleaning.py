# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:22:45 2025

@author: Liam
"""
import os
import numpy as np
import rasterio
from tqdm import tqdm
from scipy.ndimage import label, binary_dilation

def remove_spot_fires(dataset_dir, min_px=10):
    """
    Removes small isolated spot fires from burn area rasters and removes areas that do not touch the previous day's burn.
    Args:
        dataset_dir (str): Path to the dataset directory containing fire rasters.
        min_px (int): Minimum number of pixels for a connected region to be retained. Defaults to 10.
    Returns:
        None: Modifies the raster files in place, removing small isolated fires.
    """
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Removing spot fires')
    
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        fire_tif = os.path.join(folder_path, f'{folder}_burn.tif')
        
        if os.path.exists(fire_tif):
            # Read the tif
            with rasterio.open(fire_tif) as src:
                data = src.read(1)  # Read as a single band array
                profile = src.profile  # Get metadata
            
            # Get unique burn day-of-year (DOY) values excluding background (0)
            unique_doys = np.unique(data[(data != 0) & (~np.isnan(data))])
            
            # Process each DOY separately
            for doy in unique_doys:
                # Remove areas below min_px:
                # Create a binary mask for the current DOY
                doy_mask = (data == doy).astype(np.uint8)
                # Label connected components
                labeled_array, num_features = label(doy_mask)
                
                # Iterate over detected components
                for i in range(1, num_features + 1):
                    if np.sum(labeled_array == i) < min_px:
                        # Remove small fire patches
                        data[labeled_array == i] = 0
                
                # Remove areas that do not touch doy - 1
                prev_doy = doy - 1
                if prev_doy in unique_doys:
                    prev_mask = (data == prev_doy).astype(np.uint8)
                    prev_mask_dilated = binary_dilation(prev_mask)  # Expand prev_doy mask to check adjacency
                    labeled_array, num_features = label(doy_mask)
                    
                    # Identify regions that do not touch prev_doy
                    for i in range(1, num_features + 1):
                        region_mask = (labeled_array == i)
                        if not np.any(prev_mask_dilated[region_mask]):
                            data[region_mask] = 0
            
            # Save the modified raster
            no_spot = fire_tif.replace('_burn.tif', '_burn_nospot.tif')
            with rasterio.open(no_spot, 'w', **profile) as dst:
                dst.write(data, 1)
        
        pbar.update(1)
    
    pbar.close()
