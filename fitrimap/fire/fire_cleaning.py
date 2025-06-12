# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:22:45 2025

@author: Liam
"""
import os
import numpy as np
import rasterio
from tqdm import tqdm
from scipy.ndimage import label, binary_dilation, generate_binary_structure


def remove_spot_fires(dataset_dir):
    """
    Removes disconnected burn regions (spot fires) from daily wildfire burn rasters.
    For each day-of-year (DOY), removes regions not connected to previous DOY.

    Args:
        dataset_dir (str): Path to directory containing burn rasters.

    Returns:
        None
    """
    # Define 8-connectivity (diagonal connections allowed)
    structure = generate_binary_structure(2, 2)
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Removing spot fires')

    for folder in os.listdir(dataset_dir):
        fid = folder.replace('_CNFDB', '').replace('_ABoVE', '')
        folder_path = os.path.join(dataset_dir, folder)
        fire_tif = os.path.join(folder_path, f'{fid}_burn.tif')

        if not os.path.exists(fire_tif):
            pbar.update(1)
            continue

        # Open raster and read data
        with rasterio.open(fire_tif) as src:
            data = src.read(1)
            profile = src.profile

        # Ensure we're working with a copy to avoid modifying original
        out_data = np.copy(data)

        # Extract unique DOYs, excluding background
        unique_doys = np.unique(out_data[out_data > 0])
        unique_doys = unique_doys.astype(int)  # Ensure integer DOY values

        for doy in unique_doys:
            # Binary mask for current DOY
            doy_mask = (out_data == doy).astype(np.uint8)

            # Label all regions for current DOY
            labeled_array, num_features = label(doy_mask, structure=structure)

            # Prepare previous DOY mask
            prev_doy = doy - 1
            if prev_doy in unique_doys:
                prev_mask = (out_data == prev_doy).astype(np.uint8)
                prev_mask_dilated = binary_dilation(prev_mask, structure=structure)

                # Iterate through labeled regions
                for i in range(1, num_features + 1):
                    region_mask = (labeled_array == i)
                    # If not connected to previous DOY, remove region
                    if not np.any(prev_mask_dilated[region_mask]):
                        out_data[region_mask] = 0

        # Write modified raster to disk
        no_spot_path = fire_tif.replace('_burn.tif', '_burn_nospot.tif')
        profile.update(dtype=out_data.dtype)
        with rasterio.open(no_spot_path, 'w', **profile) as dst:
            dst.write(out_data, 1)

        pbar.update(1)

    pbar.close()


def remove_unburnable(data_dir):
    # Iterate over all fire directories
    for fire_id in os.listdir(data_dir):
        fire_path = os.path.join(data_dir, fire_id)
        fid = '_'.join(fire_id.split('_')[:2])

        # Check if it's a directory
        if os.path.isdir(fire_path):
            # Define file paths for fuelmap and burn file
            dem_fuel_map_path = os.path.join(fire_path, f'{fid}_fuelmap.tif')
            burn_path = os.path.join(fire_path, f'{fid}_burn.tif')

            # Ensure both files exist
            if os.path.exists(dem_fuel_map_path) and os.path.exists(burn_path):
                # Open both files
                with rasterio.open(dem_fuel_map_path) as fuelmap, rasterio.open(burn_path, 'r+') as burn_file:
                    fuel_data = fuelmap.read(1)

                    # Read krig_file as a mutable array
                    burn_data = burn_file.read(1)  # Reads the first band

                    # Define conditions for setting burn values to 0
                    mask = (fuel_data == 101) | (fuel_data == 102)

                    # Count the number of pixels that will be changed
                    pixels_changed = np.sum(mask)

                    # Apply the mask to set values in burn_data to 0
                    print(fire_id)
                    burn_data[mask] = 0

                    # Write the updated krig_data back to burn+file
                    burn_file.write(burn_data, 1)

                    # Print the number of changed pixels
                    print(f'Fire ID {fire_id}: {pixels_changed} pixels changed.')


if __name__ == '__main__':
    dataset_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\CNFDB 256 100m NEW'
    remove_spot_fires(dataset_dir, min_px=6)
