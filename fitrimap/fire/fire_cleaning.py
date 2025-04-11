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
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Removing spot fires')
    for folder in os.listdir(dataset_dir):
        fid = folder.replace('_CNFDB', '').replace('_ABoVE', '')
        folder_path = os.path.join(dataset_dir, folder)
        fire_tif = os.path.join(folder_path, f'{fid}_burn.tif')

        if os.path.exists(fire_tif):
            # Read the tif
            with rasterio.open(fire_tif) as src:
                data = src.read(1)  # Read as a single band array
                profile = src.profile  # Get metadata
            out_data = np.copy(data)
            # Get unique burn day-of-year (DOY) values excluding background (0)
            unique_doys = np.unique(data[(data != 0) & (~np.isnan(data))])

            # Process each DOY separately
            for doy in unique_doys:
                # Create a binary mask for the current DOY
                doy_mask = (data == doy).astype(np.uint8)

                # Remove areas that do not touch doy - 1
                prev_doy = doy - 1
                if prev_doy in unique_doys:
                    prev_mask = (data == prev_doy).astype(np.uint8)
                    prev_mask_dilated = binary_dilation(prev_mask)  # Expand prev_doy mask to check adjacency
                    labeled_array, num_features = label(doy_mask)

                    for i in range(1, num_features + 1):
                        region_mask = (labeled_array == i)
                        # Remove regions that do not touch prev_doy
                        if not np.any(prev_mask_dilated[region_mask]):
                            out_data[region_mask] = 0
                        # Remove areas below min_px
                        if np.sum(labeled_array == i) < min_px:
                            # Remove small fire patches
                            out_data[labeled_array == i] = 0

            # Save the modified raster
            no_spot = fire_tif.replace('_burn.tif', '_burn_nospot.tif')
            with rasterio.open(no_spot, 'w', **profile) as dst:
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
                    burn_data[mask] = 0

                    # Write the updated krig_data back to burn+file
                    burn_file.write(burn_data, 1)

                    # Print the number of changed pixels
                    print(f'Fire ID {fire_id}: {pixels_changed} pixels changed.')


if __name__ == '__main__':
    dataset_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\ABoVE Priority Hybrid 256 100m (old)'
    remove_spot_fires(dataset_dir, min_px=6)
