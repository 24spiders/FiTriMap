# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 09:39:38 2025

@author: Labadmin
"""
# TODO: Does this file need to exist?
import os
import rasterio
import numpy as np
from tqdm import tqdm

from fitrimap.fuels.recode_fuelmap import recode_fuelmap_RSI


def get_RSI(dataset_dir, fwi_nc_dir):
    """Creates RSI maps for each fuelmap in a dataset.

    Args:
        dataset_dir (str): Path to the dataset to process.
        fwi_nc_dir (str): Path to folder containing ISI netCDF files.
    """
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Getting RSIs')
    for fire_id in os.listdir(dataset_dir):
        fire_dir = os.path.join(dataset_dir, fire_id)
        if os.path.isdir(fire_dir):
            fid = '_'.join(fire_id.split('_')[:2])
            fire_raster = os.path.join(fire_id, f'{fid}_burn.tif')
            output_rsi = os.path.join(fire_id, 'Indices', f'{fid}_RSI.tif')
            if os.path.exists(output_rsi):
                continue
            fuelmap_path = os.path.join(fire_id, 'Cropped', f'{fire_id}_fuelmap.tif')
            year = int(fire_id[:4])
            with rasterio.open(fire_raster) as src:
                data = src.read()
                # Get the average value in data excluding 0 and nan
                mask = (data != 0) & (~np.isnan(data))

                # Apply mask to filter out zeros and NaNs
                filtered_data = data[mask]

                # Calculate the average value of the filtered data
                avg_value = np.median(filtered_data) if filtered_data.size > 0 else np.nan  # Handle case if no valid data

                if avg_value is np.nan:
                    continue

                # Get dates
                # TODO: Technically, we need a new fuelmap for each day
                doy = int(avg_value)

            recode_fuelmap_RSI(fuelmap_path, output_rsi, doy, year, fwi_nc4_dir)
        pbar.update(1)


if __name__ == '__main__':
    os.chdir(r'D:\!Research\01 - Python\Piyush\CNN Fire Prediction\Raw Hybrid 256')
    fwi_nc4_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\ISI'
