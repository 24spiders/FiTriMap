# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:04:07 2025

@author: Labadmin
"""

import os
import numpy as np
import rasterio
from tqdm import tqdm


def get_dataset_stats(dataset_dir):
    variables = ['RSI', 'elevation', 'slope', 'aspect']
    stats = {}

    # Initialize stats dictionary for each variable
    for var in variables:
        stats[var] = {
            'total_min': float('inf'),
            'total_max': float('-inf'),
            'total_sum': 0,
            'total_sq_sum': 0,
            'total_count': 0
        }

    # Iterate through each fire folder
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Getting normalization values')
    for fire_id in os.listdir(dataset_dir):
        fid = '_'.join(fire_id.split('_')[:2])  # Standardized fire ID
        fire_dir = os.path.join(dataset_dir, fire_id)

        if os.path.isdir(fire_dir):
            for var in variables:
                var_path = os.path.join(fire_dir, f'{fid}_{var}.tif')

                if os.path.exists(var_path):
                    with rasterio.open(var_path) as src:
                        data = src.read(1)  # Read the first band

                        # Mask out invalid values
                        valid_data = data[(data != 0) & ~np.isnan(data) & (data > -999)]

                        if valid_data.size > 0:
                            # Update min and max
                            mn = np.min(valid_data)
                            mx = np.max(valid_data)
                            stats[var]['total_min'] = min(stats[var]['total_min'], mn)
                            stats[var]['total_max'] = max(stats[var]['total_max'], mx)

                            # Compute sum and squared sum incrementally
                            stats[var]['total_sum'] += np.sum(valid_data)
                            stats[var]['total_sq_sum'] += np.sum(valid_data ** 2)
                            stats[var]['total_count'] += valid_data.size
        pbar.update(1)

    pbar.close()
    # Compute mean and standard deviation for each variable
    for var in variables:
        count = stats[var]['total_count']
        if count > 0:
            mean = stats[var]['total_sum'] / count
            stddev = np.sqrt((stats[var]['total_sq_sum'] / count) - (mean ** 2))
        else:
            mean, stddev = None, None

        stats[var]['total_mean'] = mean
        stats[var]['total_stddev'] = stddev

        # Remove intermediate sum values to clean up output
        del stats[var]['total_sum']
        del stats[var]['total_sq_sum']
        del stats[var]['total_count']

    print(stats)
    return stats


def normalize_dataset(dataset_dir, variable_stats, method='minmax'):
    assert method in ['minmax', 'z-score']
    variables = ['RSI', 'elevation', 'slope', 'aspect']

    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc=f'Normalizing dataset using {method}')

    for fire_id in os.listdir(dataset_dir):
        fid = '_'.join(fire_id.split('_')[:2])  # Standardized fire ID
        fire_dir = os.path.join(dataset_dir, fire_id)

        if os.path.isdir(fire_dir):
            for var in variables:
                var_path = os.path.join(fire_dir, f'{fid}_{var}.tif')

                if os.path.exists(var_path):
                    with rasterio.open(var_path) as src:
                        data = src.read(1)
                        profile = src.profile  # Preserve metadata

                        valid_mask = (data != 0) & ~np.isnan(data)

                        if method == 'minmax':
                            min_val = variable_stats[var]['total_min']
                            max_val = variable_stats[var]['total_max']
                            data[valid_mask] = (data[valid_mask] - min_val) / (max_val - min_val)
                        elif method == 'z-score':
                            mean_val = variable_stats[var]['total_mean']
                            stddev_val = variable_stats[var]['total_stddev']
                            data[valid_mask] = (data[valid_mask] - mean_val) / stddev_val

                    with rasterio.open(var_path, 'w', **profile) as dst:
                        dst.write(data, 1)

        pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    stats = get_dataset_stats(r'D:/!Research/01 - Python/FiTriMap/ignore_data/ABoVE 256')
