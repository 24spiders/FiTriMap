# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:16:09 2025

@author: Labadmin
"""
import os
import rasterio
import zipfile
from tqdm import tqdm
import shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np


def fire_area_quantiles(dataset_dir, quantiles=[0.25, 0.5, 0.75]):
    areas = []  # List to store the maximum dimensions (real-world area) of each fire raster

    # Iterate through folders in the dataset directory
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        # If the path is a directory
        if os.path.isdir(folder_path):
            fid = '_'.join(folder.split('_')[:2])
            fire_tif = os.path.join(folder_path, f'{fid}_burn.tif')

            # Check if the fire raster exists
            if os.path.exists(fire_tif):
                with rasterio.open(fire_tif) as src:
                    # Get pixel resolution (units per pixel)
                    res_x, res_y = abs(src.transform.a), abs(src.transform.e)
                    # Count the number of pixels not 0 or nan
                    data = src.read(1)
                    area_px = np.sum((data > 0) & (~np.isnan(data)))
                    area = area_px * res_x * res_y

                    # Convert from px area to map unit area
                    areas.append(area)

    # Compute quantiles of the maximum dimensions
    quantile_values = np.percentile(areas, np.array(quantiles) * 100)

    # Format the quantiles as 'Q10', 'Q50', 'Q75', etc.
    quantile_dict = {f'Q{int(q * 100)}': value for q, value in zip(quantiles, quantile_values)}

    return quantile_dict


def fire_extent_quantiles(dataset_dir, quantiles=[0.25, 0.5, 0.75]):
    max_dims = []  # List to store the maximum dimensions (real-world area) of each fire raster

    # Iterate through folders in the dataset directory
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        # If the path is a directory
        if os.path.isdir(folder_path):
            fire_tif = os.path.join(folder_path, f'{folder}_burn.tif')

            # Check if the fire raster exists
            if os.path.exists(fire_tif):
                with rasterio.open(fire_tif) as src:
                    # Get pixel resolution (units per pixel)
                    res_x, res_y = abs(src.transform.a), abs(src.transform.e)

                    # Compute real-world area in map units (dimensions in map units)
                    x_dim = src.width * res_x
                    y_dim = src.height * res_y

                    # Get the maximum dimension (max area in map units)
                    max_dim = max(abs(x_dim), abs(y_dim))
                    max_dims.append(max_dim)

    # Compute quantiles of the maximum dimensions
    quantile_values = np.percentile(max_dims, np.array(quantiles) * 100)

    # Format the quantiles as 'Q10', 'Q50', 'Q75', etc.
    quantile_dict = {f'Q{int(q * 100)}': value for q, value in zip(quantiles, quantile_values)}

    return quantile_dict


def get_cnfdb(dataset_dir, zip_dir, bad_fires=[]):
    # Ensure the dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Initialize progress bar for ZIP extraction
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]
    pbar = tqdm(total=len(zip_files), desc='Unzipping CNFDB')

    for zip_file in zip_files:
        zip_path = os.path.join(zip_dir, zip_file)

        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            # Extract files that are not in bad_fires
            for file_name in file_list:
                fire_id = os.path.basename(file_name).split('_')[0]  # Extract fire ID
                if fire_id not in bad_fires:
                    zip_ref.extract(file_name, dataset_dir)

        pbar.update(1)
    pbar.close()

    # Rename for consistency
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            fire_tif = os.path.join(folder_path, f'{folder}_krig.tif')
            if os.path.exists(fire_tif):
                # Rename the file if within size constraints
                new_tif_path = os.path.join(folder_path, f'{folder}_burn.tif')
                os.rename(fire_tif, new_tif_path)


def remove_cnfdb_by_size(dataset_dir, size_dict):
    # Filter extracted fires based on real-world area
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        if os.path.isdir(folder_path):
            fire_tif = os.path.join(folder_path, f'{folder}_burn.tif')

            if os.path.exists(fire_tif):
                with rasterio.open(fire_tif) as src:
                    # Get pixel resolution (units per pixel)
                    res_x, res_y = abs(src.transform.a), abs(src.transform.e)

                    # Compute real-world area in map units
                    x_dim = src.width * res_x
                    y_dim = src.height * res_y
                    max_dim = max(abs(x_dim), abs(y_dim))

                # If area is out of range, delete the folder
                if max_dim < size_dict.get('min', 0) or max_dim > size_dict.get('max', float('inf')):
                    shutil.rmtree(folder_path)  # Remove entire fire folder


def resize_cnfdb(dataset_dir, target_shape, size_dict={}, target_crs='EPSG:3979'):
    if size_dict:
        remove_cnfdb_by_size(dataset_dir, size_dict)

    target_height, target_width = target_shape
    max_width, max_height = 0, 0
    largest_fire_tif = None

    # 1. Find max dimensions of rasters
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Finding largest size')
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        fire_tif = os.path.join(folder_path, f'{folder}_burn.tif')

        if os.path.exists(fire_tif):
            with rasterio.open(fire_tif) as src:
                if src.width > max_width:
                    max_width = src.width
                if src.height > max_height:
                    max_height = src.height
                    largest_fire_tif = fire_tif
        pbar.update(1)
    pbar.close()

    # 2. Compute scaling factors and new resolution
    scale_x = max_width / target_width
    scale_y = max_height / target_height
    print(f'Largest fire tif: {largest_fire_tif}')
    with rasterio.open(largest_fire_tif) as src:
        new_res = (src.res[0] * scale_x, src.res[1] * scale_y)

    # 3. Reproject, resize, and pad rasters
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc=f'Reprojecting to {target_crs}, resizing to {new_res}, and padding to {target_shape}')
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        fire_tif = os.path.join(folder_path, f'{folder}_burn.tif')

        if os.path.exists(fire_tif):
            with rasterio.open(fire_tif) as src:
                raster_data = src.read(1)
                src_transform = src.transform
                src_crs = src.crs
                profile = src.profile.copy()
                # Get reprojected dimensions and transform
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds, resolution=new_res
                )

            # Calculate padding needed to center the data
            x_padding = max(0, target_width - width)
            y_padding = max(0, target_height - height)
            x_offset = x_padding // 2
            y_offset = y_padding // 2

            # Get the geographic bounds of the reprojected data
            west, south, east, north = rasterio.transform.array_bounds(height, width, transform)

            # Get pixel dimensions
            pixel_width = transform.a
            pixel_height = transform.e  # Note: This is typically negative

            # Create new transform that centers the data in both x and y
            new_west = west - (x_offset * pixel_width)

            # For y-axis, we need to adjust differently because pixel_height is negative
            # We want to move the north bound up (more positive) by y_offset pixels
            new_north = north - (y_offset * pixel_height)  # Since pixel_height is negative, this will increase north

            new_transform = rasterio.transform.Affine(
                pixel_width, 0.0, new_west,
                0.0, pixel_height, new_north
            )

            profile.update({
                'crs': target_crs,
                'transform': new_transform,  # Use the adjusted transform
                'width': target_width,
                'height': target_height,
                'dtype': 'int16',  # Ensure integer storage
                'compress': 'lzw',
                'nodata': -9999  # Set a valid nodata value
            })

            output_tif = os.path.join(folder_path, f'{folder}_burn.tif')

            with rasterio.open(output_tif, 'w', **profile) as dst:
                data = np.full((target_height, target_width), 0, dtype=np.int16)  # Initialize with zeros
                raster_data = np.nan_to_num(raster_data, nan=0)
                reproject(
                    source=raster_data,
                    destination=data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=new_transform,  # Use the adjusted transform
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  # Use nearest neighbor resampling
                )
                dst.write(data, 1)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    os.chdir(r'D:\!Research\01 - Python\FiTriMap\ignore_data')
    os.environ['PROJ_DATA'] = r'C:\Users\Labadmin\anaconda3\envs\weather\Lib\site-packages\pyproj\proj_dir\share\proj'
    dataset_dir = 'CNFDB 256 100m NEW'
    zip_dir = r'D:\!Research\01 - Python\Piyush\CNN Fire Prediction\Piyush Fire Dataset\Fire growth rasters'
    bad_fires = ['2002_375', '2002_389', '2002_640', '2003_64', '2003_362', '2003_393', '2003_412', '2003_586', '2003_602', '2003_633', '2004_546', '2005_2', '2005_7', '2006_366',
                 '2006_671', '2007_96', '2009_339', '2009_397', '2011_317', '2012_248', '2012_250', '2012_545', '2012_745', '2012_851', '2013_288', '2013_567', '2013_805', '2015_155',
                 '2015_1177', '2015_1693', '2016_174', '2017_1860', '2018_494', '2020_359', '2020_343']
    # get_cnfdb(dataset_dir, zip_dir, bad_fires)

    quants = fire_extent_quantiles(dataset_dir, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(quants)

    # quants = {'Q10': 6660.0, 'Q25': 8640.0, 'Q50': 12600.0, 'Q75': 19980.0, 'Q90': 32040.0}
    # size_dict = {'min': quants['Q10'],
    #              'max': quants['Q90'] - 6000}
    size_dict = None
    resize_cnfdb(dataset_dir, target_shape=(256, 256), size_dict=size_dict)
