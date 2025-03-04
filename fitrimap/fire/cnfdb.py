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


def above_extent_quantiles(dataset_dir, quantiles=[0.25, 0.5, 0.75]):
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
    # TODO: Rename folders with _cnfdb ?


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
    max_width = 0
    max_height = 0

    # Iterate over all raster files to find the largest pixel size
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Finding largest pixel size')
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        fire_tif = os.path.join(folder_path, f'{folder}_burn.tif')

        if os.path.exists(fire_tif):
            with rasterio.open(fire_tif) as src:
                height, width = src.height, src.width
                # Update the max width and height
                max_width = max(max_width, width)
                max_height = max(max_height, height)
            pbar.update(1)
    pbar.close()
    if max_width > target_width or max_height > target_height:
        raise ValueError('Max dims greater than target dims!')

    # Reproject and pad every raster to be the same size with centered image
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc=f'Reprojecting to {target_crs} and padding to {target_shape}')
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        fire_tif = os.path.join(folder_path, f'{folder}_burn.tif')

        if os.path.exists(fire_tif):
            # Create a temporary file for the reprojected data
            temp_tif = os.path.join(folder_path, f'{folder}_temp.tif')

            with rasterio.open(fire_tif) as src:
                source_crs = src.crs

                # Skip reprojection if already in target CRS
                if source_crs == target_crs:
                    # Copy the file to temp for consistent processing
                    shutil.copy(fire_tif, temp_tif)
                else:
                    # Reproject to target CRS
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        source_crs, target_crs, src.width, src.height,
                        *src.bounds
                    )

                    dst_kwargs = src.meta.copy()
                    dst_kwargs.update({
                        'crs': target_crs,
                        'transform': dst_transform,
                        'width': dst_width,
                        'height': dst_height
                    })

                    with rasterio.open(temp_tif, 'w', **dst_kwargs) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=source_crs,
                                dst_transform=dst_transform,
                                dst_crs=target_crs,
                                resampling=Resampling.nearest
                            )

            # Now open the reprojected file and pad it
            with rasterio.open(temp_tif) as src:
                # Calculate padding amounts
                height, width = src.height, src.width
                pad_width_total = target_width - width
                pad_height_total = target_height - height

                # Calculate padding on each side to center the image
                pad_left = pad_width_total // 2
                pad_right = pad_width_total - pad_left
                pad_top = pad_height_total // 2
                pad_bottom = pad_height_total - pad_top

                # Read data and metadata
                data = src.read(1)
                meta = src.meta.copy()

                # Create padded array with centered image
                padded_data = np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right)),
                                     mode='constant', constant_values=np.nan)

                # Update metadata for new dimensions
                meta.update({"height": target_height, "width": target_width})

                # Update the transform to account for the padding
                # Shift the transform to maintain the same coordinate reference
                transform = src.transform
                xoff = -pad_left * transform.a
                yoff = -pad_top * transform.e
                new_transform = rasterio.Affine(transform.a, transform.b, transform.c + xoff,
                                                transform.d, transform.e, transform.f + yoff)
                meta.update({"transform": new_transform})

                # Write final padded and reprojected file
                with rasterio.open(fire_tif, 'w', **meta) as dst:
                    dst.write(padded_data, 1)

            # Clean up the temporary file
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

            pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    os.chdir(r'D:\!Research\01 - Python\FiTriMap\ignore_data')
    dataset_dir = 'CNFDB 256'
    zip_dir = r'D:\!Research\01 - Python\Piyush\CNN Fire Prediction\Piyush Fire Dataset\Fire growth rasters'
    bad_fires = ['2002_375', '2002_389', '2002_640', '2003_64', '2003_362', '2003_393', '2003_412', '2003_586', '2003_602', '2003_633', '2004_546', '2005_2', '2005_7', '2006_366',
                 '2006_671', '2007_96', '2009_339', '2009_397', '2011_317', '2012_248', '2012_250', '2012_545', '2012_745', '2012_851', '2013_288', '2013_567', '2013_805', '2015_155',
                 '2015_1177', '2015_1693', '2016_174', '2017_1860', '2018_494', '2020_359', '2020_343']
    get_cnfdb(dataset_dir, zip_dir, bad_fires)

    # quants = above_extent_quantiles(dataset_dir, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    quants = {'Q10': 6660.0, 'Q25': 8640.0, 'Q50': 12600.0, 'Q75': 19980.0, 'Q90': 32040.0}
    size_dict = {'min': quants['Q10'],
                 'max': quants['Q90']}
    resize_cnfdb(dataset_dir, target_shape=(256, 256), size_dict=size_dict)
