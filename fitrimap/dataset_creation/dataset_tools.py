# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:18:38 2025

@author: Labadmin
"""
import os
import shutil
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def find_actioned_fires(dataset_dir, actioned_csv, remove=False):
    # Load the actioned fires CSV into a GeoDataFrame
    actioned_fires = gpd.read_file(actioned_csv)
    actioned_fires['geometry'] = actioned_fires['wkt_geom_pt'].apply(wkt.loads)
    actioned_fires = gpd.GeoDataFrame(actioned_fires, geometry='geometry')

    # Ensure the GeoDataFrame is in the correct EPSG
    actioned_fires = actioned_fires.set_crs(epsg=3978)
    actioned_fires = actioned_fires.to_crs(epsg=3979)

    # Initialize counter for actioned fires
    actioned_count = 0
    unactioned_count = 0
    actioned_fire_ids = []
    # Iterate through each fire ID in the directory
    for fire_id in os.listdir(dataset_dir):
        # Extract the fire ID without the '_piyush' suffix
        fid = '_'.join(fire_id.split('_')[:2])
        actioned = False
        if os.path.isdir(os.path.join(dataset_dir, fire_id)):
            # Define the path to the TIF file
            tif_file = os.path.join(dataset_dir, fire_id, f'{fid}_burn.tif')

            # Extract the year from the fire ID
            year = fire_id.split('_')[0]

            # Filter actioned fires for the specific year
            actioned_fires_year = actioned_fires[actioned_fires['YEAR'] == year]

            if os.path.exists(tif_file):
                with rasterio.open(tif_file) as src:
                    tif_crs = src.crs
                    tif_bounds = src.bounds

                # Transform actioned fires to the TIF's coordinate system
                actioned_fires_transformed = actioned_fires_year.to_crs(tif_crs)
                # Create a mask from the TIF bounds
                tif_geometry = box(tif_bounds[0], tif_bounds[1], tif_bounds[2], tif_bounds[3])

                for geom in actioned_fires_transformed.geometry:
                    # Check for intersection
                    if geom.intersects(tif_geometry):
                        actioned_count += 1
                        actioned = True
                        actioned_fire_ids.append(fire_id)
                        if remove:
                            shutil.rmtree(os.path.join(dataset_dir, fire_id))
                        break
                if actioned is False:
                    unactioned_count += 1

    print(f'Total actioned fires overlapping: {actioned_count}')
    return actioned_fire_ids


def combine_ABoVE_CNFDB():
    pass


def replace_dataset_nans(dataset_dir):
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Replacing NaNs')

    # Iterate through each fire directory
    for fire_id in os.listdir(dataset_dir):
        fire_path = os.path.join(dataset_dir, fire_id)

        # Skip non-directory files
        if not os.path.isdir(fire_path):
            continue

        fid = '_'.join(fire_id.split('_')[:2])

        # Define raster file paths
        raster_files = {
            'burn': os.path.join(fire_path, f'{fid}_burn.tif'),
            'elevation': os.path.join(fire_path, f'{fid}_elevation.tif'),
            'aspect': os.path.join(fire_path, f'{fid}_aspect.tif'),
            'slope': os.path.join(fire_path, f'{fid}_slope.tif'),
            'RSI': os.path.join(fire_path, f'{fid}_RSI.tif'),
        }

        # Process each raster file
        for key, raster_file in raster_files.items():
            if os.path.exists(raster_file):
                with rasterio.open(raster_file, 'r+') as src:
                    # Read the raster data as an array
                    raster_data = src.read(1)

                    # Get the no data value from the raster metadata
                    no_data_value = src.nodata

                    # Replace NaN, 'no data', and -9999 values with 0
                    if no_data_value is not None:
                        raster_data[raster_data == no_data_value] = 0
                    raster_data = np.nan_to_num(raster_data, nan=0, posinf=0, neginf=0)

                    # Write the modified data back to the raster
                    src.write(raster_data, 1)
        pbar.update(1)

    pbar.close()


def validate_dataset(dataset_dir, raise_error=False):
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Validating dataset')
    for fire_id in os.listdir(dataset_dir):
        fire_path = os.path.join(dataset_dir, fire_id)
        if not os.path.isdir(fire_path):
            continue
        fid = '_'.join(fire_id.split('_')[:2])

        raster_files = {
            'burn': os.path.join(fire_path, f'{fid}_burn.tif'),
            'elevation': os.path.join(fire_path, f'{fid}_elevation.tif'),
            'aspect': os.path.join(fire_path, f'{fid}_aspect.tif'),
            'slope': os.path.join(fire_path, f'{fid}_slope.tif'),
            'RSI': os.path.join(fire_path, f'{fid}_RSI.tif'),
        }
        passing = True
        for key, path in raster_files.items():
            if not os.path.exists(path):
                if raise_error:
                    raise FileNotFoundError(f'ERROR: {path} does not exist!')
                print(f'WARNING: {path} does not exist!')
                passing = False
        pbar.update(1)

    pbar.close()
    return passing


def compute_raster_statistics(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read first band
        data = data[data != src.nodata]  # Exclude nodata values

        return {
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'std': np.std(data)
        }


def plot_dataset_histograms(dataset_dir):
    statistics = {'elevation': [], 'aspect': [], 'slope': [], 'RSI': []}
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Plotting dataset histograms')
    for fire_id in os.listdir(dataset_dir):
        fire_path = os.path.join(dataset_dir, fire_id)
        if not os.path.isdir(fire_path):
            continue

        fid = '_'.join(fire_id.split('_')[:2])

        raster_files = {
            'elevation': os.path.join(fire_path, f'{fid}_elevation.tif'),
            'aspect': os.path.join(fire_path, f'{fid}_aspect.tif'),
            'slope': os.path.join(fire_path, f'{fid}_slope.tif'),
            'RSI': os.path.join(fire_path, f'{fid}_RSI.tif'),
        }

        for key, path in raster_files.items():
            if os.path.exists(path):
                stats = compute_raster_statistics(path)
                statistics[key].append(stats['mean'])  # Collect mean values
        pbar.update(1)
    pbar.close()

    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, (key, values) in enumerate(statistics.items()):
        if values:
            axes[i].hist(values, bins=20, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Histogram of {key} Mean Values')
            axes[i].set_xlabel('Mean Value')
            axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    actioned_csv = r'G:\Shared drives\UofA Wildfire\Project\05 - GIS\NFDB\NFDB Point\actioned_fires.csv'
    dataset_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\ABoVE 128'
    validate_dataset(dataset_dir, raise_error=True)
