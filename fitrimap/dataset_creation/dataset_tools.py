# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:18:38 2025

@author: Labadmin
"""
# Defaults
import datetime
import os
import shutil
from collections import Counter

# Spatial
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box, Polygon
import rasterio
from rasterio.features import shapes

# Other
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


def find_matching_fires(priority_fires, proposed_fire_dir):
    matched_fires = []

    # Process proposed fire
    _, prop_fire_id = os.path.split(proposed_fire_dir)
    prop_fire_year = prop_fire_id.split('_')[0]
    prop_fire_tif = os.path.join(proposed_fire_dir, f'{prop_fire_id}_burn.tif')
    with rasterio.open(prop_fire_tif) as prop_src:
        prop_transform = prop_src.transform
        prop_data = prop_src.read(1)
        prop_doys = prop_data[(prop_data != 0) & (~np.isnan(prop_data))]
        prop_doys = np.unique(prop_doys)
    prop_min_day = min(prop_doys) - 3
    prop_max_day = max(prop_doys) + 3
    prop_mask = (prop_data != 0) & ~np.isnan(prop_data)
    prop_polygons = [Polygon(geom['coordinates'][0]) for geom, value in shapes(prop_data, mask=prop_mask, transform=prop_transform)]

    # Check all priority fires
    for priority_fire_dir in priority_fires:
        if os.path.isdir(priority_fire_dir):
            _, priority_fire_id = os.path.split(priority_fire_dir)
            priority_fire_year = priority_fire_id.split('_')[0]

            # If fires occured in different years, they do not match
            if priority_fire_year != prop_fire_year:
                continue

            priority_fire_tif = os.path.join(priority_fire_dir, f'{priority_fire_id}_burn.tif')
            with rasterio.open(priority_fire_tif) as priority_src:
                priority_transform = priority_src.transform
                priority_data = priority_src.read(1)
                priority_doys = priority_data[(priority_data != 0) & (~np.isnan(priority_data))]
                priority_doys = np.unique(priority_doys)

            priority_min_day = min(priority_doys) - 3
            priority_max_day = max(priority_doys) + 3

            # If fires have very different DoY's, they do not match
            if not (priority_max_day >= prop_min_day and prop_max_day >= priority_min_day):  # Intersection of DOY values
                continue
            # Finally, if fires have occurred around the same date, check for spatial overlap
            priority_mask = (priority_data != 0) & ~np.isnan(priority_data)
            priority_polygons = [Polygon(geom['coordinates'][0]) for geom, value in shapes(priority_data, mask=priority_mask, transform=priority_transform)]

            for priority_poly in priority_polygons:
                for prop_poly in prop_polygons:
                    if priority_poly.intersects(prop_poly):
                        matched_fires.append(priority_fire_dir)

    return matched_fires


def combine_ABoVE_CNFDB(above_dir, cnfdb_dir, hybrid_dir, priority='ABoVE'):
    assert priority in ['ABoVE', 'CNFDB']
    os.makedirs(hybrid_dir, exist_ok=True)
    above_dirs = [os.path.join(above_dir, folder) for folder in os.listdir(above_dir)]
    cnfdb_dirs = [os.path.join(cnfdb_dir, folder) for folder in os.listdir(cnfdb_dir)]

    # Set dirs
    if priority == 'ABoVE':
        priority_dirs = above_dirs
        suff = '_ABoVE'
        other_dirs = cnfdb_dirs
        other_suff = '_CNFDB'

    elif priority == 'CNFDB':
        priority_dirs = cnfdb_dirs
        suff = '_CNFDB'
        other_dirs = above_dirs
        other_suff = '_ABoVE'

    # Copy all priority fires
    pbar = tqdm(total=len(priority_dirs), desc=f'Copying priority ({priority}) fires')
    for fire_dir in priority_dirs:
        if os.path.isdir(fire_dir):
            _, fire_id = os.path.split(fire_dir)
            dst = os.path.join(hybrid_dir, fire_id + suff)
            if not os.path.exists(dst):
                shutil.copytree(fire_dir, dst)
        pbar.update(1)
    pbar.close()

    # Check other fires, only copy if they do not match an existing priority fire
    pbar = tqdm(total=len(other_dirs), desc='Checking other fires')
    for fire_dir in other_dirs:
        if os.path.isdir(fire_dir):
            _, fire_id = os.path.split(fire_dir)
            dst = os.path.join(hybrid_dir, fire_id + other_suff)

            # Check for matching fires
            matched_fires = find_matching_fires(priority_dirs, fire_dir)

            if not matched_fires:
                shutil.copytree(fire_dir, dst)
        pbar.update(1)
    pbar.close()


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


def plot_date_distribution(dataset_dir):
    fire_counts = Counter()

    # Progress bar setup
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Plotting dataset histograms')

    for fire_id in os.listdir(dataset_dir):
        fire_path = os.path.join(dataset_dir, fire_id)
        if not os.path.isdir(fire_path):
            continue

        fid = '_'.join(fire_id.split('_')[:2])
        year = int(fid[:4])  # Extract the year from fire_id
        burn_tif = os.path.join(fire_path, f'{fid}_burn.tif')

        # Read the burn date raster
        with rasterio.open(burn_tif) as src:
            burn_data = src.read(1)
            burn_doys = burn_data[(burn_data != 0) & (~np.isnan(burn_data))]
            burn_doys = np.unique(burn_doys)

        if len(burn_doys) == 0:
            continue

        min_day = min(burn_doys)  # Earliest burn day-of-year

        # Convert day-of-year to month
        month = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(min_day) - 1)
        fire_counts[(year, month.month)] += 1  # Store (year, month) count

        pbar.update(1)

    pbar.close()

    # Prepare data for plotting
    years, months = zip(*fire_counts.keys())
    counts = list(fire_counts.values())

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts, tick_label=[f'{y}-{m:02d}' for y, m in zip(years, months)], color='royalblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Fires')
    plt.title('Fire Occurrences by Month and Year')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # actioned_csv = r'G:\Shared drives\UofA Wildfire\Project\05 - GIS\NFDB\NFDB Point\actioned_fires.csv'
    # dataset_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\ABoVE 128'
    # validate_dataset(dataset_dir, raise_error=True)

    above_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\ABoVE 256 100m'
    cnfdb_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\CNFDB 256 100m'
    hybrid_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\ABoVE Priority Hybrid 256 100m'
    combine_ABoVE_CNFDB(above_dir, cnfdb_dir, hybrid_dir, priority='ABoVE')
