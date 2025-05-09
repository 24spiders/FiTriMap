# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:33:44 2025

@author: Labadmin
"""
import os
import rasterio
import numpy as np
from pathlib import Path
from rasterio.enums import Resampling
import pickle
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import binary_dilation

from fitrimap.utils.geospatial_utils import crop_master_to_tif, resize_tif, reproject_to_nearest_utm, crop_raster_to_valid_data
from fitrimap.fuels.recode_fuelmap import recode_fuelmap_RSI
from fitrimap.topography.get_topo_indices import create_topo_indices
from fitrimap.fire.above_to_tif import ABoVE_shp_to_tif, load_ABoVE_shp
from fitrimap.dataset_creation.normalize import get_dataset_stats, normalize_dataset
from fitrimap.dataset_creation.dataset_tools import plot_dataset_histograms, validate_dataset, replace_dataset_nans
from fitrimap.fire.cnfdb import get_cnfdb, resize_cnfdb, reproject_cnfdb
from fitrimap.fire.fire_cleaning import remove_spot_fires, remove_unburnable


def get_data(dataset_dir,
             master_fuelmap_dir,
             master_dem_path,
             isi_nc4_dir):
    # Make the dataset directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Open the master dem (doing it here saves time)
    master_dem = rasterio.open(master_dem_path)

    # Find all relevant tif files
    fire_rasters = []  # List to store relative paths of matching files

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # Check if the file ends with '_burn.tif'
            if file.endswith('_burn.tif'):
                pth = os.path.join(root, file)
                fire_rasters.append(pth)  # Add to the list

    prev_year = None
    pbar = tqdm(total=len(fire_rasters), desc='Getting fuels, topography, and weather')
    for fire_raster in fire_rasters:
        # Get fire raster, init data paths
        h, t = os.path.split(fire_raster)
        fire_id = t.lower().replace('_burn.tif', '')
        fire_dir = os.path.join(dataset_dir, fire_id)
        fuelmap_path = os.path.join(fire_dir, f'{fire_id}_fuelmap.tif')
        topo_path = os.path.join(fire_dir, f'{fire_id}_elevation.tif')
        output_rsi = os.path.join(fire_dir, f'{fire_id}_RSI.tif')

        # Get year, open fuelmap raster
        year = int(fire_id[:4])
        if year != prev_year:
            if year == 2015 or year == 2020:
                master_fuelmap_path = os.path.join(master_fuelmap_dir, f'3979_FBP-{year - 1}-100m.tif')
            else:
                master_fuelmap_path = os.path.join(master_fuelmap_dir, f'3979_FBP-{year}-100m.tif')
            master_fuelmap = rasterio.open(master_fuelmap_path)

        # Check if data already exists; if so, skip
        if os.path.exists(fuelmap_path) and os.path.exists(topo_path) and os.path.exists(output_rsi):
            pbar.update(1)
            continue

        with rasterio.open(fire_raster) as src:
            data = src.read()
            # Get the average value in data excluding 0 and nan
            mask = (data != 0) & (~np.isnan(data))

            # Apply mask to filter out zeros and NaNs
            filtered_data = data[mask]

            # Calculate the average value of the filtered data
            avg_value = np.median(filtered_data) if filtered_data.size > 0 else np.nan  # Handle case if no valid data

            # Get dates
            # TODO: Technically, we need a new fuelmap for each day
            doy = int(avg_value)

        # Crop and recode fuel map
        crop_master_to_tif(master_fuelmap, fire_raster, fuelmap_path, buffer_distance=0)
        reproject_to_nearest_utm(fuelmap_path)
        crop_raster_to_valid_data(fuelmap_path)
        recode_fuelmap_RSI(fuelmap_path, output_rsi, doy, year, isi_nc4_dir)

        # Crop and calculate topography indices
        crop_master_to_tif(master_dem, fire_raster, topo_path, buffer_distance=0)
        reproject_to_nearest_utm(topo_path)
        crop_raster_to_valid_data(topo_path)
        create_topo_indices(topo_path, fire_id)

        # TODO: get weather
        pbar.update(1)
        prev_year = year
    pbar.close()


def resize_dataset(dataset_dir,
                   shape=(128, 128)):
    resized_imgs = []
    # Iterate through the dataset
    dataset_path = Path(dataset_dir)
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Resizing dataset')
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            for tif_file in folder.glob('*.tif'):
                if '_burn' in tif_file.name:
                    resampling_method = Resampling.mode
                else:
                    resampling_method = Resampling.nearest

                resized_tif = resize_tif(os.path.join(folder, tif_file.name),
                                         shape=shape,
                                         resampling_method=resampling_method)
                resized_imgs.append(resized_tif)
        pbar.update(1)

    pbar.close()
    return resized_imgs


def fire_stats(dataset_dir, output_csv):
    # Create an empty DataFrame to store all results
    df = pd.DataFrame(columns=['fire_id', 'burn_day', 'area', 'area_percentage', 'num_px'])

    # Iterate through each fire_id directory in the raw_dir
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Calculating fire growth stats')
    for fire_id in os.listdir(dataset_dir):
        fire_dir = os.path.join(dataset_dir, fire_id)
        if os.path.isdir(fire_dir):
            fid = '_'.join(fire_id.split('_')[:2])
            burn_file = os.path.join(fire_dir, f'{fid}_burn.tif')
            # Load the krig_file using rasterio
            with rasterio.open(burn_file) as src:
                raster_data = src.read(1)  # Read the first (and only) channel

                # Get unique values from the raster data
                vals = raster_data[(raster_data != 0) & (~np.isnan(raster_data))]
                vals = np.unique(vals)

                # Calculate the area for each unique value
                pixel_area = src.res[0] * src.res[1]  # Area of a single pixel in the same unit as the CRS
                results = []
                areas = [0]

                for val in vals:
                    # Number of pixels burning
                    num_px = np.sum(raster_data == val)

                    # Area burning
                    area = num_px * pixel_area

                    # New area burned as a percentage of previously burned area (can be > 1)
                    area_per = area / sum(areas)

                    # Handle div 0
                    if area_per > 1e99:
                        area_per = 1
                    areas.append(area)
                    results.append([fire_id, val, area, area_per, num_px])

                # Append the results to the DataFrame
                df = pd.concat([df, pd.DataFrame(results, columns=['fire_id', 'burn_day', 'area', 'area_percentage', 'num_px'])], ignore_index=True)
        pbar.update(1)
    pbar.close()
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


def make_csv(dataset_dir, stats_csv, output_csv, growth_thresh=0, val_percentage=0.1, test_percentage=0.1, subset=None):
    # Initialize vars
    df = pd.DataFrame(columns=['Split', 'Fire_ID', 'Burn Day', 'Previous Day (1)', 'Previous Day (2)'])
    jj = 0

    # Load statistics
    stats_df = pd.read_csv(stats_csv)

    if not subset:
        subset = os.listdir(dataset_dir)

    # Iterate through directories in the input directory
    pbar = tqdm(total=len(subset), desc='Making dataset csv')
    for fire_id in subset:
        if os.path.isdir(os.path.join(dataset_dir, fire_id)):
            fid = '_'.join(fire_id.split('_')[:2])

            # Open the raster file
            raster_path = os.path.join(dataset_dir, fire_id, f'{fid}_burn.tif')
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1)  # Read the first band

            # Get unique values from the raster data
            days_of_year = raster_data[(raster_data != 0) & (~np.isnan(raster_data))]
            days_of_year = np.unique(days_of_year)
            days_of_year = sorted(days_of_year)

            # Iterate through the days of year
            for i in range(2, len(days_of_year)):
                # Get the 'day of burn'
                burn_day = days_of_year[i]
                # Get the previous days
                prev_day_1 = days_of_year[i - 1]
                prev_day_2 = days_of_year[i - 2]

                # Check that there is continuous data over three days
                cond1 = prev_day_1 == (burn_day - 1)
                cond2 = prev_day_2 == (burn_day - 2)

                if cond1 and cond2:
                    fire_stats = stats_df[(stats_df['fire_id'] == fire_id) & (stats_df['burn_day'] == burn_day)]
                    per_value = fire_stats['area_percentage'].values[0]
                    num_px = fire_stats['num_px'].values[0]

                    # Check that the fire grows enough
                    if per_value < growth_thresh:
                        continue

                    # Check that is burns more than 25 pixels
                    if num_px < 25:
                        continue

                    # Check that burn_day is adjacent to the previous day
                    burn_day_mask = (raster_data == burn_day)
                    prev_day_mask = (raster_data == prev_day_1)
                    structure = np.ones((3, 3))
                    prev_day_dilated = binary_dilation(prev_day_mask, structure=structure)
                    adjacent_check = burn_day_mask & prev_day_dilated
                    if not adjacent_check.any():
                        continue

                    # Add to the output df
                    df.loc[jj] = ['train', fire_id, burn_day, prev_day_1, prev_day_2]
                    jj += 1
        pbar.update(1)
    pbar.close()

    # Get unique fire_ids, shuffle
    unique_fire_ids = df['Fire_ID'].unique()
    np.random.shuffle(unique_fire_ids)

    # Split fire_ids into train, val, and test
    num_val = int(len(unique_fire_ids) * val_percentage)
    num_test = int(len(unique_fire_ids) * test_percentage)
    val_fire_ids = unique_fire_ids[:num_val]
    test_fire_ids = unique_fire_ids[num_val:num_val + num_test]
    train_fire_ids = unique_fire_ids[num_val + num_test:]

    # Now ensure no duplicates
    test_fire_ids = list(set(test_fire_ids))

    # Split and print the final sets
    train_fire_ids = list(set(train_fire_ids) - set(test_fire_ids))  # Remove test fire_ids from train set
    val_fire_ids = list(set(val_fire_ids) - set(test_fire_ids))  # Remove test fire_ids from val set

    # Assign splits based on fire_id
    df.loc[df['Fire_ID'].isin(val_fire_ids), 'Split'] = 'val'
    df.loc[df['Fire_ID'].isin(test_fire_ids), 'Split'] = 'test'

    # Shuffle and save the dataframe to CSV
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(os.path.join(dataset_dir, output_csv), index=False)


def create_dataset(dataset_dir, processing_options):
    # TODO: Keep tidying inputs
    if processing_options['create_above_rasters']['include']:
        # Create ABoVE rasters
        above_shp_dir = processing_options['create_above_rasters']['above_shp_dir']
        shape = processing_options['create_above_rasters']['shape']
        pkl_dirs = processing_options['create_above_rasters']['pkl_dirs']
        size_dict = processing_options['create_above_rasters']['size_dict']
        save_pkl = processing_options['create_above_rasters']['save_pkl']
        fire_rasters = ABoVE_shp_to_tif(above_shp_dir,
                                        dataset_dir,
                                        shape=shape,
                                        pkl_dirs=pkl_dirs,
                                        size_dict=size_dict)
        with open(save_pkl, 'wb') as f:
            pickle.dump(fire_rasters, f)

    if processing_options['get_cnfdb_rasters']['include']:
        # Get CNFDB rasters
        zip_dir = processing_options['get_cnfdb_rasters']['zip_dir']
        bad_fires = processing_options['get_cnfdb_rasters']['bad_fires']
        size_dict = processing_options['get_cnfdb_rasters']['size_dict']
        target_shape = processing_options['get_cnfdb_rasters']['target_shape']
        get_cnfdb(dataset_dir, zip_dir, bad_fires)
        reproject_cnfdb(dataset_dir)
        resize_cnfdb(dataset_dir, target_shape=target_shape, size_dict=size_dict)

    if processing_options['get_data']['include']:
        # Get the data
        master_fuelmap_dir = processing_options['get_data']['master_fuelmap_dir']
        master_dem_path = processing_options['get_data']['master_dem_path']
        isi_nc4_dir = processing_options['get_data']['isi_nc4_dir']
        get_data(dataset_dir,
                 master_fuelmap_dir,
                 master_dem_path,
                 isi_nc4_dir)

    if processing_options['resize']['include']:
        # Resize the dataset
        shape = processing_options['resize']['shape']
        _ = resize_dataset(dataset_dir, shape=shape)

    if processing_options['sanitize']['include']:
        # Check all required images exist
        validate_dataset(dataset_dir, raise_error=True)

        # Replace NaNs with 0
        replace_dataset_nans(dataset_dir)

    if processing_options['cleaning']['include']:
        # Remove unburnable fuels
        remove_unburnable(dataset_dir)

        # Remove spot fires
        remove_spot_fires(dataset_dir, min_px=6)

    if processing_options['normalize']['include']:
        # Get dataset normalization values
        method = processing_options['normalize']['method']
        variable_stats = get_dataset_stats(dataset_dir)

        # Normalize the dataset
        normalize_dataset(dataset_dir, variable_stats, method=method)

    if processing_options['get_fire_stats']['include']:
        stats_csv = processing_options['get_fire_stats']['stats_csv']
        fire_stats(dataset_dir, stats_csv)

    if processing_options['make_csv']['include']:
        # Make the dataset csv
        stats_csv = processing_options['make_csv']['stats_csv']
        output_csv = processing_options['make_csv']['output_csv']
        growth_thresh = processing_options['make_csv']['growth_thresh']
        subset = processing_options['make_csv']['subset']
        make_csv(dataset_dir,
                 stats_csv=stats_csv,
                 output_csv=output_csv,
                 growth_thresh=growth_thresh,
                 val_percentage=0.1,
                 test_percentage=0.1,
                 subset=subset)

    if processing_options['plot']['include']:
        # Plot dataset hists
        print('Plotting...')
        plot_dataset_histograms(dataset_dir)

    print('\n Done! \n')


if __name__ == '__main__':
    os.environ['PROJ_DATA'] = r'C:\Users\Labadmin\anaconda3\envs\weather\Lib\site-packages\pyproj\proj_dir\share\proj'
    os.chdir(r'D:\!Research\01 - Python\FiTriMap\ignore_data')
    dataset_dir = 'CNFDB 256 100m NEW'
    above_shp_dir = r'D:\!Research\01 - Python\Piyush\FirePred\Data\Wildfire\Wildfires_Date_of_Burning_1559\unzipped'
    master_fuelmap_dir = r'G:\Shared drives\UofA Wildfire\Project\01 - Machine Learning\Daily Wildfire Prediction\Fuel Maps'
    master_dem_path = r'G:\Shared drives\UofA Wildfire\Project\03 - Imagery\Canada MDEM\mrdem-30-dtm.tif'
    isi_nc4_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\FWI\ISI'
    sz = 256

    # %% ABoVE
    # all_fires, fire_areas = load_ABoVE_shp(above_shp_dir)

    # with open('all_fires_above_2002_2018.pkl', 'wb') as f:
    #     pickle.dump(all_fires, f)
    # with open('fire_areas_above_2002_2018.pkl', 'wb') as f:
    #     pickle.dump(fire_areas, f)

    pkl_dirs = {'all_fires': 'all_fires_above_2002_2018.pkl',
                'fire_areas': 'fire_areas_above_2002_2018.pkl'}

    # with open(pkl_dirs['fire_areas'], 'rb') as f:
    #     fire_areas = pickle.load(f)
    # with open(pkl_dirs['all_fires'], 'rb') as f:
    #     all_fires = pickle.load(f)

    # Remember: these are extent quantiles (not area)
    quants_above = {'Q10': 49.94959039194655,
                    'Q25': 172.79736878257245,
                    'Q50': 575.3554684885603,
                    'Q75': 2875.4346012604947,
                    'Q90': 9285.051258987989}
    size_dict_above = {'min': quants_above['Q10'],
                       'max': quants_above['Q90'] + 16000}  # 114.20 m

    # %% CNFDB
    quants_cnfdb = {'Q10': 6660.0, 'Q25': 8640.0, 'Q50': 12600.0, 'Q75': 19980.0, 'Q90': 32040.0}
    size_dict_cnfdb = {'min': quants_cnfdb['Q10'],
                       'max': quants_cnfdb['Q90'] - 5000}  # adjustment makes pixel size ~ 105m

    # %% Processing
    processing_options = {
        'create_above_rasters': {
            'include': False,
            'above_shp_dir': above_shp_dir,
            'shape': (sz, sz),
            'pkl_dirs': pkl_dirs,
            'size_dict': size_dict_above,
            'save_pkl': 'fire_rasters.pkl'
        },
        'get_cnfdb_rasters': {
            'include': False,
            'zip_dir': r'D:\!Research\01 - Python\Piyush\CNN Fire Prediction\Piyush Fire Dataset\Fire growth rasters',
            'size_dict': size_dict_cnfdb,
            'bad_fires': ['2002_375', '2002_389', '2002_640', '2003_64', '2003_362', '2003_393', '2003_412', '2003_586', '2003_602', '2003_633', '2004_546', '2005_2', '2005_7', '2006_366',
                          '2006_671', '2007_96', '2009_339', '2009_397', '2011_317', '2012_248', '2012_250', '2012_545', '2012_745', '2012_851', '2013_288', '2013_567', '2013_805', '2015_155',
                          '2015_1177', '2015_1693', '2016_174', '2017_1860', '2018_494', '2020_359', '2020_343'],
            'target_shape': (sz, sz)
        },
        'get_data': {
            'include': False,
            'master_fuelmap_dir': master_fuelmap_dir,  # str
            'master_dem_path': master_dem_path,  # str
            'isi_nc4_dir': isi_nc4_dir  # str
        },
        'resize': {
            'include': False,
            'shape': (sz, sz)
        },
        'sanitize': {
            'include': False
        },
        'cleaning': {
            'include': False
        },
        'normalize': {
            'include': True,
            'method': 'minmax'  # str
        },
        'get_fire_stats': {
            'include': True,
            'stats_csv': 'cnfdb_256_100m_NEW_nospot_stats.csv'  # str
        },
        'make_csv': {
            'include': True,
            'output_csv': 'cnfdb_256_100m_NEW_nospot_10per.csv',
            'stats_csv': 'cnfdb_256_100m_NEW_nospot_stats.csv',
            'growth_thresh': 0.1,
            'subset': None
        },
        'plot': {
            'include': False
        }
    }

    # Add: hybrid dataset creation

    create_dataset(dataset_dir, processing_options)
