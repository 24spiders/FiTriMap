# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:09:14 2025

@author: Labadmin
"""
import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import numpy as np
from tqdm import tqdm
import re
from datetime import datetime
from weatherfetch.array_ops import interpolate_in_bbox, ProcessingOptions
from weatherfetch.point_ops import nearest_n_points
from weatherfetch.earthaccess_fetch import url_download_merra2, build_url

import fitrimap
from fitrimap.utils.date_utils import doy_to_month_day


def get_weather_data(dataset_dir,
                     nc4_dir,
                     weather_dataset,
                     variables,
                     n_pts):
    # Iterate over each directory in the base directory
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc=f'Downloading {weather_dataset} data')
    for fire_id in os.listdir(dataset_dir):
        fire_dir = os.path.join(dataset_dir, fire_id)
        if os.path.isdir(fire_dir):  # Check if it's a directory
            # Create folder to save unpacked weather
            weather_dir = os.path.join(fire_dir, 'Weather')
            os.makedirs(weather_dir, exist_ok=True)
            # Get fire boundary tif
            fid = '_'.join(fire_id.split('_')[:2])
            burn_file = os.path.join(fire_dir, f'{fid}_burn.tif')
            year = fid[:4]  # Get the year (first 4 characters of the fire ID)

            # Load the krig file using rasterio
            with rasterio.open(burn_file) as src:
                # Get the data from the krig file
                burn_data = src.read(1)  # Read the first band of the raster

                # Mask out 0 values and NaNs
                valid_data = burn_data[(burn_data != 0) & (~np.isnan(burn_data))]
                unique_doys = np.unique(valid_data)

                # Get the bounding box (bbox) of the fire file
                bounds = src.bounds  # (min_lon, min_lat, max_lon, max_lat)
                native_bounds = src.bounds

                # Transform the coordinates to EPSG:4326 if necessary
                if src.crs != CRS.from_epsg(4326):
                    bbox = transform_bounds(src.crs, CRS.from_epsg(4326),
                                            left=native_bounds.left,
                                            bottom=native_bounds.bottom,
                                            right=native_bounds.right,
                                            top=native_bounds.top)

                else:
                    # If already in EPSG:4326, directly use the bounds
                    bbox = bounds

            # Convert Day of Year (DOY) to 'YYYY-MM-DD' format
            for doy in unique_doys:
                date = datetime.strptime(f"{year}-{int(doy)}", "%Y-%j")  # Convert DOY to YYYY-MM-DD
                date = date.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'

                # Check if all the needed data exists
                all_exist = True
                for variable in variables:
                    output_csv_path = os.path.join(weather_dir, f'{fid}_{variable}_{int(doy)}.csv')
                    if not os.path.exists(output_csv_path):
                        all_exist = False

                # If not, download and unpack
                if not all_exist:
                    downloaded_files = url_download_merra2(nc4_dir,
                                                           weather_dataset,
                                                           dates=(date, date),
                                                           verbose=False)

                    for file in downloaded_files:
                        for variable in variables:
                            point = ((bbox[0] + bbox[2]) / 2,
                                     (bbox[1] + bbox[3]) / 2)
                            output_csv_path = os.path.join(weather_dir, f'{fid}_{variable}_{int(doy)}.csv')
                            if not os.path.exists(output_csv_path):
                                nearest_n_points(file,
                                                 variable,
                                                 point,
                                                 output_csv_path=output_csv_path,
                                                 output_shp_path=None,
                                                 n_pts=n_pts)
        pbar.update(1)
    pbar.close()


def interpolate_weather_data(dataset_dir,
                             nc4_dir,
                             weather_dataset,
                             variables,
                             n_pts,
                             avg_hours_per_var,
                             resolution):
    # Perform interpolation for each variable
    for i, variable in enumerate(variables):
        for fire_id in os.listdir(dataset_dir):
            fire_dir = os.path.join(dataset_dir, fire_id)
            if os.path.isdir(fire_dir):  # Check if it's a directory
                # Create folder to save the weather array
                weather_dir = os.path.join(fire_dir, 'Weather Arr')
                os.makedirs(weather_dir, exist_ok=True)
                # Get fire boundary tif
                fid = '_'.join(fire_id.split('_')[:2])
                burn_file = os.path.join(fire_dir, f'{fid}_burn.tif')
                year = fid[:4]  # Get the year (first 4 characters of the fire ID)

                # Load the krig file using rasterio
                with rasterio.open(burn_file) as src:
                    # Get the data from the krig file
                    burn_data = src.read(1)  # Read the first band of the raster

                    # Mask out 0 values and NaNs
                    valid_data = burn_data[(burn_data != 0) & (~np.isnan(burn_data))]
                    unique_doys = np.unique(valid_data)

                    # Get the bounding box (bbox) of the fire file
                    bounds = src.bounds  # (min_lon, min_lat, max_lon, max_lat)
                    native_bounds = src.bounds

                    # Transform the coordinates to EPSG:4326 if necessary
                    if src.crs != CRS.from_epsg(4326):
                        bbox = transform_bounds(src.crs, CRS.from_epsg(4326),
                                                left=native_bounds.left,
                                                bottom=native_bounds.bottom,
                                                right=native_bounds.right,
                                                top=native_bounds.top)

                    else:
                        # If already in EPSG:4326, directly use the bounds
                        bbox = bounds

                for doy in unique_doys:
                    month, day = doy_to_month_day(year, doy)

                    if variable == 'T2MMAX':
                        mode = 'daily'
                        hour = None
                        avg_hours = None

                    elif variable in ['ULML', 'VLML', 'PRECTOT']:
                        mode = 'avg_hourly'
                        hour = None
                        avg_hours = avg_hours_per_var[i]

                    elif variable in ['U2M', 'V2M', 'T2M']:
                        mode = 'avg_hourly'
                        hour = None
                        avg_hours = avg_hours_per_var[i]

                    # Get path to NC4 file
                    date = f'{year}-{month}-{day}'
                    filename, _ = build_url(weather_dataset, date)
                    nc4_file_path = os.path.join(nc4_dir, filename)

                    # Create the regular expression pattern
                    pattern = re.compile(f'{fid}_{variable}_{month}_{day}')

                    # Check if file has already been created
                    matching_files = [filename for filename in os.listdir(weather_dir) if pattern.search(filename)]
                    if len(matching_files) == 0:
                        # Set processing options
                        output_tif_path = os.path.join(weather_dir, f'{fid}_{variable}_{month}_{day}.tif')
                        reproj = {'epsg': '3979',
                                  'bbox': native_bounds}
                        options = ProcessingOptions(variable=variable,
                                                    bbox=bbox,
                                                    resolution=resolution,
                                                    mode=mode,
                                                    n_pts=n_pts,
                                                    d=None,
                                                    method='linear',
                                                    hour=hour,
                                                    avg_hours=avg_hours,
                                                    reproj=reproj,
                                                    convert_h5=True)
                        # Run interpolation
                        interpolate_in_bbox(nc4_file_path,
                                            output_tif_path,
                                            options)
                    else:
                        # Skip if file is already created
                        continue


if __name__ == '__main__':
    # Get data
    nc4_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\10m M2I1NXASM NC4s'
    weather_dataset = 'M2I1NXASM'
    variables = ['U10M', 'V10M', 'T10M']
    dataset_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\ABoVE 256'
    get_weather_data(dataset_dir,
                     nc4_dir,
                     weather_dataset,
                     variables,
                     n_pts=10)

    # # Interpolate data
    # nc4_base_dir = r'D:\Users\Liam\Documents\01 - University\Research\Python\Piyush\NC4s\M2I1NXASM NC4s'
    # variables = ['U2M', 'V2M', 'T2M']
    # avg_hours_per_var = [6, 6, 24]
    # resolution = 100
    # n_pts = 10
    # interpolate_weather_data(dataset_dir,
    #                          nc4_dir,
    #                          weather_dataset,
    #                          variables,
    #                          n_pts,
    #                          avg_hours_per_var,
    #                          resolution)
