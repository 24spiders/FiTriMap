# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:07:01 2025

@author: Labadmin
"""
import os
import glob
import rasterio
import numpy as np
from fitrimap.utils.nc4_utils import load_fwi_nc4, find_nearest_n_points
from fitrimap.utils.geospatial_utils import get_tif_bounds, transform_bounds


def get_abc(fuelmap):
    """Gets 'a', 'b', and 'c' values - coefficients used by Prometheus to calculate RSI.

    Args:
        fuelmap (np.array): Numpy array where cell values = FBP fuel type codes.

    Returns:
        a_arr, b_arr, c_arr (np.array): Numpy arrays of Prometheus RSI coefficients.
    """
    # Initialize arrays
    a_arr = np.zeros(fuelmap.shape)
    b_arr = np.zeros(fuelmap.shape)
    c_arr = np.zeros(fuelmap.shape)

    # a values
    a_arr[np.where(fuelmap == 1)] = 90  # C1
    a_arr[np.where(fuelmap == 2)] = 110  # C2
    a_arr[np.where(fuelmap == 3)] = 110  # C3
    a_arr[np.where(fuelmap == 4)] = 110  # C4
    a_arr[np.where(fuelmap == 5)] = 30  # C5
    a_arr[np.where(fuelmap == 7)] = 45  # C7
    a_arr[np.where(fuelmap == 11)] = 30  # D1
    a_arr[np.where(fuelmap == 13)] = 30  # D1/D3
    a_arr[np.where(fuelmap == 31)] = 190  # O1a
    a_arr[np.where(fuelmap == 101)] = 0  # Non-fuel
    a_arr[np.where(fuelmap == 102)] = 0  # Water
    a_arr[np.where(fuelmap == 415)] = 42  # 15% con mixed
    a_arr[np.where(fuelmap == 625)] = 50  # 25% con mixed
    a_arr[np.where(fuelmap == 650)] = 70  # 50% con mixed
    a_arr[np.where(fuelmap == 675)] = 90  # 75% con mixed

    # b values
    b_arr[np.where(fuelmap == 1)] = 0.0649  # C1
    b_arr[np.where(fuelmap == 2)] = 0.0282  # C2
    b_arr[np.where(fuelmap == 3)] = 0.0444  # C3
    b_arr[np.where(fuelmap == 4)] = 0.0293  # C4
    b_arr[np.where(fuelmap == 5)] = 0.0697  # C5
    b_arr[np.where(fuelmap == 7)] = 0.0305  # C7
    b_arr[np.where(fuelmap == 11)] = 0.0232  # D1
    b_arr[np.where(fuelmap == 13)] = 0.0232  # D1/D3
    b_arr[np.where(fuelmap == 31)] = 0.0310  # O1a
    b_arr[np.where(fuelmap == 101)] = 0  # Non-fuel
    b_arr[np.where(fuelmap == 102)] = 0  # Water
    b_arr[np.where(fuelmap == 415)] = 0.02395  # 15% con mixed
    b_arr[np.where(fuelmap == 625)] = 0.02445  # 25% con mixed
    b_arr[np.where(fuelmap == 650)] = 0.0257  # 50% con mixed
    b_arr[np.where(fuelmap == 675)] = 0.02695  # 75% con mixed

    # c values
    c_arr[np.where(fuelmap == 1)] = 4.5  # C1
    c_arr[np.where(fuelmap == 2)] = 1.5  # C2
    c_arr[np.where(fuelmap == 3)] = 3.0  # C3
    c_arr[np.where(fuelmap == 4)] = 1.5  # C4
    c_arr[np.where(fuelmap == 5)] = 4.0  # C5
    c_arr[np.where(fuelmap == 7)] = 2.0  # C7
    c_arr[np.where(fuelmap == 11)] = 1.6  # D1
    c_arr[np.where(fuelmap == 13)] = 1.6  # D1/D3
    c_arr[np.where(fuelmap == 31)] = 1.4  # O1a
    c_arr[np.where(fuelmap == 101)] = 0  # Non-fuel
    c_arr[np.where(fuelmap == 102)] = 0  # Water
    c_arr[np.where(fuelmap == 415)] = 1.57  # 15% con mixed
    c_arr[np.where(fuelmap == 625)] = 1.55  # 25% con mixed
    c_arr[np.where(fuelmap == 650)] = 1.5  # 50% con mixed
    c_arr[np.where(fuelmap == 675)] = 1.45  # 75% con mixed

    return a_arr, b_arr, c_arr


def get_FWI(nc_path, variable, doy, year, point):
    """Gets Fire Weather Index values from NC files from 'Global Fire Weather Indices' - https://zenodo.org/records/3626193

    Args:
        nc_path (str): Path to the netCDF file.
        variable (str): Index to extract (e.g., 'FFMC').
        doy (int): Day of year to get variable data for.
        year (int): Year to get variable data for.
        point (tuple): (lat, lon) in EPSG:4326 of the point of interest.

    Returns:
        avg_index (float): Mean value of the 4 nearest points to 'point' of 'variable'.
    """
    # Load the .nc file corresponding to the year
    df = load_fwi_nc4(nc_path, variable, doy, year, verbose=False)
    df = find_nearest_n_points(point[0], point[1], df, n_pts=4)
    avg_index = df[variable].mean()
    return avg_index


def recode_fuelmap_RSI(fuelmap_path, output_path, doy, year, nc_dir):
    """Uses ISI maps to recode the fuelmaps. ISI maps can be found here https://zenodo.org/records/3540950. Equations from 'Development and Structure of the Canadian FBP system'

    Args:
        fuelmap_path (str): Path to the fuelmap to recode.
        output_path (str): Path to save the recoded fuelmap.
        doy (int): Day of year to get variable data for.
        year (int): Year to get variable data for.
        nc_dir (str): Path to folder containing ISI netCDF files.

    Returns:
        RSI (np.array): Numpy array with same shape as 'fuelmap' containing RSI values. Also saves RSI array as GeoTIFF.
    """
    with rasterio.open(fuelmap_path) as src:
        fuelmap = src.read(1)
        profile = src.profile

    # Get the center point of the raster
    bound_dict = get_tif_bounds(fuelmap_path)
    xmin, ymin, xmax, ymax = transform_bounds(bound_dict, 4326)
    lat = (ymin + ymax) / 2
    lon = (xmin + xmax) / 2
    point = (lat, lon)

    a_arr, b_arr, c_arr = get_abc(fuelmap)
    nc4_path = glob.glob(os.path.join(nc_dir, f'*initial_spread_index*{year}*.nc'))[0]
    ISI = get_FWI(nc4_path, 'ISI', doy, year, point)
    RSI = a_arr * (1 - np.exp(-b_arr * ISI))**(c_arr)

    # Update the profile
    profile.update(dtype=rasterio.float32, count=1)

    # Save the RSI raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(RSI.astype(np.float32), 1)

    return RSI
