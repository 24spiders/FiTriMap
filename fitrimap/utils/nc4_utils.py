# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:30:44 2025

@author: Labadmin
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def load_fwi_nc4(nc_file_path, variable, doy, year, verbose=True):
    """Loads data from netCDF files from 'Global Fire Weather Indices' - https://zenodo.org/records/3626193

    Args:
        nc_file_path (str): Path to the netCDF file.
        variable (str): Name of the variable to load from the netCDF file.
        doy (int): Day of year to get variable data for.
        year (int): Year to get variable data for.
        verbose (bool, optional): If True, prints additional information. Defaults to True.

    Raises:
        ValueError: Raised when passed variable is not found in netCDF file.
        ValueError: Rasied when passed date (year, doy) is not found in netCDF file.

    Returns:
        df (Pandas DataFrame): DataFrame containing the loaded data. Has columns [Latitude, Longitude, Date, Variable].
    """
    # Open the NetCDF file
    dataset = nc.Dataset(nc_file_path, 'r')
    if verbose:
        print(f'netCDF Keys: {dataset.variables.keys()}')

    # Get the variable data
    if variable not in dataset.variables:
        raise ValueError(f'Variable {variable} not found in NetCDF file! Check keys.')

    var_data = dataset.variables[variable][:]

    # Get latitude and longitude variables
    lat_var = next((var for var in ['Latitude', 'latitude', 'lat'] if var in dataset.variables), None)
    lon_var = next((var for var in ['Longitude', 'longitude', 'lon'] if var in dataset.variables), None)
    lats = dataset.variables[lat_var][:]
    lons = dataset.variables[lon_var][:]
    lons[lons > 180] -= 360

    # Create meshgrid of latitudes and longitudes
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Get time variable
    time_var = next((var for var in ['Time', 'time'] if var in dataset.variables), None)
    time_data = dataset.variables[time_var][:]

    # Convert DoY to proper date for comparison
    input_date = (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime('%Y-%m-%d')

    # Filter data where DoY matches input DoY
    filtered_data_idx = (time_data == doy)
    if not np.any(filtered_data_idx):
        raise ValueError(f'No data found for DoY {doy} in the netCDF file.')

    # Apply the filter based on the DoY index
    filtered_var_data = var_data[filtered_data_idx]
    filtered_lats = lat_grid.flatten()
    filtered_lons = lon_grid.flatten()
    filtered_dates = np.array([input_date] * len(filtered_lats))

    # Flatten the variable data for the filtered time indices
    filtered_var_flat = filtered_var_data.flatten()

    # Create the DataFrame
    df = pd.DataFrame({
        'latitude': filtered_lats,
        'longitude': filtered_lons,
        'date': filtered_dates,
        variable: filtered_var_flat
    })

    # Remove rows where the variable is NaN or masked
    df = df.dropna()

    # Close the dataset
    dataset.close()

    return df


def find_nearest_n_points(lat, lon, df, n_pts):
    """From WeatherFetch
    Filters a DataFrame to the nearest 'n_pts' points to a given (lat, lon) based on latitude and longitude.

    Args:
        lat (float): Latitude of the target point.
        lon (float): Longitude of the target point.
        df (pd.DataFrame): DataFrame containing columns 'latitude', 'longitude', and other data.
        n_pts (int): Number of nearest points to retain.

    Returns:
        pd.DataFrame: Filtered DataFrame containing the 'n' nearest points.
    """
    # Ensure required columns are present in the DataFrame
    required_columns = {'latitude', 'longitude'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f'The DataFrame must contain the following columns: {required_columns}')

    # Convert target lat/lon to radians
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)

    # Convert DataFrame lat/lon columns to radians
    lats_rad = np.radians(df['latitude'].values)
    lons_rad = np.radians(df['longitude'].values)

    # Calculate differences in latitudes and longitudes
    dlat = lats_rad - lat_rad
    dlon = lons_rad - lon_rad

    # Haversine distance calculation
    a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = 6371 * c  # Earth's radius in kilometers

    # Add distances to the DataFrame (convert to metres)
    df['distance'] = distances * 1000

    # Sort DataFrame by distance and select the top 'n' rows
    filtered_df = df.nsmallest(n_pts, 'distance')

    # Drop the 'distance' column from the output for clarity
    filtered_df = filtered_df.drop(columns=['distance'])

    return filtered_df
