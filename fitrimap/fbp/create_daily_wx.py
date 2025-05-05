# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:24:53 2025

@author: Labadmin
"""

import os
import rasterio
import math
import numpy as np
import pandas as pd

from fitrimap.utils.date_utils import doy_to_month_day
from fitrimap.fbp.prometheus_fwi import get_fwi_indices


def specific_to_relative_humidity(q, T, P):
    """
    Converts specific humidity to relative humidity.
    https://earthscience.stackexchange.com/questions/2360/how-do-i-convert-specific-humidity-to-relative-humidity

    Args:
        q (float): Specific humidity (kg/kg)
        T (float): Air temperature (°C)
        P (float): Atmospheric pressure (Pa)

    Returns:
        RH (float): Relative humidity (%)
    """
    TK = T + 273.15
    T0 = 273.15
    RH = 0.263 * P * q * (math.exp((17.67 * (TK - T0)) / (TK - 29.65)))**(-1)
    return RH


def daily_wx(dataset_dir, weather_dir, FWI_nc_dir, fire_ids):
    for fire_id in fire_ids:
        df = pd.DataFrame(columns=['Daily', 'Min_Temp', 'Max_Temp', 'Min_RH', 'Min_WS', 'Max_WS', 'WD', 'Precip', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'])
        fid = '_'.join(fire_id.split('_')[:2])
        fire_weather_dir = os.path.join(weather_dir, fire_id, 'Weather')
        burn_tif = os.path.join(os.path.join(dataset_dir, fire_id, f'{fid}_burn.tif'))

        # Get unique values
        with rasterio.open(burn_tif) as src:
            data = src.read(1)

        _, t = os.path.split(burn_tif)
        year = t[:4]

        # Mask out 0 and nan values
        valid_data = data[(data != 0) & (~np.isnan(data))]
        unique_values = np.unique(valid_data)
        unique_values = np.sort(unique_values)

        for value in unique_values:
            # Get daily
            month, day = doy_to_month_day(year, value)
            daily = f'{day}/{month}/{year}'

            # Load temp data
            temp_csv = os.path.join(fire_weather_dir, f'{fid}_T10M_{value}.csv')
            temp_df = pd.read_csv(temp_csv)
            min_temp = temp_df['T10M'].min() - 273.15
            max_temp = temp_df['T10M'].max() - 273.15

            # Get relative humidity
            spechum_csv = os.path.join(fire_weather_dir, f'{fid}_QV10M_{value}.csv')  # Specific Humidty in kg/kg
            spechum_df = pd.read_csv(spechum_csv)
            q = spechum_df['QV10M'].min()

            # Extract identifying values
            min_spechum_row = spechum_df.loc[spechum_df['QV10M'].idxmin()]
            lat = min_spechum_row['latitude']
            lon = min_spechum_row['longitude']
            date = min_spechum_row['date']
            hour = min_spechum_row['hour']

            press_csv = os.path.join(fire_weather_dir, f'{fid}_PS_{value}.csv')  # Pressure in Pa
            press_df = pd.read_csv(press_csv)

            # Find the matching pressure value
            matching_press_row = press_df[
                (press_df['latitude'] == lat) &
                (press_df['longitude'] == lon) &
                (press_df['date'] == date) &
                (press_df['hour'] == hour)
            ]

            # Extract the PS value (as scalar, assuming one match)
            P = matching_press_row['PS'].values[0]

            # Find the matching temperature value
            matching_temp_row = temp_df[
                (temp_df['latitude'] == lat) &
                (temp_df['longitude'] == lon) &
                (temp_df['date'] == date) &
                (temp_df['hour'] == hour)
            ]

            # Extract the PS value (as scalar, assuming one match)
            T = matching_temp_row['T10M'].values[0] - 273.15

            # Get RH
            min_rh = specific_to_relative_humidity(q, T, P)

            # Get precip
            precip_csv = os.path.join(fire_weather_dir, f'{fid}_PRECTOT_{value}.csv')
            precip_df = pd.read_csv(precip_csv)
            precip = precip_df['PRECTOT'].to_numpy()
            precip = precip.sum()  # All precipitation that day [TODO: Filter to one unique point]
            precip = precip * 3600 * 24  # From kg/m2s to mm/hr to mm

            # Load wind data
            u_csv = os.path.join(fire_weather_dir, f'{fid}_U10M_{value}.csv')
            v_csv = os.path.join(fire_weather_dir, f'{fid}_V10M_{value}.csv')
            u_df = pd.read_csv(u_csv)
            v_df = pd.read_csv(v_csv)
            us = u_df['U10M'].to_numpy()
            vs = v_df['V10M'].to_numpy()
            wss = np.sqrt(us**2 + vs**2)
            min_ws = wss.min()
            max_ws = wss.max()

            max_idx = np.argmax(wss)
            u_at_max_ws = us[max_idx]
            v_at_max_ws = vs[max_idx]
            wd = math.degrees(math.atan2(u_at_max_ws, v_at_max_ws)) % 360  # atan2 is (y, x) but we swap to (x, y) to go clockwise

            # Get FWI
            fwi_dict = get_fwi_indices(dataset_dir, fire_id, int(value), FWI_nc_dir, save_csv=True)

            # Create and append new row
            new_row = {'Daily': daily,
                       'Min_Temp': min_temp,
                       'Max_Temp': max_temp,
                       'Min_RH': min_rh,
                       'Min_WS': min_ws,
                       'Max_WS': max_ws,
                       'WD': wd,
                       'Precip': precip,
                       'FFMC': fwi_dict['FFMC'],
                       'DMC': fwi_dict['DMC'],
                       'DC': fwi_dict['DC'],
                       'ISI': fwi_dict['ISI'],
                       'BUI': fwi_dict['BUI'],
                       'FWI': fwi_dict['FWI']
                       }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(os.path.join(dataset_dir, fire_id, f'{fid}_daily_wx.csv'))
