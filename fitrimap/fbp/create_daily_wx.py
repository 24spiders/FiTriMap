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


def daily_wx(dataset_dir, fire_ids):
    for fire_id in fire_ids:
        df = pd.DataFrame(columns=['Daily', 'Min_Temp', 'Max_Temp', 'Min_RH', 'Min_WS', 'Max_WS', 'WD', 'Precip', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'])
        fid = '_'.join(fire_id.split('_')[:2])
        weather_dir = os.path.join(dataset_dir, fire_id, 'Weather')
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
            temp_csv = os.path.join(weather_dir, f'{fid}_T10M_{value}.csv')
            temp_df = pd.read_csv(temp_csv)
            min_temp = temp_df['T10M'].min() - 273.15
            max_temp = temp_df['T10M'].max() - 273.15

            # Load RH data
            pass

            # Load wind data
            u_csv = os.path.join(weather_dir, f'{fid}_U10M_{value}.csv')
            v_csv = os.path.join(weather_dir, f'{fid}_V10M_{value}.csv')
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

            # Create and append new row
            new_row = {'Daily': daily,
                       'Min_Temp': min_temp,
                       'Max_Temp': max_temp,
                       # 'Min_RH': min_rh,
                       'Min_WS': min_ws,
                       'Max_WS': max_ws,
                       'WD': wd,
                       # 'Precip': precip,
                       # 'FFMC': 0,
                       # 'DMC': 0,
                       # 'DC': 0,
                       # 'ISI': 0,
                       # 'BUI': 0,
                       # 'FWI': 0
                       }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
