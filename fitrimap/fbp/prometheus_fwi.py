# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 09:18:09 2025

@author: Labadmin
"""
import os
import glob
import rasterio
import numpy as np
import pandas as pd

from fitrimap.fuels.recode_fuelmap import get_FWI
from fitrimap.utils.geospatial_utils import get_tif_bounds, transform_bounds


def get_fwi_indices(dataset_dir,
                    fire_id,
                    FWI_nc_dir,
                    save_csv=False):
    # Init dataframe and fire_dir
    df = pd.DataFrame(columns=['Fire ID', 'Year', 'DOY', 'Latitude', 'Longitude', 'Index', 'Value'])
    fire_dir = os.path.join(dataset_dir, fire_id)

    if os.path.isdir(fire_dir):
        # Get burn_tif
        fid = '_'.join(fire_id.split('_')[:2])
        year = int(fid[:4])
        burn_tif = os.path.join(fire_dir, f'{fid}_burn.tif')

        # Get DOY
        with rasterio.open(burn_tif) as src:
            data = src.read()
            # Get the average value in data excluding 0 and nan
            mask = (data != 0) & (~np.isnan(data))

            # Apply mask to filter out zeros and NaNs
            filtered_data = data[mask]

            # Get first day
            min_value = np.min(filtered_data) if filtered_data.size > 0 else np.nan  # Handle case if no valid data

            # Get dates
            # TODO: Technically, we need a new fuelmap for each day
            doy = int(min_value)

        # Get point
        bound_dict = get_tif_bounds(burn_tif)
        bound_dict['epsg'] = 3979  # TODO: Make this more robust
        xmin, ymin, xmax, ymax = transform_bounds(bound_dict, 4326)
        lat = (ymin + ymax) / 2
        lon = (xmin + xmax) / 2
        point = (lat, lon)

        # Get FWI indices
        bui_nc_dir = os.path.join(FWI_nc_dir, 'BUI')
        bui_nc_path = glob.glob(os.path.join(bui_nc_dir, f'*build_up_index*{year}*.nc'))[0]
        bui = get_FWI(bui_nc_path, 'BUI', doy, year, point)
        row = {'Fire ID': fire_id, 'Year': year, 'DOY': doy, 'Latitude': lat, 'Longitude': lon, 'Index': 'BUI', 'Value': bui}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        dc_nc_dir = os.path.join(FWI_nc_dir, 'DC')
        dc_nc_path = glob.glob(os.path.join(dc_nc_dir, f'*drought_code*{year}*.nc'))[0]
        dc = get_FWI(dc_nc_path, 'DC', doy, year, point)
        row = {'Fire ID': fire_id, 'Year': year, 'DOY': doy, 'Latitude': lat, 'Longitude': lon, 'Index': 'DC', 'Value': dc}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        dmc_nc_dir = os.path.join(FWI_nc_dir, 'DMC')
        dmc_nc_path = glob.glob(os.path.join(dmc_nc_dir, f'*duff_moisture_code*{year}*.nc'))[0]
        dmc = get_FWI(dmc_nc_path, 'DMC', doy, year, point)
        row = {'Fire ID': fire_id, 'Year': year, 'DOY': doy, 'Latitude': lat, 'Longitude': lon, 'Index': 'DMC', 'Value': dmc}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        ffmc_nc_dir = os.path.join(FWI_nc_dir, 'FFMC')
        ffmc_nc_path = glob.glob(os.path.join(ffmc_nc_dir, f'*fine_fuel_moisture_code*{year}*.nc'))[0]
        ffmc = get_FWI(ffmc_nc_path, 'FFMC', doy, year, point)
        row = {'Fire ID': fire_id, 'Year': year, 'DOY': doy, 'Latitude': lat, 'Longitude': lon, 'Index': 'FFMC', 'Value': ffmc}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        fwi_nc_dir = os.path.join(FWI_nc_dir, 'FWI')
        fwi_nc_path = glob.glob(os.path.join(fwi_nc_dir, f'*fire_weather_index*{year}*.nc'))[0]
        fwi = get_FWI(fwi_nc_path, 'FWI', doy, year, point)
        row = {'Fire ID': fire_id, 'Year': year, 'DOY': doy, 'Latitude': lat, 'Longitude': lon, 'Index': 'FWI', 'Value': fwi}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        isi_nc_dir = os.path.join(FWI_nc_dir, 'ISI')
        isi_nc_path = glob.glob(os.path.join(isi_nc_dir, f'*initial_spread_index*{year}*.nc'))[0]
        isi = get_FWI(isi_nc_path, 'ISI', doy, year, point)
        row = {'Fire ID': fire_id, 'Year': year, 'DOY': doy, 'Latitude': lat, 'Longitude': lon, 'Index': 'ISI', 'Value': isi}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        df.to_csv(os.path.join(fire_dir, 'fwi.csv'))

        index_dict = {'BUI': bui,
                      'DC': dc,
                      'DMC': dmc,
                      'FFMC': ffmc,
                      'FWI': fwi,
                      'ISI': isi}
    return index_dict


if __name__ == '__main__':
    dataset_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\CNFDB Prometheus Comparison\Below Q1'
    FWI_nc_dir = r'D:\!Research\01 - Python\FiTriMap\ignore_data\FWI'
    get_fwi_indices(dataset_dir, FWI_nc_dir, save_csv=True)
