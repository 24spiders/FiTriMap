# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:39:41 2025

@author: Labadmin
"""
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
import re
from shapely.ops import unary_union
import pickle
from tqdm import tqdm
import shutil

from fitrimap.fire.fire_size import filter_fires_by_threshold


def load_ABoVE_shp(above_shp_dir):
    all_fires = {}
    for file in os.listdir(above_shp_dir):
        if file.endswith('.shp'):
            print(f'Loading {file}...')
            m = re.search(r'\d{4}', file)
            year = m.group(0)
            gdf = gpd.read_file(os.path.join(above_shp_dir, file))
            # if int(year) > 2018:
            #     continue
            # Group by UID_Fire
            for uid, group in gdf.groupby('UID_Fire'):
                # We must exlcude Alaska as the fuel map only has data in Canada
                if 'AK' not in group['FD_Agency'].tolist():
                    fire_id = f'{year}_{uid}'
                    if fire_id in all_fires:
                        all_fires[fire_id] = pd.concat([all_fires[fire_id], group])
                    else:
                        all_fires[fire_id] = group

    # Now, measure areas and find largest dimension
    fire_areas = {}
    for fire_id, fire_gdf in all_fires.items():
        print(f'Measuring {fire_id}...')
        # Dissolve all polygons for this fire to get total area
        dissolved = unary_union(fire_gdf.geometry)
        fire_areas[fire_id] = dissolved.area

    return all_fires, fire_areas


def ABoVE_shp_to_tif(above_shp_dir,
                     fire_raster_dir,
                     shape=(128, 128),
                     pkl_dirs={},
                     size_dict={},
                     verbose=False):

    # Make fire raster directory
    os.makedirs(fire_raster_dir, exist_ok=True)

    # Set target sizes
    target_width, target_height = shape[0], shape[1]

    # Load the fires
    if pkl_dirs:
        with open(pkl_dirs['all_fires'], 'rb') as f:
            all_fires = pickle.load(f)
        with open(pkl_dirs['fire_areas'], 'rb') as f:
            fire_areas = pickle.load(f)
    else:
        all_fires, fire_areas, _ = load_ABoVE_shp(above_shp_dir)

    # Filter by area
    if size_dict:
        keys = filter_fires_by_threshold(all_fires, size_dict['min'], size_dict['max'], metric='max_extent')
        pbar = tqdm(total=len(keys), desc='Filtering fires by size')
        # Remove fires from processing
        for key in keys:
            if verbose:
                print(f'Removing {key} due to size...')
            if key in fire_areas:
                del fire_areas[key]
            if key in all_fires:
                del all_fires[key]
            pbar.update(1)
        pbar.close()

    # Identify the fire with the largest extent in width or height
    max_width = 0
    max_height = 0
    pbar = tqdm(total=len(all_fires.items()), desc='Calculating pixel size')
    for fire_id, fire_gdf in all_fires.items():
        fire_gdf = fire_gdf.to_crs(epsg=3979)
        bounds = fire_gdf.total_bounds  # Get bounding box (minx, miny, maxx, maxy)

        fire_width = bounds[2] - bounds[0]  # maxx - minx
        fire_height = bounds[3] - bounds[1]  # maxy - miny

        max_width = max(max_width, fire_width)
        max_height = max(max_height, fire_height)
        pbar.update(1)

    # Compute a pixel size that ensures all fires fit
    pixel_width = max_width / target_width
    pixel_height = max_height / target_height
    pixel_size = max(pixel_width, pixel_height)  # Maintain aspect ratio
    print(f'Using consistent pixel size: {pixel_size:.2f} meters')

    pbar = tqdm(total=len(all_fires.items()), desc='Rasterizing fires')
    fire_rasters = []
    for fire_id, fire_gdf in all_fires.items():
        # Skip the fire if it is only 1-day long
        if all(fire_gdf['JD'] == 0) or len(fire_gdf) < 2:
            if verbose:
                print(f'Skipping {fire_id} (< 2 days)...')
            pbar.update(1)
            continue

        os.makedirs(os.path.join(fire_raster_dir, fire_id), exist_ok=True)

        # Skip the fire if it already exists
        output_path = os.path.join(fire_raster_dir, fire_id, f'{fire_id}_burn.tif')
        if os.path.exists(output_path):
            pbar.update(1)
            continue

        fire_gdf = fire_gdf.to_crs(epsg=3979)

        # Get the bounds of the current fire
        bounds = fire_gdf.total_bounds

        # Calculate the center point of the fire
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        # If it does not fit, adjust pixel size
        largest_width = bounds[2] - bounds[0]
        largest_height = bounds[3] - bounds[1]

        if (largest_width) > (target_width * pixel_size) or (largest_height) > (target_height * pixel_size):
            new_pixel_width = largest_width / target_width
            new_pixel_height = largest_height / target_height
            new_pixel_size = max(new_pixel_width, new_pixel_height)

            # Calculate the required extent to fit the fire
            half_width = (target_width * new_pixel_size) / 2
            half_height = (target_height * new_pixel_size) / 2
            print(f'Fire too big, adjusted to {new_pixel_size:.2f}...')

        else:
            # Calculate the required extent to fit the fire
            half_width = (target_width * pixel_size) / 2
            half_height = (target_height * pixel_size) / 2

        # Define the extent centered on the fire
        minx = center_x - half_width
        maxx = center_x + half_width
        miny = center_y - half_height
        maxy = center_y + half_height

        # Create transform using the consistent pixel size
        transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy,
            target_width, target_height
        )

        # Create empty raster
        raster = np.zeros((target_height, target_width), dtype=np.float32)

        # Rasterize each polygon
        for idx, row in fire_gdf.iterrows():
            if row['JD'] != 0:
                day_from_start = row['JD']
                feature = {'geometry': row.geometry, 'properties': {'day_from_start': day_from_start}}

                feature_raster = rasterize(
                    [(feature['geometry'], feature['properties']['day_from_start'])],
                    out_shape=(target_height, target_width),
                    transform=transform,
                    dtype=np.float32
                )

                # Combine with main raster
                raster = np.maximum(raster, feature_raster)

        # Save as GeoTiff if there is any burn
        valid_data = raster[(raster != 0) & (~np.isnan(raster))]
        unique_doys = np.unique(valid_data)
        if len(unique_doys) > 0:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=target_height,
                width=target_width,
                count=1,
                dtype=np.float32,
                crs='EPSG:3979',
                transform=transform
            ) as dst:
                dst.write(raster, 1)
            fire_rasters.append(output_path)
        else:
            print(f'No unique DOYs in {fire_id}...')
            shutil.rmtree(os.path.join(fire_raster_dir, fire_id))
        pbar.update(1)

    pbar.close()
    return fire_rasters
