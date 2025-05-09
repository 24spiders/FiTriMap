# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:30:08 2025

@author: Labadmin
"""
import os
import rasterio
import numpy as np
import geopandas as gpd
from skimage import measure
from shapely.geometry import Polygon, shape

import fitrimap
from fitrimap.utils.date_utils import doy_to_month_day


def mask_to_polygons(data, mask, transform, value, simplify_tolerance=100):
    # Generate polygons using rasterio.features.shapes (preserves topology)
    shapes_gen = rasterio.features.shapes(data, mask=mask, transform=transform)

    # Convert shapes to shapely geometries
    polygons = []
    for geom, val in shapes_gen:
        if val == value:
            # Convert geojson geometry to shapely geometry
            poly = shape(geom).simplify(tolerance=simplify_tolerance, preserve_topology=True)
            polygons.append(poly)

    # Skip if no valid geometry
    if not polygons:
        return None
    else:
        return polygons


def create_ignitions(burn_tif):
    h, t = os.path.split(burn_tif)
    os.makedirs(os.path.join(h, 'Ignitions'), exist_ok=True)
    # Load data from raster
    with rasterio.open(burn_tif) as src:
        data = src.read(1)
        transform = src.transform
        src_crs = src.crs

    year = t[:4]

    # Mask out 0 and nan values
    valid_data = data[(data != 0) & (~np.isnan(data))]
    unique_values = np.unique(valid_data)
    unique_values = np.sort(unique_values)

    for value in unique_values:
        # Create mask for the current DOY
        mask = data == value

        # Get timestamp from DOY
        month, day = doy_to_month_day(year, value)
        timestamp = f'{day}/{month}/{year}'

        polygons = mask_to_polygons(data, mask, transform, value)

        # Create GeoDataFrame and export
        gdf = gpd.GeoDataFrame({'geometry': polygons, 'timestamp': timestamp}, crs=src_crs)
        gdf = gdf.to_crs('EPSG:4326')

        # Write to shapefile
        out_shp = t.replace('.tif', f'_{month}_{day}.shp')
        gdf.to_file(os.path.join(h, 'Ignitions', out_shp), driver='ESRI Shapefile')


if __name__ == '__main__':
    os.chdir(r'D:\!Research\01 - Python\FiTriMap\ignore_data\CNFDB 256 100m\2002_109')
    create_ignitions('2002_109_burn_nospot.tif')
