# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:54:29 2025

@author: Labadmin
"""
import os
from osgeo import gdal


def create_topo_indices(dem_file, output_dir):
    _, fire_id = os.path.split(output_dir)
    fid = '_'.join(fire_id.split('_')[:2])
    # Compute slope
    slope_path = os.path.join(output_dir, f'{fid}_slope.tif')
    if not os.path.isfile(slope_path):
        gdal.DEMProcessing(slope_path, dem_file, 'slope')

    # Compute aspect
    aspect_path = os.path.join(output_dir, f'{fid}_aspect.tif')
    if not os.path.isfile(aspect_path):
        gdal.DEMProcessing(aspect_path, dem_file, 'aspect')
