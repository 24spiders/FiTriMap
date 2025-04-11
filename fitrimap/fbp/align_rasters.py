# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:18:31 2025

@author: Labadmin

To run in Prometheus, fuel map and elevation rasters must be exactly aligned with the same pixel size and cell size.

"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
from tqdm import tqdm


def align_tif(reference_tif, other_tif, output_tif):
    """
    Aligns 'other_tif' to match the CRS, resolution, and pixel dimensions of 'reference_tif',
    and writes the aligned raster to 'output_tif'.

    Args:
        reference_tif (str): File path to the reference GeoTIFF.
        other_tif (str): File path to the GeoTIFF to align.
        output_tif (str): File path where the aligned GeoTIFF will be saved.

    Returns:
        None
    """

    # Open the reference GeoTIFF to get CRS, transform, and shape
    with rasterio.open(reference_tif) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_meta = ref.meta.copy()

    # Open the other GeoTIFF to read data and metadata
    with rasterio.open(other_tif) as src:
        src_data = src.read(1)  # Read first band
        src_crs = src.crs
        src_transform = src.transform
        src_dtype = src.meta['dtype']

        # Allocate an array for the reprojected data
        dst_data = np.empty((dst_height, dst_width), dtype=src_dtype)

        # Reproject and resample
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    # Update metadata for the output GeoTIFF
    dst_meta.update({
        'height': dst_height,
        'width': dst_width,
        'transform': dst_transform,
        'crs': dst_crs,
        'dtype': src_dtype
    })

    # Write the aligned raster to disk
    with rasterio.open(output_tif, 'w', **dst_meta) as dst:
        dst.write(dst_data, 1)


def align_fuelmap_elevation(dataset_dir):
    pbar = tqdm(total=len(os.listdir(dataset_dir)), desc='Aligning elevation and fuelmap rasters')
    for fire_id in os.listdir(dataset_dir):
        fire_dir = os.path.join(dataset_dir, fire_id)
        if os.path.isdir(fire_dir):
            fid = '_'.join(fire_id.split('_')[:2])
            fuelmap_tif = os.path.join(fire_dir, f'{fid}_fuelmap.tif')
            elev_tif = os.path.join(fire_dir, f'{fid}_elevation.tif')
            output_tif = os.path.join(fire_dir, f'{fid}_elevation_aligned.tif')
            align_tif(fuelmap_tif, elev_tif, output_tif)
            pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    os.chdir(r'D:\!Research\01 - Python\FiTriMap\ignore_data\CNFDB 256 100m\2002_109')
    reference_tif = '2002_109_fuelmap.tif'
    other_tif = '2002_109_elevation.tif'
    output_tif = '2002_109_elev_reproj.tif'
    align_tif(reference_tif, other_tif, output_tif)
