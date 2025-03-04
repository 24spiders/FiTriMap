# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:08:05 2025

@author: Labadmin
"""
import rasterio
from pyproj import Transformer
from rasterio.mask import mask
from shapely.geometry import box, mapping
from shapely.ops import transform
import geopandas as gpd
import numpy as np


def get_tif_bounds(tif_file):
    with rasterio.open(tif_file) as src:
        # Get bounds
        bounds = src.bounds

        # Get EPSG code
        epsg_code = src.crs.to_epsg()
    bound_dict = {'bounds': bounds, 'epsg': epsg_code}
    return bound_dict


def get_shp_bounds(shp_file):
    # Read the shapefile using geopandas
    gdf = gpd.read_file(shp_file)

    # Get the bounding box of the shapefile
    bounds = gdf.total_bounds  # (xmin, ymin, xmax, ymax)

    # Get the EPSG code from the CRS
    epsg_code = gdf.crs.to_epsg() if gdf.crs else None

    return {'bounds': bounds, 'epsg': epsg_code}


def transform_bounds(bound_dict, output_epsg):
    # Get bound params
    bounds = bound_dict['bounds']
    input_epsg = bound_dict['epsg']
    # Initialize transformer
    transformer = Transformer.from_crs(f'EPSG:{input_epsg}', f'EPSG:{output_epsg}', always_xy=True)

    # Transform coordinates
    xmin, ymin = transformer.transform(bounds[0], bounds[1])
    xmax, ymax = transformer.transform(bounds[2], bounds[3])

    return [xmin, ymin, xmax, ymax]


def crop_master_to_tif(master_tif, crop_to_tif, output_path, buffer=0):
    # Get the boundary of the tif you are cropping to
    bound_dict = get_tif_bounds(crop_to_tif)
    # Crop
    crop_tif_to_bounds(master_tif, bound_dict, output_path, buffer)
    return output_path


def crop_master_to_shp(master_path, crop_to_shp, output_path, buffer=0):
    print(f'Cropping {master_path} to {crop_to_shp}...')
    # Get the total boundary of the shp you are cropping to
    bound_dict = get_shp_bounds(crop_to_shp)
    # Crop fuelmap
    crop_tif_to_bounds(master_path, bound_dict, output_path, buffer)
    return output_path


def crop_tif_to_bounds(master_tif,
                       bound_dict,
                       output_path,
                       buffer_distance=0):
    # TODO: This could be perfected to ensure pixel alignment (ie, resample fuel maps)
    # Get bound params
    bounds = bound_dict['bounds']
    input_epsg = bound_dict['epsg']

    # Create a polygon from the bounds
    master_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Buffer the polygon if a buffer distance is provided
    if buffer_distance != 0:
        master_polygon = master_polygon.buffer(buffer_distance)

    master_crs = master_tif.crs
    input_crs = rasterio.crs.CRS.from_epsg(input_epsg)

    # Define a transformation function
    project = Transformer.from_crs(input_crs, master_crs, always_xy=True).transform

    # Transform the master polygon to the CRS of the other TIFF
    transformed_polygon = transform(project, master_polygon)

    # Mask the other TIFF with the transformed polygon
    out_image, out_transform = mask(master_tif, [mapping(transformed_polygon)], crop=True)
    out_meta = master_tif.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })

    # Save the cropped TIFF to the output directory
    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(out_image)


def resize_tif(tif_file, shape, resampling_method):
    assert tif_file.endswith('.tif')
    target_width, target_height = shape[0], shape[1]

    # Resize the tif_file
    with rasterio.open(tif_file) as src:
        # Get the original bounds
        left, bottom, right, top = src.bounds

        # Create a new transform to preserve the bounds and projection
        new_transform = rasterio.transform.from_bounds(left, bottom, right, top, target_width, target_height)

        # Initialize an empty list to hold resampled channels
        resampled_channels = np.zeros((src.count, target_height, target_width), dtype=src.dtypes[0])

        # Loop through each channel and resample it
        for i in range(src.count):
            # Read and resample the current channel with the specified method
            channel_data = src.read(
                i + 1,  # Channels are 1-indexed in rasterio
                out_shape=(1, target_height, target_width),
                resampling=resampling_method  # Apply  resampling method
            )
            # Append the resampled channel to the list
            resampled_channels[i] = channel_data

        # Update metadata for the output file
        output_meta = src.meta.copy()
        output_meta.update({
            'height': target_height,
            'width': target_width,
            'transform': new_transform,
        })

    # Write the resized image to disk
    with rasterio.open(tif_file, 'w', **output_meta) as dst:
        # Write each channel individually
        for i in range(resampled_channels.shape[0]):
            dst.write(resampled_channels[i], i + 1)  # Write each channel

    return tif_file
