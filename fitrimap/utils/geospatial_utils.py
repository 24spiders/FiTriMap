# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:08:05 2025

@author: Labadmin
"""
import rasterio
import os
from pyproj import Transformer
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from shapely.geometry import box, mapping
from shapely.ops import transform
from pyproj import Transformer
import geopandas as gpd
import numpy as np


def get_tif_bounds(tif_file):
    """Gets the bounds and EPSG code of a TIF file.

    Args:
        tif_file (str): Path to the TIF file.

    Returns:
        bound_dict (dict): Dictionary containing 'bounds' (xmin, ymin, xmax, ymax) and 'epsg' (epsg code).
    """
    with rasterio.open(tif_file) as src:
        # Get bounds
        bounds = src.bounds

        # Get EPSG code
        epsg_code = src.crs.to_epsg()
    bound_dict = {'bounds': bounds, 'epsg': epsg_code}
    return bound_dict


def get_shp_bounds(shp_file):
    """Gets the bounds and EPSG code of a shp file.

    Args:
        shp_file (str): Path to the SHP file.

    Returns:
        bound_dict (dict): Dictionary containing 'bounds' (xmin, ymin, xmax, ymax) and 'epsg' (epsg code).
    """
    # Read the shapefile using geopandas
    gdf = gpd.read_file(shp_file)

    # Get the bounding box of the shapefile
    bounds = gdf.total_bounds  # (xmin, ymin, xmax, ymax)

    # Get the EPSG code from the CRS
    epsg_code = gdf.crs.to_epsg() if gdf.crs else None
    bound_dict = {'bounds': bounds, 'epsg': epsg_code}
    return bound_dict


def transform_bounds(bound_dict, output_epsg):
    """Transforms bounds to a different projection

    Args:
        bound_dict (dict): Dictionary containing 'bounds' (xmin, ymin, xmax, ymax) and 'epsg' (epsg code). Likely returned from either get_tif_bounds() or get_shp_bounds().
        output_epsg (int): Output EPSG code to project bounds to.

    Returns:
        new_bounds (list): [xmin, ymin, xmax, ymax] in the output_epsg coordinate system.
    """
    # Get bound params
    bounds = bound_dict['bounds']
    input_epsg = bound_dict['epsg']
    # Initialize transformer
    transformer = Transformer.from_crs(f'EPSG:{input_epsg}', f'EPSG:{output_epsg}', always_xy=True)

    # Transform coordinates
    xmin, ymin = transformer.transform(bounds[0], bounds[1])
    xmax, ymax = transformer.transform(bounds[2], bounds[3])
    new_bounds = [xmin, ymin, xmax, ymax]
    return new_bounds


def crop_master_to_tif(master_tif, crop_to_tif, output_path, buffer_distance=0):
    """Crops a large 'master' tif to the bounds of a smaller tif.

    Args:
        master_tif (str): Path to the master tif.
        crop_to_tif (str): Path to the smaller tif (master_tif will be cropped to its bounds)
        output_path (str): Path to save the cropped master_tif
        buffer_distance (int, optional): Distance that crop_to_tif bounds will be buffered. Defaults to 0.

    Returns:
        output_path (str): Path where the cropped master_tif was save.
    """
    # Get the boundary of the tif you are cropping to
    bound_dict = get_tif_bounds(crop_to_tif)
    # Crop
    crop_tif_to_bounds(master_tif, bound_dict, output_path, buffer_distance)
    return output_path


def crop_master_to_shp(master_path, crop_to_shp, output_path, buffer_distance=0):
    """Crops a large 'master' tif to the bounds of a shapefile.

    Args:
        master_tif (str): Path to the master tif.
        crop_to_shp (str): Path to the shapefile (master_tif will be cropped to its bounds)
        output_path (str): Path to save the cropped master_tif
        buffer_distance (int, optional): Distance that crop_to_tif bounds will be buffered. Defaults to 0.

    Returns:
        output_path (str): Path where the cropped master_tif was save.
    """
    print(f'Cropping {master_path} to {crop_to_shp}...')
    # Get the total boundary of the shp you are cropping to
    bound_dict = get_shp_bounds(crop_to_shp)
    # Crop fuelmap
    crop_tif_to_bounds(master_path, bound_dict, output_path, buffer_distance)
    return output_path


def crop_tif_to_bounds(master_tif,
                       bound_dict,
                       output_path,
                       buffer_distance=0):
    """Crops a 'master' tif to passed bounds.

    Args:
        master_tif (str): Path to the master tif.
        bound_dict (dict): Dictionary containing 'bounds' (xmin, ymin, xmax, ymax) and 'epsg' (epsg code). Likely returned from either get_tif_bounds() or get_shp_bounds().
        output_path (str): Path to save the cropped master_tif
        buffer_distance (int, optional): Distance that crop_to_tif bounds will be buffered. Defaults to 0.
    """
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


def resize_tif(tif_file, shape, resampling_method, output_path=None):
    """Resizes a tif to a new shape. Updates the tifs projection to ensure it keeps the same spatial bounds.
    OVERWRITES tif_file unless output_path is passed.

    Args:
        tif_file (str): Path to the tif file to resize.
        shape (tuple of ints): (target_width, target_height) in pixels.
        resampling_method (str): Method of rasterio.enums.Resampling.
        output_path (str): Path to save the resized_tif. If not passed, tif_file is overwritten. Defaults to None.

    Returns:
        output_path (str): Path to the resized tif_file.
    """
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
    if not output_path:
        output_path = tif_file
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        # Write each channel individually
        for i in range(resampled_channels.shape[0]):
            dst.write(resampled_channels[i], i + 1)  # Write each channel

    return output_path


def reproject_to_nearest_utm(input_raster_path, output_raster_path=None):
    """
    Reprojects a raster to the nearest UTM coordinate system based on its centroid.
    Works regardless of input CRS (e.g., EPSG:3979).
    Args:
        input_raster_path (str): Path to the input raster file
        output_raster_path (str): Optional path to save the reprojected raster. If None, overwrites it.
    Returns:
        output_raster_path (str): Path to the reprojected raster file
    """
    # Open the input raster
    with rasterio.open(input_raster_path) as src:
        # Get raster attributes
        src_count = src.count
        src_transform = src.transform
        src_crs = src.crs
        data = src.read()

        # Compute the centroid in the source CRS
        bounds = src.bounds
        x_center = (bounds.left + bounds.right) / 2
        y_center = (bounds.top + bounds.bottom) / 2

        # Transform centroid to lat/lon (EPSG:4326)
        transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
        lon, lat = transformer.transform(x_center, y_center)

        # Determine UTM zone
        utm_zone = int((lon + 180) / 6) + 1

        # Use NAD83 if in typical North American range, otherwise WGS84
        if -142 <= lon <= -52 and 40 <= lat <= 85:  # Conservative NAD83 bounds
            target_epsg = 26900 + utm_zone  # NAD83 / UTM zone
        elif lat >= 0:
            target_epsg = 32600 + utm_zone  # WGS84 / UTM north
        else:
            target_epsg = 32700 + utm_zone  # WGS84 / UTM south

        target_crs = CRS.from_epsg(target_epsg)

        # Calculate transform and metadata for reprojection
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

    # Generate output path if not provided
    if output_raster_path is None:
        output_raster_path = input_raster_path

    # Perform the reprojection
    with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
        for i in range(0, src_count):
            reproject(
                source=data[i],
                destination=rasterio.band(dst, i + 1),
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

    return output_raster_path


def crop_raster_to_valid_data(raster_path):
    """
    Crops a raster to its valid data extent.
    Args:
        raster_path (str): Path to raster to crop in place
    """
    with rasterio.open(raster_path) as src:
        # Read the first band and mask
        data = src.read(1)
        _, t = os.path.split(raster_path)
        if np.isnan(src.nodata):
            mask_valid = ~np.isnan(data)
        else:
            mask_valid = data != src.nodata

        # Get row/col bounds of valid data
        rows, cols = np.where(mask_valid)
        row_start, row_stop = rows.min(), rows.max() + 1
        col_start, col_stop = cols.min(), cols.max() + 1

        # Window of valid data
        window = rasterio.windows.Window(col_start, row_start,
                                         col_stop - col_start, row_stop - row_start)

        # Read cropped data
        transform = src.window_transform(window)
        cropped_data = src.read(window=window)

        # Write back to same file or new one
        meta = src.meta.copy()
        meta.update({
            'height': cropped_data.shape[1],
            'width': cropped_data.shape[2],
            'transform': transform
        })

    with rasterio.open(raster_path, 'w', **meta) as dst:
        dst.write(cropped_data)
