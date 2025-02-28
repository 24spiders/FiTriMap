# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:23:26 2025

@author: Labadmin
"""
import matplotlib.pyplot as plt
import numpy as np


def fire_extent_quantiles(all_fires):
    """
    Computes the 10th, 25th, 50th, 75th, and 90th quantiles of fire extents (max width or height).

    Args:
        all_fires (dict): A dictionary where keys are fire IDs and values are GeoDataFrames of fire perimeters.

    Returns:
        dict: A dictionary containing the computed quantiles.
    """
    extents = []  # List to store the maximum dimension (width or height) of each fire

    for fire_id, fire_gdf in all_fires.items():
        # Reproject to a consistent CRS (EPSG:3979 for meters)
        fire_gdf = fire_gdf.to_crs(epsg=3979)

        # Get bounding box (minx, miny, maxx, maxy)
        bounds = fire_gdf.total_bounds
        fire_width = bounds[2] - bounds[0]  # maxx - minx
        fire_height = bounds[3] - bounds[1]  # maxy - miny

        # Store the larger of the two dimensions
        extents.append(max(fire_width, fire_height))

    # Define the quantiles to compute
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Compute the quantiles using NumPy
    quantile_values = np.quantile(extents, quantiles)

    # Return the quantiles as a dictionary
    quants = {f'Q{int(q*100)}': quantile_values[i] for i, q in enumerate(quantiles)}
    return quants


def fire_size_quantiles(fire_areas):
    """
    Computes the 10th, 25th, 50th, 75th, and 90th quantiles of fire areas.

    Args:
        fire_areas (dict): A dictionary where keys are fire IDs and values are fire areas.

    Returns:
        dict: A dictionary containing the computed quantiles.
    """
    # Extract the fire areas from the dictionary
    areas = np.array(list(fire_areas.values()))

    # Define the quantiles to compute
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Compute the quantiles using NumPy
    quantile_values = np.quantile(areas, quantiles)

    # Return the quantiles as a dictionary
    quants = {f'Q{int(q*100)}': quantile_values[i] for i, q in enumerate(quantiles)}
    return quants


def filter_fires_by_threshold(all_fires, min_val=None, max_val=None, metric='area'):
    """
    Identifies fire IDs that should be removed based on thresholds for area or extent.

    Args:
        all_fires (dict): A dictionary where keys are fire IDs and values are either:
                          - Fire areas (if metric='area')
                          - GeoDataFrames of fire perimeters (if metric='extent')
        min_val (float, optional): The minimum threshold. Fires below this value will be removed.
        max_val (float, optional): The maximum threshold. Fires above this value will be removed.
        metric (str): The metric to filter by. Options:
                      - 'area': Uses fire area values.
                      - 'min_extent': Uses the smaller dimension (width or height).
                      - 'max_extent': Uses the larger dimension (width or height).

    Returns:
        list: A list of fire IDs that should be removed.
    """
    fire_values = {}

    if metric == 'area':
        # If metric is area, assume all_fires is a dictionary of areas
        fire_values = all_fires
    elif metric in ['min_extent', 'max_extent']:
        for fire_id, fire_gdf in all_fires.items():
            # Reproject to a consistent CRS (EPSG:3979 for meters)
            fire_gdf = fire_gdf.to_crs(epsg=3979)

            # Get bounding box (minx, miny, maxx, maxy)
            bounds = fire_gdf.total_bounds
            fire_width = bounds[2] - bounds[0]  # maxx - minx
            fire_height = bounds[3] - bounds[1]  # maxy - miny

            # Store the required extent metric
            if metric == 'min_extent':
                fire_values[fire_id] = min(fire_width, fire_height)
            else:  # metric == 'max_extent'
                fire_values[fire_id] = max(fire_width, fire_height)

    # Identify fire IDs where the value is outside the specified range
    return [fire_id for fire_id, value in fire_values.items()
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val)]


def plot_fire_areas(fire_areas, large_threshold=None, small_threshold=None):
    '''
    Plots a histogram of fire areas and, if a threshold is provided, lists fires exceeding that threshold.

    Arguments:
        fire_areas: dict, dictionary with fire IDs as keys and fire areas in square meters as values
        threshold: float, area threshold in square meters to filter fires for reporting (default is None)

    Returns:
        large_fires: list, list of fire IDs with areas exceeding the threshold (if provided)
    '''
    # Separate fires below and above the threshold
    if large_threshold is not None:
        areas_below_threshold = [area for area in fire_areas.values() if area <= large_threshold]
        large_fires = [fire_id for fire_id, area in fire_areas.items() if area > large_threshold]
    else:
        areas_below_threshold = list(fire_areas.values())
        large_fires = []

    if small_threshold is not None:
        areas_above_threshold = [area for area in fire_areas.values() if area >= small_threshold]
        small_fires = [fire_id for fire_id, area in fire_areas.items() if area < small_threshold]
    else:
        areas_above_threshold = list(fire_areas.values())
        small_fires = []

    bad_fires = large_fires + small_fires

    # Plot histogram of fire areas
    plt.figure(figsize=(10, 6))
    plt.hist(areas_below_threshold, bins=30, color='orange', edgecolor='black')
    plt.xlabel('Fire Area (sq m)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Fire Areas')
    plt.show()

    return bad_fires
