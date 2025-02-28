# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:47:48 2025

@author: Labadmin
"""

import os
import rasterio

import fitrimap
from fitrimap.utils.geospatial_utils import get_tif_bounds, get_shp_bounds, crop_tif_to_bounds, crop_master_to_tif

if __name__ == '__main__':
    from fitrimap.fuels.recode_fuelmap import recode_fuelmap_RSI
    os.chdir(r'D:\!Research\01 - Python\FiTriMap\ignore_data\Test Fire')
    master_fuelmap_path = r'G:\Shared drives\UofA Wildfire\Project\01 - Machine Learning\Piyush Project\Fuel Maps\3979_FBP-2002-100m.tif'
    crop_to_tif = '2002_129_krig.tif'
    output_path = 'test.tif'
    crop_master_to_tif(master_fuelmap_path, crop_to_tif, output_path, buffer=0)
    recode_fuelmap_RSI(output_path, 'recoded.tif', 176, 2002, r'D:\!Research\01 - Python\FiTriMap\ignore_data\ISI')
