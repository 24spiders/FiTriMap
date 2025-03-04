# FiTriMap [WIP - v0]

FiTriMap is a Python package designed for acquiring and processing spatial 'wildfire triangle' data—fuel, topography, and weather—around wildfire perimeters.

## Features

FiTriMap automates the retrieval, processing, and standardization of wildfire-related spatial datasets. Specifically, it:

- Accepts a wildfire shapefile or raster input, expected from datasets such as:
  - [ABoVE Wildfires - Date of Burning](https://daac.ornl.gov/ABOVE/guides/Wildfires_Date_of_Burning.html)
  - [Canadian National Fire Database (CNFDB)](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb)
- Extracts and processes key environmental data within the wildfire perimeter:
  - **Fuel Data**: Crops a fuel map to the wildfire boundary and applies equations to convert from FBP fuel classes to Rate of Spread Index (RSI).
  - **Topography**: Crops a Digital Elevation Model (DEM) and derives elevation, slope, and aspect.
  - **Weather**: Uses [`WeatherFetch`](https://github.com/24spiders/WeatherFetch) to acquire and interpolate weather data on the wildfire’s burn date.
- Performs post-processing, including:
  - Reprojection to a common coordinate system
  - Resizing to a uniform pixel size
  - Normalization
  - Fire statistics calculation
  - Dataset CSV creation

## Output Format

FiTriMap generates structured wildfire datasets in the following directory format:

```
dataset_dir
├── fire_id_1
│   ├── fire_id_burn.tif         # Wildfire perimeter raster
│   ├── fire_id_fuelmap.tif      # Cropped and processed fuel map
│   ├── fire_id_RSI.tif          # Rate of Spread Index
│   ├── fire_id_elevation.tif    # Elevation data
│   ├── fire_id_slope.tif        # Slope data
│   ├── fire_id_aspect.tif       # Aspect data
├── fire_id_2
│   ├── ...
```



## Installation

FiTriMap can be installed by cloning this repository, navigating to the cloned directory, and calling 'python setup.py develop`.

FiTriMap requires [`WeatherFetch`](https://github.com/24spiders/WeatherFetch)

## Usage

[WIP]
