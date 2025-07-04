# Setup Conda Environment

A conda environment named `sentinel_env` is used to manage dependencies.

## Creating the environment

If you don’t have the environment yet, run:

```bash
conda env create -f environment.yml
conda activate sentinel_env
```

# Copernicus Sentinel-2 Data Downloader and Processor

This project automates downloading and processing Sentinel-2 satellite imagery from the Copernicus Data Space Ecosystem (CDSE) using your Copernicus account credentials.

---

## Getting Started

### 1. Create a Copernicus account

To access Sentinel-2 data, you need a Copernicus Data Space Ecosystem account:

- Visit [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- Register for an account and verify your email.
- Log in and navigate to your user profile to get your **username** and **password**.

### 2. Obtain your access tken

This script automatically fetches your access token using your Copernicus username and password.

### 3. Store Your Credentials Locally

For security, store your Copernicus credentials in a `.env` file in the root directory of this project with the following content:

CDSE_USERNAME=your_username_here

CDSE_PASSWORD=your_password_here

Alternatively, you can store your token in a `.cdse_token` file or use environment variables, this project currently uses `.env` via the `python-dotenv` package.

---

## Usage

Place your input GeoJSON files in the folder data/output_geojsons/.

Edit the geojson_paths and date_strings lists in main.py or the equivalent script to match your areas of interest and target dates.

Run the script:

```bash

python main.py
```

The script will:

Read and reproject your GeoJSON polygons.

Query the Copernicus Data Space for available Sentinel-2 products matching your area and date.

Download, merge spectral bands, and save the output GeoTIFF files to desired folder (e.g. "data/labeled/").

# Sentinel-2 Bands Order
The merged Sentinel-2 image contains the following spectral bands in this exact order:

## Band	Description	
B02	Blue	

B03	Green	

B04	Red	

B05	Vegetation Red Edge 1	

B06	Vegetation Red Edge 2	

B07	Vegetation Red Edge 3	

B08	Near Infrared (NIR)	

B8A	Narrow Near Infrared (NIR)	

B11	Shortwave Infrared (SWIR) 

B12	Shortwave Infrared (SWIR) 2	

The multi-band GeoTIFF produced by the script will have these bands in this order, so downstream analysis should use this ordering.

## How it works
The script fetches Sentinel-2 products filtered by cloud coverage, date, and location.

Downloads selected Sentinel-2 product archives.

Extracts and merges spectral bands (B02, B03, ..., B12) into multi-band GeoTIFFs.


## Notes

### Cloud coverage
Cloud coverage filter is set to 30% by default but can be adjusted in the script.

### Date (acquisition period):
The script requires a target date as input and searches for Sentinel-2 Level 2A (S2 L2A) products available within a ±3-day window around this date. This helps to find the closest available cloud-free imagery for your area of interest.

### Region of interest:
The region provided via the GeoJSON file is used to define a bounding box. The Copernicus data search looks for products that intersect with this bounding box. This means any Sentinel-2 products partially or fully covering the specified region may be downloaded to ensure complete coverage.

### Files format
Downloaded products are saved as SAFE.zip files in the specified directory to avoid repeated downloads.

Temporary files are cleaned up after merging.
