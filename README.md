# Setup Conda Environment

A conda environment named `sentinel_env` is used to manage dependencies.

## Creating the environment

If you don’t have the environment yet, run:

```bash
conda env create -f environment.yml
conda activate sentinel_env


# Copernicus Sentinel-2 Data Downloader and Processor

This project automates downloading and processing Sentinel-2 satellite imagery from the Copernicus Data Space Ecosystem (CDSE) using your Copernicus account credentials.

---

## Getting Started

### 1. Create a Copernicus Account

To access Sentinel-2 data, you need a Copernicus Data Space Ecosystem account:

- Visit [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- Register for an account and verify your email.
- Log in and navigate to your user profile to get your **username** and **password**.

### 2. Obtain Your Access Token

This script automatically fetches your access token using your Copernicus username and password.

### 3. Store Your Credentials Locally

For security, store your Copernicus credentials in a `.env` file in the root directory of this project with the following content:

CDSE_USERNAME=your_username_here
CDSE_PASSWORD=your_password_here

Alternatively, you can store your token in a `.cdse_token` file or use environment variables — this project currently uses `.env` via the `python-dotenv` package.

---

## Installation

Make sure you have Python 3.7+ installed.

Install dependencies:

```bash
pip install -r requirements.txt
Usage
Place your input GeoJSON files in the folder data/output_geojsons/.

Edit the geojson_paths and date_strings lists in main.py or the equivalent script to match your areas of interest and target dates.

Run the script:

```bash

python main.py

The script will:

Read and reproject your GeoJSON polygons.

Query the Copernicus Data Space for available Sentinel-2 products matching your area and date.

Download, merge spectral bands, and save the output GeoTIFF files to desired folder (e.g. "data/labeled/").

## How It Works
The script fetches Sentinel-2 products filtered by cloud coverage, date, and location.

Downloads selected Sentinel-2 product archives.

Extracts and merges spectral bands (B02, B03, ..., B12) into multi-band GeoTIFFs.

Visualizes product footprints for quality control (saved as PNG files).

## Notes
Cloud coverage filter is set to 30% by default but can be adjusted in the script.

Downloaded products are saved as .zip files in the specified directory to avoid repeated downloads.

Temporary files are cleaned up after merging.
