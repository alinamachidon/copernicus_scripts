import argparse
from pathlib import Path
from util.workflows import download_and_filter, get_closest_available_product_dates, validate_cdse_data_exists_resto
import json
import geopandas as gpd
from datetime import datetime
from sentinelhub import BBox, CRS
import os
import requests


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date", type=str, help="Date of interest in format: d.m.Y."
    )  
    parser.add_argument(
        "--maxcc",
        required=False,
        type=float,
        default=0.2,
        help="Max cloud coverage ratio (Default is 0.2)",
    )

    return parser


def main() -> None:
    args = get_parser().parse_args()

    target_date = args.date
    product_type = "S2MSI2A"

    geojson_path = "data/svn_border.geojson"

    # Load the GeoJSON and reproject
    geo = gpd.read_file(geojson_path)
    print("Original CRS:", geo.crs)

    # Reproject to EPSG:4326 (lat/lon for SentinelHub)
    geo = geo.to_crs(epsg=4326)

    # Get bounding box from the reprojected geometry
    min_lon, min_lat, max_lon, max_lat = geo.total_bounds

    # Define the BBox
    bbox = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)
   
    #dir_name = start_date.replace(".", "-")
    
    # Get interval [date - 2 days, date + 2 days]
    #time_of_interest = get_couple_days_span(start_date)

    # Check data availability using CDSE
    # validate_cdse_data_exists_resto(
    #     collection=product_type,
    #     bbox_list=bbox_list,
    #     interval=time_of_interest,
    #     maxcc=args.maxcc,
    # )

    #date = get_closest_available_product_dates(product_type, bbox, maxcc=args.maxcc, target_date=start_date)
    #print("Closest available product date:", date)

    date_object = datetime.strptime(target_date, '%d.%m.%Y').date()
    print(target_date)

    download_and_filter(
        bbox=bbox,
        target_date=date_object,
        cloud_coverage=0.2, #args.maxcc,
        data_collection = "SENTINEL-2",
        product_type = product_type,
        save_dir="data/products",
        max_records=100,
    )


if __name__ == "__main__":
    main()
