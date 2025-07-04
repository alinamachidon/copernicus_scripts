from util.workflows_new import download_and_filter
import geopandas as gpd
from datetime import datetime
from sentinelhub import BBox, CRS
import os



def main() -> None:
    product_type = "S2MSI2A"

    geojson_paths = ["20250326_Kamno_ESAprojekt7_Orthomosaic.geojson"]
    date_strings = ["26.03.2025"]

    # Loop through each date and call the function
    for date_str, geojson_path in zip(date_strings, geojson_paths):
        print(geojson_path)
        # Load the GeoJSON and reproject
        geo = gpd.read_file(os.path.abspath(os.path.join("data/output_geojsons/",geojson_path)))
        print("Original CRS:", geo.crs)

        # Reproject to EPSG:4326 (lat/lon for SentinelHub)
        geo = geo.to_crs(epsg=4326)

        # Get bounding box from the reprojected geometry
        min_lon, min_lat, max_lon, max_lat = geo.total_bounds

        # Define the BBox
        bbox = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)
        
        try:
            date_object = datetime.strptime(date_str, "%d.%m.%Y")

            download_and_filter(
                bbox=bbox,
                target_date=date_object.date(),
                cloud_coverage=0.3,
                data_collection="SENTINEL-2",
                product_type=product_type,
                save_dir="data/labeled"
            )
        except Exception as e:
            print(f"Error processing date {date_str}: {e}")


if __name__ == "__main__":
    main()
