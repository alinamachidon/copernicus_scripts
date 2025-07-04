import requests
import os
import pandas as pd
from sentinelhub import BBox
from rasterio.crs import CRS
from datetime import datetime, timedelta
from dotenv import load_dotenv
from shapely.geometry import box,shape
from shapely.wkt import loads as load_wkt
import geopandas as gpd
from shapely.ops import unary_union 
from shapely.geometry import shape, box
import zipfile
import rasterio
from rasterio.merge import merge
import numpy as np
import shutil
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from rasterio.enums import Resampling


def get_token():
    load_dotenv()
    # Get credentials from environment
    copernicus_user = os.getenv("CDSE_USERNAME")
    copernicus_password = os.getenv("CDSE_PASSWORD")

    data = {
        "client_id": "cdse-public",
        "username": copernicus_user,
        "password": copernicus_password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Token creation failed. Reponse from the server was: {r.json()}"
        )
    access_token = r.json()["access_token"]
    return copernicus_user, copernicus_password, access_token


def parse_geofootprint(geo):
    if isinstance(geo, dict):
        return shape(geo)
    elif isinstance(geo, list):
        print("GeoFootprint is a list. Contents:", geo[:1])
        if len(geo) > 0 and isinstance(geo[0], dict):
            return shape(geo[0])
    print("Invalid GeoFootprint:", geo)
    return None

    
def select_and_download_products(df, bbox, target_date, save_dir):
    """
    Selects the most suitable products covering the given bbox, downloads them,
    and returns paths to the downloaded zip files.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    bbox_polygon = box(*bbox)
  
    df["geometry"] = df["GeoFootprint"].apply(parse_geofootprint)
    df = df[df["geometry"].notnull()]
    plot_footprints_with_dates_and_id(df, bbox_polygon, date_str=target_date.strftime("%Y%m%d"))

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf = gdf[gdf.intersects(bbox_polygon)]
    if gdf.empty:
        print("No products intersect with the given bounding box.")
        return []

    # Sort products
    gdf["date_diff"] = gdf["Date"].apply(lambda d: abs((d - target_date).days))
    gdf = gdf.sort_values(by=["date_diff", "cloudCover"])

    selected_products = []
    covered_geom = None

    for _, row in gdf.iterrows():
        geom = row["geometry"]
        selected_products.append(row)
        if covered_geom is None:
            covered_geom = geom
        else:
            covered_geom = covered_geom.union(geom)

        if covered_geom.covers(bbox_polygon):
            print("Full area coverage achieved.")
            break

    # Convert selected rows to a DataFrame
    selected = pd.DataFrame(selected_products)

    print(f"Selected {len(selected)} products for full coverage.")

    downloaded_files = []
    session = requests.Session()
    copernicus_user, copernicus_password, access_token = get_token()
    session.headers.update({"Authorization": f"Bearer {access_token}"})

    for _, row in selected.iterrows():
        prod_id = row["Id"]
        identifier = row["Name"]
        zip_path = os.path.join(save_dir, f"{identifier}.zip")

        if os.path.exists(zip_path):
                print(f"Skipping {identifier}: zip already exists.")
                downloaded_files.append(zip_path)
        else:
            try:
                download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"
                response = session.get(download_url, allow_redirects=False)
                while response.status_code in (301, 302, 303, 307):
                    download_url = response.headers["Location"]
                    response = session.get(download_url, allow_redirects=False)
                file = session.get(download_url, verify=False, allow_redirects=True)

                with open(zip_path, "wb") as f:
                    print(f"Downloaded {identifier}")
                    f.write(file.content)

                downloaded_files.append(zip_path)
            except Exception as e:
                print(f"Failed to download {identifier}: {e}")

    return downloaded_files

def merge_bands(zip_paths, output_path="merged.tif", temp_dir="./temp_unzip"):
    """
    Unzips Sentinel-2 product archives, finds spectral bands,
    merges them across tiles (if multiple), and saves a multi-band GeoTIFF.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # Bands to merge (define order explicitly)
    bands_to_merge = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    band_paths = {band: [] for band in bands_to_merge}

    # --- Unzip all archives ---
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
            print(f"Extracted: {zip_path}")

    # --- Find IMG_DATA folders ---
    img_data_dirs = glob.glob(os.path.join(temp_dir, "*.SAFE/GRANULE/*/IMG_DATA*"), recursive=True)
    print(f"Found {len(img_data_dirs)} IMG_DATA directories.\n")

    # --- Collect paths for each band ---
    for img_data_path in img_data_dirs:
        print(f"Processing: {img_data_path}")
        jp2_files = sorted(glob.glob(os.path.join(img_data_path, "**/*.jp2"), recursive=True))

        if not jp2_files:
            print("  No .jp2 files found.")
            continue

        for f in jp2_files:
            filename = os.path.basename(f)
            for band in bands_to_merge:
                if f"{band}_" in filename:
                    band_paths[band].append(f)
                    print(f"  Found {band}: {f}")

    # Merge each band individually 
    merged_bands = []
    first_band_src = None
    final_transform = None
    final_crs = None
    final_width = None
    final_height = None
    dtype = None

    for band in bands_to_merge:
        if not band_paths[band]:
            print(f"Warning: No files found for band {band}")
            continue

        src_files = [rasterio.open(p) for p in band_paths[band]]

        # Merge multiple tiles (if needed)
        mosaic, out_trans = merge(src_files, resampling=Resampling.nearest)

        # Use first band to set overall metadata
        if first_band_src is None:
            first_band_src = src_files[0]
            final_transform = out_trans
            final_crs = first_band_src.crs
            final_width = mosaic.shape[2]
            final_height = mosaic.shape[1]
            dtype = first_band_src.dtypes[0]

        merged_bands.append(mosaic[0])

        for src in src_files:
            src.close()

    # Write merged multi-band GeoTIFF 
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": len(merged_bands),
        "width": final_width,
        "height": final_height,
        "transform": final_transform,
        "crs": final_crs,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, band_data in enumerate(merged_bands, start=1):
            dst.write(band_data, i)

    print(f"Merged image with {len(merged_bands)} bands saved to: {output_path}")

    # Clean up temp directory
    shutil.rmtree(temp_dir)



def plot_footprints_with_dates_and_id(df, bbox_polygon, date_str="unknown"):
    # Ensure geometries are valid
    df = df[df["geometry"].notnull()]
    
    # Convert to GeoDataFrame if not already
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot BBox polygon
    x, y = bbox_polygon.exterior.xy
    ax.plot(x, y, color='red', linewidth=2, label='Target BBox')

    # Assign unique colors per product within each date
    unique_dates = sorted(gdf["Date"].unique())

    for date in unique_dates:
        df_date = gdf[gdf["Date"] == date]
        product_ids = df_date["Id"].unique()
        n_products = len(product_ids)
        cmap = cm.get_cmap('Set1', n_products)  

        for i, pid in enumerate(product_ids):
            df_product = df_date[df_date["Id"] == pid]
            color = cmap(i)

            for _, row in df_product.iterrows():
                geom = row["geometry"]
                label = f"{date} | {pid[:8]}"  # Shorten ID in legend

                if geom.geom_type == "Polygon":
                    gx, gy = geom.exterior.xy
                    ax.fill(gx, gy, color=color, alpha=0.5, label=label)
                elif geom.geom_type == "MultiPolygon":
                    for part in geom:
                        gx, gy = part.exterior.xy
                        ax.fill(gx, gy, color=color, alpha=0.5, label=label)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='x-small', ncol=1)

    ax.set_title("Footprints Colored by Date and Product ID")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"product_footprints_{date_str}.png", dpi=300)
    plt.close()


def plot_footprints_with_dates(df, bbox_polygon, date_str="unknown"):
    # Ensure geometries are valid
    df = df[df["geometry"].notnull()]
    
    # Convert to GeoDataFrame if not already
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Get unique dates for colormap
    unique_dates = sorted(gdf["Date"].unique())
    norm = mcolors.Normalize(vmin=0, vmax=len(unique_dates)-1)
    cmap = cm.get_cmap('viridis', len(unique_dates))

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot BBox polygon
    x, y = bbox_polygon.exterior.xy
    ax.plot(x, y, color='red', linewidth=2, label='Target BBox')

    # Plot each footprint, colored by acquisition date
    for _, row in gdf.iterrows():
        geom = row["geometry"]
        date = row["Date"]
        color_idx = unique_dates.index(date)
        color = cmap(norm(color_idx))

        if geom.geom_type == "Polygon":
            gx, gy = geom.exterior.xy
            ax.fill(gx, gy, color=color, alpha=0.5, label=str(date))
        elif geom.geom_type == "MultiPolygon":
            for part in geom:
                gx, gy = part.exterior.xy
                ax.fill(gx, gy, color=color, alpha=0.5, label=str(date))

    # Avoid duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    ax.set_title("Product Footprints by Acquisition Date")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"product_footprints_{date_str}.png", dpi=300)
    plt.close()


def download_and_filter(
        bbox: BBox,
        target_date: datetime,
        cloud_coverage: float = 0.2,
        data_collection: str = "SENTINEL-2",
        product_type: str = "S2MSI2A",
        save_dir: str = "./data",
    ):
    """
    Search and download Sentinel-2 products for the closest available dates.

    Args:
        bbox (BBox): SentinelHub BBox (WGS84)
        available_dates (datetime.datetime): datetime objects with targeted date for retrieval
        cloud_coverage (float): Max allowed cloud cover (0–1)
        data_collection (str): e.g., "SENTINEL-2"
        product_type (str): e.g. "S2MSI2A"
        save_dir (str): Where to save the products
        max_records (int): Max number of products to fetch
    """
    os.makedirs(save_dir, exist_ok=True)

    minx, miny = bbox.lower_left
    maxx, maxy = bbox.upper_right
    bbox_str = f"{minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}"
    polygon = f"POLYGON(({bbox_str}))"

    clouds = f"{cloud_coverage*100}"

    date_str = target_date.strftime("%Y-%m-%d")
    before_target_date = target_date - timedelta(days=5)
    before_target_date_string = before_target_date.strftime("%Y-%m-%d")
    after_target_date = target_date + timedelta(days=5)
    after_target_date_string = after_target_date.strftime("%Y-%m-%d")

    print(f"Querying for data on closest date: {date_str}")  

    url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq '{data_collection}' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start ge {before_target_date_string}T00:00:00.000Z and "
        f"ContentDate/Start lt {after_target_date_string}T00:00:00.000Z and "
        f"Attributes/OData.CSC.StringAttribute/any(a: a/Name eq 'productType' and a/Value eq '{product_type}') and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {clouds})&"
        f"$expand=Attributes&$top=100&$count=true"
    )   

    response = requests.get(url)
    response.raise_for_status()

    products = pd.DataFrame.from_dict(response.json().get("value", []))
   
    entries = []
    if not products.empty:
        for product in response.json().get("value", []):
            cloud_cover = None
            for attr in product.get("Attributes", []):
                if attr["Name"] == "cloudCover":
                    cloud_cover = float(attr["Value"])
                    break
            print(product["Name"], "Cloud Cover:", cloud_cover)


            content_date = product.get("ContentDate", {}).get("Start", None)
            cloud_cover = None
            for attr in product.get("Attributes", []):
                if attr["Name"] == "cloudCover":
                    cloud_cover = float(attr["Value"])
                    break

            if content_date and cloud_cover is not None:
                entries.append({
                    "Id": product["Id"],
                    "Name": product["Name"],
                    "Date": datetime.fromisoformat(content_date.replace("Z", "+00:00")).date(),  # date only
                    "cloudCover": cloud_cover,
                    "GeoFootprint": product["GeoFootprint"]
                })
    
        df = pd.DataFrame(entries)

        df["date_diff"] = df["Date"].apply(lambda d: abs((d - target_date).days))
        df = df.sort_values(by=["date_diff", "cloudCover"])
   
        print("All available dates:", df["Date"].unique())
        
        downloaded_zips = select_and_download_products(df, bbox, target_date, save_dir)
        print(downloaded_zips)
        
        if downloaded_zips:
            sample_name = os.path.basename(downloaded_zips[0])
            print(sample_name)

            # regex to extract acquisition date and tile
            match = re.search(r'_(\d{8}T\d{6})_N\d{4}_R\d{3}_(T[0-9A-Z]{5})_', sample_name)
            acq_time, tile_id = match.groups() if match else ('DATE', 'TILE')

            output_name = f"S2_{tile_id}_{acq_time}_bands.tif"
            print(output_name)

            merge_bands(downloaded_zips, output_path=os.path.join(save_dir, output_name))
            