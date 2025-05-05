import requests
import os
import pandas as pd
from sentinelhub import BBox
from datetime import datetime, timedelta
from dotenv import load_dotenv


def validate_cdse_data_exists_resto(collection: str, bbox_list: list, interval: tuple, maxcc: float = 0.2):
    base_url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"

    start, end = interval
    start_str = start.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end.strftime("%Y-%m-%dT23:59:59Z")

    for bbox in bbox_list:
        minx, miny = bbox.lower_left
        maxx, maxy = bbox.upper_right
        box_param = f"{minx},{miny},{maxx},{maxy}"

        params = {
            "productType": collection,
            "cloudCover": f"[0,{int(maxcc * 100)}]", 
            "startDate": start_str,
            "completionDate": end_str,
            "maxRecords": 5,
            "box": box_param,
        }

        print(f"Querying RESTO API for BBox: {box_param} | Dates: {start_str} - {end_str}")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise RuntimeError(f"CDSE RESTO query failed ({response.status_code}):\n{response.text}")

        json_data = response.json()
        if not json_data.get("features"):
            raise RuntimeError(f"No data found for bbox {box_param} in given date range.")

    print("CDSE data available for all BBoxes using RESTO API.")



def get_closest_available_product_dates(product_type: str, bbox: BBox, target_date: datetime, maxcc: float = 0.2, search_window_days: int = 2):
    """
    Search for the closest available CDSE product date near the target date for each BBox.

    Args:
        product_type (str): e.g. "S2MSI2A"
        bbox (sentinelhub.BBox):  bbox of the targeted area
        target_date (datetime): Date around which to search
        maxcc (float): Max allowed cloud coverage (0–1)
        search_window_days (int): Days to search before and after target date

    Returns:
        list[datetime]: List of closest available product dates 

    Raises:
        RuntimeError: If no products are found in the search window
    """
    base_url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"

    target_date = datetime.strptime(target_date, "%d.%m.%Y")

    start = target_date - timedelta(days=search_window_days)
    end = target_date + timedelta(days=search_window_days)
    start_str = start.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end.strftime("%Y-%m-%dT23:59:59Z")

    minx, miny = bbox.lower_left
    maxx, maxy = bbox.upper_right
    box_param = f"{minx},{miny},{maxx},{maxy}"

    params = {
        "productType": product_type,
        "cloudCover": f"[0,{int(maxcc * 100)}]",
        "startDate": start_str,
        "completionDate": end_str,
        "maxRecords": 100,
        "box": box_param,
    }

    print(f"Querying RESTO API for BBox: {box_param} | Target Date: {target_date.date()} ±{search_window_days} days")
    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"CDSE RESTO query failed ({response.status_code}):\n{response.text}")

    features = response.json().get("features", [])
    if not features:
        raise RuntimeError(f"No data found for bbox {box_param} in ±{search_window_days} days of {target_date.date()}")

    # Sort by proximity to the target date
    features.sort(
        key=lambda f: abs(datetime.fromisoformat(f["properties"]["startDate"].replace("Z", "")) - target_date)
    )

    closest_iso = features[0]["properties"]["startDate"]
    closest_date = datetime.fromisoformat(closest_iso.replace("Z", ""))
    
    return closest_date


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



def download_and_filter(
        bbox: BBox,
        target_date: datetime,
        cloud_coverage: float = 0.2,
        data_collection: str = "SENTINEL-2",
        product_type: str = "S2MSI2A",
        save_dir: str = "./data",
        max_records: int = 100,
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
    before_target_date = target_date - timedelta(days=3)
    before_target_date_string = before_target_date.strftime("%Y-%m-%d")
    after_target_date = target_date + timedelta(days=3)
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

    entries = []

    products = pd.DataFrame.from_dict(response.json().get("value", []))
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
    
        # Group by date and pick product with lowest cloud cover
        best_per_date = df.loc[df.groupby("Date")["cloudCover"].idxmin()].reset_index(drop=True)

        for _, row in best_per_date.iterrows():
            print(f"Selected {row['Name']} with cloud cover: {row['cloudCover']} on {row['Date']}")
            
            prod_id = row["Id"]
            identifier = row["Name"]

            try:
                session = requests.Session()
                copernicus_user, copernicus_password, access_token = get_token()
                session.headers.update({"Authorization": f"Bearer {access_token}"})

                download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"
                response = session.get(download_url, allow_redirects=False)

                while response.status_code in (301, 302, 303, 307):
                    download_url = response.headers["Location"]
                    response = session.get(download_url, allow_redirects=False)

                file = session.get(download_url, verify=False, allow_redirects=True)

                zip_path = os.path.join(save_dir, f"{identifier}.zip")
                # if os.path.exists(zip_path):
                #     print(f"File already exists locally: {zip_path}")
                #     continue

                with open(zip_path, "wb") as f:
                    print(identifier)
                    f.write(file.content)

                print(f"Downloaded {identifier}")

            except Exception as e:
                print(f"Failed to download {identifier}: {e}")

    else:
        print(f"No data found for date {date_str}.") 

