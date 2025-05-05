import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
import numpy as np
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
import cv2
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import csv
import shutil


sentinel_bands=[4, 3, 2]


def resample_raster(input_raster, output_raster, target_resolution, resampling_method=Resampling.bilinear):
    """
    Resamples raster to a target resolution (e.g., 10m/pixel).
    
    Parameters:
        input_raster (str): Path to the input raster.
        output_raster (str): Path to save the resampled output raster.
        target_resolution (float): Target resolution in meters (e.g., 10 for Sentinel-2 L2A).
        resampling_method (rasterio.enums.Resampling): Resampling method (default: bilinear).
    """
    with rasterio.open(input_raster) as src:
        original_crs = src.crs
        #print(f"Original CRS: {original_crs}")

        # Step 1: Detect if CRS is in degrees (EPSG:4326) and convert to meters
        if original_crs.to_string().startswith("EPSG:4326"):
            print("Detected EPSG:4326 (degrees), converting to a metric projection (UTM)...")
            
            # Convert to an appropriate UTM zone (automatic selection)
            dst_crs = "EPSG:3857"  # Web Mercator (Meters)
        else:
            dst_crs = original_crs  # Already in meters

        # Step 2: Calculate new transform and size
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_resolution
        )

        # Step 3: Update profile
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,  # Update to projected CRS
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": src.dtypes[0],  # Preserve dtype
            "nodata": src.nodata,  # Keep nodata values
            "compress": "lzw"  # Enable compression
        })

        # Step 4: Resample with the correct resolution
        with rasterio.open(output_raster, "w", **profile) as dst:
            for i in range(1, src.count + 1):  # Loop through bands
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method  
                )

        print(f"Resampled {input_raster} to {target_resolution}m/pixel and saved to {output_raster}")


def reproject_raster_to_meters(input_raster, output_raster):
    """
    Reprojects a raster to CRS.
    """
    with rasterio.open(input_raster) as src:
        original_crs = src.crs
        #print(f"Original CRS: {original_crs}")

        # Step 1: Detect if CRS is in degrees (EPSG:4326) and convert to meters
        if original_crs.to_string().startswith("EPSG:4326"):
            print("Detected EPSG:4326 (degrees), converting to a metric projection (UTM)...")
            
            # Convert to an appropriate UTM zone (automatic selection)
            dst_crs = "EPSG:3857"  # Web Mercator (Meters)
        else:
            dst_crs = original_crs  # Already in meters

        with rasterio.open(input_raster) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            profile = src.profile.copy()
            profile.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            with rasterio.open(output_raster, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest  
                    )

    print(f"Reprojected raster saved to {output_raster}")
    

def clip_raster_and_save(raster_path, output_tif_path, reference_raster):
    """
    Clips a raster using the bounding box of a reference raster (drone image).
    Ensures both images cover the same area.
    """
    with rasterio.open(reference_raster) as ref_src:
        bbox = ref_src.bounds  

    # Convert bounding box to a polygon
    coordinates = [
        (bbox.left, bbox.bottom),
        (bbox.left, bbox.top),
        (bbox.right, bbox.top),
        (bbox.right, bbox.bottom),
        (bbox.left, bbox.bottom)
    ]

    gdf = gpd.GeoDataFrame(geometry=[Polygon(coordinates)], crs=ref_src.crs)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs  
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        geojson_geom = [mapping(gdf.geometry[0])]

        # Clip raster
        clipped_raster, clipped_transform = mask(src, geojson_geom, crop=True)
        profile = src.profile
        profile.update({
            "height": clipped_raster.shape[1],
            "width": clipped_raster.shape[2],
            "transform": clipped_transform
        })

        # Save the clipped raster
        with rasterio.open(output_tif_path, "w", **profile) as dst:
            dst.write(clipped_raster)

    print(f"Clipped raster saved to {output_tif_path}")




def pad_raster_to_tile_size(raster_path, padded_raster_path, tile_size_meters, pad_value=None):
    """
    Pads the raster so its dimensions are divisible by the tile size.
    Padding is added to the right and bottom. The padding value can be specified or inferred.
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform
        pixel_size_x, pixel_size_y = transform.a, -transform.e
        tile_size_x = int(tile_size_meters / pixel_size_x)
        tile_size_y = int(tile_size_meters / pixel_size_y)

        pad_x = (tile_size_x - (src.width % tile_size_x)) % tile_size_x
        pad_y = (tile_size_y - (src.height % tile_size_y)) % tile_size_y

        new_width = src.width + pad_x
        new_height = src.height + pad_y

        # Read original data
        data = src.read()
        dtype = src.dtypes[0]

        # Choose padding value
        if pad_value is None:
            if dtype == 'uint8':
                pad_value = 255
            elif dtype == 'uint16':
                pad_value = 10000
            elif 'float' in dtype:
                pad_value = 1.0
            else:
                pad_value = 0 

        # Create padded array
        padded_data = np.full((src.count, new_height, new_width), pad_value, dtype=dtype)
        padded_data[:, :src.height, :src.width] = data

        # Update profile
        profile = src.profile.copy()
        profile.update({
            "width": new_width,
            "height": new_height,
            "nodata": None,  # You can also set to pad_value if needed
            "dtype": dtype
        })

        with rasterio.open(padded_raster_path, "w", **profile) as dst:
            dst.write(padded_data)

    return padded_raster_path



def create_tiles(raster_path, output_folder, tile_size_meters):
    """
    Cuts a raster into tiles of a given geographic size.
    Ensures tiles cover the exact same spatial area.
    """
    os.makedirs(output_folder, exist_ok=True)

    padded_raster_path = raster_path.replace(".tif", "_padded.tif")
    padded_raster_path = pad_raster_to_tile_size(raster_path, padded_raster_path, tile_size_meters)

    with rasterio.open(padded_raster_path) as src:
        transform = src.transform
        pixel_size_x, pixel_size_y = transform.a, -transform.e  # Get pixel size
        tile_size_x = int(tile_size_meters / pixel_size_x)
        tile_size_y = int(tile_size_meters / pixel_size_y)

        num_tiles_x = src.width // tile_size_x
        num_tiles_y = src.height // tile_size_y

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                left = transform.c + (i * tile_size_x * pixel_size_x)
                top = transform.f - (j * tile_size_y * pixel_size_y)
                right = left + (tile_size_x * pixel_size_x)
                bottom = top - (tile_size_y * pixel_size_y)

                window = rasterio.windows.from_bounds(left, bottom, right, top, transform)
                tile = src.read(window=window)

                profile = src.profile.copy()
                profile.update({
                    "width": window.width,
                    "height": window.height,
                    "transform": rasterio.windows.transform(window, transform)
                })

                output_tile_path = f"{output_folder}/tile_{i}_{j}.tif"
                with rasterio.open(output_tile_path, "w", **profile) as dst:
                    dst.write(tile)

                print(f"Saved tile: {output_tile_path}")

def normalize_image(image):
    """Normalize image to [0, 1] range for correct visualization."""
    image = image.astype(np.float32)
    min_val = np.percentile(image, 2)  # Clip out extreme values
    max_val = np.percentile(image, 98)
    # Avoid division by zero if max_val == min_val
    if max_val - min_val == 0:
        return np.zeros_like(image)  # or return image
    
    image = np.clip((image - min_val) / (max_val - min_val), 0, 1)
    return image

def upscale_image(image, target_size):
    """Upscale small Sentinel image to match drone tile size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)  # Nearest-neighbor keeps original pixels



def plot_all_tiles_with_labels(drone_folder, sentinel_folder, labels_folder, output_folder):
    """
    Plots side-by-side comparisons for all matching Sentinel-2, Drone, and Label tiles.

    Args:
        drone_folder (str): Path to the folder containing drone tiles.
        sentinel_folder (str): Path to the folder containing sentinel tiles.
        labels_folder (str): Path to the folder containing label tiles.
        output_folder (str): Path to save the comparison PNGs.
    """
    os.makedirs(output_folder, exist_ok=True)

    drone_tiles = sorted(glob.glob(os.path.join(drone_folder, "*.tif")))
    sentinel_tiles = sorted(glob.glob(os.path.join(sentinel_folder, "*.tif")))
    label_tiles = sorted(glob.glob(os.path.join(labels_folder, "*.tif")))

    for drone_path, sentinel_path, label_path in zip(drone_tiles, sentinel_tiles, label_tiles):
        tile_name = os.path.basename(drone_path).replace(".tif", "")

        with rasterio.open(sentinel_path) as src:
            sentinel_data = src.read(sentinel_bands)
            sentinel_data = np.moveaxis(sentinel_data, 0, -1)  # Convert to (H, W, C)
            sentinel_data = normalize_image(sentinel_data)

        with rasterio.open(drone_path) as src:
            drone_data = src.read([1, 2, 3])  # Use first 3 bands as RGB
            drone_data = np.moveaxis(drone_data, 0, -1)  # Convert to (H, W, C)
            drone_data = normalize_image(drone_data)

        with rasterio.open(label_path) as src:
            labels_data = src.read(1)  # Assuming single-band label raster
            labels_data = normalize_image(labels_data)  # Normalize for visualization

        # Upscale Sentinel-2 Image and Labels to match Drone size
        upscaled_sentinel = upscale_image(sentinel_data, (drone_data.shape[1], drone_data.shape[0]))
        upscaled_labels = upscale_image(labels_data, (drone_data.shape[1], drone_data.shape[0]))

        # Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(upscaled_sentinel)
        axes[0].set_title(f"Sentinel-2: {tile_name}")
        axes[0].axis("off")

        axes[1].imshow(drone_data)
        axes[1].set_title(f"Drone: {tile_name}")
        axes[1].axis("off")

        axes[2].imshow(upscaled_labels, cmap="gray")
        axes[2].set_title(f"Labels: {tile_name}")
        axes[2].axis("off")

        plt.tight_layout()

        output_png_path = os.path.join(output_folder, f"{tile_name}.png")
        plt.savefig(output_png_path, dpi=300)
        plt.close(fig)
        
        print(f"Saved comparison: {output_png_path}")


def plot_all_tiles(drone_folder, sentinel_folder, output_folder):
    """
    Plots side-by-side comparisons for all matching Sentinel-2 and Drone tiles.

    Args:
        drone_folder (str): Path to the folder containing drone tiles.
        sentinel_folder (str): Path to the folder containing sentinel tiles.
        output_folder (str): Path to save the comparison PNGs.
    """
    os.makedirs(output_folder, exist_ok=True)

    drone_tiles = sorted(glob.glob(os.path.join(drone_folder, "*.tif")))
    sentinel_tiles = sorted(glob.glob(os.path.join(sentinel_folder, "*.tif")))

    for drone_path, sentinel_path in zip(drone_tiles, sentinel_tiles):
        tile_name = os.path.basename(drone_path).replace(".tif", "")

        with rasterio.open(sentinel_path) as src:
            sentinel_data = src.read(sentinel_bands)
            sentinel_data = np.moveaxis(sentinel_data, 0, -1)  # Convert to (H, W, C)
            sentinel_data = normalize_image(sentinel_data)

        # Load Drone Image (4-band RGBA)
        with rasterio.open(drone_path) as src:
            drone_data = src.read([1, 2, 3])  # Use first 3 bands as RGB
            drone_data = np.moveaxis(drone_data, 0, -1)  # Convert to (H, W, C)
            drone_data = normalize_image(drone_data)

        # Upscale Sentinel-2 Image to match Drone size
        upscaled_sentinel = upscale_image(sentinel_data, (drone_data.shape[1], drone_data.shape[0]))

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(upscaled_sentinel)
        axes[0].set_title(f"Sentinel-2: {tile_name}")
        axes[0].axis("off")

        axes[1].imshow(drone_data)
        axes[1].set_title(f"Drone: {tile_name}")
        axes[1].axis("off")

        plt.tight_layout()

        output_png_path = os.path.join(output_folder, f"{tile_name}.png")
        plt.savefig(output_png_path, dpi=300)
        plt.close(fig)
        
        print(f"Saved comparison: {output_png_path}")



def plot_all_tiles_grid(drone_folder, sentinel_folder):
    """
    Plots a grid of all Sentinel-2 and Drone tiles, ensuring corresponding pairs are displayed together.
    Also checks if tiles cover the same geographic extent.
    """
    drone_tiles = sorted(glob.glob(os.path.join(drone_folder, "*.tif")))
    sentinel_tiles = sorted(glob.glob(os.path.join(sentinel_folder, "*.tif")))

    num_tiles = min(len(drone_tiles), len(sentinel_tiles))  # Ensure equal tile count
    fig, axes = plt.subplots(num_tiles, 2, figsize=(10, 5 * num_tiles))  # 2 columns: Sentinel & Drone
    if num_tiles == 1:
        axes = [axes]  

    for i, (drone_path, sentinel_path) in enumerate(zip(drone_tiles, sentinel_tiles)):
        tile_name = os.path.basename(drone_path).replace(".tif", "")
        
        # Load Sentinel-2 Image 
        with rasterio.open(sentinel_path) as src:
            sentinel_data = src.read(sentinel_bands)  # Assume RGB bands
            sentinel_data = np.moveaxis(sentinel_data, 0, -1)  # Convert to (H, W, C)
            sentinel_data = normalize_image(sentinel_data)

        # Load Drone Image 
        with rasterio.open(drone_path) as src:
            drone_data = src.read([1, 2, 3])  
            drone_data = np.moveaxis(drone_data, 0, -1)  # Convert to (H, W, C)
            drone_data = normalize_image(drone_data)

        # Upscale Sentinel-2 Image to match Drone size
        upscaled_sentinel = upscale_image(sentinel_data, (drone_data.shape[1], drone_data.shape[0]))

        axes[i][0].imshow(upscaled_sentinel)
        axes[i][0].set_title(f"Sentinel-2: {tile_name}")
        axes[i][0].axis("off")

        axes[i][1].imshow(drone_data)
        axes[i][1].set_title(f"Drone: {tile_name}")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig("all_tiles_plot.png")
    plt.close(fig)


def plot_reconstructed_image(tile_folder, tile_size_meters, pixel_size, output_path=None):
    """
    Reconstructs and plots all tiles of an image by stitching them back together.
    
    Args:
        tile_folder (str): Path to the folder containing tiles.
        tile_size_meters (int): The real-world size of each tile (e.g., 100m, 200m).
        pixel_size (float): Pixel size in meters (e.g., 10m for Sentinel, ~1m for Drone).
        output_path (str, optional): Path to save the reconstructed image.
    """
    tile_paths = sorted(glob.glob(os.path.join(tile_folder, "*.tif")))

    # Extract tile indices from filenames (assuming format: tile_X_Y.tif)
    tile_info = []
    for path in tile_paths:
        filename = os.path.basename(path)
        parts = filename.replace(".tif", "").split("_")
        try:
            x_idx, y_idx = int(parts[1]), int(parts[2])
            tile_info.append((x_idx, y_idx, path))
        except ValueError:
            print(f"Invalid file: {filename}")
    
    tile_info.sort(key=lambda x: (x[1], x[0]))

    num_tiles_x = max(x for x, _, _ in tile_info) + 1
    num_tiles_y = max(y for _, y, _ in tile_info) + 1

    # Read the first tile to determine expected tile dimensions
    with rasterio.open(tile_info[0][2]) as src:
        expected_tile_height, expected_tile_width = src.shape

    # Create an empty array for the full reconstructed image (initially zeros)
    full_height = num_tiles_y * expected_tile_height
    full_width = num_tiles_x * expected_tile_width
    full_image = np.zeros((full_height, full_width), dtype=np.float32)

    for x_idx, y_idx, path in tile_info:
        with rasterio.open(path) as src:
            tile_data = src.read(1)  # Assuming single-band grayscale
            tile_height, tile_width = tile_data.shape  # Get actual tile size

        # Compute placement in the final array
        start_x, start_y = x_idx * expected_tile_width, y_idx * expected_tile_height

        # Adjust for tiles that are smaller than expected
        full_image[start_y:start_y + tile_height, start_x:start_x + tile_width] = tile_data[:tile_height, :tile_width]

    # Plot the reconstructed image
    plt.figure(figsize=(12, 8))
    plt.imshow(full_image, cmap='gray')
    plt.title(f"Reconstructed Image ({num_tiles_x}x{num_tiles_y} tiles)")
    plt.axis("off")

    # Save the reconstructed image 
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved reconstructed image to {output_path}")


def print_raster_info(raster_name):
    with rasterio.open(raster_name) as src:
        print(f"CRS: {src.crs}")
        print(f"Pixel Size: {src.res}")  
        print(f"Image Size: {src.width} x {src.height}")



def has_small_invalid_white_regions(tile_path, max_valid_white_area=3000): 
    """
    Checks if the image contains small irregular white regions,
    ignoring large white areas like rivers or masks.

    Args:
        tile_path (str): Path to the image tile (.tif).
        max_valid_white_area (int): Maximum area (in pixels) to consider a white region as "small".
    
    Returns:
        bool: True if small white regions are found, False otherwise.
    """
    with rasterio.open(tile_path) as src:
        rgb = src.read([1, 2, 3])  # Shape: (3, H, W)

    # Step 1: Detect strictly white pixels
    white_mask = np.all(rgb == 255, axis=0).astype(np.uint8)

    # Step 2: Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

    # Step 3: Loop through components, skip background (label 0)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_valid_white_area:
            return True  # Found a small white region

    return False  # No small white regions found


def is_mostly_white(tile_path, threshold=0.9):
    with rasterio.open(tile_path) as src:
        rgb = src.read([1, 2, 3])  # Shape: (3, H, W)

    # Convert from (3, H, W) to (H, W, 3) for OpenCV
    rgb_img = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # Define white pixels: grayscale value > 250
    white_pixels = np.sum(gray == 255)
    total_pixels = gray.size

    white_ratio = white_pixels / total_pixels
    return white_ratio >= threshold


def save_valid_tiles_to_csv(drone_tiles_folder, invalid_pixels_area_size, valid_csv_path, invalid_csv_path):
    tile_paths = sorted(glob.glob(os.path.join(drone_tiles_folder, "*.tif")))
    valid_tiles = []
    invalid_tiles = []

    for tile_path in tile_paths:
        has_invalid_white = has_small_invalid_white_regions(tile_path, invalid_pixels_area_size)
        mostly_white = is_mostly_white(tile_path, threshold=0.8)

        if not has_invalid_white and not mostly_white:
            valid_tiles.append(tile_path)
            print(f"VALID: {tile_path}")
        else:
            reason = []
            if has_invalid_white:
                reason.append("small white patches")
            if mostly_white:
                reason.append("mostly white")

            invalid_tiles.append((tile_path, "; ".join(reason)))
            print(f"INVALID ({', '.join(reason)}): {tile_path}")

    with open(valid_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["valid_tile_path"])
        for path in valid_tiles:
            writer.writerow([path])

    with open(invalid_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["invalid_tile_path", "reason"])
        for path, reason in invalid_tiles:
            writer.writerow([path, reason])


    print(f"\n Saved {len(valid_tiles)} valid tile paths to: {valid_csv_path}")
    print(f"\n Saved {len(invalid_tiles)} invalid tile paths to: {invalid_csv_path}")



def merge_valid_csvs(csv_folder, merged_csv_path):
    csv_files = glob.glob(os.path.join(csv_folder, "*valid*.csv"))
    dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "valid_tile_path" in df.columns:
            dfs.append(df[["valid_tile_path"]])

    merged_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged {len(csv_files)} CSVs into: {merged_csv_path} with {len(merged_df)} unique tiles.")
    return merged_df


def plot_tiles_to_pdf(valid_tile_df, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for idx, row in valid_tile_df.iterrows():
            tile_path = row["valid_tile_path"]
            try:
                with rasterio.open(tile_path) as src:
                    rgb = src.read([1, 2, 3])  # shape: (3, H, W)
                    rgb_img = np.transpose(rgb, (1, 2, 0))  # to (H, W, 3)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(rgb_img)
                ax.set_title(f"Tile: {tile_path}")
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to plot {tile_path}: {e}")
    print(f"Saved PDF report with {len(valid_tile_df)} tiles to: {pdf_path}")


def copy_valid_files_to_folder(csv_file_path, destination_folder):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Read filenames from CSV and copy each file
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header_skipped = False
        for row in reader:
            # Skip header if necessary
            if not header_skipped and not os.path.exists(row[0]):
                header_skipped = True
                continue

            filepath = row[0]
            
            if os.path.exists(filepath):
                basename = os.path.basename(filepath)
                parent_folder = os.path.basename(os.path.dirname(filepath))
                # Construct new filename using basename + parent folder
                name, ext = os.path.splitext(basename)
                new_filename = f"{name}_{parent_folder}{ext}"
                dst_path = os.path.join(destination_folder, new_filename)

                shutil.copy2(filepath, dst_path)
                print(f"Copied: {filepath} → {new_filename}")
            else:
                print(f"File not found: {filepath}")


def copy_valid_files_to_folder(csv_file_path, s2_parent_folder, destination_folder):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Read filenames from CSV and copy each file
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header_skipped = False
        for row in reader:
            # Skip header if necessary
            if not header_skipped and not os.path.exists(row[0]):
                header_skipped = True
                continue

            filepath = row[0]
            
            if os.path.exists(filepath):
                basename = os.path.basename(filepath)
                print(basename)
                drone_parent_folder = os.path.basename(os.path.dirname(filepath))
                # Construct new filename using basename + parent folder
                name, ext = os.path.splitext(basename)
                new_filename = f"{name}_{drone_parent_folder}{ext}"
                dst_path = os.path.join(destination_folder, new_filename)

                shutil.copy2(filepath, dst_path)


                filepath_s2 = os.path.join(s2_parent_folder, basename )
                new_filename_s2 = f"{name}_{s2_parent_folder}{ext}"
                dst_path = os.path.join(destination_folder, new_filename_s2)
                shutil.copy2(filepath_s2, dst_path)

                print(f"Copied: {filepath} → {new_filename}")
            else:
                print(f"File not found: {filepath}")

def main():
    # target tile size (in meters)
    tile_size = 400

    invalid_pixels_area_size = 10000

    # Paths to input raster files
    sentinel_raster_path = "sentinel/sentinel-dl/data/stacked_outputs/S2_T33TUL_20240825T100551_stack_10m.tif"
    print_raster_info(sentinel_raster_path)
    s2identifier =  os.path.basename(sentinel_raster_path)[:-4]
    
    drone_raster_path = "orthomosaics/20242808_10cmGSD_orthomosaic.tif"
   
    identifier = os.path.basename(drone_raster_path)[:-4]


    # Target resolution for the drone image (in meters)
    target_drone_res = 0.1
    # Target resolution for the satellite image (in meters)
    target_s2_res = 10

    drone_reproj = f"{identifier}_reprojected.tif"
    reproject_raster_to_meters(drone_raster_path, drone_reproj)
    #print_raster_info(drone_raster)
    
    sentinel_resampled = f"{s2identifier}_rescaled_sentinel_10m.tif"
    resample_raster(sentinel_raster_path, sentinel_resampled, target_s2_res)

    sentinel_clipped = f"{s2identifier}_clipped.tif"

    clip_raster_and_save(sentinel_resampled, sentinel_clipped, drone_reproj)
    #print_raster_info(sentinel_clipped)

    sentinel_tiles_folder = "sentinel_tiles_{s2identifier}" # output folder for generated sentinel image tiles
    drone_tiles_folder = f"drone_tiles_{identifier}" # output folder for generated drone image tiles
        
    create_tiles(sentinel_clipped, sentinel_tiles_folder, tile_size)
    create_tiles(drone_reproj, drone_tiles_folder, tile_size)


    valid_results_folder = "validity_results"
    os.makedirs(valid_results_folder, exist_ok=True)

    save_valid_tiles_to_csv(drone_tiles_folder, invalid_pixels_area_size, f"{valid_results_folder}/valid_tiles_list_{identifier}.csv", f"{valid_results_folder}/invalid_tiles_list_{identifier}.csv" )
        
    copy_valid_files_to_folder(csv_file_path = f"{valid_results_folder}/valid_tiles_list_{identifier}.csv", s2_parent_folder = sentinel_tiles_folder, destination_folder = f"{identifier}_valid_tiles")
   
   

if __name__ == "__main__":
    main()
    