import rasterio
from rasterio.enums import Resampling
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np



def save_rgb_preview(tif_path, band_files, preview_folder):
    os.makedirs(preview_folder, exist_ok=True)
    
    # Map bands to indices
    band_indices = {}
    for idx, f in enumerate(band_files, start=1):
        name = os.path.basename(f)
        if 'B02' in name:
            band_indices['blue'] = idx
        elif 'B03' in name:
            band_indices['green'] = idx
        elif 'B04' in name:
            band_indices['red'] = idx

    if not all(b in band_indices for b in ['red', 'green', 'blue']):
        print(" Skipping preview: Missing RGB bands.")
        return

    with rasterio.open(tif_path) as src:
        r = src.read(band_indices['red'])
        g = src.read(band_indices['green'])
        b = src.read(band_indices['blue'])

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    # Normalize to [0, 1] for display
    rgb /= np.percentile(rgb, 98)  # Stretch contrast a bit
    rgb = np.clip(rgb, 0, 1)

    # Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(os.path.basename(tif_path))
    plt.axis("off")

    preview_path = os.path.join(preview_folder, os.path.splitext(os.path.basename(tif_path))[0] + "_RGB_preview.png")
    plt.savefig(preview_path, bbox_inches='tight')
    plt.close()
    print(f" RGB preview saved: {preview_path}")




# Input root where .SAFE folders are located
products_root = "data/products/"
# Output folder for stacked TIFFs
output_folder = "data/stacked_outputs/"
os.makedirs(output_folder, exist_ok=True)

# Find all .SAFE folders
#safe_folders = sorted(glob.glob(os.path.join(products_root, "*.SAFE")))
#print(f"Found {len(safe_folders)} SAFE products.")

#for safe_path in safe_folders:
safe_path = "data/products/S2B_MSIL2A_20240830T100559_N0511_R022_T33TVM_20240830T134009.SAFE"
# Find IMG_DATA folders inside each SAFE
img_data_dirs = glob.glob(os.path.join(safe_path, "GRANULE/*/IMG_DATA/"))
print(img_data_dirs)

for img_data_path in img_data_dirs:
    print(f"\nProcessing IMG_DATA: {img_data_path}")

    # Get all JP2 files inside IMG_DATA subdirs (10m, 20m, 60m)
    jp2_files = sorted(glob.glob(os.path.join(img_data_path, "**/*.jp2"), recursive=True))

    # Filter relevant bands
    band_files = [f for f in jp2_files if 'B0' in os.path.basename(f)]
    band_files.sort()

    if not band_files:
        print(f"No B0* band files found in {img_data_path}")
        continue

    # Extract info from filename
    sample_name = os.path.basename(band_files[0])
    match = re.search(r'(T[0-9A-Z]{5})_(\d{8}T\d{6})', sample_name)
    tile_id, timestamp = match.groups() if match else ('TILE', 'DATE')

    output_name = f"S2_{tile_id}_{timestamp}_stack_10m.tif"
    output_path = os.path.join(output_folder, output_name)

    # Use a 10m band as reference
    ref_band = next((f for f in band_files if '_10m' in f), None)
    if not ref_band:
        print(f"No 10m resolution band found in {img_data_path}. Skipping.")
        continue

    with rasterio.open(ref_band) as ref:
        ref_meta = ref.meta.copy()
        ref_height = ref.height
        ref_width = ref.width
        ref_transform = ref.transform
        ref_crs = ref.crs

    ref_meta.update(
        count=len(band_files),
        driver="GTiff",
        height=ref_height,
        width=ref_width,
        transform=ref_transform,
        crs=ref_crs
    )

    # Write stacked bands
    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        for idx, f in enumerate(band_files, start=1):
            with rasterio.open(f) as src:
                if src.width != ref_width or src.height != ref_height:
                    data = src.read(
                        1,
                        out_shape=(ref_height, ref_width),
                        resampling=Resampling.bilinear
                    )
                else:
                    data = src.read(1)
                dst.write(data, idx)
                print(f"Band {idx}: {os.path.basename(f)} written.")

    print(f"Output saved as: {output_path}")

    # Generate and save RGB preview
    save_rgb_preview(output_path, band_files, preview_folder="data/preview_images/")


