import rasterio
from rasterio.enums import Resampling
import glob
import os
import re
import zipfile
import os

# Path to .SAFE/GRANULE/.../IMG_DATA
img_data_path = "data/products/S2B_MSIL2A_20250206T101109_N0511_R022_T33TUM_20250206T122414.SAFE/GRANULE/L2A_T33TUM_A041375_20250206T101112/IMG_DATA/"

print(img_data_path)
# Get all jp2 files inside R60m, R20m, and R10m subdirectories
jp2_files = sorted(glob.glob(os.path.join(img_data_path, "*/**/*.jp2"), recursive=True))

print("JP2 files found:", jp2_files)

band_files = [f for f in jp2_files if 'B0' in os.path.basename(f)]
print(band_files)
band_files.sort()

# Extract info from first filename
sample_name = os.path.basename(band_files[0])
match = re.search(r'(T[0-9A-Z]{5})_(\d{8}T\d{6})', sample_name)
tile_id, timestamp = match.groups() if match else ('TILE', 'DATE')

# Create output filename
output_name = f"S2_{tile_id}_{timestamp}_stack_10m.tif"

# Use first 10m band as reference
with rasterio.open([f for f in band_files if '_10m' in f][0]) as ref:
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

# Write output
with rasterio.open(output_name, 'w', **ref_meta) as dst:
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

print(f"\n Output saved as: {output_name}")
