import pandas as pd
import rasterio
from rasterio.warp import transform
import numpy as np
from pathlib import Path


base_dir = Path(__file__).resolve().parent


data_dir = base_dir / "data"
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create results folder if it does not already exist


input_excel = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area.xlsx"
output_excel = results_dir / (input_excel.stem + "_classifications" + input_excel.suffix)
tif_path = data_dir / "gwa3_250_windspeed_100m.tif"

iec_mapping = {
    0: "1A+",
    1: "1A",
    2: "1B",
    3: "1C",
    4: "2A+",
    5: "2A",
    6: "2B",
    7: "2C",
    8: "3A+",
    9: "3A",
    10: "3B",
    11: "3C",
    12: "S"
}

def extract_numeric_class(iec_class):
    if iec_class and iec_class[0].isdigit():
        return int(iec_class[0])
    return np.nan

# Read the Excel file from the input path.
df = pd.read_excel(input_excel)

# Create a list of coordinate pairs from the "Longitude" and "Latitude" columns.
coords = list(zip(df["Longitude"], df["Latitude"]))

# Open the TIFF file and sample its values at the coordinate locations.
with rasterio.open(tif_path) as src:
    raster_crs = src.crs
    if raster_crs and raster_crs.to_string() != "EPSG:4326":
        lons, lats = zip(*coords)
        xs, ys = transform("EPSG:4326", raster_crs, lons, lats)
        sample_coords = list(zip(xs, ys))
    else:
        sample_coords = coords
    sampled_values = list(src.sample(sample_coords))

# Extract pixel values and map them to IEC classes.
pixel_values = [int(arr[0]) if arr[0] is not None else None for arr in sampled_values]
iec_classifications = [iec_mapping.get(val) for val in pixel_values]
iec_class_numeric = [extract_numeric_class(cl) if cl is not None else np.nan for cl in iec_classifications]

df["IEC_Class"] = iec_classifications
df["IEC_Class_Num"] = iec_class_numeric

df.to_excel(output_excel, index=False)
print("Final data saved to:", output_excel)
