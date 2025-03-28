import pandas as pd
import rasterio
from rasterio.warp import transform
import numpy as np
import os

input_excel = r"D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_with_IEC_Elevation_v2_area.xlsx"
base, ext = os.path.splitext(input_excel)
output_excel = base + "_classifications" + ext
tif_path = r"D:\gwa3_250_windspeed_100m.tif"

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

df = pd.read_excel(input_excel)
coords = list(zip(df["Longitude"], df["Latitude"]))

with rasterio.open(tif_path) as src:
    raster_crs = src.crs
    if raster_crs and raster_crs.to_string() != "EPSG:4326":
        lons, lats = zip(*coords)
        xs, ys = transform("EPSG:4326", raster_crs, lons, lats)
        sample_coords = list(zip(xs, ys))
    else:
        sample_coords = coords
    sampled_values = list(src.sample(sample_coords))

pixel_values = [int(arr[0]) if arr[0] is not None else None for arr in sampled_values]
iec_classifications = [iec_mapping.get(val) for val in pixel_values]
iec_class_numeric = [extract_numeric_class(cl) if cl is not None else np.nan for cl in iec_classifications]

df["IEC_Class"] = iec_classifications
df["IEC_Class_Num"] = iec_class_numeric

df.to_excel(output_excel, index=False)
