import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
import concurrent.futures
import math

# File paths
input_excel = r"D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_final_2.xlsx"
output_excel = r"D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_with_IEC_Elevation_v2.xlsx"
dem_file = r"D:\eurodem.tif"

# Define European bounding box and convert to arcseconds
europe_bounds_deg = (-20, 35, 40, 70)
europe_bounds_arcsec = tuple(bound * 3600 for bound in europe_bounds_deg)

with rasterio.open(dem_file) as src:
    out_trans = src.transform
    window = from_bounds(europe_bounds_arcsec[0], europe_bounds_arcsec[1],
                         europe_bounds_arcsec[2], europe_bounds_arcsec[3],
                         transform=out_trans)
    elev_europe = src.read(1, window=window)
    new_transform = rasterio.windows.transform(window, out_trans)

c_height, c_width = elev_europe.shape
dem_left, dem_top = new_transform * (0, 0)
dem_right, dem_bottom = new_transform * (c_width, c_height)
extent_europe = (dem_left / 3600, dem_right / 3600, dem_bottom / 3600, dem_top / 3600)

# Compute pixel spacing in meters
pixel_width_deg = abs(new_transform.a) / 3600
pixel_height_deg = abs(new_transform.e) / 3600
pixel_height_m = pixel_height_deg * 111320
center_lat = (extent_europe[2] + extent_europe[3]) / 2.0
pixel_width_m = pixel_width_deg * 111320 * math.cos(math.radians(center_lat))

# Process wind turbine data
df = pd.read_excel(input_excel)
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude_arcsec"] = df["Longitude"] * 3600
df["Latitude_arcsec"] = df["Latitude"] * 3600

df["in_dem"] = (
    (df["Longitude_arcsec"] >= dem_left) &
    (df["Longitude_arcsec"] <= dem_right) &
    (df["Latitude_arcsec"] >= dem_bottom) &
    (df["Latitude_arcsec"] <= dem_top)
)

valid = df["in_dem"]
rows, cols = rowcol(new_transform,
                    df.loc[valid, "Longitude_arcsec"].values,
                    df.loc[valid, "Latitude_arcsec"].values)
df.loc[valid, "row"] = rows
df.loc[valid, "col"] = cols

# Compute TRI and slope using a 5x5 window
half_window_x = 2
half_window_y = 2
tri_threshold = 3.0
slope_threshold = 5.0

def compute_local_metrics(idx):
    row_pix, col_pix = idx
    row_min = max(row_pix - half_window_y, 0)
    row_max = min(row_pix + half_window_y + 1, c_height)
    col_min = max(col_pix - half_window_x, 0)
    col_max = min(col_pix + half_window_x + 1, c_width)
    window_vals = elev_europe[row_min:row_max, col_min:col_max]
    if window_vals.size == 0:
        return np.nan, np.nan
    center_index = (window_vals.shape[0] // 2, window_vals.shape[1] // 2)
    center_val = window_vals[center_index]
    tri = np.mean(np.abs(window_vals - center_val))
    if window_vals.shape[0] > 1 and window_vals.shape[1] > 1:
        grad_y, grad_x = np.gradient(window_vals, pixel_height_m, pixel_width_m)
        grad_center_y = grad_y[center_index]
        grad_center_x = grad_x[center_index]
        slope_radians = np.arctan(np.sqrt(grad_center_x**2 + grad_center_y**2))
        slope_deg = np.degrees(slope_radians)
    else:
        slope_deg = np.nan
    return tri, slope_deg

indices_valid = list(zip(df.loc[valid, "row"].astype(int),
                         df.loc[valid, "col"].astype(int)))
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(compute_local_metrics, indices_valid))

tri_list, slope_list = zip(*results)
df.loc[valid, "TRI"] = tri_list
df.loc[valid, "Slope_deg"] = slope_list

df.loc[valid, "Terrain_Type"] = np.where(
    (df.loc[valid, "TRI"] > tri_threshold) | (df.loc[valid, "Slope_deg"] > slope_threshold),
    "Complex", "Flat"
)
df.loc[~valid, "TRI"] = 0.0
df.loc[~valid, "Slope_deg"] = 0.0
df.loc[~valid, "Terrain_Type"] = "Flat"

df.drop(columns=["row", "col", "Longitude_arcsec", "Latitude_arcsec", "in_dem"], inplace=True)
df.to_excel(output_excel, index=False)
