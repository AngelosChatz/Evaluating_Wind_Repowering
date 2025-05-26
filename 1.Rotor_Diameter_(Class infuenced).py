import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import rasterio
from rasterio.warp import transform
from pathlib import Path


base_dir = Path(__file__).resolve().parent


data_dir = base_dir / "data"
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create results folder if it does not exist.

# Define the input and output file names.
input_file = "Windfarms_World_20230530.xlsx"
output_file = "Windfarms_World_20230530_final_1.xlsx"  # Final output

# Construct full paths using the relative directories.
full_input_path = data_dir / input_file
full_output_path = results_dir / output_file


tif_file = "gwa3_250_windspeed_100m.tif"  # IEC classification TIFF
tif_path = data_dir / tif_file

# Read the Excel file from the relative path.
df = pd.read_excel(full_input_path, sheet_name=2, header=0)
df = df.iloc[:, :27]
df.columns = [
    "ID", "Continent", "ISO code", "Country", "State code", "Area", "City", "Name", "2nd name",
    "Latitude", "Longitude", "Altitude/Depth", "Location accuracy", "Offshore", "Manufacturer",
    "Turbine", "Hub height", "Number of turbines", "Total power", "Developer", "Operator", "Owner",
    "Commissioning date", "Status", "Decommissioning date", "Link", "Update"
]

# Filter for onshore turbines in Europe.
df["Offshore"] = df["Offshore"].astype(str).str.strip().str.lower()
df = df[df["Offshore"] == "no"].copy()
df["Continent"] = df["Continent"].astype(str).str.strip().str.lower()
df = df[df["Continent"] == "europe"].copy()

# Convert coordinates to numeric; drop rows with invalid entries.
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df = df.dropna(subset=["Latitude", "Longitude"]).copy()

# Extract Rotor Diameter based on patterns per manufacturer.
def extract_rotor_diameter(turbine_str, manufacturer):
    if not isinstance(turbine_str, str):
        return None
    turbine_str = turbine_str.strip()
    manufacturer = manufacturer.strip()

    if manufacturer in ["Gamesa", "Nordex", "Vestas", "Enercon"]:
        try:
            first_chunk = turbine_str.split('/')[0]
            rotor_str = first_chunk[1:]
            return float(rotor_str)
        except:
            return None

    if manufacturer in ["Siemens", "Siemens-Gamesa"]:
        try:
            matches = re.findall(r"-\s*(\d+(\.\d+)?)", turbine_str)
            if matches:
                return float(matches[-1][0])
            else:
                return None
        except:
            return None

    if manufacturer == "GE Energy":
        try:
            matches = re.findall(r"-\s*(\d+(\.\d+)?)", turbine_str)
            if matches:
                return float(matches[-1][0])
            else:
                return None
        except:
            return None

    if manufacturer == "Senvion":
        try:
            if turbine_str.startswith("MM"):
                slash_pos = turbine_str.find('/')
                rotor_str = turbine_str[2:slash_pos] if slash_pos != -1 else turbine_str[2:]
                return float(rotor_str)
            else:
                match = re.search(r"M(\d+(\.\d+)?)(\D|$)", turbine_str)
                if match:
                    return float(match.group(1))
                else:
                    return None
        except:
            return None

    return None

df["Rotor Diameter"] = df.apply(lambda row: extract_rotor_diameter(row["Turbine"], row["Manufacturer"]), axis=1)
missing_pct = df["Rotor Diameter"].isna().mean() * 100
print(f"Percentage of missing values in Rotor Diameter (onshore, Europe): {missing_pct:.2f}%")

# Compute Single WT Capacity.
df["Total power"] = pd.to_numeric(df["Total power"], errors='coerce')
df["Number of turbines"] = pd.to_numeric(df["Number of turbines"], errors='coerce')
df = df.dropna(subset=["Total power", "Number of turbines"]).copy()
df["SingleWT_Capacity"] = df["Total power"] / df["Number of turbines"]

# Add IEC Classification from TIFF File.
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
    else:
        return np.nan

# Prepare coordinates for raster sampling (rasterio expects (lon, lat)).
coords = list(zip(df["Longitude"], df["Latitude"]))
with rasterio.open(tif_path) as src:
    print("Raster CRS:", src.crs)
    print("Raster Bounds:", src.bounds)
    raster_crs = src.crs
    if raster_crs is not None and raster_crs.to_string() != "EPSG:4326":
        lons, lats = zip(*coords)
        xs, ys = transform("EPSG:4326", raster_crs, lons, lats)
        sample_coords = list(zip(xs, ys))
    else:
        sample_coords = coords
    sampled_values = list(src.sample(sample_coords))

pixel_values = [int(arr[0]) if arr[0] is not None else None for arr in sampled_values]
iec_classifications = [iec_mapping.get(val, None) for val in pixel_values]
df["IEC_Class"] = iec_classifications
df["IEC_Class_Num"] = [extract_numeric_class(cl) if cl is not None else np.nan for cl in iec_classifications]
df["IEC_Class_Group"] = df["IEC_Class_Num"].apply(lambda x: f"Class {int(x)}" if not np.isnan(x) else "S")

# Per-Class Log–Log Regression.
df_reg = df.dropna(subset=["Rotor Diameter", "SingleWT_Capacity"]).copy()
df_reg = df_reg[df_reg["SingleWT_Capacity"] > 0]

# Define overall x-range to extend all regression lines equally.
overall_x_min = df_reg["SingleWT_Capacity"].min()
overall_x_max = df_reg["SingleWT_Capacity"].max()

groups = [grp for grp in ["Class 1", "Class 2", "Class 3", "S"] if grp in df_reg["IEC_Class_Group"].unique()]
regression_params = {}
group_colors = {"Class 1": "blue", "Class 2": "green", "Class 3": "red", "S": "orange"}

plt.figure(figsize=(10, 8))
for grp in groups:
    sub_df = df_reg[df_reg["IEC_Class_Group"] == grp]
    X = sub_df["SingleWT_Capacity"].values
    y = sub_df["Rotor Diameter"].values
    if len(X) < 2:
        print(f"Not enough data for regression in group {grp}.")
        continue
    # Perform log–log regression.
    X_log = np.log10(X)
    y_log = np.log10(y)
    slope, intercept = np.polyfit(X_log, y_log, 1)
    regression_params[grp] = (intercept, slope)

    # Compute error metrics in log space.
    y_log_pred = intercept + slope * X_log
    rmse = np.sqrt(np.mean((y_log - y_log_pred) ** 2))
    r2 = 1 - np.sum((y_log - y_log_pred) ** 2) / np.sum((y_log - np.mean(y_log)) ** 2)
    print(f"{grp} regression: intercept = {intercept:.2f}, slope = {slope:.2f}, RMSE = {rmse:.2f}, R² = {r2:.2f}")

    # Use a common x-range for the fit line.
    X_fit = np.linspace(overall_x_min, overall_x_max, 100)
    y_fit = 10 ** (intercept + slope * np.log10(X_fit))

    plt.scatter(X, y, s=20, alpha=0.6, color=group_colors.get(grp))
    plt.plot(X_fit, y_fit, linewidth=2.5, color=group_colors.get(grp),
             label=f"{grp} fit: y=10^({intercept:.2f})·X^({slope:.2f}), RMSE={rmse:.2f}, R²={r2:.2f}")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Single WT Capacity (MW) [log scale]")
plt.ylabel("Rotor Diameter (m) [log scale]")
plt.title("Per-Class Log–Log Regression (with Error Metrics)")
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

# Per-Class Regression on Linear Scale.
plt.figure(figsize=(10, 8))
for grp in groups:
    sub_df = df_reg[df_reg["IEC_Class_Group"] == grp]
    if grp not in regression_params:
        continue
    intercept, slope = regression_params[grp]
    # Use a common x-range for all fits.
    X_fit = np.linspace(overall_x_min, overall_x_max, 100)
    y_fit = 10 ** (intercept + slope * np.log10(X_fit))

    # Plot data and fit.
    plt.scatter(sub_df["SingleWT_Capacity"].values, sub_df["Rotor Diameter"].values,
                s=20, alpha=0.6, color=group_colors.get(grp))
    plt.plot(X_fit, y_fit, linewidth=2.5, color=group_colors.get(grp),
             label=f"{grp} fit: y=10^({intercept:.2f})·X^({slope:.2f})")

plt.xlabel("Single WT Capacity (MW)")
plt.ylabel("Rotor Diameter (m)")
plt.title("Per-Class Regression on Linear Scale")
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Overall Regression Graphs.
X_overall = df_reg["SingleWT_Capacity"].values
y_overall = df_reg["Rotor Diameter"].values
X_log_overall = np.log10(X_overall)
y_log_overall = np.log10(y_overall)
b_overall, a_overall = np.polyfit(X_log_overall, y_log_overall, 1)

# Compute overall error metrics.
y_log_overall_pred = a_overall + b_overall * X_log_overall
rmse_overall = np.sqrt(np.mean((y_log_overall - y_log_overall_pred) ** 2))
r2_overall = 1 - np.sum((y_log_overall - y_log_overall_pred) ** 2) / np.sum(
    (y_log_overall - np.mean(y_log_overall)) ** 2)

# Overall Log–Log Regression Plot.
plt.figure(figsize=(8, 6))
plt.scatter(X_overall, y_overall, s=20, alpha=0.6, label="Data", color="gray")
X_fit_overall = np.linspace(X_overall.min(), X_overall.max(), 100)
y_fit_overall = 10 ** (a_overall + b_overall * np.log10(X_fit_overall))
plt.plot(X_fit_overall, y_fit_overall, linewidth=2.5, color="red",
         label=f"Overall fit: y=10^({a_overall:.2f})·X^({b_overall:.2f}), RMSE={rmse_overall:.2f}, R²={r2_overall:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Single WT Capacity (MW) [log scale]")
plt.ylabel("Rotor Diameter (m) [log scale]")
plt.title("Overall Log–Log Regression")
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Overall Regression Plot on Linear Scale.
plt.figure(figsize=(8, 6))
plt.scatter(X_overall, y_overall, s=20, alpha=0.6, label="Data", color="gray")
plt.plot(X_fit_overall, y_fit_overall, linewidth=2.5, color="red",
         label=f"Overall fit: y=10^({a_overall:.2f})·X^({b_overall:.2f}), RMSE={rmse_overall:.2f}, R²={r2_overall:.2f}")
plt.xlabel("Single WT Capacity (MW)")
plt.ylabel("Rotor Diameter (m)")
plt.title("Overall Regression (Linear Scale)")
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Histograms with Improved Aesthetics.

# Histogram of Predicted Rotor Diameters (Overall Regression).
mask_overall = df["Rotor Diameter"].isna() & df["SingleWT_Capacity"].notna()
X_missing_overall = df.loc[mask_overall, "SingleWT_Capacity"].values
predicted_overall = 10 ** (a_overall + b_overall * np.log10(X_missing_overall))
plt.figure(figsize=(8, 6))
plt.hist(predicted_overall, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Predicted Rotor Diameter (m)")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Rotor Diameters (Overall Regression)")
plt.tight_layout()
plt.show()

# Histogram of Overall Rotor Diameters (Known Values).
plt.figure(figsize=(8, 6))
plt.hist(df_reg["Rotor Diameter"].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Rotor Diameter (m)")
plt.ylabel("Frequency")
plt.title("Overall Distribution of Rotor Diameters")
plt.tight_layout()
plt.show()

# Fill Missing Rotor Diameters Using Class-Specific Regression.
predicted_class = []
mask_class = df["Rotor Diameter"].isna() & df["SingleWT_Capacity"].notna() & df["IEC_Class_Group"].notna()
for idx, row in df.loc[mask_class].iterrows():
    grp = row["IEC_Class_Group"]
    if grp in regression_params:
        cap = row["SingleWT_Capacity"]
        intercept, slope = regression_params[grp]
        predicted_d = 10 ** (intercept + slope * np.log10(cap))
        df.at[idx, "Rotor Diameter"] = predicted_d
        predicted_class.append(predicted_d)

print("Filled missing Rotor Diameter values using class-specific log–log regression.")

plt.figure(figsize=(8, 6))
if predicted_class:
    plt.hist(predicted_class, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Predicted Rotor Diameter (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Rotor Diameters (Class-Specific)")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df["Rotor Diameter"].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Rotor Diameter (m)")
plt.ylabel("Frequency")
plt.title("Overall Distribution of Rotor Diameters (After Filling)")
plt.tight_layout()
plt.show()

# Save the final data to the output Excel file using the relative results directory.
df.to_excel(full_output_path, index=False)
print("Final data saved to:", full_output_path)
