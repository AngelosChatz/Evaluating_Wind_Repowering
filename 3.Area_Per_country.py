import pandas as pd
import numpy as np
from pathlib import Path
import os

def parse_custom_date(date_str):
    """Parse date in 'yyyy/mm' or 'yyyy' format; return NaT if invalid."""
    if pd.isnull(date_str) or date_str == '#ND' or str(date_str).strip() == '':
        return pd.NaT
    try:
        return pd.to_datetime(date_str, format='%Y/%m', errors='raise')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%Y', errors='raise')
        except ValueError:
            return pd.NaT

# Set up directories: base, data (input), and results (output)
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create results directory if it doesn't exist

# Define input and output file paths
input_excel = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2.xlsx"
output_excel = results_dir / (input_excel.stem + "_area" + input_excel.suffix)

# Read data from the Excel file
df = pd.read_excel(input_excel)

# Parse the commissioning and decommissioning date columns
df["Commissioning date"] = df["Commissioning date"].apply(parse_custom_date)
df["Decommissioning date"] = df["Decommissioning date"].apply(parse_custom_date)

def is_active_in_2022(row):
    comm = row["Commissioning date"]
    decom = row["Decommissioning date"]
    # A turbine must be commissioned before end of 2022.
    if pd.isnull(comm) or comm > pd.Timestamp(2022, 12, 31):
        return False
    # If decommissioning date is missing, assume a 20-year operational life.
    if pd.isnull(decom):
        decom = comm + pd.DateOffset(years=20)
    return decom >= pd.Timestamp(2022, 1, 1)

df["Active in 2022"] = df.apply(is_active_in_2022, axis=1)
df["Rotor Diameter"] = pd.to_numeric(df["Rotor Diameter"], errors='coerce')
df["Number of turbines"] = pd.to_numeric(df["Number of turbines"], errors='coerce')
# Convert total power to MW (assuming original data is in Watts)
df["Total power"] = pd.to_numeric(df["Total power"], errors='coerce').fillna(0) / 1000

if "Terrain_Type" not in df.columns:
    raise ValueError("Missing 'Terrain_Type' column.")

def calculate_turbine_area(row):
    """Calculate the area of a turbine based on rotor diameter and terrain type."""
    D = row["Rotor Diameter"]
    if pd.isnull(D):
        return np.nan
    terrain = str(row["Terrain_Type"]).strip().lower()
    # For flat terrain use one formula; for other (complex) terrain use another.
    return 7 * D * 4 * D if terrain == "flat" else 9 * D * 6 * D

df["Turbine Area (m²)"] = np.where(
    df["Active in 2022"],
    df.apply(calculate_turbine_area, axis=1),
    np.nan
)
df["Total Park Area (m²)"] = df["Turbine Area (m²)"] * df["Number of turbines"]

# Perform aggregated calculations for active wind farms only
active_df = df[df["Active in 2022"]].copy()

# Group by Country to compute total wind park area for active wind farms
country_area = active_df.groupby("Country")["Total Park Area (m²)"].sum().reset_index()

# Group by Country to compute total installed power (in MW)
country_power = active_df.groupby("Country")["Total power"].sum().reset_index()

# Merge the aggregated results on Country
merged_density_df = pd.merge(country_power, country_area, on="Country", how="outer")
# Compute Capacity Density (MW per km²) based on the computed wind park area
merged_density_df["Park Area (km²)"] = merged_density_df["Total Park Area (m²)"] / 1e6
merged_density_df["Capacity Density (MW/km²)"] = (
    merged_density_df["Total power"] / merged_density_df["Park Area (km²)"]
)

# Print the aggregated capacity density results to the console
print("Capacity Density (MW/km²) by Country (based on wind park area):")
print(merged_density_df.to_string(index=False))

# Save the modified DataFrame with area calculations to the output Excel file.
df.to_excel(output_excel, index=False)
print("\nFinal modified data saved to:", output_excel)
