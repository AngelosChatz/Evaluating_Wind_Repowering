import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

input_excel = r"D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_with_IEC_Elevation_v2.xlsx"
base, ext = os.path.splitext(input_excel)
output_excel = base + "_area" + ext

df = pd.read_excel(input_excel)
df["Commissioning date"] = df["Commissioning date"].apply(parse_custom_date)
df["Decommissioning date"] = df["Decommissioning date"].apply(parse_custom_date)

def is_active_in_2022(row):
    comm = row["Commissioning date"]
    decom = row["Decommissioning date"]
    if pd.isnull(comm) or comm > pd.Timestamp(2022, 12, 31):
        return False
    if pd.isnull(decom):
        decom = comm + pd.DateOffset(years=20)
    return decom >= pd.Timestamp(2022, 1, 1)

df["Active in 2022"] = df.apply(is_active_in_2022, axis=1)
df["Rotor Diameter"] = pd.to_numeric(df["Rotor Diameter"], errors='coerce')
df["Number of turbines"] = pd.to_numeric(df["Number of turbines"], errors='coerce')
df["Total power"] = pd.to_numeric(df["Total power"], errors='coerce').fillna(0) / 1000

if "Terrain_Type" not in df.columns:
    raise ValueError("Missing 'Terrain_Type' column.")

def calculate_turbine_area(row):
    D = row["Rotor Diameter"]
    if pd.isnull(D):
        return np.nan
    terrain = str(row["Terrain_Type"]).strip().lower()
    return 7 * D * 4 * D if terrain == "flat" else 9 * D * 6 * D

df["Turbine Area (m²)"] = np.where(
    df["Active in 2022"],
    df.apply(calculate_turbine_area, axis=1),
    np.nan
)
df["Total Park Area (m²)"] = df["Turbine Area (m²)"] * df["Number of turbines"]

active_df = df[df["Active in 2022"]].copy()
country_area = active_df.groupby("Country")["Total Park Area (m²)"].sum().reset_index()

land_area_data = {
    "Country": [
        "Albania", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
        "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
        "Faroe Islands", "Finland", "France", "Germany", "Greece", "Hungary",
        "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Lithuania",
        "Luxembourg", "Montenegro", "Netherlands", "North Macedonia", "Norway",
        "Poland", "Portugal", "Romania", "Serbia", "Slovakia", "Slovenia",
        "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom"
    ],
    "Land Area (m²)": [
        28748e6, 83871e6, 207600e6, 30528e6, 51197e6, 110879e6, 56594e6, 9250e6,
        78867e6, 43094e6, 45228e6, 1393e6, 338145e6, 551695e6, 357114e6,
        131957e6, 93028e6, 103000e6, 70273e6, 301340e6, 10887e6, 64589e6,
        65300e6, 2586e6, 13812e6, 41543e6, 25713e6, 323802e6, 312696e6,
        92090e6, 238397e6, 77474e6, 49035e6, 20273e6, 505992e6, 450295e6,
        41290e6, 603550e6, 243610e6
    ]
}
land_area_df = pd.DataFrame(land_area_data)

merged_area_df = pd.merge(country_area, land_area_df, on="Country", how="right")
merged_area_df["Land Area (km²)"] = merged_area_df["Land Area (m²)"] / 1e6
merged_area_df["Occupied Percentage (%)"] = (
    merged_area_df["Total Park Area (m²)"] / merged_area_df["Land Area (m²)"] * 100
)
merged_area_df = merged_area_df.sort_values("Occupied Percentage (%)", ascending=False)

country_power = active_df.groupby("Country")["Total power"].sum().reset_index()
merged_density_df = pd.merge(
    country_power,
    merged_area_df[["Country", "Total Park Area (m²)"]],
    on="Country",
    how="right"
)
merged_density_df["Park Area (km²)"] = merged_density_df["Total Park Area (m²)"] / 1e6
merged_density_df["Capacity Density (MW/km²)"] = (
    merged_density_df["Total power"] / merged_density_df["Park Area (km²)"]
)
merged_density_df = merged_density_df.sort_values("Capacity Density (MW/km²)", ascending=False)

print("Occupied Percentage by Country (Wind Park Area):")
print(merged_area_df.to_string(index=False))
print("\nCapacity Density (MW/km²) by Country:")
print(merged_density_df.to_string(index=False))

sns.set(style="whitegrid")

plt.figure(figsize=(14, 8))
sns.barplot(x="Occupied Percentage (%)", y="Country", data=merged_area_df, palette="viridis")
plt.xlabel("Percentage of Land Occupied by Wind Parks (%)", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.title("Land Occupied by Wind Parks in European Countries", fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
sns.barplot(x="Capacity Density (MW/km²)", y="Country", data=merged_density_df, palette="magma")
plt.xlabel("Capacity Density (MW/km²)", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.title("Wind Turbine Capacity Density in European Countries", fontsize=14)
plt.tight_layout()
plt.show()

merged_df = pd.merge(
    merged_area_df[['Country', 'Occupied Percentage (%)']],
    merged_density_df[['Country', 'Capacity Density (MW/km²)']],
    on="Country"
).sort_values("Occupied Percentage (%)", ascending=False)

fig, ax = plt.subplots(figsize=(14, 10))
ax.barh(merged_df["Country"], merged_df["Occupied Percentage (%)"], color="skyblue")
ax.set_xlabel("Occupied Percentage (%)", fontsize=12)
ax.set_title("Wind Park Occupancy with Capacity Density Annotations", fontsize=14)
ax.invert_yaxis()
for i, row in merged_df.iterrows():
    ax.text(row["Occupied Percentage (%)"] + 0.5, i, f"{row['Capacity Density (MW/km²)']:.2f} MW/km²",
            va='center', fontsize=10, color="salmon")
plt.tight_layout()
plt.show()
