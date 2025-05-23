import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
import matplotlib.colors as mcolors

NEW_FILE = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf.xlsx"
OLD_FILE = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Cf_old_updated.xlsx"
SPATIAL_FILE = r"D:\SET 2023\Thesis Delft\Model\atlite_example\data\NUTS_RG_01M_2016_4326.geojson"
CUSTOM_SPATIAL_FILE = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\data\custom.geo.json"  # Custom GeoJSON for out-of-bounds points

# Load NUTS3 regions
nuts = gpd.read_file(SPATIAL_FILE)
nuts3_base = nuts[nuts["LEVL_CODE"] == 3].copy()

# Load custom GeoJSON file (for out-of-bounds areas)
custom_regions = gpd.read_file(CUSTOM_SPATIAL_FILE)


def agg_nuts3(df):
    # Convert wind farm data into GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.Longitude, df.Latitude)],
        crs="EPSG:4326"
    )

    # Perform spatial join with NUTS3 regions
    joined = gpd.sjoin(
        gdf,
        nuts3_base[["NUTS_ID", "geometry"]],
        how="left",  # 'left' join so we keep all data from wind farms
        predicate="within"
    )

    # Aggregate the energy by NUTS_ID
    sums = joined.groupby("NUTS_ID")["Annual_Energy_TWh"].sum().reset_index()
    nuts3 = nuts3_base.merge(sums, on="NUTS_ID", how="left")
    nuts3["Annual_Energy_TWh"] = nuts3["Annual_Energy_TWh"].fillna(0)
    nuts3["Annual_Energy_TWh"] = nuts3["Annual_Energy_TWh"]  # already in TWh
    nuts3["Annual_Energy_GWh"] = nuts3["Annual_Energy_TWh"] * 1000  # for any required GWh display
    return nuts3.to_crs(epsg=3857)


# Input data
df_new = pd.read_excel(NEW_FILE, sheet_name="Sheet1")
df_old = pd.read_excel(OLD_FILE, sheet_name="Sheet1")

# Yield-based selection
mask_yield = df_new["Annual_Energy_TWh"] > df_old["Annual_Energy_TWh"]
df_yield = df_new.copy()
df_yield["Annual_Energy_TWh"] = np.where(
    mask_yield,
    df_new["Annual_Energy_TWh"],
    df_old["Annual_Energy_TWh"]
)

# Aggregate per scenario
nuts3_new = agg_nuts3(df_new)
nuts3_old = agg_nuts3(df_old)
nuts3_yield = agg_nuts3(df_yield)

# Compute difference map
nuts3_diff = nuts3_yield.copy()
nuts3_diff["Diff_Energy_TWh"] = (
    nuts3_yield["Annual_Energy_TWh"]
    - nuts3_old["Annual_Energy_TWh"]
)

# Prepare parks GeoDataFrame and status
parks = gpd.GeoDataFrame(
    df_new,
    geometry=[Point(xy) for xy in zip(df_new.Longitude, df_new.Latitude)],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Updated status map: Blue if repowered energy yield is higher, else black
mask_base = df_new["Annual_Energy_TWh"] > 0
parks["status_map"] = np.where(
    df_new["Annual_Energy_TWh"] > df_old["Annual_Energy_TWh"], 'yield_selected',
    'base_repowered'
)

status_colors = {
    'base_repowered': '#000000',  # black
    'yield_selected': '#377eb8'  # blue
}

# Map extent (WebMercator)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = transformer.transform(-10.0, 35.0)
xmax, ymax = transformer.transform(30.0, 70.0)


# Plotting functions
def plot_nuts3(gdf, col, title, cmap='plasma', norm=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        column=col, cmap=cmap, norm=norm,
        edgecolor='gray', linewidth=0.2, legend=True,
        legend_kwds={'label': col + " (TWh)"}, ax=ax  # Update units in legend
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


def plot_parks_status():
    fig, ax = plt.subplots(figsize=(12, 10))
    for status, color in status_colors.items():
        parks[parks['status_map'] == status].plot(
            ax=ax, marker='o', markersize=3,
            color=color, alpha=0.6,
            label=status.replace('_', ' ').title()
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
    ax.legend(title='Park Status', loc='upper right')
    ax.set_axis_off()
    ax.set_title('5) Park Status – Base vs Yield-Selected', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


# 1)–3) Scenario maps
plot_nuts3(nuts3_new, 'Annual_Energy_TWh', '1) Repowering (New)')  # Units are now TWh
plot_nuts3(nuts3_old, 'Annual_Energy_TWh', '2) Old')  # Units are now TWh
plot_nuts3(nuts3_yield, 'Annual_Energy_TWh', '3) Yield-based Selection')  # Units are now TWh

# 4) Difference map with vertical colorbar
max_diff = nuts3_diff['Diff_Energy_TWh'].max()  # Using TWh
vmin = 0
stops = [0, 0.5, max_diff]  # Adjust color stops for TWh range
colors = ['white', 'blue', 'red']
stops_norm = [s / max_diff for s in stops]
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_diff', list(zip(stops_norm, colors)))
norm = plt.Normalize(vmin=vmin, vmax=max_diff)

fig, ax = plt.subplots(figsize=(12, 10))
nuts3_diff.plot(
    column='Diff_Energy_TWh',  # Change to TWh
    cmap=custom_cmap,
    norm=norm,
    edgecolor='gray',
    linewidth=0.2,
    legend=True,
    legend_kwds={
        'label': 'Δ Annual Energy (TWh)',  # Updated legend label
        'orientation': 'vertical'
    },
    ax=ax
)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
ax.set_axis_off()
ax.set_title('4) Yield minus Old: Δ Annual Energy (TWh)', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# 5) Parks status map
plot_parks_status()
