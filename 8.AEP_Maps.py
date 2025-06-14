import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
import matplotlib.colors as mcolors
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
this_dir     = Path(__file__).resolve().parent
results_dir  = this_dir / "results"
figures_dir  = results_dir / "figures"
data_dir     = this_dir / "data"

# ensure figures directory exists
figures_dir.mkdir(parents=True, exist_ok=True)

NEW_FILE           = results_dir / "Approach_2_Cf.xlsx"
OLD_FILE           = results_dir / "Cf_old_updated.xlsx"
SPATIAL_FILE       = data_dir / "NUTS_RG_01M_2016_4326.geojson"
CUSTOM_SPATIAL_FILE= data_dir / "custom.geo.json"
# ─────────────────────────────────────────────────────────────────────────────

# Load spatial boundaries
nuts = gpd.read_file(SPATIAL_FILE)
nuts3_base = nuts[nuts["LEVL_CODE"] == 3].copy()

def agg_nuts3(df):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.Longitude, df.Latitude)],
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(
        gdf,
        nuts3_base[["NUTS_ID", "geometry"]],
        how="left",
        predicate="within"
    )
    sums = joined.groupby("NUTS_ID")["Annual_Energy_TWh"].sum().reset_index()
    nuts3 = nuts3_base.merge(sums, on="NUTS_ID", how="left")
    nuts3["Annual_Energy_TWh"] = nuts3["Annual_Energy_TWh"].fillna(0)
    nuts3["Annual_Energy_GWh"] = nuts3["Annual_Energy_TWh"] * 1000
    return nuts3.to_crs(epsg=3857)

# ─────────────────────────────────────────────────────────────────────────────
# Read Excel files & standardize units
df_new = pd.read_excel(NEW_FILE, sheet_name="Sheet1")
df_old = pd.read_excel(OLD_FILE, sheet_name="Sheet1")

df_new["Annual_Energy_TWh"] = df_new["Annual_Energy_MWh_new"] / 1e6
if "Annual_Energy_TWh" not in df_old.columns:
    if "Annual_Energy_MWh_old" in df_old.columns:
        df_old["Annual_Energy_TWh"] = df_old["Annual_Energy_MWh_old"] / 1e6
    else:
        raise KeyError("Old DataFrame lacks both 'Annual_Energy_TWh' and 'Annual_Energy_MWh_old'.")
# ─────────────────────────────────────────────────────────────────────────────

# Yield‐based selection
mask_yield   = df_new["Annual_Energy_TWh"] > df_old["Annual_Energy_TWh"]
df_yield     = df_new.copy()
df_yield["Annual_Energy_TWh"] = np.where(
    mask_yield,
    df_new["Annual_Energy_TWh"],
    df_old["Annual_Energy_TWh"]
)

# Spatial aggregates
nuts3_new   = agg_nuts3(df_new)
nuts3_old   = agg_nuts3(df_old)
nuts3_yield = agg_nuts3(df_yield)

# Δ‐map data
nuts3_diff = nuts3_yield.copy()
nuts3_diff["Diff_Energy_TWh"] = (
    nuts3_yield["Annual_Energy_TWh"]
    - nuts3_old["Annual_Energy_TWh"]
)

# Parks status layer
parks = gpd.GeoDataFrame(
    df_new,
    geometry=[Point(xy) for xy in zip(df_new.Longitude, df_new.Latitude)],
    crs="EPSG:4326"
).to_crs(epsg=3857)

parks["status_map"] = np.where(
    df_new["Annual_Energy_TWh"] > df_old["Annual_Energy_TWh"],
    'yield_selected',
    'base_repowered'
)

status_colors = {
    'base_repowered': '#000000',
    'yield_selected': '#377eb8'
}

# Common map extent
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = transformer.transform(-10.0, 35.0)
xmax, ymax = transformer.transform(30.0, 70.0)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions (now saving PNGs)

def plot_nuts3(gdf, col, title, fname, cmap='plasma', norm=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        column=col, cmap=cmap, norm=norm,
        edgecolor='gray', linewidth=0.2, legend=True,
        legend_kwds={'label': col + " (TWh)"}, ax=ax
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()
    fig.savefig(str(figures_dir / fname), dpi=300)
    plt.close(fig)

def plot_parks_status(fname):
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
    fig.savefig(str(figures_dir / fname), dpi=300)
    plt.close(fig)
# ─────────────────────────────────────────────────────────────────────────────

# Generate & save all maps
plot_nuts3(nuts3_new,   'Annual_Energy_TWh', '1) Repowering (New)',         '1_repowered_new.png')
plot_nuts3(nuts3_old,   'Annual_Energy_TWh', '2) Old',                       '2_old.png')
plot_nuts3(nuts3_yield, 'Annual_Energy_TWh', '3) Yield-based Selection',     '3_yield_selection.png')

# Δ‐map with custom colormap
max_diff    = nuts3_diff['Diff_Energy_TWh'].max()
vmin        = 0
stops       = [0, max_diff/2, max_diff]
colors_list = ['white', 'blue', 'red']
stops_norm  = [s / max_diff for s in stops]
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_diff', list(zip(stops_norm, colors_list)))
norm        = plt.Normalize(vmin=vmin, vmax=max_diff)

fig, ax = plt.subplots(figsize=(12, 10))
nuts3_diff.plot(
    column='Diff_Energy_TWh',
    cmap=custom_cmap,
    norm=norm,
    edgecolor='gray',
    linewidth=0.2,
    legend=True,
    legend_kwds={'label': 'Δ Annual Energy (TWh)', 'orientation': 'vertical'},
    ax=ax
)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
ax.set_axis_off()
ax.set_title('4) Yield minus Old: Δ Annual Energy (TWh)', fontsize=16, pad=20)
plt.tight_layout()
fig.savefig(str(figures_dir / "4_delta_map.png"), dpi=300)
plt.close(fig)

plot_parks_status('5_parks_status.png')
