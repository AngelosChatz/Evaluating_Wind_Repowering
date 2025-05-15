import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer

# ─── 0. File paths & load NUTS3 ───────────────────────────────────────────────
NEW_FILE     = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Energy_Yield_Parks.xlsx"
OLD_FILE     = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf_old.xlsx"
SPATIAL_FILE = r"D:\SET 2023\Thesis Delft\Model\atlite_example\data\NUTS_RG_01M_2016_4326.geojson"

nuts = gpd.read_file(SPATIAL_FILE)
nuts3_base = nuts[nuts["LEVL_CODE"] == 3].copy()

# ─── 1. Aggregation helper ────────────────────────────────────────────────────
def agg_nuts3(df):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.Longitude, df.Latitude)],
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(
        gdf,
        nuts3_base[["NUTS_ID", "geometry"]],
        how="inner",
        predicate="within"
    )
    sums = joined.groupby("NUTS_ID")["Annual_Energy_TWh"].sum().reset_index()
    nuts3 = nuts3_base.merge(sums, on="NUTS_ID", how="left")
    nuts3["Annual_Energy_TWh"] = nuts3["Annual_Energy_TWh"].fillna(0)
    nuts3["Annual_Energy_GWh"] = nuts3["Annual_Energy_TWh"] * 1000
    return nuts3.to_crs(epsg=3857)

# ─── 2. Read & build scenarios ─────────────────────────────────────────────────
df_new = pd.read_excel(NEW_FILE, sheet_name="Sheet1")
df_old = pd.read_excel(OLD_FILE, sheet_name="Sheet1")

# Repowering (new)
nuts3_new = agg_nuts3(df_new)

# Hybrid = max(new, old) at park level
df_hybrid = df_new.copy()
df_hybrid["Annual_Energy_TWh"] = np.where(
    df_new["Annual_Energy_TWh"] < df_old["Annual_Energy_TWh"],
    df_old["Annual_Energy_TWh"],
    df_new["Annual_Energy_TWh"]
)
nuts3_hybrid = agg_nuts3(df_hybrid)

# Old
nuts3_old = agg_nuts3(df_old)

# Compute difference
nuts3_diff = nuts3_hybrid.copy()
nuts3_diff["Diff_Energy_GWh"] = (
    nuts3_hybrid["Annual_Energy_GWh"]
    - nuts3_old["Annual_Energy_GWh"]
)

# Build parks GeoDataFrame in WebMercator
parks = gpd.GeoDataFrame(
    df_new,
    geometry=[Point(xy) for xy in zip(df_new.Longitude, df_new.Latitude)],
    crs="EPSG:4326"
).to_crs(epsg=3857)
parks["status"] = np.where(
    df_new["Annual_Energy_TWh"] > df_old["Annual_Energy_TWh"],
    "repowered",
    "old"
)

# Prepare map extent
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin = transformer.transform(-10.0, 35.0)
xmax, ymax = transformer.transform(30.0, 70.0)

# Heatmap plotting function
def plot_heatmap(gdf, column, title):
    vmin, vmax = gdf[column].min(), gdf[column].max()
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        column=column,
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        edgecolor="gray",
        linewidth=0.2,
        legend=True,
        legend_kwds={"label": column},
        ax=ax
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

# Park locations plotting function
def plot_parks(title):
    fig, ax = plt.subplots(figsize=(12, 10))
    for status, color in [("repowered", "red"), ("old", "black")]:
        parks[parks["status"] == status].plot(
            ax=ax,
            marker="o",
            markersize=2,
            color=color,
            alpha=0.4,
            label=status
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7)
    ax.legend(title="Park status")
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

# ─── Draw each figure separately ──────────────────────────────────────────────
plot_heatmap(nuts3_new,   "Annual_Energy_GWh",            "1) Repowering (New)")
plot_heatmap(nuts3_hybrid,"Annual_Energy_GWh",            "2) Hybrid (Max new/old)")
plot_heatmap(nuts3_old,   "Annual_Energy_GWh",            "3) Old")
plot_parks("4) Wind Park Locations — Red = repowered, Black = old")
plot_heatmap(nuts3_diff,  "Diff_Energy_GWh",              "5) Hybrid minus Old: Δ Annual Energy (GWh)")


# ─── After you construct df_hybrid ────────────────────────────────────────────

# Boolean mask: True when we pulled from df_old (because df_new was zero)
from_old_mask = df_new["Annual_Energy_TWh"] == 0

n_total        = len(df_new)
n_from_old     = from_old_mask.sum()
n_from_new     = n_total - n_from_old

# Power‐increase mask: True when new > old
power_increase_mask = df_new["Annual_Energy_TWh"] > df_old["Annual_Energy_TWh"]
n_power_increase    = power_increase_mask.sum()

print("Hybrid sourcing summary:")
print(f"  • Total parks:            {n_total}")
print(f"  • Used NEW data for:      {n_from_new} rows ({n_from_new/n_total*100:.1f}%)")
print(f"  • Fell back to OLD data:  {n_from_old} rows ({n_from_old/n_total*100:.1f}%)\n")

print("Power‐increase summary:")
print(f"  • Parks where NEW > OLD:  {n_power_increase} rows ({n_power_increase/n_total*100:.1f}%)")
