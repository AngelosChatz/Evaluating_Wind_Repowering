import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pyproj import Transformer
from pathlib import Path
import matplotlib.patches as mpatches

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
this_dir       = Path(__file__).resolve().parent
results_dir    = this_dir / "results"
data_dir       = this_dir / "data"

INPUT_XLS      = results_dir / "selected_full_results.xlsx"
NUTS3_GEOJSON  = data_dir / "NUTS_RG_01M_2016_4326.geojson"
CUSTOM_GEOJSON = data_dir / "custom.geo.json"
OUTPUT_PNG     = results_dir / "repowering_economic_map_eu.png"

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_excel(INPUT_XLS)
required = {'Latitude','Longitude','NPV_rep','NPV_dec'}
if not required.issubset(df.columns):
    raise KeyError(f"Missing required columns in {INPUT_XLS}: {required - set(df.columns)}")

# ─── COMPUTE COUNTS & PERCENTAGES ───────────────────────────────────────────────
df['rep_viable'] = df['NPV_rep'] > df['NPV_dec']
total_parks      = len(df)
rep_count        = int(df['rep_viable'].sum())
keep_count       = total_parks - rep_count
rep_pct          = rep_count / total_parks * 100
keep_pct         = keep_count / total_parks * 100

print(f"Total sites: {total_parks}")
print(f"Repowering wins: {rep_count} ({rep_pct:.1f}%)")
print(f"Keep existing wins: {keep_count} ({keep_pct:.1f}%)\n")

# ─── LOAD REGION BOUNDARIES ──────────────────────────────────────────────────────
nuts = gpd.read_file(NUTS3_GEOJSON)
nuts3 = nuts[nuts["LEVL_CODE"] == 3].to_crs(epsg=3857)

# ─── MAKE PARKS GeoDataFrame ────────────────────────────────────────────────────
parks = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df.Longitude, df.Latitude)],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# ─── COUNTRY‐LEVEL SUMMARY ──────────────────────────────────────────────────────
# Spatially join to get each park’s NUTS3 region, then extract country code
parks_with_nuts = gpd.sjoin(
    parks,
    nuts3[['NUTS_ID','geometry']],
    how='left',
    predicate='within'
)
# first two letters of NUTS_ID are the country ISO code
parks_with_nuts['country'] = parks_with_nuts['NUTS_ID'].str[:2]

# Group by country
country_summary = (
    parks_with_nuts
      .groupby('country')
      .agg(
          total_sites=('rep_viable','size'),
          rep_count=('rep_viable','sum')
      )
      .assign(
          rep_pct=lambda d: d['rep_count'] / d['total_sites'] * 100
      )
      .sort_values('rep_pct', ascending=False)
)

print("Repowering viability by country (descending repowering %):")
print(country_summary[['total_sites','rep_count','rep_pct']].to_string(formatters={'rep_pct':'{:.1f}%'.format}))

# ─── DEFINE EU BOUNDS & ZOOM OUT ────────────────────────────────────────────────
lon_min, lat_min = -10.0, 34.0
lon_max, lat_max = 31.0, 72.0
transformer      = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
xmin, ymin       = transformer.transform(lon_min, lat_min)
xmax, ymax       = transformer.transform(lon_max, lat_max)
dx = (xmax - xmin) * 0.05
dy = (ymax - ymin) * 0.05
xmin, xmax = xmin - dx, xmax + dx
ymin, ymax = ymin - dy, ymax + dy

# ─── LOAD CUSTOM BOUNDARIES ─────────────────────────────────────────────────────
custom = gpd.read_file(CUSTOM_GEOJSON).to_crs(epsg=3857)

# ─── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

# Plot region boundaries
nuts3.boundary.plot(ax=ax, edgecolor="lightgrey", linewidth=0.5)
custom.boundary.plot(ax=ax, edgecolor="grey", linewidth=0.7)

# Plot parks as tiny dots
parks.plot(
    ax=ax,
    markersize=4,
    color=parks['rep_viable'].map({True: 'red', False: 'black'}),
    alpha=0.7
)

# Legend in top-left
handles = [
    mpatches.Patch(color='red',   label='NPV_rep > NPV_dec'),
    mpatches.Patch(color='black', label='NPV_rep ≤ NPV_dec')
]
ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.8)

# Apply EU extent
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_axis_off()
ax.set_title(
    "Wind‐Farm Sites in EU: NPV Comparison Repowering vs Existing",
    fontsize=16, pad=20
)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()

print(f"\nMap saved to {OUTPUT_PNG}")
