#!/usr/bin/env python
import atlite
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ─── 1. Define base directory and build relative paths ────────────────────────
BASE_DIR = Path(__file__).resolve().parent

path_source         = BASE_DIR / "results" / "Approach_2.xlsx"
path_spatial_units  = BASE_DIR/ "data" / "NUTS_RG_01M_2016_4326.geojson"
path_custom_geo     = BASE_DIR / "data" / "custom.geo.json"
path_cutout         = BASE_DIR / "data" / "era5.nc"

# **Two** separate power‐curve CSVs:
path_pc1            = BASE_DIR / "data" / "Power Curves.csv"
path_pc2            = BASE_DIR / "data" / "Power Curves 2.csv"

# Where to save the output
output_path         = BASE_DIR / "results" / "Approach_2_Cf.xlsx"

# ─── 2. Load repowering results & spatial units ────────────────────────────────
print("Loading repowering results…")
rep = pd.read_excel(path_source, index_col=0)
print(f"  → {len(rep)} parks loaded")

print("Loading NUTS polygons…")
nuts = gpd.read_file(path_spatial_units).set_index("NUTS_ID")

print("Loading custom-country polygons…")
custom_shapes = (
    gpd.read_file(path_custom_geo)
       .reset_index()[["index", "geometry"]]
       .rename(columns={"index": "shape_id"})
)

# ─── 3. Build parks GeoDataFrame ───────────────────────────────────────────────
parks = gpd.GeoDataFrame(
    rep,
    geometry=gpd.points_from_xy(rep.Longitude, rep.Latitude),
    crs=nuts.crs
)

USE_CUSTOM = {"Ukraine", "Bosnia and Herzegovina", "Belarus", "Kosovo", "Iceland", "Faroe Islands", "Slovenia"}
parks["use_custom"] = parks["Country"].isin(USE_CUSTOM)

# ─── 4. Spatial join to assign shape_id ────────────────────────────────────────
# 4a) NUTS join
nuts_shapes = (
    nuts.reset_index()[["NUTS_ID", "geometry"]]
        .rename(columns={"NUTS_ID": "shape_id"})
)
join_n = gpd.sjoin(
    parks[~parks.use_custom],
    nuts_shapes,
    how="left",
    predicate="within"
)
parks.loc[join_n.index, "shape_id"] = join_n["shape_id"]

# 4b) Custom join
join_c = gpd.sjoin(
    parks[parks.use_custom],
    custom_shapes,
    how="left",
    predicate="within"
)
# GeoPandas calls the right-hand field "shape_id_right"
parks.loc[join_c.index, "shape_id"] = join_c["shape_id_right"]

# ─── 5. Consolidate used shapes ────────────────────────────────────────────────
used_ids     = parks["shape_id"].dropna().unique().tolist()
shapes_total = pd.concat(
    [nuts_shapes.set_index("shape_id"), custom_shapes.set_index("shape_id")],
    axis=0
).loc[used_ids, ["geometry"]]

# ─── 6. Build ERA5 cutout ───────────────────────────────────────────────────────
print("Building ERA5 cutout…")
margin = 0.1
xmin, ymin, xmax, ymax = (
    parks.geometry.x.min() - margin,
    parks.geometry.y.min() - margin,
    parks.geometry.x.max() + margin,
    parks.geometry.y.max() + margin
)
cutout = atlite.Cutout(
    str(path_cutout),
    bounds=[xmin, ymin, xmax, ymax],
    time=slice("2015-01-01", "2020-12-31"),
    module="10m_wind_speed"
)

# ─── 7. Load BOTH power‐curve CSVs ──────────────────────────────────────────────
print("Loading power curves…")
pc_dfs = []
for p in (path_pc1, path_pc2):
    df = pd.read_csv(p)
    # ensure first column is named "Wind speed"
    first = df.columns[0].strip().lower()
    if first in {"speed", "wind speed"}:
        df = df.rename(columns={df.columns[0]: "Wind speed"})
    pc_dfs.append(df)

def get_turbine_cfg(model):
    """Search both dataframes for a column matching `model`."""
    for df in pc_dfs:
        if model in df.columns:
            return {
                "hub_height": 100,
                "V": df["Wind speed"].tolist(),
                "POW": df[model].tolist(),
                "P":   float(df[model].max())
            }
    return None

# prepare to store capacity factors
rep["CapacityFactor"] = pd.NA

# ─── 8. Compute CF per turbine model ───────────────────────────────────────────
for model in rep["Recommended_WT_Model"].dropna().unique():
    cfg = get_turbine_cfg(model)
    if cfg is None:
        print(f"  ⚠ No power curve for {model}, skipping")
        continue

    mask      = rep["Recommended_WT_Model"] == model
    pts_model = parks[mask]
    if pts_model.empty:
        continue

    # build atlite layout
    layout_df = pts_model.rename(columns={"Longitude":"x", "Latitude":"y"})[
        ["x", "y", "Total_New_Capacity"]
    ]
    layout   = cutout.layout_from_capacity_list(layout_df, col="Total_New_Capacity")
    cf_ts    = cutout.wind(turbine=cfg, layout=layout, shapes=shapes_total, per_unit=True)

    # average over time and map back
    shape_dim = [d for d in cf_ts.dims if d != "time"][0]
    mean_cf   = (
        cf_ts.mean(dim="time")
             .to_dataframe(name="CF")
             .reset_index()
             .rename(columns={shape_dim: "shape_id"})
    )
    cf_map    = mean_cf.set_index("shape_id")["CF"].to_dict()
    rep.loc[mask, "CapacityFactor"] = parks.loc[mask, "shape_id"].map(cf_map)

#  9. Compute annual yield & save
rep["CapacityFactor"] = pd.to_numeric(rep["CapacityFactor"], errors="coerce").fillna(0.0)

rep["Annual_Energy_MWh_new"] = (
    rep["Total_New_Capacity"].astype(float)
  * rep["CapacityFactor"]
  * 8760.0
)

print(f"\nSaving results (with CF & Annual_Energy_MWh_new) to:\n  → {output_path}")
rep.to_excel(output_path)
print("Done ✅")
