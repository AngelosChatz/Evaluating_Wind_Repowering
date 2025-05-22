import atlite
import pandas as pd
import geopandas as gpd
from pathlib import Path

# 1. CONFIGURATION
DATA_DIR            = Path("data")
RESULTS_DIR         = Path(r"D:/SET 2023/Thesis Delft/Model/Evaluating_Wind_Repowering/results")
REPOWER_FILE        = RESULTS_DIR / "Approach_4.xlsx"
OUTPUT_FILE         = RESULTS_DIR / "Approach_4_Cf.xlsx"
ERA5_FILE           = DATA_DIR / "era5.nc"
SPATIAL_NUTS_FILE   = DATA_DIR / "NUTS_RG_01M_2016_4326.geojson"
SPATIAL_CUSTOM_FILE = DATA_DIR / "custom.geo.json"
POWER_CURVE_FILES   = [
    DATA_DIR / "Power Curves.csv",
    DATA_DIR / "Power Curves 2.csv"
]
DEFAULT_HUB_HEIGHT  = 100  # meters

# These countries use the custom geometry
USE_CUSTOM = {
    "Ukraine",
    "Bosnia and Herzegovina",
    "Belarus",
    "Kosovo",
    "Iceland",
    "Faroe Islands",
    "Slovenia"
}


# 2. HELPERS
def load_power_curves(paths):
    print("  ▶ Loading power curves…")
    dfs = []
    for p in paths:
        print(f"    • {p.name}")
        df = pd.read_csv(p)
        if df.columns[0].strip().lower() != "wind speed":
            df = df.rename(columns={df.columns[0]: "Wind speed"})
        dfs.append(df)
    return dfs

def find_turbine_config(curves, model):
    for df in curves:
        if model in df.columns:
            return {
                "hub_height": DEFAULT_HUB_HEIGHT,
                "V":    df["Wind speed"].tolist(),
                "POW":  df[model].tolist(),
                "P":    df[model].max()
            }
    return None


# 3. MAIN
def main():
    print("\n--- START Capacity‐Factor Assignment ---\n")

    # 3.1 Read repower table
    print(f"1) Reading repower table: {REPOWER_FILE}")
    repower = pd.read_excel(REPOWER_FILE, index_col=0)
    print(f"   → {len(repower)} parks loaded")
    required = {"Country","Longitude","Latitude","Recommended_WT_Model","Total_New_Capacity"}
    if not required.issubset(repower.columns):
        missing = required - set(repower.columns)
        raise ValueError(f"Missing columns in {REPOWER_FILE}: {missing}")

    # 3.2 Load spatial layers
    print("\n2) Loading spatial layers…")
    nuts     = gpd.read_file(SPATIAL_NUTS_FILE).set_index("NUTS_ID")
    custom   = gpd.read_file(SPATIAL_CUSTOM_FILE).set_index("admin")
    print(f"   • NUTS:   {len(nuts)} polygons")
    print(f"   • Custom: {len(custom)} polygons")

    # 3.3 Build parks GeoDataFrame
    print("\n3) Building parks GeoDataFrame…")
    parks = gpd.GeoDataFrame(
        repower,
        geometry=gpd.points_from_xy(repower.Longitude, repower.Latitude),
        crs=nuts.crs
    )
    parks["use_custom"] = parks["Country"].isin(USE_CUSTOM)
    print(f"   → {parks['use_custom'].sum()} parks use custom, {len(parks)-parks['use_custom'].sum()} use NUTS")

    # 3.4 Spatial‐join per layer to get shape_id
    print("\n4) Spatial‐joining parks to assign shape_id…")
    # prepare RHS tables with explicit shape_id column
    nuts_shapes   = nuts.reset_index()[["NUTS_ID","geometry"]].rename(columns={"NUTS_ID":"shape_id"})
    custom_shapes = custom.reset_index()[["admin","geometry"]].rename(columns={"admin":"shape_id"})

    # join NUTS parks
    join_n = gpd.sjoin(
        parks[~parks.use_custom],
        nuts_shapes,
        how="left",
        predicate="within"
    )
    parks.loc[join_n.index, "shape_id"] = join_n["shape_id"]
    print(f"   • {len(join_n)} parks matched to NUTS polygons")

    # join custom parks
    join_c = gpd.sjoin(
        parks[ parks.use_custom],
        custom_shapes,
        how="left",
        predicate="within"
    )
    # GeoPandas will name the rhs shape_id column "shape_id_right"
    parks.loc[join_c.index, "shape_id"] = join_c["shape_id_right"]
    print(f"   • {len(join_c)} parks matched to custom polygons")

    # 3.5 Build minimal shapes_total
    used_ids     = parks["shape_id"].dropna().unique().tolist()
    shapes_total = pd.concat(
        [nuts_shapes.set_index("shape_id"), custom_shapes.set_index("shape_id")],
        axis=0
    ).loc[used_ids, ["geometry"]]
    print(f"\n5) shapes_total contains {len(shapes_total)} unique polygons")

    # 3.6 Load ERA5 and power curves
    print("\n6) Loading ERA5 cutout & power curves…")
    cutout = atlite.Cutout(str(ERA5_FILE))
    curves = load_power_curves(POWER_CURVE_FILES)

    # prepare output column
    repower["CapacityFactor"] = pd.NA

    # 3.7 Compute CF per turbine model
    print("\n7) Computing CapacityFactor for each park (grouped by model)…")
    for model in repower["Recommended_WT_Model"].dropna().unique():
        print(f"   ▶ Model: {model}")
        cfg = find_turbine_config(curves, model)
        if cfg is None:
            print("     ⚠ No power curve, skipping")
            continue

        mask      = repower["Recommended_WT_Model"] == model
        pts_model = parks[mask]
        print(f"     • {len(pts_model)} parks to process")

        # build layout DataFrame for atlite
        df_layout = pts_model.rename(
            columns={"Longitude":"x","Latitude":"y"}
        )[['x','y','Total_New_Capacity']]

        # compute CF time series once
        wl    = cutout.layout_from_capacity_list(df_layout, col="Total_New_Capacity")
        cf_ts = cutout.wind(
            turbine=cfg,
            layout=wl,
            shapes=shapes_total,
            per_unit=True
        )

        # identify shape dimension
        shape_dim = next(d for d in cf_ts.dims if d != "time")
        print(f"     • CF dims: {cf_ts.dims}; using '{shape_dim}'")

        # average over time
        mean_cf = (
            cf_ts.mean(dim="time")
                 .to_dataframe(name="CF")
                 .reset_index()
                 .rename(columns={shape_dim:"shape_id"})
        )

        # map CF back to parks
        cf_map = mean_cf.set_index("shape_id")["CF"].to_dict()
        repower.loc[mask, "CapacityFactor"] = pts_model["shape_id"].map(cf_map)
        print(f"     ✔ Assigned CF to {mask.sum()} parks\n")

    # 3.8 Save full DataFrame with per-park CapacityFactor
    print(f"8) Saving detailed results to {OUTPUT_FILE}")
    repower.to_excel(OUTPUT_FILE)
    print("\n--- DONE ---\n")


if __name__ == "__main__":
    main()
