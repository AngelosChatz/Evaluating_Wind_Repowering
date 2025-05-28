#!/usr/bin/env python
import atlite
import pandas as pd
import geopandas as gpd

# ─── 1. File paths ────────────────────────────────────────────────────────────────
path_source          = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2.xlsx"
path_old_cf          = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf_old.xlsx"
path_spatial_units   = r"D:\SET 2023\Thesis Delft\Model\atlite_example\data\NUTS_RG_01M_2016_4326.geojson"
path_custom_geo      = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\data\custom.geo.json"
path_cutout          = r"D:\SET 2023\Thesis Delft\Model\atlite_example\data\era5.nc"
path_power_curve     = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\data\Power Curves.csv"
output_path          = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\CF_old_updated.xlsx"

# ─── 2. Load original repowering results & spatial units ────────────────────────
print("Loading source repowering results…")
rep = pd.read_excel(path_source, sheet_name="Sheet1", index_col=0)
spatial = gpd.read_file(path_spatial_units).set_index("NUTS_ID")

# ─── 3. Assign representative turbines & compute capacities ─────────────────────

rep_turbines = {
    'Acciona.AW77.1500': 1500,
    'Alstom.Eco.74': 1670,
    'Alstom.Eco.80': 1670,
    'Alstom.Eco.110': 3000,
    'Bonus.B23.150': 150,
    'Bonus.B33.300': 300,
    'Bonus.B37.450': 450,
    'Bonus.B41.500': 500,
    'Bonus.B44.600': 600,
    'Bonus.B54.1000': 1000,
    'Bonus.B62.1300': 1300,
    'Bonus.B82.2300': 2300,
    'Dewind.D4.41.500': 500,
    'Dewind.D6.1000': 1000,
    'Enercon.E40.500': 500,
    'Enercon.E40.600': 600,
    'Enercon.E44.900': 900,
    'Enercon.E48.800': 800,
    'Enercon.E53.800': 800,
    'Enercon.E66.1500': 1500,
    'Enercon.E66.1800': 1800,
    'Enercon.E66.2000': 2000,
    'Enercon.E70.2000': 2000,
    'Enercon.E70.2300': 2300,
    'Enercon.E82.1800': 1800,
    'Enercon.E82.2000': 2000,
    'Enercon.E82.2300': 2300,
    'Enercon.E82.3000': 3000,
    'Enercon.E92.2300': 2300,
    'Enercon.E92.2350': 2350,
    'Enercon.E101.3000': 3000,
    'Enercon.E112.4500': 4500,
    'Enercon.E126.6500': 6500,
    'Enercon.E126.7000': 7000,
    'Enercon.E126.7500': 7500,
    'EWT.DirectWind.52.900': 900,
    'Gamesa.G47.660': 660,
    'Gamesa.G52.850': 850,
    'Gamesa.G58.850': 850,
    'Gamesa.G80.2000': 2000,
    'Gamesa.G87.2000': 2000,
    'Gamesa.G90.2000': 2000,
    'Gamesa.G128.4500': 4500,
    'GE.900S': 900,
    'GE.1.5s': 1500,
    'GE.1.5se': 1500,
    'GE.1.5sl': 1500,
    'GE.1.5sle': 1500,
    'GE.1.5xle': 1500,
    'GE.1.6': 1600,
    'GE.1.7': 1700,
    'GE.2.5xl': 2500,
    'GE.2.75.103': 2750,
    'Goldwind.GW82.1500': 1500,
    'NEG.Micon.M1500.500': 500,
    'NEG.Micon.M1500.750': 750,
    'NEG.Micon.NM48.750': 750,
    'NEG.Micon.NM52.900': 900,
    'NEG.Micon.NM60.1000': 1000,
    'NEG.Micon.NM64c.1500': 1500,
    'NEG.Micon.NM80.2750': 2750,
    'Nordex.N27.150': 150,
    'Nordex.N29.250': 250,
    'Nordex.N43.600': 600,
    'Nordex.N50.800': 800,
    'Nordex.N60.1300': 1300,
    'Nordex.N80.2500': 2500,
    'Nordex.N90.2300': 2300,
    'Nordex.N90.2500': 2500,
    'Nordex.N100.2500': 2500,
    'Nordex.N131.3000': 3000,
    'Nordex.N131.3300': 3300,
    'Nordtank.NTK500': 500,
    'Nordtank.NTK600': 600,
    'PowerWind.56.900': 900,
    'REpower.MD70.1500': 1500,
    'REpower.MD77.1500': 1500,
    'REpower.MM70.2000': 2000,
    'REpower.MM82.2000': 2000,
    'REpower.MM92.2000': 2000,
    'REpower.3.4M': 3400,
    'REpower.5M': 5000,
    'REpower.6M': 6000,
    'Siemens.SWT.1.3.62': 1300,
    'Siemens.SWT.2.3.82': 2300,
    'Siemens.SWT.2.3.93': 2300,
    'Siemens.SWT.2.3.101': 2300,
    'Siemens.SWT.3.0.101': 3000,
    'Siemens.SWT.3.6.107': 3600,
    'Siemens.SWT.3.6.120': 3600,
    'Siemens.SWT.4.0.130': 4000,
    'Suzlon.S88.2100': 2100,
    'Suzlon.S97.2100': 2100,
    'Tacke.TW600.43': 600,
    'Vestas.V27.225': 225,
    'Vestas.V29.225': 225,
    'Vestas.V39.500': 500,
    'Vestas.V42.600': 600,
    'Vestas.V44.600': 600,
    'Vestas.V47.660': 660,
    'Vestas.V52.850': 850,
    'Vestas.V66.1650': 1650,
    'Vestas.V66.1750': 1750,
    'Vestas.V66.2000': 2000,
    'Vestas.V80.1800': 1800,
    'Vestas.V80.2000': 2000,
    'Vestas.V90.1800': 1800,
    'Vestas.V90.2000': 2000,
    'Vestas.V90.3000': 3000,
    'Vestas.V100.1800': 1800,
    'Vestas.V100.2000': 2000,
    'Vestas.V110.2000': 2000,
    'Vestas.V112.3000': 3000,
    'Vestas.V112.3300': 3300,
    'Vestas.V164.7000': 7000,
    'Wind.World.W3700': 3700,
    'Wind.World.W4200': 4200,
    'Windmaster.WM28.300': 300,
    'Windmaster.WM43.750': 750,
    'Windflow.500': 500,
    'XANT.M21.100': 100
}

# keep one model per unique capacity
unique_rep = {}
for model, cap in rep_turbines.items():
    if cap not in unique_rep:
        unique_rep[cap] = model

def assign_rep_turbine(capacity):
    if pd.isna(capacity):
        return None, None
    # find closest capacity
    closest, _ = min(unique_rep.items(), key=lambda x: abs(x[0] - capacity))
    return unique_rep[closest], closest

# apply assignment
rep[["Representative_New_Model", "Representative_New_Capacity"]] = (
    rep["SingleWT_Capacity"]
       .apply(lambda x: assign_rep_turbine(x))
       .apply(pd.Series)
)
rep["Capacity_Diff"] = (rep["SingleWT_Capacity"] - rep["Representative_New_Capacity"]).abs()

# compute total new capacity
rep["Total_New_Capacity"] = rep["Representative_New_Capacity"] * rep["Number of turbines"]

# ─── 4. Prepare layout and power‐curve helper ────────────────────────────────────
layout = rep[["Latitude","Longitude","Representative_New_Model","Total_New_Capacity"]].rename(
    columns={"Longitude":"x","Latitude":"y"}
)

# load power curves
pc_df = pd.read_csv(path_power_curve)
if pc_df.columns[0].strip().lower() == "speed":
    pc_df.rename(columns={pc_df.columns[0]: "Wind speed"}, inplace=True)

def get_turbine_cfg(name):
    if name not in pc_df.columns:
        return None
    return {
        "hub_height": 100,
        "V":   pc_df["Wind speed"].tolist(),
        "POW": pc_df[name].tolist(),
        "P":   float(pc_df[name].max())
    }

# build ERA5 cutout
margin = 0.1
bounds = [
    layout.x.min() - margin, layout.y.min() - margin,
    layout.x.max() + margin, layout.y.max() + margin
]
print("Building ERA5 cutout…")
cutout = atlite.Cutout(
    path_cutout,
    bounds=bounds,
    time=slice("2010-01-01","2010-12-31"),
    module="10m_wind_speed"
)
print("Cutout ready\n")

# ─── 5. First pass: compute CF over NUTS polygons ───────────────────────────────
print("Running first-pass CF calculation over NUTS regions…")
for turb in rep["Representative_New_Model"].dropna().unique():
    cfg = get_turbine_cfg(turb)
    if cfg is None:
        print(f"⚠️  No power curve for {turb}, skipping.")
        continue

    pts = layout[layout.Representative_New_Model == turb].copy()
    if pts.empty:
        continue

    layout_list = cutout.layout_from_capacity_list(pts, col="Total_New_Capacity")
    cf = cutout.wind(turbine=cfg, layout=layout_list, shapes=spatial, per_unit=True)
    mean_cf = cf.mean(dim="time").to_dataframe(name="CF").rename_axis("NUTS_ID")
    cf_map = spatial.merge(mean_cf, left_index=True, right_index=True)

    gdf = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts.x, pts.y),
        crs=spatial.crs
    )
    joined = gpd.sjoin(gdf, cf_map[["CF","geometry"]], how="left", predicate="within")
    rep.loc[joined.index, "CapacityFactor"] = joined["CF"]

# ─── 6. Fallback: custom-country pass for any remaining NaNs ────────────────────
mask_nan = rep["CapacityFactor"].isna()
print(f"{mask_nan.sum()} rows still missing CF → running fallback on custom countries\n")
if mask_nan.any():
    USE_CUSTOM = [
        "Ukraine","Belarus","Kosovo","Iceland",
        "Slovenia","Bosnia and Herz.","Faeroe Is."
    ]
    custom = gpd.read_file(path_custom_geo).set_index("name").loc[USE_CUSTOM]

    layout2 = layout[mask_nan].copy()
    for turb in layout2["Representative_New_Model"].dropna().unique():
        cfg = get_turbine_cfg(turb)
        if cfg is None:
            print(f"⚠️  No power curve for {turb} in fallback, skipping.")
            continue

        pts2 = layout2[layout2.Representative_New_Model == turb]
        print(f"Fallback for {turb}: {len(pts2)} point(s)")
        layout_list = cutout.layout_from_capacity_list(pts2, col="Total_New_Capacity")
        cf2 = cutout.wind(turbine=cfg, layout=layout_list, shapes=custom, per_unit=True)
        mean_cf2 = cf2.mean(dim="time").to_dataframe(name="CF").rename_axis("name")
        cf_map2 = custom[["geometry"]].merge(mean_cf2, left_index=True, right_index=True)

        gdf2 = gpd.GeoDataFrame(
            pts2,
            geometry=gpd.points_from_xy(pts2.x, pts2.y),
            crs=custom.crs
        )
        joined2 = gpd.sjoin(gdf2, cf_map2[["CF","geometry"]], how="left", predicate="within")
        rep.loc[joined2.index, "CapacityFactor"] = joined2["CF"]
        for idx, row in joined2.iterrows():
            print(f"  • Row {idx}: CF = {row['CF']:.3f}")

# ─── 7. Save final results ───────────────────────────────────────────────────────
print("\nSaving completed CF results to:", output_path)
rep.to_excel(output_path)
print("Done ✅")
