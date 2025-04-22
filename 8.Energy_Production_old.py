#!/usr/bin/env python
import atlite
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt



# Input files and output file paths:
path_repowering_results = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2.xlsx"
path_cutout = r"D:\SET 2023\Thesis Delft\Model\atlite_example\data\era5.nc"  # ERA5 netCDF file
path_spatial_units = r"D:\SET 2023\Thesis Delft\Model\atlite_example\data\NUTS_RG_01M_2016_4326.geojson"
path_power_curve = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\data\Power Curves.csv"
output_path = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf_old.xlsx"


# 2. LOAD REPLOWERING RESULTS & SPATIAL UNITS
repowering_results = pd.read_excel(path_repowering_results, sheet_name="Sheet1", index_col=0)
spatial_units = gpd.read_file(path_spatial_units).set_index("NUTS_ID")


# 3. ASSIGN REPRESENTATIVE WIND TURBINE MODELS
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
#keep only one model per unique capacity.
unique_rep_turbines = {}
for model, capacity in rep_turbines.items():
    if capacity not in unique_rep_turbines:
        unique_rep_turbines[capacity] = model



def assign_representative_turbine(single_capacity):
    if pd.isna(single_capacity):
        return None, None
    closest_cap = None
    min_diff = float('inf')
    for cap in unique_rep_turbines.keys():
        diff = abs(single_capacity - cap)
        if diff < min_diff:
            min_diff = diff
            closest_cap = cap
    return unique_rep_turbines[closest_cap], closest_cap


# Create new columns: Representative_New_Model, Representative_New_Capacity, and Capacity_Diff
repowering_results[['Representative_New_Model', 'Representative_New_Capacity']] = (
    repowering_results['SingleWT_Capacity']
    .apply(lambda x: assign_representative_turbine(x))
    .apply(pd.Series)
)
repowering_results['Capacity_Diff'] = abs(
    repowering_results['SingleWT_Capacity'] - repowering_results['Representative_New_Capacity']
)


# 4. PREPARE THE LAYOUT FOR ATLITE
# Use the new representative assignment for layout.
cols_layout = ["Latitude", "Longitude", "Representative_New_Model", "Total_New_Capacity"]
layout_point = repowering_results[cols_layout].copy()
layout_point = layout_point.rename(columns={"Longitude": "x", "Latitude": "y"})

# Compute spatial bounds from the layout points (adding a small margin)
margin = 0.1
x_min = layout_point["x"].min() - margin
x_max = layout_point["x"].max() + margin
y_min = layout_point["y"].min() - margin
y_max = layout_point["y"].max() + margin
bounds = [x_min, y_min, x_max, y_max]


# 5. LOAD POWER CURVE DATA FROM THE CSV FILE
pc_df = pd.read_csv(path_power_curve)
if pc_df.columns[0].strip().lower() == "speed":
    pc_df.rename(columns={pc_df.columns[0]: "Wind speed"}, inplace=True)


def get_turbine_config_from_df(df, turbine_name):
    """
    Returns a turbine configuration dictionary for turbine_name if found as a column in df.
    Assumes the first column is "Wind speed".
    """
    if turbine_name in df.columns:
        wind_speeds = df["Wind speed"].values
        power_values = df[turbine_name].values
        config = {
            "hub_height": 100,  # Adjust this if needed.
            "V": wind_speeds.tolist(),
            "POW": power_values.tolist(),
            "P": max(power_values)
        }
        return config
    return None

# 7. CREATE THE ATLITE CUTOUT WITH TIME, MODULE, AND BOUNDS
cutout = atlite.Cutout(
    path_cutout,
    bounds=bounds,
    time=slice("2010-01-01", "2010-12-31"),
    module="10m_wind_speed"
)

# 8. CALCULATE CAPACITY FACTORS FOR EACH UNIQUE REPRESENTATIVE TURBINE MODEL
turbine_models = repowering_results["Representative_New_Model"].unique()

for turbine in turbine_models:
    # Get the turbine configuration from the power curve CSV.
    turbine_config = get_turbine_config_from_df(pc_df, turbine)
    if turbine_config is None:
        print(f"No matching power curve data found for turbine: {turbine}")
        continue

    # Filter layout points for entries with this representative turbine.
    layout_point_turbine = layout_point[layout_point["Representative_New_Model"] == turbine].copy()
    if layout_point_turbine.empty:
        continue

    # Create the layout using the 'Total_New_Capacity' column.
    layout = cutout.layout_from_capacity_list(layout_point_turbine, col="Total_New_Capacity")

    # Calculate wind capacity factors using atlite.
    capacityfactors = cutout.wind(
        turbine=turbine_config,
        layout=layout,
        shapes=spatial_units,
        per_unit=True,
    )

    # Save the time-resolved capacity factors to a NetCDF file.
    safe_turbine_name = turbine.replace("/", "_").replace(" ", "_")
    capacityfactors.to_netcdf(f"cf_wind_{safe_turbine_name}.nc")

    # Compute the mean capacity factor (over time) and convert to a DataFrame.
    mean_cf = capacityfactors.mean(dim="time").to_dataframe(name=turbine)

    # Merge the mean capacity factors with the spatial units (polygons).
    cf_map = spatial_units.merge(mean_cf, on="NUTS_ID")

    # Create a GeoDataFrame for the turbine locations for this turbine model.
    gdf_turbine = gpd.GeoDataFrame(
        layout_point_turbine,
        geometry=gpd.points_from_xy(layout_point_turbine["x"], layout_point_turbine["y"]),
        crs=spatial_units.crs
    )

    # Spatially join the mean capacity factor from the polygons onto each turbine point.
    gdf_joined = gpd.sjoin(gdf_turbine, cf_map[[turbine, "geometry"]], how="left", predicate="within")
    gdf_joined = gdf_joined.rename(columns={turbine: "CapacityFactor"})

    # Update the repowering_results DataFrame with the capacity factor.
    repowering_results.loc[gdf_joined.index, "CapacityFactor"] = gdf_joined["CapacityFactor"]

repowering_results.to_excel(output_path)
print(f"Updated repowering results saved to: {output_path}")
