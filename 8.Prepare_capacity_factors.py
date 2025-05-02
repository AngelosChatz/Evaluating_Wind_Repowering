import atlite
import pandas as pd
import geopandas as gpd
from pathlib import Path


# 1. CONFIGURATION
DATA_DIR = Path("data")
RESULTS_DIR = Path(r"D:/SET 2023/Thesis Delft/Model/Evaluating_Wind_Repowering/results")

# Input files
REPOWER_FILE = RESULTS_DIR / "Approach_2.xlsx"
ERA5_FILE = DATA_DIR / "era5.nc"
SPATIAL_UNITS_FILE = DATA_DIR / "NUTS_RG_01M_2016_4326.geojson"
POWER_CURVE_FILES = [
    DATA_DIR / "Power Curves.csv",
    DATA_DIR / "Power Curves 2.csv"
]

# Output file
OUTPUT_FILE = RESULTS_DIR / "Approach_2_Cf+old.xlsx"

# Default hub height for capacity factor calculation
DEFAULT_HUB_HEIGHT = 100  # in meters


# 2. HELPER FUNCTIONS
def load_power_curves(file_paths):
    """
    Load power curve CSVs and ensure the wind speed column is standardized.
    Returns a list of DataFrames.
    """
    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path)
        # Ensure first column is 'Wind speed'
        first_col = df.columns[0]
        if first_col.strip().lower() != "wind speed":
            df = df.rename(columns={first_col: "Wind speed"})
        dataframes.append(df)
    return dataframes


def find_turbine_config(power_dfs, model_name):
    """
    Given a list of power curve DataFrames, find the one containing the
    specified turbine model and return a config dict (hub height, wind speeds, power).
    """
    for df in power_dfs:
        if model_name in df.columns:
            speeds = df["Wind speed"].values.tolist()
            powers = df[model_name].values.tolist()
            return {
                "hub_height": DEFAULT_HUB_HEIGHT,
                "V": speeds,
                "POW": powers,
                "P": max(powers)
            }
    return None


# 3. MAIN WORKFLOW
def main():
    # 3.1 Load repowering recommendations
    print("Loading repowering results...")
    repower = pd.read_excel(REPOWER_FILE, sheet_name="Sheet1", index_col=0)

    # 3.2 Load spatial data and ERA5 cutout
    print("Reading spatial units and ERA5 data...")
    spatial_units = gpd.read_file(SPATIAL_UNITS_FILE).set_index("NUTS_ID")
    cutout = atlite.Cutout(str(ERA5_FILE))

    # 3.3 Prepare turbine location data
    print("Preparing turbine layout points...")
    layout = (
        repower
        .rename(columns={"Longitude": "x", "Latitude": "y"})
        [["x", "y", "Recommended_WT_Model", "Total_New_Capacity"]]
    )

    # 3.4 Load power curves
    print("Loading power curve data...")
    power_curves = load_power_curves(POWER_CURVE_FILES)

    # 3.5 Iterate through unique turbine models
    for model in repower["Recommended_WT_Model"].unique():
        print(f"Processing turbine model: {model}")
        config = find_turbine_config(power_curves, model)
        if not config:
            print(f"  Warning: No power curve found for '{model}'. Skipping.")
            continue

        # Filter locations for this model
        pts = layout[layout["Recommended_WT_Model"] == model]
        if pts.empty:
            print(f"  No locations for '{model}'.")
            continue

        # Build layout and compute capacity factors
        wind_layout = cutout.layout_from_capacity_list(pts, col="Total_New_Capacity")
        cf_timeseries = cutout.wind(
            turbine=config,
            layout=wind_layout,
            shapes=spatial_units,
            per_unit=True
        )

        # Save detailed time series
        safe_name = model.replace("/", "_").replace(" ", "_")
        cf_timeseries.to_netcdf(f"cf_timeseries_{safe_name}.nc")

        # Compute mean capacity factor
        mean_cf = cf_timeseries.mean(dim="time").to_dataframe(name=model)

        # Merge capacity factor by spatial unit
        cf_map = spatial_units.merge(mean_cf, on="NUTS_ID")

        # Spatial join back to turbine points
        gdf_pts = gpd.GeoDataFrame(
            pts,
            geometry=gpd.points_from_xy(pts["x"], pts["y"]),
            crs=spatial_units.crs
        )
        joined = gpd.sjoin(gdf_pts, cf_map[[model, "geometry"]], how="left", predicate="within")
        repower.loc[joined.index, "CapacityFactor"] = joined[model]

    # 3.6 Save updated results
    print(f"Saving updated results to {OUTPUT_FILE}...")
    repower.to_excel(OUTPUT_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
