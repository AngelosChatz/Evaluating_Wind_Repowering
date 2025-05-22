import pandas as pd
import math
import re
from pathlib import Path

# Define the base directory (the directory containing this script)
base_dir = Path(__file__).resolve().parent

# Define directories for input files and output files
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create the results folder if it doesn't exist

# Set file paths relative to the defined directories
input_excel = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
output_excel = results_dir / "Approach_1.xlsx"

# List of turbine dictionaries with properties for each IEC class
turbines = [
    # IEC Class 1 turbines
    {"Model": "Siemens.SWT.3.0.101", "Capacity": 3.0, "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 4.3-120", "Capacity": 4.3, "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "Siemens.SWT.3.6.107", "Capacity": 3.6, "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6-154", "Capacity": 6.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.5-167", "Capacity": 8.5, "RotorDiameter": 167, "IEC_Class_Num": 1},
    {"Model": "Nordex 100-3300", "Capacity": 3.3, "RotorDiameter": 100, "IEC_Class_Num": 1},

    # IEC Class 2 turbines
    {"Model": "Enercon.E82.3000", "Capacity": 3.0, "RotorDiameter": 82, "IEC_Class_Num": 2},
    {"Model": "Vestas V90-3.0", "Capacity": 3.0, "RotorDiameter": 90, "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.0", "Capacity": 4.0, "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 4.5-145", "Capacity": 4.5, "RotorDiameter": 145, "IEC_Class_Num": 2},

    # IEC Class 3 turbines
    {"Model": "Enercon E-115-3.000", "Capacity": 3.0, "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "Siemens SWT 6.6-170", "Capacity": 6.6, "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Nordex N131-3000", "Capacity": 3.0, "RotorDiameter": 131, "IEC_Class_Num": 3},

    # IEC Class S turbines
    {"Model": "Enercon E126-4000", "Capacity": 4.0, "RotorDiameter": 126, "IEC_Class_Num": 0},
    {"Model": "Enercon E175-6000", "Capacity": 5.0, "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "Vestas V150-6.0", "Capacity": 6.0, "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "Vestas V164-9500", "Capacity": 9.5, "RotorDiameter": 164, "IEC_Class_Num": 0},
    {"Model": "Nordex 149-4500", "Capacity": 4.5, "RotorDiameter": 149, "IEC_Class_Num": 0}
]

def land_area(diameter, terrain_type):
    """
    Returns the estimated land area required (m²) per turbine depending on the terrain type:
      - Flat terrain    : 28 * D²   (derived from 7D x 4D spacing)
      - Complex terrain : 54 * D²   (derived from 9D x 6D spacing)
    """
    t = str(terrain_type).lower()
    if t == "flat":
        return 28 * (diameter ** 2)
    elif t == "complex":
        return 54 * (diameter ** 2)
    else:
        return 28 * (diameter ** 2)

def best_fitting_turbine_for_iec_class(turbine_list, terrain_type, iec_class_num, available_area):
    # Filter turbines by matching IEC class
    matching_turbines = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]

    if not matching_turbines:
        return None, 0, 0, None

    # Sort turbines by energy density (capacity per required area) in descending order
    sorted_turbines = sorted(
        matching_turbines,
        key=lambda t: t["Capacity"] / land_area(t["RotorDiameter"], terrain_type),
        reverse=True
    )

    # Try each turbine until one fits at least one unit
    for t in sorted_turbines:
        turbine_area = land_area(t["RotorDiameter"], terrain_type)
        new_turb_count = int(available_area // turbine_area)  # floor division to get full units
        if new_turb_count >= 1:
            best_ratio = t["Capacity"] / turbine_area
            return t, best_ratio, new_turb_count, turbine_area

    return None, 0, 0, None

def main():
    # Read the wind farm data from Excel using the relative input path
    df = pd.read_excel(input_excel)
    print(f"Read {len(df)} rows from {input_excel}.")

    # Filter to keep only rows where "Active in 2022" is True
    df = df[df["Active in 2022"] == True]

    # Convert IEC_Class_Num to integer if needed
    if "IEC_Class_Num" in df.columns:
        df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce").fillna(0).astype(int)
    else:
        print("Warning: IEC_Class_Num column not found in the Excel file.")

    # Prepare lists to store the recommendation results
    recommended_models = []
    recommended_capacities = []
    new_turbine_counts = []
    total_new_capacities = []

    for idx, row in df.iterrows():
        # Extract site data
        terrain = row.get("Terrain_Type", "flat")
        iec_num = row.get("IEC_Class_Num", 0)  # after conversion, 0 means invalid/missing

        # Use the Total Park Area (m²) column from the DataFrame
        old_area = row.get("Total Park Area (m²)")
        if old_area is None or (isinstance(old_area, float) and math.isnan(old_area)):
            print(f"Row {idx}: Total Park Area (m²) is missing. Skipping calculation for this row.")
            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
            continue

        # Find the best replacement turbine that fits in the available area
        best_turb, best_ratio, new_turb_count, new_area = best_fitting_turbine_for_iec_class(
            turbine_list=turbines,
            terrain_type=terrain,
            iec_class_num=iec_num,
            available_area=old_area
        )

        if best_turb is None:
            print(f"Row {idx}: No turbine fits within the old park area = {old_area:.2f} m².")
            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
        else:
            # Calculate total new capacity based on the number of turbines that can fit
            new_total_capacity = new_turb_count * best_turb["Capacity"]

            # Store results
            recommended_models.append(best_turb["Model"])
            recommended_capacities.append(best_turb["Capacity"])
            new_turbine_counts.append(new_turb_count)
            total_new_capacities.append(new_total_capacity)

    # Add new columns to the DataFrame with the calculated recommendations
    df["Recommended_WT_Model"] = recommended_models
    df["Recommended_WT_Capacity"] = recommended_capacities
    df["New_Turbine_Count"] = new_turbine_counts
    df["Total_New_Capacity"] = total_new_capacities

    # Save the updated DataFrame to Excel using the relative output path
    df.to_excel(output_excel, index=False)
    print(f"\nUpdated database saved to {output_excel}")

if __name__ == "__main__":
    main()
