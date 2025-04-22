import pandas as pd
import math
import re
from pathlib import Path

# Define the base directory and set up folders for input and output
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"


# Define relative file paths for the input and output Excel files
input_excel = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
output_excel = results_dir / "Approach_4.xlsx"

# Wind turbine dataset (updated with new models)
turbines = [
    # IEC Class 1 turbines
    {"Model": "Siemens SWT 3-101", "Capacity": 3.0, "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 4.3-120", "Capacity": 4.3, "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 8-154", "Capacity": 8.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens.SWT.3.6.107", "Capacity": 3.6, "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6-154", "Capacity": 6.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.5-167", "Capacity": 8.5, "RotorDiameter": 167, "IEC_Class_Num": 1},
    {"Model": "Nordex 100-3300", "Capacity": 3.3, "RotorDiameter": 100, "IEC_Class_Num": 1},

    # IEC Class 2 turbines
    {"Model": "Enercon.E82.3000", "Capacity": 3.0, "RotorDiameter": 82, "IEC_Class_Num": 2},
    {"Model": "Vestas.V90.3000", "Capacity": 3.0, "RotorDiameter": 90, "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.0", "Capacity": 4.0, "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 4.5-145", "Capacity": 4.5, "RotorDiameter": 145, "IEC_Class_Num": 2},

    # IEC Class 3 turbines
    {"Model": "Enercon E-115-3.000", "Capacity": 3.0, "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "Siemens SWT 6.6-170", "Capacity": 6.6, "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Vestas V136-3.45", "Capacity": 3.45, "RotorDiameter": 136, "IEC_Class_Num": 3},
    {"Model": "Nordex.N131.3000", "Capacity": 3.0, "RotorDiameter": 131, "IEC_Class_Num": 3},

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
      - Flat terrain: 28 * D^2 (derived from 7D x 4D spacing)
      - Complex terrain: 54 * D^2 (derived from 9D x 6D spacing)
    """
    t = str(terrain_type).lower()
    if t == "flat":
        return 28 * (diameter ** 2)
    elif t == "complex":
        return 54 * (diameter ** 2)
    else:
        return 28 * (diameter ** 2)

def best_fitting_turbine_for_iec_class_min_turbines(turbine_list, terrain_type, iec_class_num, available_area):
    matching_turbines = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]
    if not matching_turbines:
        return None, 0, 0, None

    candidates = []
    for t in matching_turbines:
        turbine_area = land_area(t["RotorDiameter"], terrain_type)
        n_float = available_area / turbine_area
        if n_float >= 1:
            floor_n = math.floor(n_float)
            # If fractional part is ≥ 0.8, round up.
            count = math.ceil(n_float) if (n_float - floor_n) >= 0.8 else floor_n
            total_capacity = count * t["Capacity"]
            candidates.append((t, count, turbine_area, total_capacity))

    if not candidates:
        return None, 0, 0, None

    # Pick candidate maximizing total capacity; if tied, one with fewer turbines.
    best_candidate_tuple = max(candidates, key=lambda x: (x[3], -x[1]))
    best_candidate, best_turb_count, best_turbine_area, best_total_capacity = best_candidate_tuple
    best_ratio = best_candidate["Capacity"] / best_turbine_area if best_turbine_area else 0
    return best_candidate, best_ratio, best_turb_count, best_turbine_area

def main():
    # Read input Excel file using the relative path
    df = pd.read_excel(input_excel)
    print(f"Read {len(df)} rows from {input_excel}.")

    # Filter for active wind farms
    df = df[df["Active in 2022"] == True]

    # Ensure IEC_Class_Num column is integer
    if "IEC_Class_Num" in df.columns:
        df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce").fillna(0).astype(int)
    else:
        print("Warning: IEC_Class_Num column not found in the Excel file.")

    # Prepare lists to hold computed result values
    recommended_models = []
    recommended_capacities = []
    new_turbine_counts = []
    total_new_capacities = []
    new_total_park_areas = []

    for idx, row in df.iterrows():
        terrain = row.get("Terrain_Type", "flat")
        iec_num = row.get("IEC_Class_Num", 0)
        old_area = row.get("Total Park Area (m²)")
        if old_area is None or (isinstance(old_area, float) and math.isnan(old_area)):
            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
            new_total_park_areas.append(None)
            continue

        best_turb, best_ratio, new_turb_count, turbine_area = best_fitting_turbine_for_iec_class_min_turbines(
            turbine_list=turbines,
            terrain_type=terrain,
            iec_class_num=iec_num,
            available_area=old_area
        )

        if best_turb is None:
            print(f"Row {idx}: No turbine fits within the available park area = {old_area:.2f} m².")
            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
            new_total_park_areas.append(None)
        else:
            new_total_capacity = new_turb_count * best_turb["Capacity"]
            new_total_park_area = turbine_area * new_turb_count

            recommended_models.append(best_turb["Model"])
            recommended_capacities.append(best_turb["Capacity"])
            new_turbine_counts.append(new_turb_count)
            total_new_capacities.append(new_total_capacity)
            new_total_park_areas.append(new_total_park_area)

    # Append result columns to the DataFrame
    df["Recommended_WT_Model"] = recommended_models
    df["Recommended_WT_Capacity"] = recommended_capacities
    df["New_Turbine_Count"] = new_turbine_counts
    df["Total_New_Capacity"] = total_new_capacities
    df["New_Total_Park_Area (m²)"] = new_total_park_areas

    # Save the updated DataFrame to Excel using the relative output path
    df.to_excel(output_excel, index=False)
    print(f"\nUpdated database saved to {output_excel}")

if __name__ == "__main__":
    main()
