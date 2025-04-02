import pandas as pd
import math
import re

# New wind turbine dataset (updated with new models)
turbines = [
    # IEC Class 1 turbines
    {"Model": "swt 3 101", "Capacity": 3.0, "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "swt 4.3 120", "Capacity": 4.3, "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "SWT 8 154", "Capacity": 8.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "swt 3.6 107", "Capacity": 3.6, "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6 154", "Capacity": 6.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8,5 167", "Capacity": 8.5, "RotorDiameter": 167, "IEC_Class_Num": 1},
    {"Model": "nordex 100 3300", "Capacity": 3.3, "RotorDiameter": 100, "IEC_Class_Num": 1},

    # IEC Class 2 turbines
    {"Model": "e82 3000", "Capacity": 3.0, "RotorDiameter": 82, "IEC_Class_Num": 2},
    {"Model": "V90 3 MW", "Capacity": 3.0, "RotorDiameter": 90, "IEC_Class_Num": 2},
    {"Model": "v136 4 MW", "Capacity": 4.0, "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG  4.5 145", "Capacity": 4.5, "RotorDiameter": 145, "IEC_Class_Num": 2},

    # IEC Class 3 turbines
    {"Model": "Enercon E-115 3.000", "Capacity": 3.0, "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "SWT 6.6 170", "Capacity": 6.6, "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Nordex 131 4000", "Capacity": 4.0, "RotorDiameter": 131, "IEC_Class_Num": 3},
    {"Model": "Nordex N131 3000", "Capacity": 3.0, "RotorDiameter": 131, "IEC_Class_Num": 3},

    # IEC Class S turbines
    {"Model": "e126 4000", "Capacity": 4.0, "RotorDiameter": 126, "IEC_Class_Num": 0},
    {"Model": "e175 6000", "Capacity": 5.0, "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "v150 6 MW", "Capacity": 6.0, "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "v164 9500", "Capacity": 9.5, "RotorDiameter": 164, "IEC_Class_Num": 0},
    {"Model": "Nordex 149 4500", "Capacity": 4.5, "RotorDiameter": 149, "IEC_Class_Num": 0}
]

def land_area(diameter, terrain_type):
    """
    Returns the estimated land area required (m^2) per turbine depending on the terrain type:
      - Flat terrain    : 28 * D^2   (derived from 7D x 4D spacing)
      - Complex terrain : 54 * D^2   (derived from 9D x 6D spacing)
    """
    t = str(terrain_type).lower()
    if t == "flat":
        return 28 * (diameter ** 2)
    elif t == "complex":
        return 54 * (diameter ** 2)
    else:
        # Default to flat spacing if unknown
        return 28 * (diameter ** 2)

def best_fitting_turbine_for_iec_class_min_turbines(turbine_list, terrain_type, iec_class_num, available_area):
    matching_turbines = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]
    if not matching_turbines:
        return None, 0, 0, None

    best_candidate = None
    best_total_capacity = -1
    best_turb_count = None
    best_turbine_area = None

    for t in matching_turbines:
        turbine_area = land_area(t["RotorDiameter"], terrain_type)
        new_turb_count = int(available_area // turbine_area)
        if new_turb_count < 1:
            continue  # Skip if at least one turbine cannot fit
        total_capacity = new_turb_count * t["Capacity"]

        # Choose candidate with higher total capacity; if tied, choose one with fewer turbines.
        if (total_capacity > best_total_capacity) or \
           (total_capacity == best_total_capacity and new_turb_count < best_turb_count):
            best_candidate = t
            best_total_capacity = total_capacity
            best_turb_count = new_turb_count
            best_turbine_area = turbine_area

    if best_candidate is None:
        return None, 0, 0, None

    best_ratio = best_candidate["Capacity"] / best_turbine_area
    return best_candidate, best_ratio, best_turb_count, best_turbine_area

def main():
    # File paths
    input_excel = r"D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
    output_excel = r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new_ninja_models.xlsx"

    # Read input Excel file
    df = pd.read_excel(input_excel)
    print(f"Read {len(df)} rows from {input_excel}.")

    # Filter rows where "Active in 2022" is True
    df = df[df["Active in 2022"] == True]

    # Convert IEC_Class_Num to integer
    if "IEC_Class_Num" in df.columns:
        df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce").fillna(0).astype(int)
    else:
        print("Warning: IEC_Class_Num column not found in the Excel file.")

    # Initialize result columns
    recommended_models = []
    recommended_capacities = []
    new_turbine_counts = []
    total_new_capacities = []
    new_total_park_areas = []

    for idx, row in df.iterrows():
        terrain = row.get("Terrain_Type", "flat")
        iec_num = row.get("IEC_Class_Num", 0)  # 0 indicates invalid if conversion fails

        # Use "Total Park Area (m²)" column
        old_area = row.get("Total Park Area (m²)")
        if old_area is None or (isinstance(old_area, float) and math.isnan(old_area)):
            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
            new_total_park_areas.append(None)
            continue

        # Find the best replacement turbine that fits in the available area
        best_turb, best_ratio, new_turb_count, turbine_area = best_fitting_turbine_for_iec_class_min_turbines(
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
            new_total_park_areas.append(None)
        else:
            # Calculate new total capacity and park area
            new_total_capacity = new_turb_count * best_turb["Capacity"]
            new_total_park_area = turbine_area * new_turb_count

            # Store results
            recommended_models.append(best_turb["Model"])
            recommended_capacities.append(best_turb["Capacity"])
            new_turbine_counts.append(new_turb_count)
            total_new_capacities.append(new_total_capacity)
            new_total_park_areas.append(new_total_park_area)

    # Add results as new columns in the DataFrame
    df["Recommended_WT_Model"] = recommended_models
    df["Recommended_WT_Capacity"] = recommended_capacities
    df["New_Turbine_Count"] = new_turbine_counts
    df["Total_New_Capacity"] = total_new_capacities
    df["New_Total_Park_Area (m²)"] = new_total_park_areas

    # Save the updated DataFrame to Excel
    df.to_excel(output_excel, index=False)
    print(f"\nUpdated database saved to {output_excel}")

if __name__ == "__main__":
    main()
