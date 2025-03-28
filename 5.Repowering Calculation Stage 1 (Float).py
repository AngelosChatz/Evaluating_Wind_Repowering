import pandas as pd
import math
import re


turbines = [
    # Class I
    {"Model": "Vestas V90-3.0 MW", "Capacity": 3.00, "RotorDiameter": 90, "IEC_Class_Num": 1},
    {"Model": "E-82 EP2 E4", "Capacity": 2.35, "RotorDiameter": 82, "IEC_Class_Num": 1},
    {"Model": "Enercon E-136 EP5", "Capacity": 4.65, "RotorDiameter": 136, "IEC_Class_Num": 1},
    {"Model": "Vestas V117-4.5 MW", "Capacity": 4.50, "RotorDiameter": 117, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6.6-155", "Capacity": 6.60, "RotorDiameter": 155, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.0-167", "Capacity": 8.00, "RotorDiameter": 167, "IEC_Class_Num": 1},

    # Class II
    {"Model": "Siemens Gamesa SG 3.0-132", "Capacity": 3.00, "RotorDiameter": 132, "IEC_Class_Num": 2},
    {"Model": "E-82 EP2 E4", "Capacity": 3.00, "RotorDiameter": 82, "IEC_Class_Num": 2},
    {"Model": "Nordex N90-2.5", "Capacity": 2.50, "RotorDiameter": 90, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 3.4-132", "Capacity": 3.40, "RotorDiameter": 132, "IEC_Class_Num": 2},
    {"Model": "Enercon E-138 EP3", "Capacity": 4.20, "RotorDiameter": 138, "IEC_Class_Num": 2},
    {"Model": "Vestas V117-4.2 MW", "Capacity": 4.20, "RotorDiameter": 117, "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.5", "Capacity": 4.50, "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Nordex N155/5.X", "Capacity": 5.00, "RotorDiameter": 155, "IEC_Class_Num": 2},
    {"Model": "Enercon E-147 EP5", "Capacity": 5.00, "RotorDiameter": 147, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 5.8-155", "Capacity": 5.80, "RotorDiameter": 155, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 6.6-170", "Capacity": 6.60, "RotorDiameter": 170, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 7.0-170", "Capacity": 7.00, "RotorDiameter": 155, "IEC_Class_Num": 2},

    # Class III
    {"Model": "Siemens Gamesa SG 3.4-145", "Capacity": 3.40, "RotorDiameter": 145, "IEC_Class_Num": 3},
    {"Model": "Vestas V150-4.5", "Capacity": 4.50, "RotorDiameter": 150, "IEC_Class_Num": 3},
    {"Model": "Enercon E-160 EP5", "Capacity": 4.60, "RotorDiameter": 160, "IEC_Class_Num": 3},
    {"Model": "Nordex N163/5.X", "Capacity": 5.70, "RotorDiameter": 163, "IEC_Class_Num": 3},
    {"Model": "Siemens Gamesa SG 5.8-170", "Capacity": 5.80, "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Nordex N175/6.X", "Capacity": 6.80, "RotorDiameter": 175, "IEC_Class_Num": 3},

    # Class S
    {"Model": "Vestas V120-2.2 MW", "Capacity": 2.20, "RotorDiameter": 120, "IEC_Class_Num": 0},
    {"Model": "Vestas V105-3.45 MW", "Capacity": 3.45, "RotorDiameter": 105, "IEC_Class_Num": 0},
    {"Model": "Nordex N133/4.8", "Capacity": 4.80, "RotorDiameter": 133, "IEC_Class_Num": 0},
    {"Model": "Vestas V150-6.0 MW", "Capacity": 6.00, "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "Enercon E-175 EP5", "Capacity": 6.50, "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "Vestas V172-7.2 MW", "Capacity": 7.20, "RotorDiameter": 172, "IEC_Class_Num": 0}
]


def land_area(diameter, terrain_type):

    t = str(terrain_type).lower()
    if t == "flat":
        return 28 * (diameter ** 2)
    elif t == "complex":
        return 54 * (diameter ** 2)
    else:
        # If unknown or missing, default to flat spacing
        return 28 * (diameter ** 2)

def best_turbine_for_iec_class(turbine_list, terrain_type, iec_class_num):

    # Filter turbines by matching IEC class
    matching_turbines = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]

    # Debug: show how many matching turbines we got


    if not matching_turbines:
        return None, 0

    best_ratio = -1
    best_turbine = None

    for t in matching_turbines:
        d = t["RotorDiameter"]
        capacity = t["Capacity"]
        a = land_area(d, terrain_type)
        ratio = capacity / a

        if ratio > best_ratio:
            best_ratio = ratio
            best_turbine = t

    return best_turbine, best_ratio


def main():
    # Adjust these to your actual Excel file paths
    input_excel = r"D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
    output_excel = r"D:\SET 2023\Thesis Delft\Model\Repowering_Stage_1_float.xlsx"

    # Read the wind farm data from Excel
    df = pd.read_excel(input_excel)
    print(f"Read {len(df)} rows from {input_excel}.")

    # Filter to keep only the rows where Active in 2022 is True
    df = df[df["Active in 2022"] == True]

    # Convert IEC_Class_Num to integer if needed
    if "IEC_Class_Num" in df.columns:
        df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce").fillna(0).astype(int)
    else:
        print("Warning: IEC_Class_Num column not found in the Excel file.")

    # Prepare columns for results
    recommended_models = []
    recommended_capacities = []
    new_turbine_counts = []
    total_new_capacities = []

    for idx, row in df.iterrows():
        # 3.1. Extract site data
        terrain = row.get("Terrain_Type", "flat")
        iec_num = row.get("IEC_Class_Num", 0)  # after numeric conversion, 0 means invalid

        # Use the Total Park Area (m²) column directly (assumed to be provided)
        old_area = row.get("Total Park Area (m²)")
        if old_area is None or (isinstance(old_area, float) and math.isnan(old_area)):
            print(f"Row {idx}: Total Park Area (m²) is missing. Skipping calculation for this row.")
            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
            continue

        # 3.2. Find the best replacement turbine
        best_turb, best_ratio = best_turbine_for_iec_class(
            turbine_list=turbines,
            terrain_type=terrain,
            iec_class_num=iec_num
        )

        if best_turb is None:

            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
        else:
            # 3.3. Calculate how many new turbines fit in the total park area
            new_area = land_area(best_turb["RotorDiameter"], terrain)
            new_turb_count = old_area / new_area

            # 3.4. Calculate total new capacity
            new_total_capacity = new_turb_count * best_turb["Capacity"]


            # 3.5. Store results
            recommended_models.append(best_turb["Model"])
            recommended_capacities.append(best_turb["Capacity"])
            new_turbine_counts.append(new_turb_count)
            total_new_capacities.append(new_total_capacity)

    # 3.6. Add new columns to the DataFrame
    df["Recommended_WT_Model"] = recommended_models
    df["Recommended_WT_Capacity"] = recommended_capacities
    df["New_Turbine_Count"] = new_turbine_counts
    df["Total_New_Capacity"] = total_new_capacities

    # 3.7. Save the updated DataFrame to Excel
    df.to_excel(output_excel, index=False)
    print(f"\nUpdated database saved to {output_excel}")

if __name__ == "__main__":
    main()
