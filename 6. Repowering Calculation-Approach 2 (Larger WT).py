import pandas as pd
import math
import re


turbines = [
    # Class I
    {"Model": "Vestas V90-3.0 MW", "Capacity": 3.00, "RotorDiameter": 90, "IEC_Class_Num": 1},
    {"Model": "E-82 EP2 E4",        "Capacity": 2.35, "RotorDiameter": 82, "IEC_Class_Num": 1},
    {"Model": "Enercon E-136 EP5",  "Capacity": 4.65, "RotorDiameter": 136,"IEC_Class_Num": 1},
    {"Model": "Vestas V117-4.5 MW", "Capacity": 4.50, "RotorDiameter": 117,"IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6.6-155","Capacity": 6.60, "RotorDiameter": 155,"IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.0-167","Capacity": 8.00, "RotorDiameter": 167,"IEC_Class_Num": 1},

    # Class II
    {"Model": "Siemens Gamesa SG 3.0-132","Capacity": 3.00, "RotorDiameter": 132,"IEC_Class_Num": 2},
    {"Model": "E-82 EP2 E4",        "Capacity": 3.00, "RotorDiameter": 82, "IEC_Class_Num": 2},
    {"Model": "Nordex N90-2.5",     "Capacity": 2.50, "RotorDiameter": 90, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 3.4-132","Capacity": 3.40, "RotorDiameter": 132,"IEC_Class_Num": 2},
    {"Model": "Enercon E-138 EP3",  "Capacity": 4.20, "RotorDiameter": 138,"IEC_Class_Num": 2},
    {"Model": "Vestas V117-4.2 MW", "Capacity": 4.20, "RotorDiameter": 117,"IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.5",    "Capacity": 4.50, "RotorDiameter": 136,"IEC_Class_Num": 2},
    {"Model": "Nordex N155/5.X",    "Capacity": 5.00, "RotorDiameter": 155,"IEC_Class_Num": 2},
    {"Model": "Enercon E-147 EP5",  "Capacity": 5.00, "RotorDiameter": 147,"IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 5.8-155","Capacity": 5.80, "RotorDiameter": 155,"IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 6.6-170","Capacity": 6.60, "RotorDiameter": 170,"IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 7.0-170","Capacity": 7.00, "RotorDiameter": 155,"IEC_Class_Num": 2},

    # Class III
    {"Model": "Siemens Gamesa SG 3.4-145","Capacity": 3.40, "RotorDiameter": 145,"IEC_Class_Num": 3},
    {"Model": "Vestas V150-4.5",    "Capacity": 4.50, "RotorDiameter": 150,"IEC_Class_Num": 3},
    {"Model": "Enercon E-160 EP5",  "Capacity": 4.60, "RotorDiameter": 160,"IEC_Class_Num": 3},
    {"Model": "Nordex N163/5.X",    "Capacity": 5.70, "RotorDiameter": 163,"IEC_Class_Num": 3},
    {"Model": "Siemens Gamesa SG 5.8-170","Capacity": 5.80, "RotorDiameter": 170,"IEC_Class_Num": 3},
    {"Model": "Nordex N175/6.X",    "Capacity": 6.80, "RotorDiameter": 175,"IEC_Class_Num": 3},

    # Class S
    {"Model": "Vestas V120-2.2 MW", "Capacity": 2.20, "RotorDiameter": 120,"IEC_Class_Num": 4},
    {"Model": "Vestas V105-3.45 MW","Capacity": 3.45, "RotorDiameter": 105,"IEC_Class_Num": 4},
    {"Model": "Nordex N133/4.8",    "Capacity": 4.80, "RotorDiameter": 133,"IEC_Class_Num": 4},
    {"Model": "Vestas V150-6.0 MW", "Capacity": 6.00, "RotorDiameter": 150,"IEC_Class_Num": 4},
    {"Model": "Enercon E-175 EP5",  "Capacity": 6.50, "RotorDiameter": 175,"IEC_Class_Num": 4},
    {"Model": "Vestas V172-7.2 MW", "Capacity": 7.20, "RotorDiameter": 172,"IEC_Class_Num": 4}
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
        # If unknown or missing, default to flat spacing
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

        # Choose the candidate with a higher total capacity;
        # if tied, choose the one with a lower turbine count.
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

    input_excel = r"D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
    output_excel = r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new.xlsx"


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
    new_total_park_areas = []

    for idx, row in df.iterrows():

        terrain = row.get("Terrain_Type", "flat")
        iec_num = row.get("IEC_Class_Num", 0)  # after numeric conversion, 0 means invalid

        # Use the Total Park Area (m²) column directly
        old_area = row.get("Total Park Area (m²)")
        if old_area is None or (isinstance(old_area, float) and math.isnan(old_area)):
            recommended_models.append(None)
            recommended_capacities.append(None)
            new_turbine_counts.append(None)
            total_new_capacities.append(None)
            new_total_park_areas.append(None)
            continue

        # 3.2. Find the best replacement turbine that fits in the available area
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
            # 3.3. Calculate total new capacity
            new_total_capacity = new_turb_count * best_turb["Capacity"]

            # Calculate new total park area based on the chosen turbine spacing layout
            new_total_park_area = turbine_area * new_turb_count

            # 3.4. Store results
            recommended_models.append(best_turb["Model"])
            recommended_capacities.append(best_turb["Capacity"])
            new_turbine_counts.append(new_turb_count)
            total_new_capacities.append(new_total_capacity)
            new_total_park_areas.append(new_total_park_area)

    # 3.5. Add new columns to the DataFrame
    df["Recommended_WT_Model"] = recommended_models
    df["Recommended_WT_Capacity"] = recommended_capacities
    df["New_Turbine_Count"] = new_turbine_counts
    df["Total_New_Capacity"] = total_new_capacities
    df["New_Total_Park_Area (m²)"] = new_total_park_areas

    # 3.6. Save the updated DataFrame to Excel
    df.to_excel(output_excel, index=False)
    print(f"\nUpdated database saved to {output_excel}")

if __name__ == "__main__":
    main()
