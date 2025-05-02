import pandas as pd
import math
from pathlib import Path

# Define base and results directories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

# Input/output paths
input_excel = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
output_excel = results_dir / "Approach_5.xlsx"

# Turbine database (same as before)
turbines = [
    # IEC Class 1
    {"Model": "Siemens SWT 3-101", "Capacity": 3.0, "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 4.3-120", "Capacity": 4.3, "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 8-154", "Capacity": 8.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens.SWT.3.6.107", "Capacity": 3.6, "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6-154", "Capacity": 6.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.5-167", "Capacity": 8.5, "RotorDiameter": 167, "IEC_Class_Num": 1},
    {"Model": "Nordex 100-3300", "Capacity": 3.3, "RotorDiameter": 100, "IEC_Class_Num": 1},
    # IEC Class 2
    {"Model": "Enercon.E82.3000", "Capacity": 3.0, "RotorDiameter": 82, "IEC_Class_Num": 2},
    {"Model": "Vestas.V90.3000", "Capacity": 3.0, "RotorDiameter": 90, "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.0", "Capacity": 4.0, "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 4.5-145", "Capacity": 4.5, "RotorDiameter": 145, "IEC_Class_Num": 2},
    # IEC Class 3
    {"Model": "Enercon E-115-3.000", "Capacity": 3.0, "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "Siemens SWT 6.6-170", "Capacity": 6.6, "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Vestas V136-3.45", "Capacity": 3.45, "RotorDiameter": 136, "IEC_Class_Num": 3},
    {"Model": "Nordex.N131.3000", "Capacity": 3.0, "RotorDiameter": 131, "IEC_Class_Num": 3},
    # IEC Class S
    {"Model": "Enercon E126-4000", "Capacity": 4.0, "RotorDiameter": 126, "IEC_Class_Num": 0},
    {"Model": "Enercon E175-6000", "Capacity": 5.0, "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "Vestas V150-6.0", "Capacity": 6.0, "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "Vestas V164-9500", "Capacity": 9.5, "RotorDiameter": 164, "IEC_Class_Num": 0},
    {"Model": "Nordex 149-4500", "Capacity": 4.5, "RotorDiameter": 149, "IEC_Class_Num": 0}
]


def land_area(diameter, terrain_type):
    t = str(terrain_type).strip().lower()
    return 28 * diameter**2 if t == "flat" or t not in {"flat", "complex"} else 54 * diameter**2


def best_fitting_turbine_for_iec_class_min_turbines(turbine_list, terrain_type, iec_class_num, available_area):
    matching = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]
    best_candidate, best_total_capacity, best_count, best_area_per = None, -1, 0, None
    for t in matching:
        area_per = land_area(t["RotorDiameter"], terrain_type)
        count = int(available_area // area_per)
        if count < 1:
            continue
        total_cap = count * t["Capacity"]
        if total_cap > best_total_capacity or (total_cap == best_total_capacity and count < best_count):
            best_candidate, best_total_capacity, best_count, best_area_per = t, total_cap, count, area_per
    if not best_candidate:
        return None, 0, 0, None
    return best_candidate, best_candidate["Capacity"]/best_area_per, best_count, best_area_per


def main():
    df = pd.read_excel(input_excel)
    print(f"Read {len(df)} rows from {input_excel}.")

    # Filter active farms
    df = df[df.get("Active in 2022", True) == True]
    if "IEC_Class_Num" in df.columns:
        df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce").fillna(0).astype(int)

    # Identify columns
    number_col = next((c for c in df.columns if c.lower() == "number of turbines"), None)
    power_col = next((c for c in df.columns if c.lower() == "total power"), None)
    manuf_col = next((c for c in df.columns if c.lower() == "manufacturer"), None)
    turb_col = next((c for c in df.columns if c.lower() == "turbine"), None)

    # Prepare result lists
    rec_model, rec_cap, new_count, tot_new_cap, new_area = ([] for _ in range(5))

    for _, row in df.iterrows():
        terrain = row.get("Terrain_Type", "flat")
        iec = row.get("IEC_Class_Num", 0)
        area_old = row.get("Total Park Area (m²)")
        orig_n = row.get(number_col) if number_col else None
        orig_p = row.get(power_col) if power_col else None
        # merge manufacturer and turbine columns
        orig_manuf = row.get(manuf_col, "")
        orig_turb = row.get(turb_col, "")
        orig_model = f"{orig_manuf} {orig_turb}".strip() or None

        # No area or invalid: preserve original specs
        if pd.isna(area_old):
            rec_model.append(orig_model)
            rec_cap.append(orig_p/orig_n if orig_n and orig_p else None)
            new_count.append(orig_n)
            tot_new_cap.append(orig_p)
            new_area.append(area_old)
            continue

        # Attempt repower
        best, _, count, area_per = best_fitting_turbine_for_iec_class_min_turbines(
            turbines, terrain, iec, area_old)

        if not best:
            # Not repowered: use same original merged model
            rec_model.append(orig_model)
            rec_cap.append(orig_p/orig_n if orig_n and orig_p else None)
            new_count.append(orig_n)
            tot_new_cap.append(orig_p)
            new_area.append(area_old)
        else:
            # Repowered: recommend new model
            rec_model.append(best["Model"])
            rec_cap.append(best["Capacity"])
            new_count.append(count)
            tot_new_cap.append(count * best["Capacity"])
            new_area.append(count * area_per)

    # Assign result columns
    df["Recommended_WT_Model"] = rec_model
    df["Recommended_WT_Capacity"] = rec_cap
    df["New_Turbine_Count"] = new_count
    df["Total_New_Capacity"] = tot_new_cap
    df["New_Total_Park_Area (m²)"] = new_area

    df.to_excel(output_excel, index=False)
    print(f"\nUpdated database saved to {output_excel}")

if __name__ == "__main__":
    main()
