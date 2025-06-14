import pandas as pd
import math
import re
from pathlib import Path

# Set up directories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

# File paths
input_excel_path = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
output_excel_path = results_dir / "Repowering_Stage_1_float.xlsx"

# Turbine specs
turbines = [
    # IEC Class 1 turbines
    {"Model": "Siemens SWT 3-101", "Capacity": 3.0, "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 4.3-120", "Capacity": 4.3, "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 8-154", "Capacity": 8.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 3.6-107", "Capacity": 3.6, "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6-154", "Capacity": 6.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.5-167", "Capacity": 8.5, "RotorDiameter": 167, "IEC_Class_Num": 1},
    {"Model": "Nordex 100-3300", "Capacity": 3.3, "RotorDiameter": 100, "IEC_Class_Num": 1},
    # IEC Class 2 turbines
    {"Model": "E82 3000", "Capacity": 3.0, "RotorDiameter": 82, "IEC_Class_Num": 2},
    {"Model": "Vestas V90-3.0", "Capacity": 3.0, "RotorDiameter": 90, "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.0", "Capacity": 4.0, "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 4.5-145", "Capacity": 4.5, "RotorDiameter": 145, "IEC_Class_Num": 2},
    # IEC Class 3 turbines
    {"Model": "Enercon E-115-3.000", "Capacity": 3.0, "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "Siemens SWT 6.6-170", "Capacity": 6.6, "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Nordex 131-4000", "Capacity": 4.0, "RotorDiameter": 131, "IEC_Class_Num": 3},
    {"Model": "Nordex N131-3000", "Capacity": 3.0, "RotorDiameter": 131, "IEC_Class_Num": 3},
    # IEC Class S turbines
    {"Model": "Enercon E126-4000", "Capacity": 4.0, "RotorDiameter": 126, "IEC_Class_Num": 0},
    {"Model": "Enercon E175-6000", "Capacity": 5.0, "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "Vestas V150-6.0", "Capacity": 6.0, "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "Vestas V164-9500", "Capacity": 9.5, "RotorDiameter": 164, "IEC_Class_Num": 0},
    {"Model": "Nordex 149-4500", "Capacity": 4.5, "RotorDiameter": 149, "IEC_Class_Num": 0}
]

def land_area(diameter, terrain_type):
    t = str(terrain_type).lower()
    if t == "flat":
        return 28 * (diameter ** 2)
    elif t == "complex":
        return 54 * (diameter ** 2)
    else:
        return 28 * (diameter ** 2)  # default to flat

def best_turbine_for_iec_class(turbine_list, terrain_type, iec_class_num):
    matching = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]
    if not matching:
        return None, 0
    best = max(
        matching,
        key=lambda t: t["Capacity"] / land_area(t["RotorDiameter"], terrain_type)
    )
    best_ratio = best["Capacity"] / land_area(best["RotorDiameter"], terrain_type)
    return best, best_ratio

# --- Read and filter data ---
df = pd.read_excel(input_excel_path)
print(f"Read {len(df)} rows from {input_excel_path}")

df = df[df["Active in 2022"] == True]

# Ensure IEC_Class_Num is int
if "IEC_Class_Num" in df.columns:
    df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce").fillna(0).astype(int)
else:
    print("Warning: IEC_Class_Num not found; defaulting to 0")

# Prepare result columns
recommended_models    = []
recommended_caps      = []
new_turbine_counts    = []
total_new_capacities  = []

for idx, row in df.iterrows():
    terrain = row.get("Terrain_Type", "flat")
    iec_num = row.get("IEC_Class_Num", 0)
    old_area = row.get("Total Park Area (m²)")
    if pd.isna(old_area):
        recommended_models.append(None)
        recommended_caps.append(None)
        new_turbine_counts.append(None)
        total_new_capacities.append(None)
        continue

    best_turb, ratio = best_turbine_for_iec_class(turbines, terrain, iec_num)
    if best_turb is None:
        recommended_models.append(None)
        recommended_caps.append(None)
        new_turbine_counts.append(None)
        total_new_capacities.append(None)
    else:
        new_area_per_turb = land_area(best_turb["RotorDiameter"], terrain)
        count = old_area / new_area_per_turb
        total_cap = count * best_turb["Capacity"]
        recommended_models.append(best_turb["Model"])
        recommended_caps.append(best_turb["Capacity"])
        new_turbine_counts.append(count)
        total_new_capacities.append(total_cap)

# Write results back
df["Recommended_WT_Model"]       = recommended_models
df["Recommended_WT_Capacity"]    = recommended_caps
df["New_Turbine_Count"]          = new_turbine_counts
df["Total_New_Capacity"]         = total_new_capacities

df.to_excel(output_excel_path, index=False)
print(f"Updated database saved to {output_excel_path}")
