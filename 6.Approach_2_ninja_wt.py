import pandas as pd
import math
import re
from pathlib import Path

# --- Setup directories and file paths ---
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

input_excel = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
output_excel = results_dir / "Approach_2.xlsx"

# --- Turbine definitions ---
turbines = [
    # IEC Class 1
    {"Model": "Siemens SWT 3-101",     "Capacity": 3.0, "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 4.3-120",   "Capacity": 4.3, "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 8-154",     "Capacity": 8.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens.SWT.3.6.107",   "Capacity": 3.6, "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6-154","Capacity": 6.0, "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.5-167","Capacity": 8.5,"RotorDiameter": 167,"IEC_Class_Num": 1},
    {"Model": "Nordex 100-3300",       "Capacity": 3.3, "RotorDiameter": 100, "IEC_Class_Num": 1},
    # IEC Class 2
    {"Model": "Enercon.E82.3000",      "Capacity": 3.0, "RotorDiameter": 82,  "IEC_Class_Num": 2},
    {"Model": "Vestas.V90.3000",       "Capacity": 3.0, "RotorDiameter": 90,  "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.0",       "Capacity": 4.0, "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 4.5-145","Capacity": 4.5,"RotorDiameter": 145,"IEC_Class_Num": 2},
    # IEC Class 3
    {"Model": "Enercon E-115-3.000",    "Capacity": 3.0, "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "Siemens SWT 6.6-170",   "Capacity": 6.6, "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Vestas V136-3.45",      "Capacity": 3.45,"RotorDiameter": 136, "IEC_Class_Num": 3},
    {"Model": "Nordex.N131.3000",      "Capacity": 3.0, "RotorDiameter": 131, "IEC_Class_Num": 3},
    # IEC Class S
    {"Model": "Enercon E126-4000",     "Capacity": 4.0, "RotorDiameter": 126, "IEC_Class_Num": 0},
    {"Model": "Enercon E175-6000",     "Capacity": 5.0, "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "Vestas V150-6.0",       "Capacity": 6.0, "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "Vestas V164-9500",      "Capacity": 9.5, "RotorDiameter": 164, "IEC_Class_Num": 0},
    {"Model": "Nordex 149-4500",       "Capacity": 4.5, "RotorDiameter": 149, "IEC_Class_Num": 0},
]

def land_area(diameter, terrain_type):
    t = str(terrain_type).lower()
    if t == "flat":
        return 28 * diameter**2
    elif t == "complex":
        return 54 * diameter**2
    else:
        return 28 * diameter**2  # default to flat

def best_fitting_turbine_for_iec_class_min_turbines(turbine_list, terrain_type, iec_class_num, available_area):
    matching = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]
    best_candidate = None
    best_total_cap = -1
    best_count = 0
    best_area = None

    for t in matching:
        area_per = land_area(t["RotorDiameter"], terrain_type)
        count = int(available_area // area_per)
        if count < 1:
            continue
        total_cap = count * t["Capacity"]
        if total_cap > best_total_cap or (total_cap == best_total_cap and count < best_count):
            best_candidate = t
            best_total_cap = total_cap
            best_count = count
            best_area = area_per

    if not best_candidate:
        return None, 0, 0, None

    ratio = best_candidate["Capacity"] / best_area
    return best_candidate, ratio, best_count, best_area

# --- Read and filter data ---
df = pd.read_excel(input_excel)
print(f"Read {len(df)} rows from {input_excel}")

df = df[df["Active in 2022"] == True]

if "IEC_Class_Num" in df.columns:
    df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce").fillna(0).astype(int)
else:
    print("Warning: IEC_Class_Num column not found; defaulting to 0")

# --- Compute recommendations ---
models = []
capacities = []
counts = []
total_caps = []
areas = []

for idx, row in df.iterrows():
    terrain = row.get("Terrain_Type", "flat")
    iec = row.get("IEC_Class_Num", 0)
    old_area = row.get("Total Park Area (m²)")
    if pd.isna(old_area):
        models.append(None); capacities.append(None)
        counts.append(None); total_caps.append(None); areas.append(None)
        continue

    t, ratio, cnt, a = best_fitting_turbine_for_iec_class_min_turbines(
        turbines, terrain, iec, old_area
    )
    if not t:
        models.append(None); capacities.append(None)
        counts.append(None); total_caps.append(None); areas.append(None)
    else:
        models.append(t["Model"])
        capacities.append(t["Capacity"])
        counts.append(cnt)
        total_caps.append(cnt * t["Capacity"])
        areas.append(cnt * a)

# --- Write results ---
df["Recommended_WT_Model"]       = models
df["Recommended_WT_Capacity"]    = capacities
df["New_Turbine_Count"]          = counts
df["Total_New_Capacity"]         = total_caps
df["New_Total_Park_Area (m²)"]   = areas

df.to_excel(output_excel, index=False)
print(f"Updated database saved to {output_excel}")
