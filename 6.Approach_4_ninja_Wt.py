import pandas as pd
import math
from pathlib import Path

# --- Setup directories and file paths ---
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

input_excel = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
output_excel = results_dir / "Approach_4.xlsx"

# --- Turbine definitions ---
turbines = [
    # IEC Class 1
    {"Model": "Siemens SWT 3-101",       "Capacity": 3.0,  "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 4.3-120",     "Capacity": 4.3,  "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "Siemens.SWT.3.6.107",     "Capacity": 3.6,  "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6-154", "Capacity": 6.0,  "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.5-167","Capacity": 8.5,  "RotorDiameter": 167, "IEC_Class_Num": 1},
    {"Model": "Nordex 100-3300",         "Capacity": 3.3,  "RotorDiameter": 100, "IEC_Class_Num": 1},
    # IEC Class 2
    {"Model": "Enercon.E82.3000",        "Capacity": 3.0,  "RotorDiameter": 82,  "IEC_Class_Num": 2},
    {"Model": "Vestas.V90.3000",         "Capacity": 3.0,  "RotorDiameter": 90,  "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.0",         "Capacity": 4.0,  "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 4.5-145","Capacity": 4.5, "RotorDiameter": 145, "IEC_Class_Num": 2},
    # IEC Class 3
    {"Model": "Enercon E-115-3.000",      "Capacity": 3.0,  "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "Siemens SWT 6.6-170",     "Capacity": 6.6,  "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Vestas V136-3.45",        "Capacity": 3.45, "RotorDiameter": 136, "IEC_Class_Num": 3},
    {"Model": "Nordex.N131.3000",        "Capacity": 3.0,  "RotorDiameter": 131, "IEC_Class_Num": 3},
    # IEC Class S (0)
    {"Model": "Enercon E126-4000",       "Capacity": 4.0,  "RotorDiameter": 126, "IEC_Class_Num": 0},
    {"Model": "Enercon E175-6000",       "Capacity": 5.0,  "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "Vestas V150-6.0",         "Capacity": 6.0,  "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "Vestas V164-9500",        "Capacity": 9.5,  "RotorDiameter": 164, "IEC_Class_Num": 0},
    {"Model": "Nordex 149-4500",         "Capacity": 4.5,  "RotorDiameter": 149, "IEC_Class_Num": 0}
]

def land_area(diameter, terrain_type):
    t = str(terrain_type).lower()
    if t == "flat":
        return 28 * diameter**2
    elif t == "complex":
        return 54 * diameter**2
    else:
        return 28 * diameter**2

def best_fitting_turbine_for_iec_class_min_turbines(turbine_list, terrain_type, iec_class_num, available_area):
    matching = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]
    if not matching:
        return None, 0, 0, None, False

    candidates_normal = []
    candidates_forced = []

    for t in matching:
        A = land_area(t["RotorDiameter"], terrain_type)
        n_float = available_area / A
        if n_float >= 1:
            floor_n = math.floor(n_float)
            count = math.ceil(n_float) if (n_float - floor_n) >= 0.8 else floor_n
            total_cap = count * t["Capacity"]
            candidates_normal.append((t, count, A, total_cap))
        else:
            candidates_forced.append((t, 1, A, t["Capacity"]))

    if candidates_normal:
        best = max(candidates_normal, key=lambda x: (x[3], -x[1]))
        is_forced = False
    elif candidates_forced:
        best = min(candidates_forced, key=lambda x: x[0]["RotorDiameter"])
        is_forced = True
    else:
        return None, 0, 0, None, False

    tdict, count, A, total_cap = best
    ratio = tdict["Capacity"] / A if A else 0
    return tdict, ratio, count, A, is_forced

# --- Read and filter data ---
df = pd.read_excel(input_excel)
print(f"Read {len(df)} rows from {input_excel}")

df = df[df["Active in 2022"] == True]

if "IEC_Class_Num" in df.columns:
    df["IEC_Class_Num"] = pd.to_numeric(df["IEC_Class_Num"], errors="coerce")\
                            .fillna(0).astype(int)
else:
    print("Warning: IEC_Class_Num column missing — defaulting to 0")

# --- Compute recommendations ---
rec_models = []
rec_caps = []
rec_counts = []
rec_totcaps = []
rec_areas = []
forced_updates = 0

for idx, row in df.iterrows():
    terrain = row.get("Terrain_Type", "flat")
    iec_num = row.get("IEC_Class_Num", 0)
    old_area = row.get("Total Park Area (m²)")
    if pd.isna(old_area):
        rec_models.append(None)
        rec_caps.append(None)
        rec_counts.append(None)
        rec_totcaps.append(None)
        rec_areas.append(None)
        continue

    best_turb, best_ratio, cnt, A, is_forced = best_fitting_turbine_for_iec_class_min_turbines(
        turbines, terrain, iec_num, old_area
    )
    if is_forced:
        forced_updates += 1

    if best_turb is None:
        rec_models.append(None)
        rec_caps.append(None)
        rec_counts.append(None)
        rec_totcaps.append(None)
        rec_areas.append(None)
    else:
        rec_models.append(best_turb["Model"])
        rec_caps.append(best_turb["Capacity"])
        rec_counts.append(cnt)
        rec_totcaps.append(cnt * best_turb["Capacity"])
        rec_areas.append(cnt * A)

# --- Assign and save ---
df["Recommended_WT_Model"]       = rec_models
df["Recommended_WT_Capacity"]    = rec_caps
df["New_Turbine_Count"]          = rec_counts
df["Total_New_Capacity"]         = rec_totcaps
df["New_Total_Park_Area (m²)"]   = rec_areas

df.to_excel(output_excel, index=False)
print(f"\nUpdated database saved to {output_excel}")
print(f"Total forced replacements applied: {forced_updates}")
