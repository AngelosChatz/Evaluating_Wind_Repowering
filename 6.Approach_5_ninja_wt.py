import pandas as pd
import math
from pathlib import Path

# --- Setup directories and file paths ---
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

input_excel  = results_dir / "Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx"
output_excel = results_dir / "Approach_5.xlsx"

# --- Turbine definitions ---
turbines = [
    {"Model": "Siemens SWT 3-101",       "Capacity": 3.0,  "RotorDiameter": 101, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 4.3-120",     "Capacity": 4.3,  "RotorDiameter": 120, "IEC_Class_Num": 1},
    {"Model": "Siemens SWT 8-154",       "Capacity": 8.0,  "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens.SWT.3.6.107",     "Capacity": 3.6,  "RotorDiameter": 107, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 6-154", "Capacity": 6.0,  "RotorDiameter": 154, "IEC_Class_Num": 1},
    {"Model": "Siemens Gamesa SG 8.5-167","Capacity": 8.5,  "RotorDiameter": 167, "IEC_Class_Num": 1},
    {"Model": "Nordex 100-3300",         "Capacity": 3.3,  "RotorDiameter": 100, "IEC_Class_Num": 1},

    {"Model": "Enercon.E82.3000",        "Capacity": 3.0,  "RotorDiameter": 82,  "IEC_Class_Num": 2},
    {"Model": "Vestas.V90.3000",         "Capacity": 3.0,  "RotorDiameter": 90,  "IEC_Class_Num": 2},
    {"Model": "Vestas V136-4.0",         "Capacity": 4.0,  "RotorDiameter": 136, "IEC_Class_Num": 2},
    {"Model": "Siemens Gamesa SG 4.5-145","Capacity":4.5,  "RotorDiameter":145, "IEC_Class_Num": 2},

    {"Model": "Enercon E-115-3.000",     "Capacity": 3.0,  "RotorDiameter": 115, "IEC_Class_Num": 3},
    {"Model": "Siemens SWT 6.6-170",     "Capacity": 6.6,  "RotorDiameter": 170, "IEC_Class_Num": 3},
    {"Model": "Vestas V136-3.45",        "Capacity": 3.45, "RotorDiameter": 136, "IEC_Class_Num": 3},
    {"Model": "Nordex.N131.3000",        "Capacity": 3.0,  "RotorDiameter": 131, "IEC_Class_Num": 3},

    {"Model": "Enercon E126-4000",       "Capacity": 4.0,  "RotorDiameter": 126, "IEC_Class_Num": 0},
    {"Model": "Enercon E175-6000",       "Capacity": 5.0,  "RotorDiameter": 175, "IEC_Class_Num": 0},
    {"Model": "Vestas V150-6.0",         "Capacity": 6.0,  "RotorDiameter": 150, "IEC_Class_Num": 0},
    {"Model": "Vestas V164-9500",        "Capacity": 9.5,  "RotorDiameter": 164, "IEC_Class_Num": 0},
    {"Model": "Nordex 149-4500",         "Capacity": 4.5,  "RotorDiameter": 149, "IEC_Class_Num": 0},
]

def land_area(diameter, terrain_type):
    """Estimate land area needed per turbine based on rotor diameter and terrain."""
    t = str(terrain_type).strip().lower()
    return 54*diameter**2 if t == "complex" else 28*diameter**2

def best_fitting_turbine_for_iec_class_min_turbines(turbine_list, terrain_type,
                                                    iec_class_num, available_area):
    """
    Pick turbine of given IEC class that maximizes total capacity in available_area.
    Returns (best_turbine or None, efficiency, count, area_per_turbine).
    """
    matching = [t for t in turbine_list if t["IEC_Class_Num"] == iec_class_num]
    best_turbine = None
    best_total_capacity = 0
    best_count = 0
    best_area_per = None

    for t in matching:
        area_per = land_area(t["RotorDiameter"], terrain_type)
        count = int(available_area // area_per)
        if count < 1:
            continue
        total_cap = count * t["Capacity"]
        if (total_cap > best_total_capacity) or (total_cap == best_total_capacity and (best_count == 0 or count < best_count)):
            best_turbine = t
            best_total_capacity = total_cap
            best_count = count
            best_area_per = area_per

    if not best_turbine:
        return None, 0, 0, None

    efficiency = best_turbine["Capacity"] / best_area_per
    return best_turbine, efficiency, best_count, best_area_per

# --- Read and filter data ---
df = pd.read_excel(input_excel)
print(f"Read {len(df)} rows from {input_excel}")

df = df[df.get("Active in 2022", True) == True]

if "IEC_Class_Num" in df.columns:
    df["IEC_Class_Num"] = (pd.to_numeric(df["IEC_Class_Num"], errors="coerce")
                             .fillna(0).astype(int))

# Identify key columns if present
number_col = next((c for c in df.columns if c.lower()=="number of turbines"), None)
power_col  = next((c for c in df.columns if c.lower()=="total power"), None)
manuf_col  = next((c for c in df.columns if c.lower()=="manufacturer"), None)
turb_col   = next((c for c in df.columns if c.lower()=="turbine"), None)

# Prepare result lists
rec_model    = []
rec_capacity = []
rec_count    = []
rec_totcap   = []
rec_newarea  = []
repowered    = 0
replaced     = 0

# Process each wind farm
for _, row in df.iterrows():
    terrain = row.get("Terrain_Type", "flat")
    iec = row.get("IEC_Class_Num", 0)
    area_old = row.get("Total Park Area (m²)")
    orig_n = row.get(number_col) if number_col else None
    orig_p = row.get(power_col)  if power_col  else None
    orig_manuf = row.get(manuf_col, "")
    orig_turb = row.get(turb_col, "")
    orig_model = f"{orig_manuf} {orig_turb}".strip() or None

    # Missing area: keep original
    if pd.isna(area_old):
        rec_model.append(orig_model)
        if orig_n and orig_p:
            rec_capacity.append(orig_p/orig_n)
            rec_count.append(orig_n)
            rec_totcap.append(orig_p)
        else:
            rec_capacity.append(None); rec_count.append(None); rec_totcap.append(None)
        rec_newarea.append(area_old)
        replaced += 1
        continue

    best, eff, count, area_per = best_fitting_turbine_for_iec_class_min_turbines(
        turbines, terrain, iec, area_old
    )
    orig_total_cap = orig_p if pd.notna(orig_p) else 0

    # Decide to repower or keep original
    if not best or count*best["Capacity"] < orig_total_cap:
        rec_model.append(orig_model)
        if orig_n and orig_p:
            rec_capacity.append(orig_p/orig_n)
            rec_count.append(orig_n)
            rec_totcap.append(orig_p)
        else:
            rec_capacity.append(None); rec_count.append(None); rec_totcap.append(None)
        rec_newarea.append(area_old)
        replaced += 1
    else:
        rec_model.append(best["Model"])
        rec_capacity.append(best["Capacity"])
        rec_count.append(count)
        rec_totcap.append(count*best["Capacity"])
        rec_newarea.append(count*area_per)
        repowered += 1

# Attach results and save
df["Recommended_WT_Model"]      = rec_model
df["Recommended_WT_Capacity"]   = rec_capacity
df["New_Turbine_Count"]         = rec_count
df["Total_New_Capacity"]        = rec_totcap
df["New_Total_Park_Area (m²)"]  = rec_newarea

df.to_excel(output_excel, index=False)
print(f"Updated database saved to {output_excel}")
print(f"Parks repowered: {repowered}")
print(f"Parks replaced:  {replaced}")
