import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import random
import math


excel_path = r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new.xlsx"
wind_tif_path = r"D:\gwa3_250_windspeed_100m.tif"
output_excel = r"Repowered Energy yield_with_hhconsideration.xlsx"


df = pd.read_excel(excel_path)
turbine_specs = [
    {
        "name": "E-82 EP2 E4",
        "capacity_mw": 3.0,
        "rated_speed": 16,
        "cut_in": 3,
        "cut_out": 25,
        "hub_height": 84
    },
    {
        "name": "Siemens Gamesa SG 3.4-145",
        "capacity_mw": 3.465,
        "rated_speed": 10,
        "cut_in": 3,
        "cut_out": 20,
        "hub_height": 127.5
    },
    {
        "name": "Nordex N175/6.X",
        "capacity_mw": 6.22,
        "rated_speed": 12.5,
        "cut_in": 3,
        "cut_out": 20,
        "hub_height": 179
    },
    {
        "name": "Vestas V150-4.5",
        "capacity_mw": 4.5,
        "rated_speed": 10,
        "cut_in": 3,
        "cut_out": 24.5,
        "hub_height": 120
    },
    {
        "name": "Enercon E-160 EP5",
        "capacity_mw": 5.56,
        "rated_speed": 11,
        "cut_in": 2.5,
        "cut_out": 28,
        "hub_height": 166
    },
    {
        "name": "Nordex N163/5.X",
        "capacity_mw": 5.7,
        "rated_speed": 12,
        "cut_in": 3,
        "cut_out": 26,
        "hub_height": 164
    },
    {
        "name": "Siemens Gamesa SG 5.8-170",
        "capacity_mw": 5.8,
        "rated_speed": 11,
        "cut_in": 3,
        "cut_out": 20,
        "hub_height": 115
    },
    {
        "name": "Vestas V90-3.0",
        "capacity_mw": 3.0,
        "rated_speed": 15,
        "cut_in": 4,
        "cut_out": 25,
        "hub_height": 105
    }
]

# Dictionary for quick lookup:
turbine_specs_dict = {t["name"]: t for t in turbine_specs}

# 4) Define Real Power Curves

wind_speeds_real = np.arange(0, 26, 1)
real_data = {
    "E-82 EP2 E4": [0, 0, 0, 25, 82, 174, 321, 525, 800, 1135,
                    1500, 1880, 2200, 2500, 2770, 2910, 3000, 3000, 3000, 3000,
                    3000, 3000, 3000, 3000, 3000, 3000],
    "Siemens Gamesa SG 3.4-145": None,
    "Vestas V150-4.5": None,
    "Enercon E-160 EP5": None,
    "Nordex N163/5.X": None,
    "Siemens Gamesa SG 5.8-170": None,
    "Vestas V105-3.45": [0, 0, 0, 0, 67, 151, 310, 518, 845,
                         1264, 1860, 2500, 3140, 3400, 3450, 3450, 3450, 3450, 3450,
                         3450, 3450, 3450, 3450, 3450, 3450, 3450],
    "Vestas V90-3.0": [0, 0, 0, 0, 0, 190, 353, 581, 886, 1272,
                       1700, 2100, 2500, 2800, 2950, 3000, 3000, 3000, 3000, 3000,
                       3000, 3000, 3000, 3000, 3000, 3000],
}


# Piecewise Cubic Ramp Power Curve Function
def approximate_power_curve(v, capacity_kW, cut_in, rated_speed, cut_out):
    if v < cut_in:
        return 0.0
    elif v < rated_speed:
        return capacity_kW * ((v ** 3 - cut_in ** 3) / (rated_speed ** 3 - cut_in ** 3))
    elif v < cut_out:
        return capacity_kW
    else:
        return 0.0


# Function to Get Power Output at a Given Wind Speed
def get_power_output(turbine_model, v):
    specs = turbine_specs_dict.get(turbine_model)
    if specs is None:
        return 0.0
    if v > specs["cut_out"]:
        return 0.0
    capacity_kW = specs["capacity_mw"] * 1000.0
    if real_data.get(turbine_model) is not None:
        return np.interp(v, wind_speeds_real, real_data[turbine_model])
    else:
        return approximate_power_curve(v, capacity_kW, specs["cut_in"], specs["rated_speed"], specs["cut_out"])


# Function to Sample and Adjust Wind Speed from the Raster
# Use a power-law correction from 100 m (raster) to the turbine hub height.

terrain_shear_exponents = {
    "flat": 0.1,
    "complex": 0.3
}

def sample_wind_speed(lat, lon, src, hub_height, terrain_type, ref_height=100.0):

    try:
        row, col = src.index(lon, lat)
        v_ref = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
    except Exception:
        return np.nan

    terrain_type_lower = str(terrain_type).lower().strip() if terrain_type is not None else "flat"
    shear_exp = terrain_shear_exponents.get(terrain_type_lower, 0.14)
    # Adjust wind speed from ref_height to hub_height:
    v_hub = v_ref * (hub_height / ref_height) ** shear_exp
    return float(v_hub)


# 8) Weibull PDF Function
def weibull_pdf(v, k, lam):

    v = np.array(v)
    return np.where(v < 0, 0.0, (k / lam) * (v / lam) ** (k - 1) * np.exp(-(v / lam) ** k))


def parse_iec_class(iec_str):
    if not isinstance(iec_str, str) or len(iec_str) < 2:
        return (None, None)
    if iec_str.endswith("A+"):
        turb_class = "A+"
        wind_class = iec_str[:-2]
    else:
        turb_class = iec_str[-1]
        wind_class = iec_str[:-1]
    return (wind_class.strip(), turb_class.strip())


#  Weibull-based AEP Calculation Function (per single turbine)

def compute_aep_weibull_single_turbine(turbine_model, avg_ws, iec_class_str=None):
    """
    Computes the Annual Energy Production (AEP) for a single turbine by integrating
    the turbine’s power curve over a Weibull-distributed wind speed spectrum.

    For this analysis, the Weibull shape parameter is fixed at k = 2.0.
    The scale parameter λ is determined so that the mean wind speed equals avg_ws:
      λ = avg_ws / Γ(1 + 1/k)
    """
    specs = turbine_specs_dict.get(turbine_model)
    if specs is None or np.isnan(avg_ws) or avg_ws <= 0:
        return 0.0

    # Fixed Weibull shape parameter:
    shape_k = 2.0

    # Compute scale parameter:
    gamma_term = math.gamma(1.0 + 1.0 / shape_k)  # For k=2, Γ(1.5) ≈ 0.8862
    lam = avg_ws / gamma_term

    # Numerical integration over wind speeds from 0 to 30 m/s:
    v_array = np.linspace(0, 30, 300)
    power_vals = [get_power_output(turbine_model, v) for v in v_array]
    pdf_vals = [(shape_k / lam) * ((v / lam) ** (shape_k - 1)) * np.exp(-((v / lam) ** shape_k)) for v in v_array]
    delta_v = v_array[1] - v_array[0]
    expected_power_kW = np.sum(np.array(power_vals) * np.array(pdf_vals)) * delta_v

    # Convert kW to MWh/yr:
    aep_mwh = expected_power_kW * 8760.0 / 1000.0
    return aep_mwh


# Loop Over Each Wind Park: Sample Raster, Compute AEP, and CO₂ Savings

CO2_FACTOR = 0.5  # tonnes CO₂ per MWh
aep_list_mwh = []
co2_list = []
random_indices = random.sample(range(len(df)), 5)

with rasterio.open(wind_tif_path) as src:
    for idx, row in df.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]
        turbine_model = row["Recommended_WT_Model"]
        new_turbine_count = row["New_Turbine_Count"]
        if pd.isna(new_turbine_count):
            new_turbine_count = 0

        terrain_type = row.get("Terrain_Type", "flat")
        specs = turbine_specs_dict.get(turbine_model, {})
        hub_height = specs.get("hub_height", 100)

        # Sample and adjust wind speed using the raster (which contains average speed at 100 m)
        avg_ws_hub = sample_wind_speed(lat, lon, src, hub_height, terrain_type, ref_height=100.0)
        iec_class_str = row.get("IEC_Class", None)
        aep_mwh_single = compute_aep_weibull_single_turbine(turbine_model, avg_ws_hub, iec_class_str)
        total_aep_mwh = aep_mwh_single * new_turbine_count
        aep_list_mwh.append(total_aep_mwh)
        co2_tonnes = total_aep_mwh * CO2_FACTOR
        co2_list.append(co2_tonnes)

        if idx in random_indices:
            print(f"Debug Park {idx}:")
            print(f"  Model: {turbine_model}")
            print(f"  Coordinates: lat={lat}, lon={lon}")
            print(f"  Terrain_Type: {terrain_type}")
            print(f"  IEC_Class: {iec_class_str}")
            print(f"  Sampled 100 m WS adjusted to Hub ({hub_height} m): {avg_ws_hub:.2f} m/s")
            print(f"  AEP (single turbine): {aep_mwh_single:.2f} MWh/yr")
            print(f"  New Turbine Count: {new_turbine_count}")
            print(f"  Total AEP: {total_aep_mwh:.2f} MWh/yr")
            print(f"  CO₂ Savings: {co2_tonnes:.2f} tonnes/yr")
            print("-----")

df["Annual_Energy_Yield_MWh"] = aep_list_mwh
df["Annual_Energy_Yield_TTh"] = df["Annual_Energy_Yield_MWh"] / 1e6
df["CO2_Saved_tonnes"] = co2_list


#Calculate Capacity Factor for Each Wind Park (per entry)

def compute_capacity_factor(row):
    turbine_model = row["Recommended_WT_Model"]
    new_turbine_count = row["New_Turbine_Count"]
    if pd.isna(new_turbine_count) or turbine_model not in turbine_specs_dict:
        return np.nan
    capacity_mw = turbine_specs_dict[turbine_model]["capacity_mw"]
    capacity_kw_total = capacity_mw * 1000 * new_turbine_count
    annual_energy_mwh = row["Annual_Energy_Yield_MWh"]
    if capacity_kw_total == 0:
        return np.nan
    capacity_factor = ((annual_energy_mwh * 1000) / (capacity_kw_total * 8760)) * 100
    return capacity_factor

df["Capacity_Factor (%)"] = df.apply(compute_capacity_factor, axis=1)
print(df[["Recommended_WT_Model", "New_Turbine_Count", "Annual_Energy_Yield_MWh", "Capacity_Factor (%)"]].head())


#Save Detailed Results to Excel (including the Capacity Factor column)
df.to_excel(output_excel, index=False)
print(f"Saved detailed wind park results to {output_excel}")


#Aggregate and Plot by Country (Energy Yield and CO₂ Savings)

country_yield = df.groupby("Country")["Annual_Energy_Yield_TTh"].sum().reset_index().sort_values(
    "Annual_Energy_Yield_TTh", ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(country_yield["Country"], country_yield["Annual_Energy_Yield_TTh"], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Total Repowered AEP (TWh/yr)")
plt.title("Repowered Energy Yield per Country")
plt.tight_layout()
plt.show()

country_co2 = df.groupby("Country")["CO2_Saved_tonnes"].sum().reset_index()
country_co2["CO2_Saved_Mt"] = country_co2["CO2_Saved_tonnes"] / 1e6
country_co2 = country_co2.sort_values("CO2_Saved_Mt", ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(country_co2["Country"], country_co2["CO2_Saved_Mt"], color='lightgreen')
plt.xticks(rotation=45, ha='right')
plt.ylabel("CO₂ Saved (Million Tonnes/yr)")
plt.title("Annual CO₂ Emissions Saved per Country")
plt.tight_layout()
plt.show()


#Plot Power Curves for All 11 Turbines in a 4x3 Grid

num_turbines = len(turbine_specs)
cols = 3
rows = 4  # 12 subplots (4 rows x 3 columns)
fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes = axes.flatten()
wind_speeds_modeled = np.linspace(0, 30, 300)

def plot_power_curve(ax, specs):
    turbine_name = specs["name"]
    capacity_kW = specs["capacity_mw"] * 1000.0
    c_in = specs["cut_in"]
    r_spd = specs["rated_speed"]
    c_out = specs["cut_out"]
    modeled_curve = [approximate_power_curve(v, capacity_kW, c_in, r_spd, c_out) for v in wind_speeds_modeled]
    ax.plot(wind_speeds_modeled, modeled_curve, label="Modeled Curve", color="tab:blue")
    max_val = max(modeled_curve)
    real_curve_data = real_data.get(turbine_name)
    if real_curve_data is not None:
        real_curve = np.interp(wind_speeds_modeled, wind_speeds_real, real_curve_data)
        real_curve = np.where(wind_speeds_modeled > c_out, 0, real_curve)
        ax.plot(wind_speeds_modeled, real_curve, label="Real Data", color="tab:orange", linestyle="--")
        max_val = max(max_val, max(real_curve))
    ax.set_title(turbine_name, fontsize=10)
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, max_val * 1.1)
    ax.grid(True)
    ax.legend(fontsize=8)

for i, specs in enumerate(turbine_specs):
    ax = axes[i]
    plot_power_curve(ax, specs)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()


# Compute and Print Percentage Error Metrics for Power Curve Fitting

print("\nPercentage Error Metrics for Power Curve Fitting (relative to rated capacity):\n")
for turbine, measured in real_data.items():
    if measured is not None:
        specs = turbine_specs_dict[turbine]
        capacity_kW = specs["capacity_mw"] * 1000.0
        cut_in = specs["cut_in"]
        rated_speed = specs["rated_speed"]
        cut_out = specs["cut_out"]
        modeled = np.array([approximate_power_curve(v, capacity_kW, cut_in, rated_speed, cut_out) for v in wind_speeds_real])
        measured = np.array(measured)
        errors = np.abs(modeled - measured)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean((modeled - measured) ** 2))
        max_error = np.max(errors)
        mae_pct = (mae / capacity_kW) * 100
        rmse_pct = (rmse / capacity_kW) * 100
        max_error_pct = (max_error / capacity_kW) * 100
        print(f"Turbine: {turbine}")
        print(f"  MAE: {mae:.2f} kW ({mae_pct:.2f}%)")
        print(f"  RMSE: {rmse:.2f} kW ({rmse_pct:.2f}%)")
        print(f"  Maximum Error: {max_error:.2f} kW ({max_error_pct:.2f}%)")
        print("-----")


#  Plot Weibull Distributions for Each IEC Class (using k = 2.0) with Average Speeds in Labels

df["Wind_Class"] = df["IEC_Class"].apply(lambda x: parse_iec_class(x)[0] if pd.notna(x) else None)
df["Turb_Class"] = df["IEC_Class"].apply(lambda x: parse_iec_class(x)[1] if pd.notna(x) else None)
df_valid = df[df["IEC_Class"].notna() & df["Turb_Class"].notna()]
# Group by IEC_Class and select the first entry as representative for each class.
representative_sites = df_valid.groupby("IEC_Class").first().reset_index()

plt.figure(figsize=(10, 6))
for i, row in representative_sites.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    turbine_model = row["Recommended_WT_Model"]
    terrain_type = row.get("Terrain_Type", "flat")
    iec_class_str = row["IEC_Class"]
    turb_class = row["Turb_Class"]

    specs = turbine_specs_dict.get(turbine_model, {})
    hub_height = specs.get("hub_height", 100)
    with rasterio.open(wind_tif_path) as src:
        avg_ws_hub = sample_wind_speed(lat, lon, src, hub_height, terrain_type, ref_height=100.0)

    # Fixed Weibull shape parameter:
    shape_k = 2.0
    gamma_term = math.gamma(1.0 + 1.0 / shape_k)  # For k=2, Γ(1.5) ≈ 0.8862
    lam = avg_ws_hub / gamma_term

    v_range = np.linspace(0, 30, 300)
    pdf_vals = weibull_pdf(v_range, shape_k, lam)
    # Include the average wind speed in the label:
    label = f"IEC {iec_class_str} (k={shape_k:.2f}, λ={lam:.2f}, v_avg={avg_ws_hub:.2f} m/s)"
    plt.plot(v_range, pdf_vals, label=label)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Probability Density")
plt.title("Weibull Distributions for Each IEC Class (Average Wind Speeds in Labels)")
plt.xlim(0, 30)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
