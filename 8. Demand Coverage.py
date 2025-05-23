import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === FILE PATHS ===
repowered_path = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf.xlsx"
existing_path  = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Cf_old_updated.xlsx"

# Read data
df = pd.read_excel(repowered_path, index_col=0)
df["Total_New_Capacity"] = pd.to_numeric(df["Total_New_Capacity"], errors="coerce").fillna(0)
df["CapacityFactor"]     = pd.to_numeric(df["CapacityFactor"],     errors="coerce").fillna(0)

# Load and rename old baseline data
df_old = pd.read_excel(existing_path, index_col=0)
df_old = df_old.rename(columns={
    "Total_New_Capacity": "Total_Baseline_Capacity",  # Renaming to Baseline
    "Annual_Energy_TWh":  "Annual_Energy_TWh_Baseline"  # Renaming to Baseline
})
df_old["Total_Baseline_Capacity"]    = pd.to_numeric(df_old["Total_Baseline_Capacity"],    errors="coerce").fillna(0)
df_old["Annual_Energy_TWh_Baseline"] = pd.to_numeric(df_old["Annual_Energy_TWh_Baseline"], errors="coerce").fillna(0)

# = Merge old Baseline into new DataFrame
df = df.join(df_old[["Total_Baseline_Capacity", "Annual_Energy_TWh_Baseline"]], how="left")

hours_per_year         = 8760
df["Annual_Energy_TWh"] = df["Total_New_Capacity"] * df["CapacityFactor"] * hours_per_year / 1e6

# === 5) Create selection columns ===
df["Selected_by_Capacity"] = np.where(
    df["Total_Baseline_Capacity"] > df["Total_New_Capacity"],
    df["Annual_Energy_TWh_Baseline"],
    df["Annual_Energy_TWh"]
)
df["Selected_by_Yield"] = np.where(
    df["Annual_Energy_TWh_Baseline"] > df["Annual_Energy_TWh"],
    df["Annual_Energy_TWh_Baseline"],
    df["Annual_Energy_TWh"]
)

# === Consumption Data ===
consumption_data = {
    "Country": [
        "Russia","Germany","France","Italy","UK","Turkey","Spain","Poland","Sweden","Norway",
        "Netherlands","Ukraine","Belgium","Finland","Austria","Czechia","Switzerland","Portugal",
        "Romania","Greece","Hungary","Bulgaria","Denmark","Belarus","Serbia","Ireland","Slovakia",
        "Iceland","Croatia","Slovenia","Bosnia & Herzegovina","Lithuania","Estonia","Albania",
        "Latvia","North Macedonia","Luxembourg","Moldova","Cyprus","Montenegro","Malta",
        "Faroe Islands","Gibraltar"
    ],
    "Electricity_Consumption_TWh": [
        1020.67,512.19,431.94,301.48,288.93,287.32,232.84,167.54,131.12,124.91,
        111.71,98.01,82.23,80.55,68.17,63.75,57.19,50.57,50.44,49.54,
        43.83,35.47,34.30,33.79,33.49,29.72,26.35,19.33,16.73,13.52,
        12.45,11.28,8.62,6.94,6.93,6.23,6.22,5.59,5.12,2.99,2.75,0.46,0.21
    ]
}
df_cons = pd.DataFrame(consumption_data)

# Summing the data
wind_cap = df.groupby("Country")["Selected_by_Capacity"].sum().rename("Wind_CapBased_TWh")
wind_yld = df.groupby("Country")["Selected_by_Yield"].sum().rename("Wind_YieldBased_TWh")

# Merge consumption and wind energy data
df_comb = (
    df_cons
    .merge(wind_cap, left_on="Country", right_index=True, how="inner")
    .merge(wind_yld, left_on="Country", right_index=True, how="inner")
)

# Compute coverage %
df_comb["Coverage_Cap_%"] = df_comb["Wind_CapBased_TWh"] / df_comb["Electricity_Consumption_TWh"] * 100
df_comb["Coverage_Yld_%"] = df_comb["Wind_YieldBased_TWh"] / df_comb["Electricity_Consumption_TWh"] * 100

# Sort by capacity coverage
df_sorted = df_comb.sort_values("Coverage_Cap_%", ascending=False).reset_index(drop=True)

import matplotlib as mpl

# Reset to default style
mpl.rcParams.update(mpl.rcParamsDefault)

x     = np.arange(len(df_sorted))
width = 0.35
colors = ['#4c72b0', '#55a868']  # blue & green

fig, ax = plt.subplots(figsize=(18, 8))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Draw bars
ax.bar(
    x - width/2,
    df_sorted["Coverage_Cap_%"],
    width=width,
    label="Baseline - Capacity-Based",  # Updated label for Baseline
    color=colors[0],
    edgecolor='black',
    linewidth=0.7
)
ax.bar(
    x + width/2,
    df_sorted["Coverage_Yld_%"],
    width=width,
    label="Repowering NLH-Energy Yield",  # Updated label for Repowering NLH-Energy Yield
    color=colors[1],
    edgecolor='black',
    linewidth=0.7
)

# Grid only on y-axis
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.xaxis.grid(False)

# Ticks & labels
ax.set_xticks(x)
ax.set_xticklabels(df_sorted["Country"], rotation=45, ha='right', fontsize=10)
ax.tick_params(axis='y', labelsize=12)

# Labels & title
ax.set_ylabel("Demand Coverage (%)", fontsize=14, labelpad=10)
ax.set_title("EU Countries: Demand Coverage by Repowering Approach",
             fontsize=18, fontweight='bold', pad=15)

# Legend
ax.legend(loc="upper right", fontsize=12, frameon=False)

plt.tight_layout()
plt.show()

# Compute overall EU coverage
eu_members = {
    "Germany","France","Italy","Spain","Poland","Sweden","Netherlands","Belgium",
    "Finland","Austria","Czechia","Portugal","Romania","Greece","Hungary","Bulgaria",
    "Denmark","Slovakia","Ireland","Croatia","Lithuania","Slovenia","Latvia","Estonia",
    "Luxembourg","Malta","Cyprus"
}
eu_df = df_comb[df_comb["Country"].isin(eu_members)]
total_demand   = eu_df["Electricity_Consumption_TWh"].sum()
total_wind_cap = eu_df["Wind_CapBased_TWh"].sum()
total_wind_yld = eu_df["Wind_YieldBased_TWh"].sum()
coverage_cap   = total_wind_cap / total_demand * 100
coverage_yld   = total_wind_yld / total_demand * 100

print("=== Overall EU Coverage ===")
print(f"Total EU demand:             {total_demand:.1f} TWh")
print(f"Capacity-based wind supply:  {total_wind_cap:.1f} TWh → {coverage_cap:.1f}% of demand")
print(f"Yield-based wind supply:     {total_wind_yld:.1f} TWh → {coverage_yld:.1f}% of demand")
