from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths to your Excel files
this_dir = Path(__file__).resolve().parent
results_dir = this_dir / "results"
repowered_path = results_dir / "Approach_2_Cf.xlsx"
existing_path  = results_dir / "Cf_old_updated.xlsx"

# --- 1) Load repowered ("new") data
df = pd.read_excel(repowered_path, index_col=0)
df["Total_New_Capacity"] = pd.to_numeric(df["Total_New_Capacity"], errors="coerce").fillna(0)
df["CapacityFactor"]     = pd.to_numeric(df["CapacityFactor"],     errors="coerce").fillna(0)

# If CFs are given as percentages (>1), convert to fractions
df["CapacityFactor"] = df["CapacityFactor"].apply(lambda x: x/100 if x > 1 else x)

# --- 2) Load baseline (old‐turbine) data
df_old = pd.read_excel(existing_path, index_col=0)

# Make sure inputs are numeric
df_old["Representative_New_Capacity"] = pd.to_numeric(
    df_old["Representative_New_Capacity"], errors="coerce"
).fillna(0)
df_old["Number of turbines"] = pd.to_numeric(
    df_old["Number of turbines"], errors="coerce"
).fillna(0)
df_old["Annual_Energy_TWh"] = pd.to_numeric(
    df_old["Annual_Energy_TWh"], errors="coerce"
).fillna(0)

# Compute per‐row totals for baseline capacity (in MW) and energy (in TWh)
df_old["Total_Baseline_Capacity"] = (
    df_old["Representative_New_Capacity"]  # kW per turbine
    * df_old["Number of turbines"]
    / 1e3                                  # → MW
)
df_old["Annual_Energy_TWh_Baseline"] = (
    df_old["Annual_Energy_TWh"]           # TWh per turbine
    * df_old["Number of turbines"]
)

# --- 3) Merge baseline into repowered DataFrame
df = df.join(
    df_old[["Total_Baseline_Capacity", "Annual_Energy_TWh_Baseline"]],
    how="left"
)

# --- 4) Compute new‐repowered annual energy (TWh)
hours_per_year = 8760
df["Annual_Energy_TWh"] = (
    df["Total_New_Capacity"]   # MW
    * df["CapacityFactor"]     # fraction
    * hours_per_year           # hours/year
    / 1e6                      # → TWh
)

# --- 5) Pick the larger of baseline vs repowered energy per row
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

# --- 6) Country‐level consumption data
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

# --- 7) Aggregate per‐country and compute coverage %
wind_cap = df.groupby("Country")["Selected_by_Capacity"].sum().rename("Wind_CapBased_TWh")
wind_yld = df.groupby("Country")["Selected_by_Yield"].sum().rename("Wind_YieldBased_TWh")

df_comb = (
    df_cons
      .merge(wind_cap, left_on="Country", right_index=True)
      .merge(wind_yld, left_on="Country", right_index=True)
)
df_comb["Coverage_Cap_%"] = (
    df_comb["Wind_CapBased_TWh"] / df_comb["Electricity_Consumption_TWh"] * 100
)
df_comb["Coverage_Yld_%"] = (
    df_comb["Wind_YieldBased_TWh"] / df_comb["Electricity_Consumption_TWh"] * 100
)
df_sorted = df_comb.sort_values("Coverage_Cap_%", ascending=False).reset_index(drop=True)

# --- 8) Plot bar chart
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

x     = np.arange(len(df_sorted))
width = 0.35
colors = ['#4c72b0', '#55a868']

fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.bar(
    x - width/2,
    df_sorted["Coverage_Cap_%"],
    width=width,
    label="Baseline - Capacity-Based",
    color=colors[0], edgecolor='black', linewidth=0.7
)
ax.bar(
    x + width/2,
    df_sorted["Coverage_Yld_%"],
    width=width,
    label="Repowering NLH-Energy Yield",
    color=colors[1], edgecolor='black', linewidth=0.7
)

ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(df_sorted["Country"], rotation=45, ha='right', fontsize=10)
ax.tick_params(axis='y', labelsize=12)

ax.set_ylabel("Demand Coverage (%)", fontsize=14, labelpad=10)
ax.set_title(
    "EU Countries: Demand Coverage by Repowering Approach",
    fontsize=18, fontweight='bold', pad=15
)

ax.legend(loc="upper right", fontsize=12, frameon=False)
plt.tight_layout()
plt.show()

# --- 9) Overall EU coverage summary
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
