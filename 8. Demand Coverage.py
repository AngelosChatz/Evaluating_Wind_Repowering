import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === FILE PATHS ===
repowered_path = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Energy_Yield_Parks.xlsx"
existing_path  = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf_old.xlsx"

# === LOAD DATASETS ===
df_rep = pd.read_excel(repowered_path)
df_existing = pd.read_excel(existing_path)

# === FIX COLUMN MISMATCH ERROR ===
common_columns = df_existing.columns.intersection(df_rep.columns)
df_existing = df_existing[common_columns]

# === REPLACE ROWS WITH MISSING REPOWERED CAPACITY ===
missing_mask = df_rep["Recommended_WT_Capacity"].isna()
df_combined = df_rep.copy()
df_combined.loc[missing_mask] = df_existing.loc[missing_mask]

# === GROUP BY COUNTRY ===
country_energy = df_combined.groupby("Country")["Annual_Energy_TWh"].sum().reset_index()

# === ELECTRICITY CONSUMPTION DATA (TWh, 2022) ===
consumption_data = {
    "Country": [
        "Russia", "Germany", "France", "Italy", "UK", "Turkey", "Spain", "Poland", "Sweden", "Norway",
        "Netherlands", "Ukraine", "Belgium", "Finland", "Austria", "Czechia", "Switzerland", "Portugal",
        "Romania", "Greece", "Hungary", "Bulgaria", "Denmark", "Belarus", "Serbia", "Ireland", "Slovakia",
        "Iceland", "Croatia", "Slovenia", "Bosnia & Herzegovina", "Lithuania", "Estonia", "Albania",
        "Latvia", "North Macedonia", "Luxembourg", "Moldova", "Cyprus", "Montenegro", "Malta",
        "Faroe Islands", "Gibraltar"
    ],
    "Electricity_Consumption_TWh": [
        1020.67, 512.19, 431.94, 301.48, 288.93, 287.32, 232.84, 167.54, 131.12, 124.91, 111.71, 98.01,
        82.23, 80.55, 68.17, 63.75, 57.19, 50.57, 50.44, 49.54, 43.83, 35.47, 34.3, 33.79, 33.49, 29.72,
        26.35, 19.33, 16.73, 13.52, 12.45, 11.28, 8.62, 6.94, 6.93, 6.23, 6.22, 5.59, 5.12, 2.99, 2.75,
        0.46, 0.21
    ]
}
df_consumption = pd.DataFrame(consumption_data)

# === MERGE & CALCULATE WIND SHARE ===
merged = pd.merge(country_energy, df_consumption, on="Country", how="inner")
merged["Wind_Share_%"] = (merged["Annual_Energy_TWh"] / merged["Electricity_Consumption_TWh"]) * 100
merged = merged.sort_values("Electricity_Consumption_TWh", ascending=False).reset_index(drop=True)

# === PLOT ===
fig, ax1 = plt.subplots(figsize=(18, 8))
ax2 = ax1.twinx()

x = np.arange(len(merged))
width = 0.6

# Bars (Consumption and Wind)
ax1.bar(x, merged["Electricity_Consumption_TWh"], color='lightgray', width=width, label="Electricity Consumption")
ax1.bar(x, merged["Annual_Energy_TWh"], color='green', width=width, label="Wind Energy (Repowered/Existing)")

# Wind Share (%)
ax2.scatter(x, merged["Wind_Share_%"], color='blue', label="Wind Share (%)", zorder=5)

# Formatting
ax1.set_ylabel("Electricity (TWh)")
ax2.set_ylabel("Wind Share of Demand (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(merged["Country"], rotation=90)
ax1.set_title("Electricity Consumption vs Wind Energy (Vertical) with Demand Coverage (%)")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()


# === SUMMARY STATISTICS ===
total_demand       = merged["Electricity_Consumption_TWh"].sum()
total_wind_supply  = merged["Annual_Energy_TWh"].sum()
pct_covered        = (total_wind_supply / total_demand) * 100

print(f"Total Electricity Demand (TWh): {total_demand:,.2f}")
print(f"Total Wind Energy Supply (TWh): {total_wind_supply:,.2f}")
print(f"Percent of Demand Covered by Wind: {pct_covered:.1f}%")