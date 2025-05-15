import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Define file paths
excel_path            = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf.xlsx"
old_excel_path        = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Approach_2_Cf_old.xlsx"
output_yield_file     = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Energy_Yield_Parks.xlsx"
dual_axis_chart_path  = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\dual_axis_bar_chart.png"
pie_chart_path        = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\annual_energy_pie_TWh.png"

#%% Read main data and clean
df = pd.read_excel(excel_path, index_col=0)
df["Total_New_Capacity"]      = pd.to_numeric(df["Total_New_Capacity"], errors='coerce').fillna(0)
df["CapacityFactor"]          = pd.to_numeric(df["CapacityFactor"], errors='coerce').fillna(0)
df["Recommended_WT_Capacity"] = pd.to_numeric(df.get("Recommended_WT_Capacity", 0),
                                               errors='coerce')  # leave NaN for now

#%% Read old file and prepare for merge
df_old = pd.read_excel(old_excel_path, index_col=0)
# ensure we have a numeric Annual_Energy_TWh in the old sheet
df_old["Annual_Energy_TWh"] = pd.to_numeric(df_old["Annual_Energy_TWh"], errors='coerce').fillna(0)

#%% Join old annual‐energy into main df
# this will create a helper column 'Annual_Energy_TWh_old'
df = df.join(
    df_old[["Annual_Energy_TWh"]]
    .rename(columns={"Annual_Energy_TWh": "Annual_Energy_TWh_old"}),
    how="left"
)

#%% Fill missing Recommended_WT_Capacity from old Annual_Energy_TWh
df["Recommended_WT_Capacity"] = df["Recommended_WT_Capacity"].fillna(
    df["Annual_Energy_TWh_old"]
)

# (optional) drop the helper column
df = df.drop(columns="Annual_Energy_TWh_old")

#%% Compute Annual Energy Production
hours_per_year = 8760
df["Annual_Energy_MWh"] = df["Total_New_Capacity"] * df["CapacityFactor"] * hours_per_year
df["Annual_Energy_TWh"] = df["Annual_Energy_MWh"] / 1e6

total_energy_TWh = df["Annual_Energy_TWh"].sum()
print(f"Total Annual Energy Production (TWh): {total_energy_TWh:.3f}")

# Save per-park yields
df.to_excel(output_yield_file)
print(f"Energy yield data for every park saved to: {output_yield_file}")

#%% Aggregate by Country
energy_by_country   = df.groupby("Country")["Annual_Energy_TWh"].sum().reset_index()
capacity_by_country = df.groupby("Country")["Total_New_Capacity"].sum().reset_index()

combined = pd.merge(energy_by_country,
                    capacity_by_country,
                    on="Country")
combined = combined.sort_values(by="Annual_Energy_TWh",
                                ascending=False).reset_index(drop=True)

print("Combined aggregated data (per country):")
print(combined)

#%% Plot 1: Dual-axis Bar Chart
x     = np.arange(len(combined))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 8))
ax2      = ax1.twinx()

bars1 = ax1.bar(
    x - width/2,
    combined['Annual_Energy_TWh'],
    width,
    color='blue',
    label='Annual Energy (TWh)'
)
bars2 = ax2.bar(
    x + width/2,
    combined['Total_New_Capacity'],
    width,
    color='red',
    alpha=0.7,
    label='Installed Capacity (MW)'
)

ax1.set_xticks(x)
ax1.set_xticklabels(combined['Country'], rotation=90)
ax1.set_ylabel("Annual Energy Production (TWh)")
ax2.set_ylabel("Installed Capacity (MW)")

# Combine legends
handles, labels = [], []
for ax in (ax1, ax2):
    h, l = ax.get_legend_handles_labels()
    handles += h; labels += l
ax1.legend(handles, labels, loc='upper left')

plt.title("Comparison of Annual Energy Yield and Installed Capacity per Country")
plt.tight_layout()
plt.savefig(dual_axis_chart_path, dpi=300)
print(f"Dual-axis chart saved to: {dual_axis_chart_path}")
plt.show()

#%% Plot 2: Pie Chart of Energy Shares
# Calculate share
total_energy = energy_by_country["Annual_Energy_TWh"].sum()
energy_by_country["Share"] = energy_by_country["Annual_Energy_TWh"] / total_energy

# Split large vs small
above = energy_by_country[energy_by_country["Share"] >= 0.01].copy()
below = energy_by_country[energy_by_country["Share"] <  0.01].copy()

if not below.empty:
    other_total = below["Annual_Energy_TWh"].sum()
    other_row   = pd.DataFrame({
        "Country": ["Other"],
        "Annual_Energy_TWh": [other_total]
    })
    plot_df = pd.concat([above[["Country", "Annual_Energy_TWh"]],
                         other_row],
                        ignore_index=True)
else:
    plot_df = above[["Country", "Annual_Energy_TWh"]]

# Sort descending
plot_df = plot_df.sort_values("Annual_Energy_TWh",
                              ascending=False).reset_index(drop=True)

# Prepare labels
sizes       = plot_df["Annual_Energy_TWh"]
labels      = plot_df["Country"]
percentages = sizes / sizes.sum() * 100
labels_with_pct = [f"{lbl}\n{pct:.1f}%" for lbl, pct in zip(labels, percentages)]

print("Data for pie chart (sorted, with stacked labels):")
print(plot_df.assign(Pct=percentages.round(1)))

# Plot
colors = plt.cm.tab20(np.linspace(0, 1, len(plot_df)))

plt.figure(figsize=(10, 8))
plt.pie(
    sizes,
    labels=labels_with_pct,
    labeldistance=1.05,
    startangle=90,
    counterclock=False,
    colors=colors,
    textprops={'fontsize': 8}
)
plt.title("Share of Annual Energy Production by Country (TWh)", fontsize=12)
plt.tight_layout()
plt.savefig(pie_chart_path, dpi=300)
print(f"Pie chart saved to: {pie_chart_path}")
plt.show()
