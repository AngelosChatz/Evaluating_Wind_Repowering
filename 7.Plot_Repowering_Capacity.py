import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# Define base directory and subdirectories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"

# Define file paths for each approach using relative paths
file_approach_1 = results_dir / "Approach_1.xlsx"
file_approach_2 = results_dir / "Approach_2.xlsx"
file_approach_3 = results_dir / "Approach_3.xlsx"
file_approach_4 = results_dir / "Approach_4.xlsx"
file_approach_5 = results_dir / "Approach_5.xlsx"

# Ensure naming consistency
approaches = {
    "Approach 1-Power Density": file_approach_1,
    "Approach 2-Capacity Maximization": file_approach_2,
    "Approach 3-Rounding Up": file_approach_3,
    "Approach 4-Single Turbine Flex": file_approach_4,
    "Approach 5-No-Loss Hybrid": file_approach_5,
}

years = pd.date_range(start='2000', end='2050', freq='YE')

def load_and_clean_data(file_path, rename_total_power=True):
    df = pd.read_excel(file_path)
    rename_dict = {}
    if rename_total_power and 'Total power' in df.columns:
        rename_dict['Total power'] = 'Total Power (MW)'
    if 'Total_New_Capacity' in df.columns:
        rename_dict['Total_New_Capacity'] = 'Repowered Total Capacity (MW)'
    df = df.rename(columns=rename_dict)

    if 'Total Power (MW)' in df.columns:
        df = df[df['Total Power (MW)'].notna()]
        df = df[df['Total Power (MW)'] != '#ND']
    df = df[df['Commissioning date'].notna()]
    df = df[df['Commissioning date'] != '#ND']

    def normalize_date(date):
        try:
            if '/' in str(date):
                return pd.to_datetime(date, format='%Y/%m', errors='coerce')
            else:
                return pd.to_datetime(date, format='%Y', errors='coerce')
        except Exception:
            return pd.NaT

    df['Commissioning date'] = df['Commissioning date'].apply(normalize_date)
    if 'Decommissioning date' not in df.columns:
        df['Decommissioning date'] = pd.NaT
    df['Decommissioning date'] = df['Decommissioning date'].apply(normalize_date)
    df['Decommissioning date'] = df['Decommissioning date'].fillna(
        df['Commissioning date'] + pd.DateOffset(years=20)
    )

    if 'Repowered Total Capacity (MW)' in df.columns:
        df['Repowered Total Capacity (MW)'] = pd.to_numeric(
            df['Repowered Total Capacity (MW)'], errors='coerce').fillna(0)
    else:
        df['Repowered Total Capacity (MW)'] = 0
    if 'Repowered Decommissioning date' not in df.columns:
        df['Repowered Decommissioning date'] = pd.NaT

    return df

def calculate_no_replacement_capacity(df):
    capacity = pd.DataFrame({'Year': years.year}).set_index('Year')
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating = df[(df['Commissioning date'].dt.year <= year) &
                       (df['Decommissioning date'].dt.year >= year)]
        capacity.loc[year, 'Operating Capacity'] = operating['Total Power (MW)'].sum()
    return capacity['Operating Capacity'] / 1000  # GW

def calculate_replacement_same_capacity(df):
    capacity = pd.DataFrame({'Year': years.year}).set_index('Year')
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating = df[(df['Commissioning date'].dt.year <= year) &
                       (df['Decommissioning date'].dt.year >= year)]
        replacement = 0
        if year >= 2023:
            decomm = df[df['Decommissioning date'].dt.year == year]
            replacement = decomm['Total Power (MW)'].sum()
            for idx in decomm.index:
                df.loc[idx, 'Commissioning date'] = pd.to_datetime(f'{year}')
                df.loc[idx, 'Decommissioning date'] = (
                    pd.to_datetime(f'{year}') + pd.DateOffset(years=20)
                )
        capacity.loc[year, 'Operating Capacity'] = operating['Total Power (MW)'].sum() + replacement
    return capacity['Operating Capacity'] / 1000  # GW

def calculate_capacity(df, start_repowering_year, repowered_lifetime=50):
    capacity = pd.DataFrame({'Year': years.year}).set_index('Year')
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating = df[(df['Commissioning date'].dt.year <= year) &
                       (df['Decommissioning date'].dt.year >= year)]
        net_capacity = operating['Total Power (MW)'].sum()
        if year >= start_repowering_year:
            repowered = df[(df['Decommissioning date'].dt.year == year) &
                           (df['Repowered Total Capacity (MW)'] > 0)]
            old_cap = repowered['Total Power (MW)'].sum()
            new_cap = repowered['Repowered Total Capacity (MW)'].sum()
            df.loc[repowered.index, 'Repowered Decommissioning date'] = (
                pd.to_datetime(f'{year}') + pd.DateOffset(years=repowered_lifetime)
            )
            net_capacity = net_capacity - old_cap + new_cap
        repowering_operating = df[(df['Repowered Decommissioning date'].notna()) &
                                  (df['Repowered Decommissioning date'].dt.year >= year) &
                                  (year >= df['Decommissioning date'].dt.year)]
        repowering_net = repowering_operating['Repowered Total Capacity (MW)'].sum()
        capacity.loc[year, 'Operating Capacity'] = net_capacity + repowering_net
    return capacity['Operating Capacity'] / 1000  # GW

def calculate_growth_rates(df, col, year_range):
    growth = [np.nan]
    cap_list = []
    for yr in year_range:
        cap = df[df['Decommissioning date'].dt.year == yr][col].sum() / 1000
        cap_list.append(cap)
    cap_array = np.array(cap_list)
    for i in range(1, len(year_range)):
        prev, curr = cap_array[i-1], cap_array[i]
        growth.append(((curr - prev) / prev) * 100 if prev != 0 else np.nan)
    return growth

# --- Prepare Data and Plotting ---

df_baseline = load_and_clean_data(file_approach_1)
baseline_capacity = calculate_no_replacement_capacity(df_baseline.copy())
replacement_capacity = calculate_replacement_same_capacity(df_baseline.copy())

repowering_capacity_dict = {}
decom_growth_dict = {}
repower_growth_dict = {}
year_range = np.arange(2022, 2043)

approach_colors = {
    "Approach 0-Old (Decomm+Repl)": "black",
    "Approach 1-Power Density": "blue",
    "Approach 2-Capacity Maximization": "orange",
    "Approach 3-Rounding Up": "green",
    "Approach 4-Single Turbine Flex": "red",
    "Approach 5-No-Loss Hybrid": "brown",
    "Approach 6-No-Loss Hybrid (Yield-based)": "purple"
}

for name, path in approaches.items():
    df = load_and_clean_data(path)
    repowering_capacity_dict[name] = calculate_capacity(df.copy(), 2023, 50)
    decom_growth_dict[name] = calculate_growth_rates(df.copy(), 'Total Power (MW)', year_range)
    repower_growth_dict[name] = calculate_growth_rates(df.copy(), 'Repowered Total Capacity (MW)', year_range)

# 1. Combined Line Plot
plt.figure(figsize=(12, 6))
plt.plot(years.year, baseline_capacity, label="Approach 0-Old (Decomm+Repl)", color=approach_colors["Approach 0-Old (Decomm+Repl)"], linewidth=2)
plt.plot(years.year, replacement_capacity, label="Replacement with Same Capacity", linestyle='--', color="gray", linewidth=2)
for name in approaches:
    plt.plot(years.year, repowering_capacity_dict[name], label=name, linestyle='-.', color=approach_colors[name], linewidth=2)
plt.axvline(x=2022, color='gray', linestyle='--', linewidth=1, label='Modeling Starts (2022)')
plt.title("Combined Capacity Scenarios Over Time (2000–2050)")
plt.xlabel("Year")
plt.ylabel("Operating Capacity (GW)")
plt.legend(loc="upper left", fontsize=9)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("Combined Capacity Scenarios Over Time.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Combined Capacity by Country Bar Chart
baseline_by_country = (df_baseline.groupby('Country')['Total Power (MW)'].sum() / 1000).rename("Approach 0-Old (Decomm+Repl)")

repowered_country_dict = {}
for name, path in approaches.items():
    df = load_and_clean_data(path)
    repowered_country_dict[name] = (df.groupby('Country')['Repowered Total Capacity (MW)'].sum() / 1000).rename(f"{name} (2050)")

df_bar = pd.concat([baseline_by_country] + list(repowered_country_dict.values()), axis=1).fillna(0)
df_bar = df_bar.sort_values("Approach 0-Old (Decomm+Repl)", ascending=False)
countries = df_bar.index.tolist()
n_bars = df_bar.shape[1]
x = np.arange(len(countries))
bar_width = 0.8 / n_bars
plt.figure(figsize=(14, 6))
for i, col in enumerate(df_bar.columns):
    offset = (i - n_bars/2) * bar_width + bar_width/2
    if col == "Approach 0-Old (Decomm+Repl)":
        color = "black"
    else:
        approach_name = col.replace(" (2050)", "")
        color = approach_colors.get(approach_name, "gray")
    plt.bar(x + offset, df_bar[col], width=bar_width, label=col, color=color)
plt.xticks(x, countries, rotation=45, ha="right")
plt.ylabel("Capacity (GW)")
plt.title("Combined Capacity by Country: Baseline (2022) & Repowered (2050)")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig("Combined Capacity by Country: Baseline (2022) & Repowered (2050).png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Growth Rate Subplots (3x2)
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs_flat = axs.flatten()
for idx, name in enumerate(approaches):
    ax = axs_flat[idx]
    ax.plot(year_range, decom_growth_dict[name], label="Decommissioning Growth", marker="o", linewidth=2, color=approach_colors[name])
    ax.plot(year_range, repower_growth_dict[name], label="Repowering Growth", marker="s", linestyle='--', linewidth=2, color=approach_colors[name])
    ax.set_title(f"{name} Growth Rates (2022–2042)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth Rate (%)")
    ax.legend(fontsize=9)
    ax.grid(True)
for ax in axs_flat[len(approaches):]:
    fig.delaxes(ax)
plt.tight_layout()
plt.savefig("Growth Rates.png", dpi=300, bbox_inches="tight")
plt.show()

# Section 4: ΔCapacity vs Single-Turbine Share
df1 = load_and_clean_data(file_approach_1)
df5 = load_and_clean_data(file_approach_5)

baseline_country = df_baseline.groupby('Country')['Total Power (MW)'].sum() / 1000
cap1_country = df1.groupby('Country')['Repowered Total Capacity (MW)'].sum() / 1000
cap5_country = df5.groupby('Country')['Repowered Total Capacity (MW)'].sum() / 1000
delta1 = (cap1_country - baseline_country).fillna(0)
delta5 = (cap5_country - baseline_country).fillna(0)

df_t = df_baseline.copy()
df_t["NumTurbines"] = pd.to_numeric(df_t.get("Number of turbines", 0), errors="coerce").fillna(0)
total_parks = df_t.groupby("Country").size()
single_parks = df_t[df_t["NumTurbines"] == 1].groupby("Country").size()
share = (single_parks / total_parks * 100).reindex(delta1.index).fillna(0)

order = share.sort_values().index.tolist()
d1 = delta1.reindex(order)
d5 = delta5.reindex(order)
s = share.reindex(order)

x = np.arange(len(order))
w = 0.3

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.bar(x - w/2, d1, width=w, label="ΔCapacity A1", color=approach_colors["Approach 1-Power Density"], alpha=0.8)
ax1.bar(x + w/2, d5, width=w, label="ΔCapacity A5", color=approach_colors["Approach 5-No-Loss Hybrid"], alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(order, rotation=45, ha='right')
ax1.set_ylabel("Δ Capacity (GW)")
ax1.set_title("Δ Capacity vs Single‐Turbine Share (A1 & A5)")

ax2 = ax1.twinx()
ax2.plot(x, s, 'o-', color='red', label="Single‐Turbine Share (%)")
ax2.set_ylabel("Single‐Turbine Share (%)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
plt.tight_layout()
plt.savefig("Δ Capacity vs Single‐Turbine Share.png", dpi=300, bbox_inches="tight")
plt.show()

# Section 5: ΔCapacity vs Avg Turbine Age
df_age = df_baseline.copy()
df_age["Age"] = datetime.date.today().year - df_age["Commissioning date"].dt.year
age = df_age.groupby("Country")["Age"].mean().reindex(delta1.index).fillna(0)

order2 = age.sort_values().index.tolist()
d1_2 = delta1.reindex(order2)
d5_2 = delta5.reindex(order2)
a2 = age.reindex(order2)

x2 = np.arange(len(order2))

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.bar(x2 - w/2, d1_2, width=w, label="ΔCapacity A1", color=approach_colors["Approach 1-Power Density"], alpha=0.8)
ax1.bar(x2 + w/2, d5_2, width=w, label="ΔCapacity A5", color=approach_colors["Approach 5-No-Loss Hybrid"], alpha=0.8)
ax1.set_xticks(x2)
ax1.set_xticklabels(order2, rotation=45, ha='right')
ax1.set_ylabel("Δ Capacity (GW)")
ax1.set_title("Δ Capacity vs Avg Turbine Age (A1 & A5)")

ax2 = ax1.twinx()
ax2.plot(x2, a2, 'o-', color='red', label="Avg Turbine Age (years)")
ax2.set_ylabel("Avg Turbine Age (years)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
plt.tight_layout()
plt.savefig("Capacity vs Avg Turbine Age.png", dpi=300, bbox_inches="tight")
plt.show()
