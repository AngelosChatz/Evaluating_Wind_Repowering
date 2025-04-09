import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
from pathlib import Path

# Define base directory and subdirectories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"

# Define file paths for each approach using relative paths
file_approach_1 = results_dir / "Approach_1.xlsx"
file_approach_2 = results_dir / "Approach_2.xlsx"
file_approach_3 = results_dir / "Approach_3.xlsx"
file_approach_4 = results_dir / "Approach_4.xlsx"

approaches = {
    "Approach 1": file_approach_1,
    "Approach 2": file_approach_2,
    "Approach 3": file_approach_3,
    "Approach 4": file_approach_4,
}

years = pd.date_range(start='2000', end='2050', freq='YE')


def load_and_clean_data(file_path, rename_total_power=True):
    """
    Reads an Excel file, renames columns if needed, filters out invalid rows,
    normalizes dates, and fills missing decommissioning dates.
    """
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

    # Normalize date columns: try YYYY/MM if '/' in string, else YYYY
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
        df['Repowered Total Capacity (MW)'] = pd.to_numeric(df['Repowered Total Capacity (MW)'],
                                                            errors='coerce').fillna(0)
    else:
        df['Repowered Total Capacity (MW)'] = 0

    if 'Repowered Decommissioning date' not in df.columns:
        df['Repowered Decommissioning date'] = pd.NaT

    return df


def calculate_no_replacement_capacity(df):
    capacity = pd.DataFrame({'Year': years.year})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating = df[(df['Commissioning date'].dt.year <= year) &
                       (df['Decommissioning date'].dt.year >= year)]
        capacity.loc[year, 'Operating Capacity'] = operating['Total Power (MW)'].sum()
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']


def calculate_replacement_same_capacity(df):
    """
    Calculates yearly operating capacity assuming each decommissioned park is
    replaced by one of the same capacity from 2023 onward.
    """
    capacity = pd.DataFrame({'Year': years.year})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating = df[(df['Commissioning date'].dt.year <= year) &
                       (df['Decommissioning date'].dt.year >= year)]
        if year >= 2023:
            decomm = df[df['Decommissioning date'].dt.year == year]
            replacement = decomm['Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0
            # Extend operational life by 20 years for replaced parks
            for idx in decomm.index:
                df.loc[idx, 'Commissioning date'] = pd.to_datetime(f'{year}')
                df.loc[idx, 'Decommissioning date'] = pd.to_datetime(f'{year}') + pd.DateOffset(years=20)
        else:
            replacement = 0
        capacity.loc[year, 'Operating Capacity'] = operating['Total Power (MW)'].sum() + replacement
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']


def calculate_capacity(df, start_repowering_year, repowered_lifetime=50):
    """
    Calculates yearly operating capacity under a repowering scenario.
    When repowering occurs (if Repowered Total Capacity > 0), the park’s
    operational lifetime is extended.
    """
    capacity = pd.DataFrame({'Year': years.year})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating = df[(df['Commissioning date'].dt.year <= year) &
                       (df['Decommissioning date'].dt.year >= year)]
        if year >= start_repowering_year:
            repowered = df[(df['Decommissioning date'].dt.year == year) &
                           (df['Repowered Total Capacity (MW)'] > 0)]
            df.loc[repowered.index, 'Repowered Decommissioning date'] = pd.to_datetime(f'{year}') + pd.DateOffset(
                years=repowered_lifetime)
            old_cap = repowered['Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0
            new_cap = repowered['Repowered Total Capacity (MW)'].sum()
            net = operating['Total Power (MW)'].sum() - old_cap + new_cap
        else:
            net = operating['Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0

        repowering_operating = df[(df['Repowered Decommissioning date'].notna()) &
                                  (df['Repowered Decommissioning date'].dt.year >= year) &
                                  (year >= df['Decommissioning date'].dt.year)]
        repowering_net = repowering_operating['Repowered Total Capacity (MW)'].sum()
        capacity.loc[year, 'Operating Capacity'] = net + repowering_net
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']


def calculate_growth_rates(df, col, year_range):
    """
    Calculates annual growth rates (in percent) for capacity (in GW) based on
    values of the specified column for parks with decommissioning in each year.
    """
    growth = [np.nan]
    cap_list = []
    for yr in year_range:
        cap = df[df['Decommissioning date'].dt.year == yr][col].sum() / 1000
        cap_list.append(cap)
    cap_array = np.array(cap_list)
    for i in range(1, len(year_range)):
        prev = cap_array[i - 1]
        curr = cap_array[i]
        growth.append(((curr - prev) / prev) * 100 if prev != 0 else np.nan)
    return growth


# Prepare Data for Combined Plots

df_baseline = load_and_clean_data(file_approach_1)
baseline_capacity = calculate_no_replacement_capacity(df_baseline.copy())

# Compute a single "Replacement with Same Capacity" curve
replacement_capacity = calculate_replacement_same_capacity(df_baseline.copy())

# Dictionaries to hold the repowering curves and growth rates for each approach.
repowering_capacity_dict = {}
decom_growth_dict = {}
repower_growth_dict = {}
year_range = np.arange(2022, 2043)

# Define specific colors for each approach.
approach_colors = {
    "Approach 1": "blue",
    "Approach 2": "orange",
    "Approach 3": "green",
    "Approach 4": "red"
}

for approach_name, file_path in approaches.items():
    df = load_and_clean_data(file_path)
    repowering_curve = calculate_capacity(df.copy(), start_repowering_year=2023, repowered_lifetime=50)
    repowering_capacity_dict[approach_name] = repowering_curve

    decom_growth = calculate_growth_rates(df.copy(), 'Total Power (MW)', year_range)
    repower_growth = calculate_growth_rates(df.copy(), 'Repowered Total Capacity (MW)', year_range)
    decom_growth_dict[approach_name] = decom_growth
    repower_growth_dict[approach_name] = repower_growth


# 1. Combined Line Plot

plt.figure(figsize=(12, 6))
# Plot baseline (No Replacement) curve
plt.plot(years.year, baseline_capacity, label="No Replacement (Baseline)", color="black", linewidth=2)
# Plot one replacement curve (common to all approaches)
plt.plot(years.year, replacement_capacity, label="Replacement with Same Capacity", linestyle='--',
         color="gray", linewidth=2)
# Plot repowering curves for each approach
for approach_name in approaches.keys():
    plt.plot(years.year, repowering_capacity_dict[approach_name],
             label=f"{approach_name} Repowering", linestyle='-.',
             color=approach_colors[approach_name], linewidth=2)

plt.axvline(x=2022, color='gray', linestyle='--', linewidth=1, label='Modeling Starts (2022)')
plt.title("Combined Capacity Scenarios Over Time (2000-2050)")
plt.xlabel("Year")
plt.ylabel("Operating Capacity (GW)")
plt.legend(loc="upper left", fontsize=9)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# 2. Combined Capacity by Country Bar Chart

if 'Country' in df_baseline.columns:
    baseline_by_country = (df_baseline.groupby('Country')['Total Power (MW)'].sum() / 1000).rename("Baseline (2022)")
else:
    raise ValueError("Column 'Country' not found in baseline data.")

repowered_country_dict = {}
for approach_name, file_path in approaches.items():
    df = load_and_clean_data(file_path)
    if 'Country' in df.columns:
        repowered = (df.groupby('Country')['Repowered Total Capacity (MW)'].sum() / 1000).rename(
            f"{approach_name} Repowered (2050)"
        )
        repowered_country_dict[approach_name] = repowered
    else:
        print(f"Column 'Country' not found in {approach_name}; skipping repowered data for this approach.")

df_bar = pd.concat([baseline_by_country] + list(repowered_country_dict.values()), axis=1).fillna(0)
df_bar = df_bar.sort_values("Baseline (2022)", ascending=False)
countries = df_bar.index.tolist()
n_bars = df_bar.shape[1]
x = np.arange(len(countries))
bar_width = 0.8 / n_bars

plt.figure(figsize=(14, 6))
for i, col in enumerate(df_bar.columns):
    offset = (i - n_bars / 2) * bar_width + bar_width / 2
    plt.bar(x + offset, df_bar[col], width=bar_width, label=col)

plt.xticks(x, countries, rotation=45, ha="right")
plt.ylabel("Capacity (GW)")
plt.title("Combined Capacity by Country: Baseline (2022) & Repowered (2050)")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.show()


# 3. Growth Rate Subplots (2x2)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
for ax, approach_name in zip(axs, approaches.keys()):
    ax.plot(year_range, decom_growth_dict[approach_name], label="Decommissioning Growth",
            color="purple", marker="o", linewidth=2)
    ax.plot(year_range, repower_growth_dict[approach_name], label="Repowering Growth",
            color="teal", marker="s", linewidth=2)
    ax.set_title(f"{approach_name} Growth Rates (2022-2042)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth Rate (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)

plt.tight_layout()
plt.show()
