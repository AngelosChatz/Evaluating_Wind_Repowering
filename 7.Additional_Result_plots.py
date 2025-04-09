import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
from pathlib import Path

# Set up base directory and subdirectories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"


# Define file paths for each approach using relative paths
files = {
    "Approach 1": results_dir / "Approach_1.xlsx",
    "Approach 2": results_dir / "Approach_2.xlsx",
    "Approach 3": results_dir / "Approach_3.xlsx",
    "Approach 4": results_dir / "Approach_4.xlsx"
}

# Define labels for approaches for plot legends
approach_labels = {1: "Approach 1", 2: "Approach 2", 3: "Approach 3", 4: "Approach 4"}



# SECTION 1: LINE CHART – CAPACITY 2000–2050
def compute_lines(file_path):
    # Read data from Excel file
    df = pd.read_excel(file_path)

    # Clean key columns
    for col in ['Commissioning date', 'Decommissioning date', 'Total power', 'Total_New_Capacity']:
        if col in df.columns:
            df[col] = df[col].replace(['#ND', ''], np.nan)
        else:
            print(f"Warning: Column '{col}' not found in the Excel file.")

    # Convert data types
    df['Commissioning date'] = pd.to_datetime(df['Commissioning date'], errors='coerce')
    df['Decommissioning date'] = pd.to_datetime(df['Decommissioning date'], errors='coerce')
    df['Total power'] = pd.to_numeric(df['Total power'], errors='coerce')
    df['Total_New_Capacity'] = pd.to_numeric(df['Total_New_Capacity'], errors='coerce')

    # PART A: Baseline capacity changes
    capacity_changes_hist = {}

    def add_change(year, change):
        capacity_changes_hist[year] = capacity_changes_hist.get(year, 0) + change

    for _, row in df.iterrows():
        comm = row['Commissioning date']
        tot_power = row['Total power']
        if pd.isna(comm) or pd.isna(tot_power):
            continue
        start_year = comm.year
        if pd.isna(row['Decommissioning date']):
            end_year = start_year + 30
        else:
            end_year = row['Decommissioning date'].year
        add_change(start_year, tot_power)
        add_change(end_year, -tot_power)

    hist_capacity = {}
    running_sum = 0.0
    for y in range(1980, 2023):
        if y in capacity_changes_hist:
            running_sum += capacity_changes_hist[y]
        hist_capacity[y] = running_sum

    base_growth = {}
    for y in range(2000, 2023):
        base_growth[y] = hist_capacity[y]
    if 2000 in hist_capacity and 2022 in hist_capacity:
        growth_per_year = (hist_capacity[2022] - hist_capacity[2000]) / (2022 - 2000)
    else:
        growth_per_year = 0
    for y in range(2023, 2051):
        base_growth[y] = hist_capacity[2022] + growth_per_year * (y - 2022)

    # PART B: Repowering increments
    repower_changes = {}

    def add_repower_change(year, change):
        repower_changes[year] = repower_changes.get(year, 0) + change

    for _, row in df.iterrows():
        comm = row['Commissioning date']
        repower_cap = row['Total_New_Capacity']
        if pd.isna(comm) or pd.isna(repower_cap):
            continue
        if pd.isna(row['Decommissioning date']):
            decomm_year = comm.year + 20
        else:
            decomm_year = row['Decommissioning date'].year
        repower_start = decomm_year
        FINAL_YEAR = 2050
        while repower_start <= FINAL_YEAR:
            add_repower_change(repower_start, repower_cap)
            repower_end = repower_start + 20
            add_repower_change(repower_end, -repower_cap)
            repower_start = repower_end

    repower_line = {}
    running_sum = 0.0
    for y in range(1980, 2051):
        if y in repower_changes:
            running_sum += repower_changes[y]
        repower_line[y] = running_sum

    # PART C: Combined capacity = baseline + repowering
    combined_line = {}
    for y in range(2000, 2051):
        combined_line[y] = base_growth[y] + repower_line.get(y, 0)

    years_list = list(range(2000, 2051))
    baseline_line = [base_growth[y] for y in years_list]
    combined = [combined_line[y] for y in years_list]

    return years_list, baseline_line, combined


# Compute baseline and combined capacities for the line chart using one approach (e.g., Approach 1)
years, baseline_mw, _ = compute_lines(files["Approach 1"])
combined_results = {}
for name, file_path in files.items():
    _, _, combined_mw = compute_lines(file_path)
    combined_results[name] = combined_mw

plt.figure(figsize=(12, 8))
baseline_gw = [v / 1000.0 for v in baseline_mw]
years_plot = years

plt.plot(years_plot, baseline_gw, color="black", linestyle="dashed", label="Baseline (Base Growth)")
colors = {'Approach 1': 'blue', 'Approach 2': 'green', 'Approach 3': 'red', 'Approach 4': 'purple'}
for name, combined_mw in combined_results.items():
    combined_gw = [v / 1000.0 for v in combined_mw]
    plt.plot(years_plot, combined_gw, color=colors[name], linestyle="solid", label=f"{name} Combined")

plt.title("Capacity 2000–2050 Comparison: Baseline vs. Repowered per Approach", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Capacity (GW)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# SECTION 2: HORIZONTAL BAR CHART – REPOWERED POWER DENSITY (Only for Approach 3)
repowered_df = pd.read_excel(files["Approach 3"])
cols_to_clean = ['Country', 'Commissioning date', 'Decommissioning date', 'Total_New_Capacity',
                 'New_Total_Park_Area (m²)']
for col in cols_to_clean:
    if col in repowered_df.columns:
        repowered_df[col] = repowered_df[col].replace(['#ND', ''], np.nan)
    else:
        print(f"Warning: Column '{col}' not found in Approach 3 data.")

if 'Commissioning date' in repowered_df.columns:
    repowered_df['Commissioning date'] = pd.to_datetime(repowered_df['Commissioning date'], errors='coerce')
if 'Decommissioning date' in repowered_df.columns:
    repowered_df['Decommissioning date'] = pd.to_datetime(repowered_df['Decommissioning date'], errors='coerce')
if 'Total_New_Capacity' in repowered_df.columns:
    repowered_df['Total_New_Capacity'] = pd.to_numeric(repowered_df['Total_New_Capacity'], errors='coerce')
if 'New_Total_Park_Area (m²)' in repowered_df.columns:
    repowered_df['New_Total_Park_Area (m²)'] = pd.to_numeric(repowered_df['New_Total_Park_Area (m²)'], errors='coerce')

repowered_df = repowered_df[repowered_df['Total_New_Capacity'] > 0]
grouped_rep = repowered_df.groupby('Country').agg({
    'Total_New_Capacity': 'sum',
    'New_Total_Park_Area (m²)': 'sum'
}).reset_index()

grouped_rep['Repowered Power Density (MW/km²)'] = (
        grouped_rep['Total_New_Capacity'] / (grouped_rep['New_Total_Park_Area (m²)'] / 1e6)
)

grouped_rep = grouped_rep.sort_values('Repowered Power Density (MW/km²)', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(grouped_rep['Country'], grouped_rep['Repowered Power Density (MW/km²)'],
         color=plt.cm.viridis(np.linspace(0, 1, len(grouped_rep))))
plt.title("Repowered Power Density per Country (Approach 3)", fontsize=16, fontweight='bold')
plt.xlabel("Repowered Power Density (MW/km²)", fontsize=14)
plt.ylabel("Country", fontsize=14)
plt.ticklabel_format(style='plain', axis='x')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# SECTION 3: COMPARISON – ORIGINAL vs. REPOWERED POWER DENSITY (Approach 3)
original_df = pd.read_excel(files["Approach 3"])
cols_to_clean_orig = ['Country', 'Total power', 'Total Park Area (m²)']
for col in cols_to_clean_orig:
    if col in original_df.columns:
        original_df[col] = original_df[col].replace(['#ND', ''], np.nan)
    else:
        print(f"Warning: Column '{col}' not found for original data.")
if 'Total power' in original_df.columns:
    original_df['Total power'] = pd.to_numeric(original_df['Total power'], errors='coerce')
if 'Total Park Area (m²)' in original_df.columns:
    original_df['Total Park Area (m²)'] = pd.to_numeric(original_df['Total Park Area (m²)'], errors='coerce')

original_df = original_df[(original_df['Total power'] > 0) & (original_df['Total Park Area (m²)'] > 0)]
grouped_orig = original_df.groupby('Country').agg({
    'Total power': 'sum',
    'Total Park Area (m²)': 'sum'
}).reset_index()
grouped_orig['Original Power Density (MW/km²)'] = (
        grouped_orig['Total power'] / (grouped_orig['Total Park Area (m²)'] / 1e6)
)

compare_df = pd.merge(grouped_orig[['Country', 'Original Power Density (MW/km²)']],
                      grouped_rep[['Country', 'Repowered Power Density (MW/km²)']],
                      on='Country', how='outer').fillna(0)
compare_df = compare_df.sort_values('Repowered Power Density (MW/km²)', ascending=False)

countries = compare_df['Country']
y_pos = np.arange(len(countries))
bar_height = 0.35

plt.figure(figsize=(12, 8))
plt.barh(y_pos - bar_height / 2, compare_df['Original Power Density (MW/km²)'], height=bar_height,
         label='Original', color='skyblue')
plt.barh(y_pos + bar_height / 2, compare_df['Repowered Power Density (MW/km²)'], height=bar_height,
         label='Repowered', color='salmon')
plt.yticks(y_pos, countries, fontsize=12)
plt.xlabel("Power Density (MW/km²)", fontsize=14)
plt.title("Comparison: Original vs. Repowered Power Density per Country (Approach 3)",
          fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()



# SECTION 4: 2050 COUNTRY LAND AREA COMPARISON – STACKED BARS
def compute_country_capacities(file_path):
    """
    Compute the 2050 capacity for each country for:
      - Baseline (original) scenario
      - Combined (baseline plus repowering increments) scenario
    Returns a DataFrame with columns: Country, Baseline_2050, Combined_2050.
    """
    df = pd.read_excel(file_path)
    # Clean key columns
    for col in ['Country', 'Commissioning date', 'Decommissioning date', 'Total power', 'Total_New_Capacity']:
        if col in df.columns:
            df[col] = df[col].replace(['#ND', ''], np.nan)
        else:
            print(f"Warning: Column '{col}' not found in the Excel file.")
    df['Commissioning date'] = pd.to_datetime(df['Commissioning date'], errors='coerce')
    df['Decommissioning date'] = pd.to_datetime(df['Decommissioning date'], errors='coerce')
    df['Total power'] = pd.to_numeric(df['Total power'], errors='coerce')
    df['Total_New_Capacity'] = pd.to_numeric(df['Total_New_Capacity'], errors='coerce')

    countries = df['Country'].dropna().unique()
    results = []
    for country in countries:
        sub = df[df['Country'] == country]
        capacity_changes_hist = {}
        repower_changes = {}

        def add_change(year, change, dic):
            dic[year] = dic.get(year, 0) + change

        # Process each record for this country
        for _, row in sub.iterrows():
            comm = row['Commissioning date']
            tot_power = row['Total power']
            if pd.isna(comm) or pd.isna(tot_power):
                continue
            start_year = comm.year
            if pd.isna(row['Decommissioning date']):
                end_year = start_year + 30
            else:
                end_year = row['Decommissioning date'].year
            add_change(start_year, tot_power, capacity_changes_hist)
            add_change(end_year, -tot_power, capacity_changes_hist)

            repower_cap = row['Total_New_Capacity']
            if pd.isna(comm) or pd.isna(repower_cap):
                continue
            if pd.isna(row['Decommissioning date']):
                decomm_year = comm.year + 20
            else:
                decomm_year = row['Decommissioning date'].year
            repower_start = decomm_year
            FINAL_YEAR = 2050
            while repower_start <= FINAL_YEAR:
                add_change(repower_start, repower_cap, repower_changes)
                repower_end = repower_start + 20
                add_change(repower_end, -repower_cap, repower_changes)
                repower_start = repower_end

        # Build cumulative baseline capacity from 1980 to 2022
        baseline = {}
        running_sum = 0.0
        for y in range(1980, 2023):
            if y in capacity_changes_hist:
                running_sum += capacity_changes_hist[y]
            baseline[y] = running_sum
        if 2000 in baseline and 2022 in baseline:
            growth_per_year = (baseline[2022] - baseline[2000]) / (2022 - 2000)
        else:
            growth_per_year = 0
        for y in range(2023, 2051):
            baseline[y] = baseline[2022] + growth_per_year * (y - 2022)
        baseline_2050 = baseline[2050]

        # Build cumulative repowering increments from 1980 to 2050
        repowering = {}
        running_sum = 0.0
        for y in range(1980, 2051):
            if y in repower_changes:
                running_sum += repower_changes[y]
            repowering[y] = running_sum

        # Combined scenario = baseline + repowering
        combined = {}
        for y in range(2000, 2051):
            combined[y] = baseline[y] + repowering.get(y, 0)
        combined_2050 = combined[2050]

        results.append({'Country': country, 'Baseline_2050': baseline_2050, 'Combined_2050': combined_2050})
    return pd.DataFrame(results)


def compute_land_area_data(file_path):
    """
    Computes the land area data for each country for a given approach.
    Returns a DataFrame with columns: Country, Baseline Land Area (km²), Repowered Land Area (km²).
    """
    cap_df = compute_country_capacities(file_path)
    cap_df['Repowered Capacity (MW)'] = cap_df['Combined_2050'] - cap_df['Baseline_2050']
    orig_df = pd.read_excel(file_path)
    cols_to_clean_orig = ['Country', 'Total power', 'Total Park Area (m²)']
    for col in cols_to_clean_orig:
        if col in orig_df.columns:
            orig_df[col] = orig_df[col].replace(['#ND', ''], np.nan)
        else:
            print(f"Warning: Column '{col}' not found for original data.")
    orig_df['Total power'] = pd.to_numeric(orig_df['Total power'], errors='coerce')
    orig_df['Total Park Area (m²)'] = pd.to_numeric(orig_df['Total Park Area (m²)'], errors='coerce')
    orig_df = orig_df[(orig_df['Total power'] > 0) & (orig_df['Total Park Area (m²)'] > 0)]
    grouped_orig = orig_df.groupby('Country').agg({
        'Total power': 'sum',
        'Total Park Area (m²)': 'sum'
    }).reset_index()
    grouped_orig['Original Power Density (MW/km²)'] = (
            grouped_orig['Total power'] / (grouped_orig['Total Park Area (m²)'] / 1e6)
    )
    merged_df = pd.merge(cap_df, grouped_orig[['Country', 'Original Power Density (MW/km²)']],
                         on='Country', how='left')
    merged_df['Baseline Land Area (km²)'] = merged_df['Baseline_2050'] / merged_df['Original Power Density (MW/km²)']
    merged_df['Repowered Land Area (km²)'] = merged_df['Repowered Capacity (MW)'] / merged_df[
        'Original Power Density (MW/km²)']
    return merged_df[['Country', 'Baseline Land Area (km²)', 'Repowered Land Area (km²)']]


# Compute land area data for Approach 2 and Approach 3 using relative paths
land_data_2 = compute_land_area_data(files["Approach 2"])
land_data_3 = compute_land_area_data(files["Approach 3"])
land_compare = pd.merge(land_data_2, land_data_3, on='Country', how='outer', suffixes=('_2', '_3')).fillna(0)
land_compare['Total_Area_2'] = land_compare['Baseline Land Area (km²)_2'] + land_compare['Repowered Land Area (km²)_2']
land_compare = land_compare.sort_values('Total_Area_2', ascending=False)

countries = land_compare['Country']
x = np.arange(len(countries))
width = 0.35

plt.figure(figsize=(14, 8))
plt.bar(x - width / 2, land_compare['Baseline Land Area (km²)_2'], width,
        label='Approach 2 Baseline', color='slateblue')
plt.bar(x - width / 2, land_compare['Repowered Land Area (km²)_2'], width,
        bottom=land_compare['Baseline Land Area (km²)_2'], label='Approach 2 Repowered', color='mediumseagreen')
plt.bar(x + width / 2, land_compare['Baseline Land Area (km²)_3'], width,
        label='Approach 3 Baseline', color='darkorange')
plt.bar(x + width / 2, land_compare['Repowered Land Area (km²)_3'], width,
        bottom=land_compare['Baseline Land Area (km²)_3'], label='Approach 3 Repowered', color='tomato')
plt.xticks(x, countries, rotation=45, ha='right', fontsize=12)
plt.xlabel("Country", fontsize=14)
plt.ylabel("Land Area (km²)", fontsize=14)
plt.title("Required Land Area Comparison (Approaches 2 & 3)\nStacked: Baseline vs. Repowered", fontsize=16,
          fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
