import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        # Only skip if repowered value is missing; if zero, it will simply add zero.
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

    years = list(range(2000, 2051))
    baseline_line = [base_growth[y] for y in years]
    combined = [combined_line[y] for y in years]

    return years, baseline_line, combined

# Define file paths for each approach (update paths as needed)
files = {
    "Approach 1": r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_ninjawt.xlsx",
    "Approach 2": r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new_ninja_models.xlsx",
    "Approach 3": r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new_rounding_ninjawt.xlsx",
    "Approach 4": r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new_rounding_and_singlereplacement_ninja_wt.xlsx"
}

# =========================
# LINE CHART: CAPACITY 2000–2050 (All Approaches)
# =========================

years, baseline_mw, _ = compute_lines(files["Approach 1"])
combined_results = {}
for name, file_path in files.items():
    _, _, combined_mw = compute_lines(file_path)
    combined_results[name] = combined_mw

plt.figure(figsize=(12, 8))
baseline_gw = [v / 1000.0 for v in baseline_mw]
years_plot = years

plt.plot(years_plot, baseline_gw, color="black", linestyle="dashed",
         label="Baseline (Base Growth)")
colors = {'Approach 1': 'blue', 'Approach 2': 'green', 'Approach 3': 'red', 'Approach 4': 'purple'}
for name, combined_mw in combined_results.items():
    combined_gw = [v / 1000.0 for v in combined_mw]
    plt.plot(years_plot, combined_gw, color=colors[name], linestyle="solid",
             label=f"{name} Combined")

plt.title("Capacity 2000–2050 Comparison: Baseline vs. Repowered per Approach", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Capacity (GW)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# HORIZONTAL BAR CHART: REPOWERED POWER DENSITY (Approach 3 Only, MW/km²)
# =========================

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

# Only consider rows with positive repowered capacity for this analysis
repowered_df = repowered_df[repowered_df['Total_New_Capacity'] > 0]
grouped_rep = repowered_df.groupby('Country').agg({
    'Total_New_Capacity': 'sum',
    'New_Total_Park_Area (m²)': 'sum'
}).reset_index()

grouped_rep['Repowered Power Density (MW/km²)'] = (
    grouped_rep['Total_New_Capacity'] / (grouped_rep['New_Total_Park_Area (m²)'] / 1e6)
)

# Sort descending by repowered power density
grouped_rep = grouped_rep.sort_values('Repowered Power Density (MW/km²)', ascending=False)

plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.set_facecolor('white')
bars = plt.barh(
    grouped_rep['Country'],
    grouped_rep['Repowered Power Density (MW/km²)'],
    color=plt.cm.viridis(np.linspace(0, 1, len(grouped_rep)))
)
plt.title("Repowered Power Density per Country (Approach 3)", fontsize=16, fontweight='bold')
plt.xlabel("Repowered Power Density (MW/km²)", fontsize=14)
plt.ylabel("Country", fontsize=14)
plt.ticklabel_format(style='plain', axis='x')
plt.gca().invert_yaxis()  # Highest value on top
plt.tight_layout()
plt.show()

# =========================
# COMPARISON: ORIGINAL vs. REPOWERED POWER DENSITY (Approach 3)
# =========================

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
ax = plt.gca()
ax.set_facecolor('white')
plt.barh(y_pos - bar_height / 2, compare_df['Original Power Density (MW/km²)'], height=bar_height,
         label='Original', color='skyblue')
plt.barh(y_pos + bar_height / 2, compare_df['Repowered Power Density (MW/km²)'], height=bar_height,
         label='Repowered', color='salmon')
plt.yticks(y_pos, countries, fontsize=12)
plt.xlabel("Power Density (MW/km²)", fontsize=14)
plt.title("Comparison: Original vs. Repowered Power Density per Country (Approach 3)", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 2050 COUNTRY CAPACITIES: BASELINE vs. COMBINED SCENARIOS (Approach 3)
# =========================

def compute_country_capacities(file_path):
    """
    For Approach 3, compute the 2050 capacity for each country for:
      - Baseline (original) scenario
      - Combined (baseline plus repowering increments) scenario
    Returns a DataFrame with columns: Country, Baseline_2050, Combined_2050.
    """
    df = pd.read_excel(file_path)
    # Clean key columns
    for col in ['Country', 'Commissioning date', 'Decommissioning date', 'Total power',
                'Total_New_Capacity']:
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

        # Process each turbine for this country
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
            # Only skip if repowered capacity is missing.
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

country_cap_df = compute_country_capacities(files["Approach 3"])
country_cap_df = country_cap_df.sort_values('Combined_2050', ascending=False)

plt.figure(figsize=(12,8))
x_pos = np.arange(len(country_cap_df))
width = 0.35

# Convert capacities from MW to GW
baseline_GW = country_cap_df['Baseline_2050'] / 1000.0
combined_GW = country_cap_df['Combined_2050'] / 1000.0

plt.bar(x_pos - width/2, baseline_GW, width, label='Baseline', color='skyblue')
plt.bar(x_pos + width/2, combined_GW, width, label='Combined', color='salmon')

plt.xticks(x_pos, country_cap_df['Country'], fontsize=12, rotation=45, ha='right')
plt.ylabel("Capacity in 2050 (GW)", fontsize=14)
plt.title("Approach 3: 2050 Capacity by Country\nBaseline vs Combined Scenario", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
