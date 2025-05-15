import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up base directory and subdirectories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"

# Define file paths for each approach
files = {
    "Approach 1": results_dir / "Approach_1.xlsx",
    "Approach 2": results_dir / "Approach_2.xlsx",
    "Approach 3": results_dir / "Approach_3.xlsx",
    "Approach 4": results_dir / "Approach_4.xlsx",
    "Approach 5": results_dir / "Approach_5.xlsx",
}

# Define colors for line-chart combined series
colors = {
    'Approach 1': 'blue',
    'Approach 2': 'green',
    'Approach 3': 'red',
    'Approach 4': 'purple',
    'Approach 5': 'orange'
}

# Function to compute baseline and combined (repowered) time series

def compute_lines(file_path):
    df = pd.read_excel(file_path)
    # Clean columns
    for col in ['Commissioning date', 'Decommissioning date', 'Total power', 'Total_New_Capacity']:
        if col in df.columns:
            df[col] = df[col].replace(['#ND', ''], np.nan)
    # Convert types
    df['Commissioning date'] = pd.to_datetime(df['Commissioning date'], errors='coerce')
    df['Decommissioning date'] = pd.to_datetime(df['Decommissioning date'], errors='coerce')
    df['Total power'] = pd.to_numeric(df['Total power'], errors='coerce')
    df['Total_New_Capacity'] = pd.to_numeric(df['Total_New_Capacity'], errors='coerce')
    # Baseline changes
    cap_changes = {}
    rep_changes = {}
    def add(dic, year, val): dic[year] = dic.get(year, 0) + val
    for _, row in df.iterrows():
        if pd.notna(row['Commissioning date']) and pd.notna(row['Total power']):
            start = row['Commissioning date'].year
            end = row['Decommissioning date'].year if pd.notna(row['Decommissioning date']) else start + 30
            add(cap_changes, start, row['Total power'])
            add(cap_changes, end, -row['Total power'])
        if pd.notna(row['Commissioning date']) and pd.notna(row['Total_New_Capacity']):
            rep_start = row['Decommissioning date'].year if pd.notna(row['Decommissioning date']) else row['Commissioning date'].year + 20
            cap = row['Total_New_Capacity']
            year = rep_start
            while year <= 2050:
                add(rep_changes, year, cap)
                add(rep_changes, year + 20, -cap)
                year += 20
    # Build baseline
    hist_baseline = {}
    running = 0.0
    for y in range(1980, 2023):
        running += cap_changes.get(y, 0)
        hist_baseline[y] = running
    # Linear trend beyond 2022
    if 2000 in hist_baseline and 2022 in hist_baseline:
        trend = (hist_baseline[2022] - hist_baseline[2000]) / 22
    else:
        trend = 0
    for y in range(2023, 2051):
        hist_baseline[y] = hist_baseline[2022] + trend * (y - 2022)
    # Build repowered series
    hist_repower = {}
    running = 0.0
    for y in range(1980, 2051):
        running += rep_changes.get(y, 0)
        hist_repower[y] = running
    # Extract series
    years = list(range(2000, 2051))
    baseline = [hist_baseline[y] for y in years]
    combined = [hist_baseline[y] + hist_repower.get(y, 0) for y in years]
    return years, baseline, combined

# SECTION 1: LINE CHART – CAPACITY 2000–2050
combined_results = {}
for name, path in files.items():
    years, base, comb = compute_lines(path)
    combined_results[name] = {'years': years, 'baseline': base, 'combined': comb}

plt.figure(figsize=(12, 8))
# Plot baseline dashed
plt.plot(combined_results['Approach 1']['years'],
         [v / 1000 for v in combined_results['Approach 1']['baseline']],
         color='black', linestyle='dashed', label='Baseline')
# Plot combined for each approach
for name in ['Approach 1', 'Approach 2', 'Approach 3', 'Approach 4', 'Approach 5']:
    plt.plot(combined_results[name]['years'],
             [v / 1000 for v in combined_results[name]['combined']],
             color=colors[name], label=f"{name} Combined")
plt.title("Capacity 2000–2050 Comparison: Baseline vs. Repowered per Approach", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Capacity (GW)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# SECTION 2: HORIZONTAL BAR CHART – REPOWERED POWER DENSITY (Approach 5)
rep5 = pd.read_excel(files['Approach 5']).replace({'#ND': np.nan, '': np.nan})
# Clean and convert
rep5['Commissioning date'] = pd.to_datetime(rep5.get('Commissioning date'), errors='coerce')
rep5['Decommissioning date'] = pd.to_datetime(rep5.get('Decommissioning date'), errors='coerce')
rep5['Total_New_Capacity'] = pd.to_numeric(rep5.get('Total_New_Capacity'), errors='coerce')
rep5['New_Total_Park_Area (m²)'] = pd.to_numeric(rep5.get('New_Total_Park_Area (m²)'), errors='coerce')
rep5 = rep5[rep5['Total_New_Capacity'] > 0]
# Aggregate and compute density
group5 = rep5.groupby('Country').agg({
    'Total_New_Capacity': 'sum',
    'New_Total_Park_Area (m²)': 'sum'
}).reset_index()
group5['Repowered Power Density (MW/km²)'] = group5['Total_New_Capacity'] / (group5['New_Total_Park_Area (m²)'] / 1e6)
group5 = group5.sort_values('Repowered Power Density (MW/km²)', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(group5['Country'], group5['Repowered Power Density (MW/km²)'],
         color=plt.cm.viridis(np.linspace(0, 1, len(group5))))
plt.title("Repowered Power Density per Country (Approach 5)", fontsize=16, fontweight='bold')
plt.xlabel("Repowered Power Density (MW/km²)", fontsize=14)
plt.ylabel("Country", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# SECTION 3: COMPARISON – ORIGINAL vs. REPOWERED POWER DENSITY (Approach 5)
orig5 = pd.read_excel(files['Approach 5']).replace({'#ND': np.nan, '': np.nan})
# Clean and convert
orig5['Total power'] = pd.to_numeric(orig5.get('Total power'), errors='coerce')
orig5['Total Park Area (m²)'] = pd.to_numeric(orig5.get('Total Park Area (m²)'), errors='coerce')
orig5 = orig5[(orig5['Total power'] > 0) & (orig5['Total Park Area (m²)'] > 0)]
# Aggregate and compute densities
group_o5 = orig5.groupby('Country').agg({
    'Total power': 'sum',
    'Total Park Area (m²)': 'sum'
}).reset_index()
group_o5['Original Power Density (MW/km²)'] = (
    group_o5['Total power'] / (group_o5['Total Park Area (m²)'] / 1e6)
)
# Merge and plot
compare5 = pd.merge(
    group_o5[['Country', 'Original Power Density (MW/km²)']],
    group5[['Country', 'Repowered Power Density (MW/km²)']],
    on='Country', how='outer'
).fillna(0)
compare5 = compare5.sort_values('Repowered Power Density (MW/km²)', ascending=False)

countries = compare5['Country']
y_pos = np.arange(len(countries))

plt.figure(figsize=(12, 8))
plt.barh(y_pos - 0.2, compare5['Original Power Density (MW/km²)'], height=0.4, label='Original')
plt.barh(y_pos + 0.2, compare5['Repowered Power Density (MW/km²)'], height=0.4, label='Repowered')
plt.yticks(y_pos, countries)
plt.xlabel("Power Density (MW/km²)", fontsize=14)
plt.title("Comparison: Original vs. Repowered Power Density per Country (Approach 5)", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# SECTION 4: 2050 COUNTRY LAND AREA COMPARISON – STACKED BARS (Approaches 2, 3 & 5)

def compute_country_capacities(file_path):
    df = pd.read_excel(file_path).replace({'#ND': np.nan, '': np.nan})
    df['Commissioning date'] = pd.to_datetime(df.get('Commissioning date'), errors='coerce')
    df['Decommissioning date'] = pd.to_datetime(df.get('Decommissioning date'), errors='coerce')
    df['Total power'] = pd.to_numeric(df.get('Total power'), errors='coerce')
    df['Total_New_Capacity'] = pd.to_numeric(df.get('Total_New_Capacity'), errors='coerce')
    countries = df['Country'].dropna().unique()
    records = []
    for country in countries:
        sub = df[df['Country'] == country]
        base_changes = {};
        rep_changes = {};
        def add(dic, year, val): dic[year] = dic.get(year, 0) + val
        for _, r in sub.iterrows():
            if pd.notna(r['Commissioning date']) and pd.notna(r['Total power']):
                s = r['Commissioning date'].year
                e = r['Decommissioning date'].year if pd.notna(r['Decommissioning date']) else s + 30
                add(base_changes, s, r['Total power']); add(base_changes, e, -r['Total power'])
            if pd.notna(r['Commissioning date']) and pd.notna(r['Total_New_Capacity']):
                s2 = r['Decommissioning date'].year if pd.notna(r['Decommissioning date']) else r['Commissioning date'].year + 20
                cap2 = r['Total_New_Capacity']; year = s2
                while year <= 2050:
                    add(rep_changes, year, cap2); add(rep_changes, year + 20, -cap2); year += 20
        # baseline cumulative
        hist = {}; run = 0.0
        for y in range(1980, 2023): run += base_changes.get(y, 0); hist[y] = run
        growth = ((hist[2022] - hist[2000]) / 22) if 2000 in hist and 2022 in hist else 0
        for y in range(2023, 2051): hist[y] = hist[2022] + growth * (y - 2022)
        base_2050 = hist[2050]
        # repowered cumulative
        run = 0.0; rep_hist = {}
        for y in range(1980, 2051): run += rep_changes.get(y, 0); rep_hist[y] = run
        combined_2050 = base_2050 + rep_hist[2050]
        records.append({'Country': country, 'Baseline_2050': base_2050, 'Combined_2050': combined_2050})
    return pd.DataFrame(records)


# Loop over all approaches and plot one figure each
for i in range(1, 6):
    file_i = results_dir / f"Approach_{i}.xlsx"
    caps = compute_country_capacities(file_i)

    # Sort by combined capacity descending
    caps = caps.sort_values('Combined_2050', ascending=False).reset_index(drop=True)

    countries = caps['Country']
    x = np.arange(len(countries))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, caps['Baseline_2050'], width, label='Baseline')
    plt.bar(x + width / 2, caps['Combined_2050'], width, label='Combined')
    plt.xticks(x, countries, rotation=45, ha='right', fontsize=12)
    plt.ylabel('Capacity (MW)', fontsize=14)
    plt.title(f'Approach {i}: Baseline vs Combined Capacity per Country (2050)',
              fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()
# Compute land area per approach

def compute_land_area(approach):
    caps = compute_country_capacities(files[f'Approach {approach}'])
    caps['Repowered Capacity (MW)'] = caps['Combined_2050'] - caps['Baseline_2050']
    orig = pd.read_excel(files[f'Approach {approach}']).replace({'#ND': np.nan, '': np.nan})
    orig['Total power'] = pd.to_numeric(orig.get('Total power'), errors='coerce')
    orig['Total Park Area (m²)'] = pd.to_numeric(orig.get('Total Park Area (m²)'), errors='coerce')
    orig = orig[(orig['Total power'] > 0) & (orig['Total Park Area (m²)'] > 0)]
    grp = orig.groupby('Country').agg({'Total power':'sum','Total Park Area (m²)':'sum'}).reset_index()
    grp['Orig Density'] = grp['Total power']/(grp['Total Park Area (m²)']/1e6)
    merged = pd.merge(caps, grp[['Country','Orig Density']], on='Country', how='left')
    merged['Baseline Area'] = merged['Baseline_2050']/merged['Orig Density']
    merged['Repowered Area'] = merged['Repowered Capacity (MW)']/merged['Orig Density']
    return merged[['Country','Baseline Area','Repowered Area']]

land2 = compute_land_area(2)
land3 = compute_land_area(3)
land5 = compute_land_area(5)

# Merge data for approaches 2,3,5
cmp = land2.merge(land3, on='Country', how='outer', suffixes=('_2','_3')) \
          .merge(land5.rename(columns={'Baseline Area':'Baseline_5','Repowered Area':'Repowered_5'}), on='Country', how='outer')
cmp.fillna(0, inplace=True)
cmp['Total_2'] = cmp['Baseline Area_2'] + cmp['Repowered Area_2']
cmp.sort_values('Total_2', ascending=False, inplace=True)

# Plot: uniform baseline color, distinct repowered per approach
countries = cmp['Country']
idx = np.arange(len(countries))
width = 0.2
plt.figure(figsize=(14, 8))

# Baseline bars in light gray
plt.bar(idx - width, cmp['Baseline Area_2'], width, color='lightgray', label='Baseline')
plt.bar(idx, cmp['Baseline Area_3'], width, color='lightgray')
plt.bar(idx + width, cmp['Baseline_5'], width, color='lightgray')

# Repowered bars
plt.bar(idx - width, cmp['Repowered Area_2'], width,
        bottom=cmp['Baseline Area_2'], color='green', label='Approach 2 Repowered')
plt.bar(idx, cmp['Repowered Area_3'], width,
        bottom=cmp['Baseline Area_3'], color='red', label='Approach 3 Repowered')
plt.bar(idx + width, cmp['Repowered_5'], width,
        bottom=cmp['Baseline_5'], color='orange', label='Approach 5 Repowered')

plt.xticks(idx, countries, rotation=45, ha='right', fontsize=12)
plt.xlabel("Country", fontsize=14)
plt.ylabel("Land Area (km²)", fontsize=14)
plt.title("Required Land Area Comparison (Approaches 2, 3 & 5)\nStacked: Baseline vs. Repowered", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
