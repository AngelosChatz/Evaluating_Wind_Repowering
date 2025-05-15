import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define base directory and subdirectories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"

# File paths for each approach (Excel files stored in the results folder)
file_approach_1 = results_dir / "Approach_1.xlsx"
file_approach_2 = results_dir / "Approach_2.xlsx"
file_approach_3 = results_dir / "Approach_3.xlsx"
file_approach_4 = results_dir / "Approach_4.xlsx"
file_approach_5 = results_dir / "Approach_5.xlsx"

# Define approach labels (acronyms) for plotting
approach_labels = {
    1: "ED",   # Energy Density
    2: "CM",   # Capacity Maximization
    3: "RU",   # Rounding Up
    4: "STF",  # Single Turbine Flex
    5: "NLH"   # No-Loss Hybrid
}

# Data loading and cleaning function
def load_and_clean_data(file_path, rename_total_power=True):
    df = pd.read_excel(file_path)
    rename_dict = {}
    if rename_total_power and 'Total power' in df.columns:
        rename_dict['Total power'] = 'Total Power (MW)'
    if 'Total_New_Capacity' in df.columns:
        rename_dict['Total_New_Capacity'] = 'Repowered Total Capacity (MW)'
    df = df.rename(columns=rename_dict)

    # Filter out rows with missing/invalid data
    if 'Total Power (MW)' in df.columns:
        df = df[df['Total Power (MW)'].notna()]
        df = df[df['Total Power (MW)'] != '#ND']
    df = df[df['Commissioning date'].notna()]
    df = df[df['Commissioning date'] != '#ND']

    # Normalize dates
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
            df['Repowered Total Capacity (MW)'], errors='coerce'
        ).fillna(0)
    else:
        df['Repowered Total Capacity (MW)'] = 0
    if 'Repowered Decommissioning date' not in df.columns:
        df['Repowered Decommissioning date'] = pd.NaT
    return df

# Global time horizon
years = np.arange(2000, 2051)

# Scenario function: capacity over time with repowering

def calculate_capacity(df, start_repowering_year=2023, repowered_lifetime=50):
    capacity = pd.DataFrame({'Year': years}).set_index('Year')
    capacity['Operating Capacity'] = 0.0
    df_local = df.copy()
    for year in years:
        operating = df_local[(df_local['Commissioning date'].dt.year <= year) &
                             (df_local['Decommissioning date'].dt.year >= year)]
        if year >= start_repowering_year:
            rep = df_local[(df_local['Decommissioning date'].dt.year == year) &
                           (df_local['Repowered Total Capacity (MW)'] > 0)]
            df_local.loc[rep.index, 'Repowered Decommissioning date'] = (
                pd.to_datetime(f'{year}') + pd.DateOffset(years=repowered_lifetime)
            )
            old_cap = rep['Total Power (MW)'].sum()
            new_cap = rep['Repowered Total Capacity (MW)'].sum()
            net = operating['Total Power (MW)'].sum() - old_cap + new_cap
        else:
            net = operating['Total Power (MW)'].sum() or 0
        rep_op = df_local[(df_local['Repowered Decommissioning date'].notna()) &
                          (df_local['Repowered Decommissioning date'].dt.year >= year) &
                          (year >= df_local['Decommissioning date'].dt.year)]
        rep_net = rep_op['Repowered Total Capacity (MW)'].sum()
        capacity.loc[year, 'Operating Capacity'] = net + rep_net
    return capacity['Operating Capacity'] / 1000  # GW

# Load and prepare data for each approach

df_a1 = load_and_clean_data(file_approach_1)
df_a2 = load_and_clean_data(file_approach_2)
df_a3 = load_and_clean_data(file_approach_3)
df_a4 = load_and_clean_data(file_approach_4)
df_a5 = load_and_clean_data(file_approach_5)

# Compute capacity curves
cap_a1 = calculate_capacity(df_a1)
cap_a2 = calculate_capacity(df_a2)
cap_a3 = calculate_capacity(df_a3)
cap_a4 = calculate_capacity(df_a4)
cap_a5 = calculate_capacity(df_a5)

# Dictionary for easy access to DataFrames by approach
approach_dfs = {
    1: df_a1,
    2: df_a2,
    3: df_a3,
    4: df_a4,
    5: df_a5
}

# ----- 2×3 Subplot: % Successful Upgrades by Country -----
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs_flat = axs.flatten()
for idx, (approach, df) in enumerate(approach_dfs.items()):
    ax = axs_flat[idx]
    df_success = df.copy()
    df_success['Successful Upgrade'] = (
        df_success['Repowered Total Capacity (MW)'] > df_success['Total Power (MW)']
    )
    country_stats = df_success.groupby('Country').agg(
        total_parks=('Successful Upgrade', 'count'),
        successful_upgrades=('Successful Upgrade', 'sum')
    )
    country_stats['Success Percentage'] = (
        country_stats['successful_upgrades'] / country_stats['total_parks'] * 100
    )
    country_stats = country_stats.sort_values('Success Percentage', ascending=False)
    country_stats['Success Percentage'].plot(
        kind='bar', ax=ax, color='skyblue'
    )
    ax.set_title(f"{approach_labels[approach]}: % Successful Upgrades by Country")
    ax.set_xlabel("Country")
    ax.set_ylabel("Success (%)")
    ax.grid(axis='y')
# Remove extra axis
for ax in axs_flat[len(approach_dfs):]:
    fig.delaxes(ax)
plt.tight_layout()
plt.show()

# ----- Bar Plot: Average Repowering Improvement by Approach -----
mean_improvements = []
std_improvements = []
labels = []
for approach, df in approach_dfs.items():
    df_success = df[df['Repowered Total Capacity (MW)'] > df['Total Power (MW)']]
    improvements = (
        (df_success['Repowered Total Capacity (MW)'] - df_success['Total Power (MW)']) /
        df_success['Total Power (MW)'] * 100
    )
    mean_improvements.append(improvements.mean())
    std_improvements.append(improvements.std())
    labels.append(approach_labels[approach])

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(labels, mean_improvements, yerr=std_improvements, capsize=5, color='skyblue')
ax.set_title("Average Repowering Upgrade Improvement by Approach")
ax.set_xlabel("Approach")
ax.set_ylabel("Mean Improvement (%)")
ax.grid(axis='y')
plt.tight_layout()
plt.show()
