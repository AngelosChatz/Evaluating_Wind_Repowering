import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import re

# Define base directory and subdirectories
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"

# File paths for each approach (Excel files stored in the data folder)
file_approach_1 = results_dir / "Approach_1.xlsx"
file_approach_2 = results_dir / "Approach_2.xlsx"
file_approach_3 = results_dir / "Approach_3.xlsx"
file_approach_4 = results_dir / "Approach_4.xlsx"

# Define approach labels for plotting
approach_labels = {1: "Approach 1", 2: "Approach 2", 3: "Approach 3", 4: "Approach 4"}

# Data loading and cleaning function
def load_and_clean_data(file_path, rename_total_power=True):
    df = pd.read_excel(file_path)
    rename_dict = {}
    if rename_total_power and 'Total power' in df.columns:
        rename_dict['Total power'] = 'Total Power (MW)'
    if 'Total_New_Capacity' in df.columns:
        rename_dict['Total_New_Capacity'] = 'Repowered Total Capacity (MW)'
    df = df.rename(columns=rename_dict)

    # Filter out rows with missing/invalid total power and commissioning date
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

# Scenario functions (capacity over time)
def calculate_capacity(df, start_repowering_year=2023, repowered_lifetime=50):
    capacity = pd.DataFrame({'Year': years})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    df_local = df.copy()
    for year in years:
        op = df_local[(df_local['Commissioning date'].dt.year <= year) &
                      (df_local['Decommissioning date'].dt.year >= year)]
        if year >= start_repowering_year:
            rep = df_local[(df_local['Decommissioning date'].dt.year == year) &
                           (df_local['Repowered Total Capacity (MW)'] > 0)]
            df_local.loc[rep.index, 'Repowered Decommissioning date'] = pd.to_datetime(f'{year}') + pd.DateOffset(
                years=repowered_lifetime)
            old_cap = rep['Total Power (MW)'].sum() if 'Total Power (MW)' in df_local.columns else 0
            new_cap = rep['Repowered Total Capacity (MW)'].sum()
            net = op['Total Power (MW)'].sum() - old_cap + new_cap
        else:
            net = op['Total Power (MW)'].sum() if 'Total Power (MW)' in df_local.columns else 0
        rep_op = df_local[(df_local['Repowered Decommissioning date'].notna()) &
                          (df_local['Repowered Decommissioning date'].dt.year >= year) &
                          (year >= df_local['Decommissioning date'].dt.year)]
        rep_net = rep_op['Repowered Total Capacity (MW)'].sum()
        capacity.loc[year, 'Operating Capacity'] = net + rep_net
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']

def calculate_replacement_delayed_capacity(df, replacement_delay=1, replacement_lifetime=50,
                                           replacement_start_year=2022):
    capacity_series = pd.Series(0.0, index=years)
    for idx, row in df.iterrows():
        cap = row['Total Power (MW)']
        orig_start = row['Commissioning date'].year
        decomm_year = row['Decommissioning date'].year
        for yr in years:
            if yr >= orig_start and yr <= decomm_year:
                capacity_series.loc[yr] += cap
        if decomm_year >= replacement_start_year:
            repl_start = decomm_year + replacement_delay
            repl_end = repl_start + replacement_lifetime - 1
            for yr in years:
                if yr >= repl_start and yr <= repl_end:
                    capacity_series.loc[yr] += cap
    return capacity_series / 1000

def calculate_no_replacement_capacity(df):
    capacity = pd.DataFrame({'Year': years})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    for year in years:
        op = df[(df['Commissioning date'].dt.year <= year) &
                (df['Decommissioning date'].dt.year >= year)]
        cap = op['Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0
        capacity.loc[year, 'Operating Capacity'] = cap
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']

def calculate_repowered_increment(df, replacement_delay=1, replacement_start_year=2022):
    years_proj = np.arange(2022, 2051)
    repowered_series = pd.Series(0.0, index=years_proj)
    for idx, row in df.iterrows():
        if row['Repowered Total Capacity (MW)'] > row['Total Power (MW)']:
            decomm_year = row['Decommissioning date'].year
            if decomm_year >= replacement_start_year:
                rep_year = decomm_year + replacement_delay
                for yr in years_proj:
                    if yr >= rep_year:
                        repowered_series.loc[yr] += row['Repowered Total Capacity (MW)']
    return repowered_series / 1000

# Load data for each approach using the load_and_clean_data function
df_a1 = load_and_clean_data(file_approach_1)
df_a2 = load_and_clean_data(file_approach_2)
df_a3 = load_and_clean_data(file_approach_3)
df_a4 = load_and_clean_data(file_approach_4)

# Compute capacity curves for each approach (if needed for further analysis)
cap_a1 = calculate_capacity(df_a1)
cap_a2 = calculate_capacity(df_a2)
cap_a3 = calculate_capacity(df_a3)
cap_a4 = calculate_capacity(df_a4)

# Compute replacement-delayed capacity curves (if needed for further analysis)
rep_delayed_a1 = calculate_replacement_delayed_capacity(df_a1, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)
rep_delayed_a2 = calculate_replacement_delayed_capacity(df_a2, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)
rep_delayed_a3 = calculate_replacement_delayed_capacity(df_a3, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)
rep_delayed_a4 = calculate_replacement_delayed_capacity(df_a4, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)

# Dictionary for easy access to DataFrames by approach for plotting
approach_dfs = {1: df_a1, 2: df_a2, 3: df_a3, 4: df_a4}

# ----- Produce Only the 2×2 Subplot: Successful Upgrade Percentages by Country -----
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()
for idx, (approach, df) in enumerate(approach_dfs.items()):
    ax = axs[idx]
    df_success = df.copy()
    df_success['Successful Upgrade'] = df_success['Repowered Total Capacity (MW)'] > df_success['Total Power (MW)']
    country_stats = df_success.groupby('Country').agg(
        total_parks=('Successful Upgrade', 'count'),
        successful_upgrades=('Successful Upgrade', 'sum')
    )
    country_stats['Success Percentage'] = (country_stats['successful_upgrades'] / country_stats['total_parks']) * 100
    country_stats = country_stats.sort_values(by='Success Percentage', ascending=False)
    country_stats['Success Percentage'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(f"{approach_labels[approach]}: % Successful Upgrades by Country")
    ax.set_xlabel("Country")
    ax.set_ylabel("Success (%)")
    ax.grid(axis='y')
plt.tight_layout()
plt.show()

# Bar Plot for Mean Repowering Upgrade
mean_improvements = []
std_improvements = []
labels = []
for approach, df in approach_dfs.items():
    df_success = df.copy()
    df_success = df_success[df_success['Repowered Total Capacity (MW)'] > df_success['Total Power (MW)']]
    improvements = ((df_success['Repowered Total Capacity (MW)'] - df_success['Total Power (MW)'])
                    / df_success['Total Power (MW)']) * 100
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
