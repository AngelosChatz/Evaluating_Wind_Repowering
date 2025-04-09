import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths for each approach (adjust paths as necessary)
file_approach_1 = r"D:\SET 2023\Thesis Delft\Model\excels\Approach 1.xlsx"
file_approach_2 = r"D:\SET 2023\Thesis Delft\Model\excels\Approach 2.xlsx"
file_approach_3 = r"D:\SET 2023\Thesis Delft\Model\excels\Approach 3.xlsx"
file_approach_4 = r"D:\SET 2023\Thesis Delft\Model\excels\Approach 4.xlsx"

approach_labels = {1: "Approach 1", 2: "Approach 2", 3: "Approach 3", 4: "Approach 4"}



# Data loading and cleaning function
def load_and_clean_data(file_path, rename_total_power=True):
    df = pd.read_excel(file_path)
    # Rename columns for consistency if needed
    rename_dict = {}
    if rename_total_power and 'Total power' in df.columns:
        rename_dict['Total power'] = 'Total Power (MW)'
    if 'Total_New_Capacity' in df.columns:
        rename_dict['Total_New_Capacity'] = 'Repowered Total Capacity (MW)'
    df = df.rename(columns=rename_dict)

    # Filter out rows with missing/invalid baseline capacity and commissioning date
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
    # Fill missing decommissioning dates (defaulting to +20 years from commissioning)
    df['Decommissioning date'] = df['Decommissioning date'].fillna(
        df['Commissioning date'] + pd.DateOffset(years=20)
    )
    # Ensure repowered capacity is numeric
    if 'Repowered Total Capacity (MW)' in df.columns:
        df['Repowered Total Capacity (MW)'] = pd.to_numeric(
            df['Repowered Total Capacity (MW)'], errors='coerce'
        ).fillna(0)
    else:
        df['Repowered Total Capacity (MW)'] = 0
    if 'Repowered Decommissioning date' not in df.columns:
        df['Repowered Decommissioning date'] = pd.NaT
    return df



# Scenario functions (capacity over time)
years = np.arange(2000, 2051)


def calculate_capacity(df, start_repowering_year=2023, repowered_lifetime=50):
    """
    Repowering scenario:
    When a wind park is decommissioned (from start_repowering_year onward) and has a positive
    repowered capacity, its original capacity is replaced by the repowered value.
    """
    capacity = pd.DataFrame({'Year': years})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    df_local = df.copy()
    for year in years:

        op = df_local[(df_local['Commissioning date'].dt.year <= year) &
                      (df_local['Decommissioning date'].dt.year >= year)]
        if year >= start_repowering_year:
            # Identify turbines decommissioning this year with repowered capacity
            rep = df_local[(df_local['Decommissioning date'].dt.year == year) &
                           (df_local['Repowered Total Capacity (MW)'] > 0)]


            df_local.loc[rep.index, 'Repowered Decommissioning date'] = pd.to_datetime(f'{year}') + pd.DateOffset(
                years=repowered_lifetime)
            old_cap = rep['Total Power (MW)'].sum() if 'Total Power (MW)' in df_local.columns else 0
            new_cap = rep['Repowered Total Capacity (MW)'].sum()
            net = op['Total Power (MW)'].sum() - old_cap + new_cap
        else:
            net = op['Total Power (MW)'].sum() if 'Total Power (MW)' in df_local.columns else 0

        # Add capacity from repowered turbines that are still operating
        rep_op = df_local[(df_local['Repowered Decommissioning date'].notna()) &
                          (df_local['Repowered Decommissioning date'].dt.year >= year) &
                          (year >= df_local['Decommissioning date'].dt.year)]
        rep_net = rep_op['Repowered Total Capacity (MW)'].sum()
        capacity.loc[year, 'Operating Capacity'] = net + rep_net
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']


def calculate_replacement_delayed_capacity(df, replacement_delay=1, replacement_lifetime=50,
                                           replacement_start_year=2022):
    """
    Replacement scenario with a one-year delay:
    - The original turbine operates from its commissioning year until its decommissioning year.
    - In its decommissioning year the turbine goes offline.
    - One year later, the turbine is replaced with the same capacity (operating for a fixed lifetime).
    """
    capacity_series = pd.Series(0.0, index=years)
    for idx, row in df.iterrows():
        cap = row['Total Power (MW)']
        orig_start = row['Commissioning date'].year
        decomm_year = row['Decommissioning date'].year

        # Add original capacity during its operating period
        for yr in years:
            if yr >= orig_start and yr <= decomm_year:
                capacity_series.loc[yr] += cap

        # If turbine decommissions at/after replacement_start_year, add its capacity back starting one year later
        if decomm_year >= replacement_start_year:
            repl_start = decomm_year + replacement_delay
            repl_end = repl_start + replacement_lifetime - 1
            for yr in years:
                if yr >= repl_start and yr <= repl_end:
                    capacity_series.loc[yr] += cap
    return capacity_series / 1000  # convert MW to GW


def calculate_no_replacement_capacity(df):
    """
    Baseline scenario (no replacement): computes operating capacity (GW) without any replacement.
    """
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
    """
    For each turbine with a successful upgrade (Repowered Total Capacity > Original Capacity),
    add its repowered capacity starting one year after its decommissioning (if decommissioning >= replacement_start_year).
    The capacity is added cumulatively (and does not compound) for each year.
    """
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
    return repowered_series / 1000  # convert MW to GW



# Load data for each approach
df_a1 = load_and_clean_data(file_approach_1)
df_a2 = load_and_clean_data(file_approach_2)
df_a3 = load_and_clean_data(file_approach_3)
df_a4 = load_and_clean_data(file_approach_4)


# Compute capacity curves for each approach
cap_a1 = calculate_capacity(df_a1)
cap_a2 = calculate_capacity(df_a2)
cap_a3 = calculate_capacity(df_a3)
cap_a4 = calculate_capacity(df_a4)

# Replacement-delayed capacity curves
rep_delayed_a1 = calculate_replacement_delayed_capacity(df_a1, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)
rep_delayed_a2 = calculate_replacement_delayed_capacity(df_a2, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)
rep_delayed_a3 = calculate_replacement_delayed_capacity(df_a3, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)
rep_delayed_a4 = calculate_replacement_delayed_capacity(df_a4, replacement_delay=1, replacement_lifetime=50,
                                                        replacement_start_year=2022)
# Aggregate replacement-delayed capacity (average across approaches)
rep_delayed_avg = (rep_delayed_a1 + rep_delayed_a2 + rep_delayed_a3 + rep_delayed_a4) / 4


# Plot 1: Combined Capacity Plot (Repowering vs. Replacement Strategy)

plt.figure(figsize=(12, 6))
years_int = years

# Define colors for each approach
colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}

# Plot repowering curves (solid lines)
plt.plot(years_int, cap_a1, linestyle='-', color=colors[1], label=f"{approach_labels[1]} Repowering")
plt.plot(years_int, cap_a2, linestyle='-', color=colors[2], label=f"{approach_labels[2]} Repowering")
plt.plot(years_int, cap_a3, linestyle='-', color=colors[3], label=f"{approach_labels[3]} Repowering")
plt.plot(years_int, cap_a4, linestyle='-', color=colors[4], label=f"{approach_labels[4]} Repowering")


plt.plot(years_int, rep_delayed_avg, linestyle='--', color='black', linewidth=2,
         label="Replacement Strategy (1-year Delay)")

plt.title("Total Operating Capacity – Repowering vs. Replacement Strategy (2000-2050)", fontsize=16, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Capacity (GW)")
plt.axvline(x=2022, color='grey', linestyle='--', linewidth=1, label="Model Start (2022)")
plt.legend(loc='best', fontsize=10, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot 2: Combine 4 Successful Upgrades by Country Plots in a 2x2 Subplot
approach_dfs = {1: df_a1, 2: df_a2, 3: df_a3, 4: df_a4}


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

# Loop through each approach, computing and plotting successful upgrade percentages by country
for idx, (approach, df) in enumerate(approach_dfs.items()):
    ax = axs[idx]
    df_success = df.copy()
    # Determine successful upgrades where repowered capacity exceeds original capacity
    df_success['Successful Upgrade'] = df_success['Repowered Total Capacity (MW)'] > df_success['Total Power (MW)']

    # Group data by Country
    country_stats = df_success.groupby('Country').agg(
        total_parks=('Successful Upgrade', 'count'),
        successful_upgrades=('Successful Upgrade', 'sum')
    )
    # Calculate percentage of successful upgrades
    country_stats['Success Percentage'] = (country_stats['successful_upgrades'] / country_stats['total_parks']) * 100
    country_stats = country_stats.sort_values(by='Success Percentage', ascending=False)

    # Create bar plot on corresponding axis
    country_stats['Success Percentage'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(f"{approach_labels[approach]}: % Successful Upgrades by Country")
    ax.set_xlabel("Country")
    ax.set_ylabel("Success (%)")
    ax.grid(axis='y')

plt.tight_layout()
plt.show()
