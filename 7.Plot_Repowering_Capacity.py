import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path_stage_3 = r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new_rounding_and_singlereplacement_ninja_wt.xlsx"
file_path_stage_2 = r"D:\SET 2023\Thesis Delft\Model\Repowering_Calculation_Stage_2_int_new_rounding_and_singlereplacement_ninja_wt.xlsx"

def load_and_clean_data(file_path, rename_total_power=True):
    df = pd.read_excel(file_path)

    rename_dict = {}
    if rename_total_power and 'Total power' in df.columns:
        rename_dict['Total power'] = 'Total Power (MW)'
    if 'Total_New_Capacity' in df.columns:
        rename_dict['Total_New_Capacity'] = 'Repowered Total Capacity (MW)'
    df = df.rename(columns=rename_dict)

    # Filter out rows with missing or invalid baseline capacity and commissioning date
    if 'Total Power (MW)' in df.columns:
        df = df[df['Total Power (MW)'].notna()]
        df = df[df['Total Power (MW)'] != '#ND']
    df = df[df['Commissioning date'].notna()]
    df = df[df['Commissioning date'] != '#ND']

    # Normalize date columns
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

    # Fill missing decommissioning dates with +20 years from commissioning
    df['Decommissioning date'] = df['Decommissioning date'].fillna(
        df['Commissioning date'] + pd.DateOffset(years=20)
    )

    # Ensure 'Repowered Total Capacity (MW)' is numeric
    if 'Repowered Total Capacity (MW)' in df.columns:
        df['Repowered Total Capacity (MW)'] = pd.to_numeric(
            df['Repowered Total Capacity (MW)'], errors='coerce'
        ).fillna(0)
    else:
        df['Repowered Total Capacity (MW)'] = 0

    # Create a column for repowered decommissioning date
    if 'Repowered Decommissioning date' not in df.columns:
        df['Repowered Decommissioning date'] = pd.NaT

    return df

df_stage_3 = load_and_clean_data(file_path_stage_3, rename_total_power=True)
df_stage_2 = load_and_clean_data(file_path_stage_2, rename_total_power=True)


years = pd.date_range(start='2000', end='2050', freq='YE')

def calculate_no_replacement_capacity(df):
    capacity = pd.DataFrame({'Year': years.year})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating_parks = df[
            (df['Commissioning date'].dt.year <= year) &
            (df['Decommissioning date'].dt.year >= year)
            ]
        net_capacity = operating_parks['Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0
        capacity.loc[year, 'Operating Capacity'] = net_capacity
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']

def calculate_replacement_same_capacity(df):
    capacity = pd.DataFrame({'Year': years.year})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating_parks = df[
            (df['Commissioning date'].dt.year <= year) &
            (df['Decommissioning date'].dt.year >= year)
            ]
        if year >= 2023:
            decommissioned_parks = df[df['Decommissioning date'].dt.year == year]
            replacement_capacity = decommissioned_parks[
                'Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0
            # Extend the operational life by 20 years
            for idx in decommissioned_parks.index:
                df.loc[idx, 'Commissioning date'] = pd.to_datetime(f'{year}')
                df.loc[idx, 'Decommissioning date'] = pd.to_datetime(f'{year}') + pd.DateOffset(years=20)
        else:
            replacement_capacity = 0
        net_capacity = operating_parks[
                           'Total Power (MW)'].sum() + replacement_capacity if 'Total Power (MW)' in df.columns else 0
        capacity.loc[year, 'Operating Capacity'] = net_capacity
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']


def calculate_capacity(df, start_repowering_year, repowered_lifetime=50):
    capacity = pd.DataFrame({'Year': years.year})
    capacity.set_index('Year', inplace=True)
    capacity['Operating Capacity'] = 0.0
    for year in capacity.index:
        operating_parks = df[
            (df['Commissioning date'].dt.year <= year) &
            (df['Decommissioning date'].dt.year >= year)
            ]
        if year >= start_repowering_year:
            repowered_parks = df[
                (df['Decommissioning date'].dt.year == year) &
                (df['Repowered Total Capacity (MW)'] > 0)
                ]
            df.loc[repowered_parks.index, 'Repowered Decommissioning date'] = pd.to_datetime(f'{year}') + pd.DateOffset(
                years=repowered_lifetime)
            old_capacity = repowered_parks['Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0
            new_capacity = repowered_parks['Repowered Total Capacity (MW)'].sum()
            net_capacity = operating_parks[
                               'Total Power (MW)'].sum() - old_capacity + new_capacity if 'Total Power (MW)' in df.columns else new_capacity
        else:
            net_capacity = operating_parks['Total Power (MW)'].sum() if 'Total Power (MW)' in df.columns else 0
        repowered_operating_parks = df[
            (df['Repowered Decommissioning date'].notna()) &
            (df['Repowered Decommissioning date'].dt.year >= year) &
            (year >= df['Decommissioning date'].dt.year)
            ]
        repowered_net_capacity = repowered_operating_parks['Repowered Total Capacity (MW)'].sum()
        capacity.loc[year, 'Operating Capacity'] = net_capacity + repowered_net_capacity
    capacity['Operating Capacity (GW)'] = capacity['Operating Capacity'] / 1000
    return capacity['Operating Capacity (GW)']



capacity_repower_50yr_stage_3 = calculate_capacity(df_stage_3.copy(), start_repowering_year=2023, repowered_lifetime=50)
capacity_no_replacement_stage_3 = calculate_no_replacement_capacity(df_stage_3.copy())
capacity_replacement_same_stage_3 = calculate_replacement_same_capacity(df_stage_3.copy())
capacity_repower_50yr_stage_2 = calculate_capacity(df_stage_2.copy(), start_repowering_year=2023, repowered_lifetime=50)


# 6. Plot line chart comparing Stage 3 vs Stage 2 scenarios

plt.figure(figsize=(12, 6))
plt.plot(years.year, capacity_repower_50yr_stage_3, label='Stage 3: Repowering Double', linestyle='--', color='blue')
plt.plot(years.year, capacity_no_replacement_stage_3, label='Stage 3: No Replacement', linestyle='--', color='red')
plt.plot(years.year, capacity_replacement_same_stage_3, label='Stage 3: Replacement w/ Same Cap.', linestyle='-.',
         color='green')
plt.plot(years.year, capacity_repower_50yr_stage_2, label='Stage 2: Repowering Double', linestyle='-', color='orange')
plt.axvline(x=2022, color='black', linestyle='--', linewidth=1, label='Modeling Starts (2022)')
plt.title('Comparison of Stage 3 vs Stage 2 Repowering (2000-2050)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Operating Capacity (GW)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# 7. Bar chart for Stage 3 by Country

capacity_by_country = df_stage_3.groupby('Country').agg({
    'Total Power (MW)': 'sum',
    'Repowered Total Capacity (MW)': 'sum'
})
capacity_by_country['Total Power (GW)'] = capacity_by_country['Total Power (MW)'] / 1000
capacity_by_country['Repowered Total Capacity (GW)'] = capacity_by_country['Repowered Total Capacity (MW)'] / 1000
capacity_by_country = capacity_by_country.sort_values('Total Power (GW)', ascending=False)
fig, ax = plt.subplots(figsize=(14, 6))
bar_width = 0.4
bar_positions = range(len(capacity_by_country))
ax.bar(bar_positions, capacity_by_country['Total Power (GW)'], width=bar_width, color='skyblue', edgecolor='black',
       label='Total Capacity (2022)')
ax.bar([p + bar_width for p in bar_positions], capacity_by_country['Repowered Total Capacity (GW)'], width=bar_width,
       color='lightgreen', edgecolor='black', label='Repowered Capacity (2050)')
ax.set_xticks([p + bar_width / 2 for p in bar_positions])
ax.set_xticklabels(capacity_by_country.index, rotation=45, ha='right')
ax.set_title('Stage 3 Capacity by Country: 2022 vs 2050 after Repowering', fontsize=14, fontweight='bold')
ax.set_ylabel('Capacity (GW)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()


# 8. Plot Annual Growth Rates for Decommissioning and Repowering (2022-2042)
years_range = np.arange(2022, 2043)


decom_capacity_list = []  #(GW)
repower_capacity_list = []  #(GW)

for year in years_range:
    # Decommissioning: sum baseline capacities (in MW) for parks decommissioning in that year
    decom_capacity = df_stage_3[df_stage_3['Decommissioning date'].dt.year == year]['Total Power (MW)'].sum()
    decom_capacity_list.append(decom_capacity / 1000)  # convert MW to GW

    # Repowering: sum repowered capacities (in MW) for parks decommissioning in that year
    repower_capacity = df_stage_3[
        (df_stage_3['Decommissioning date'].dt.year == year) &
        (df_stage_3['Repowered Total Capacity (MW)'] > 0)
        ]['Repowered Total Capacity (MW)'].sum()
    repower_capacity_list.append(repower_capacity / 1000)  # convert MW to GW

# Convert lists to numpy arrays
decom_capacity_array = np.array(decom_capacity_list)
repower_capacity_array = np.array(repower_capacity_list)

# Calculate Annual Growth Percentages (compared to the previous year)
# Formula: ((current - previous) / previous) * 100. The first year (2022) has no previous year.
decom_growth = [np.nan]  # Start with NaN for 2022
repower_growth = [np.nan]

for i in range(1, len(years_range)):
    prev_decom = decom_capacity_array[i - 1]
    curr_decom = decom_capacity_array[i]
    if prev_decom != 0:
        decom_growth.append(((curr_decom - prev_decom) / prev_decom) * 100)
    else:
        decom_growth.append(np.nan)

    prev_repower = repower_capacity_array[i - 1]
    curr_repower = repower_capacity_array[i]
    if prev_repower != 0:
        repower_growth.append(((curr_repower - prev_repower) / prev_repower) * 100)
    else:
        repower_growth.append(np.nan)

# Plot the Annual Growth Percentages
plt.figure(figsize=(12, 6))
plt.plot(years_range, decom_growth, label="Decommissioning Growth Rate (%)", linestyle='-', color='red')
plt.plot(years_range, repower_growth, label="Repowering Growth Rate (%)", linestyle='--', color='blue')
plt.title("Annual Growth Rates for Decommissioning vs. Repowering (2022-2042)", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Annual Growth Rate (%)", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Summary Table by Country

required_cols = ['Country', 'Number of turbines', 'Total Power (MW)', 'Repowered Total Capacity (MW)']
if all(col in df_stage_3.columns for col in required_cols):
    # Group by "Country" and compute the metrics
    df_summary = df_stage_3.groupby('Country').apply(lambda g: pd.Series({
        'Total Wind Parks': len(g),
        'Wind Parks with 1 Turbine': (g['Number of turbines'] == 1).sum(),
        'Successful Power Increases': (g['Repowered Total Capacity (MW)'] > g['Total Power (MW)']).sum()
    }))


    df_summary['Percentage of Successful Power Increases'] = (
            df_summary['Successful Power Increases'] / df_summary['Total Wind Parks'] * 100
    )


    df_summary['Percentage of Successful Power Increases'] = (
            df_summary['Percentage of Successful Power Increases']
            .round(2)
            .astype(str) + '%'
    )

    print("Summary Table by Country (Stage 3):")
    print(df_summary)


    df_summary_reset = df_summary.reset_index()


    output_file = "Stage3_Summary_By_Country.xlsx"
    df_summary_reset.to_excel(output_file, index=False, sheet_name="Summary")
