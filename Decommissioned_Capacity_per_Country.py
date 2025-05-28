import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ----- 1. Load & Clean Data -----
file_path = r'D:\SET 2023\Thesis Delft\Model\Windfarms_World_20230530_with_IEC_Elevation_v2_area_classifications.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

data = data.iloc[:, [3, 18, 22, 24, 13, 23]]
data.columns = [
    'Country', 'Total Power (kW)',
    'Commissioning Date', 'Decommissioning Date',
    'Onshore/Offshore', 'Status'
]
data = data[data['Status'] == 'Production']
data = data[~data['Commissioning Date'].astype(str).str.contains('#ND')]
data['Total Power (kW)'] = pd.to_numeric(data['Total Power (kW)'], errors='coerce')
data = data.dropna(subset=['Total Power (kW)'])

def extract_year(val):
    try:
        if isinstance(val, pd.Timestamp):
            return val.year
        s = str(val).strip()
        if s.replace('.', '', 1).isdigit():
            return int(float(s))
        if '/' in s:
            return datetime.strptime(s, '%Y/%m').year
        return int(s)
    except:
        return None

data['Commissioning Year'] = data['Commissioning Date'].apply(extract_year)
data = data.dropna(subset=['Commissioning Year'])
data['Commissioning Year'] = data['Commissioning Year'].astype(int)

# Compute decommissioning year and convert to GW
data['Decommissioning Year'] = data['Commissioning Year'] + 20
data['Total Power (GW)'] = data['Total Power (kW)'] / 1e6 * 1000

# Annual & cumulative
annual_decom_data = (
    data
    .groupby(['Country', 'Decommissioning Year'])['Total Power (GW)']
    .sum()
    .unstack(fill_value=0)
)
cumulative_decom_data = annual_decom_data.cumsum(axis=1)

# Years & sensitivity
lifespans = [15, 18, 20, 22, 25]
start_year = data['Commissioning Year'].min()
end_year = data['Commissioning Year'].max() + max(lifespans)
years = np.arange(start_year, end_year+1)

operational_capacity = {}
for L in lifespans:
    caps = []
    for yr in years:
        cap = data.loc[
            (data['Commissioning Year'] <= yr) &
            (yr < data['Commissioning Year'] + L),
            'Total Power (GW)'
        ].sum()
        caps.append(cap)
    operational_capacity[L] = caps

# ----- 2. Plot 1: Heatmap -----
fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(cumulative_decom_data, aspect='auto', cmap='plasma')
fig.colorbar(im, label='Cumulative Decommissioned Capacity (GW)')
ax.set_xticks(np.arange(len(cumulative_decom_data.columns)))
ax.set_xticklabels(cumulative_decom_data.columns, rotation=45)
ax.set_yticks(np.arange(len(cumulative_decom_data.index)))
ax.set_yticklabels(cumulative_decom_data.index)
ax.set_xlabel('Decommissioning Year')
ax.set_ylabel('Country')
ax.set_title('Cumulative Decommissioned Capacity (20-Year Life)')

# vertical line at 2022
x2022 = list(cumulative_decom_data.columns).index(2022)
ax.axvline(x=x2022, color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('heatmap_20yr.png')
plt.show()

# ----- 3. Plot 2: Sensitivity Line Plot -----
fig, ax = plt.subplots(figsize=(12, 8))
for L, caps in operational_capacity.items():
    ax.plot(years, caps, label=f'{L}-yr life')
ax.set_xlabel('Year')
ax.set_ylabel('Oper. Capacity (GW)')
ax.set_title('Capacity Decline Sensitivity')
ax.legend()
ax.grid(alpha=0.3)
ax.axvline(x=2022, color='red', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig('sensitivity.png')
plt.show()

# ----- 4. Plot 3: Stacked Bar -----
fig, ax = plt.subplots(figsize=(14, 8))
annual_decom_data.T.plot(kind='bar', stacked=True, ax=ax, width=0.8, cmap='tab20')
ax.set_xlabel('Decommissioning Year')
ax.set_ylabel('Decom. Capacity (GW)')
ax.set_title('Annual Decommissioning Rate by Country')
ax.set_xticklabels(annual_decom_data.columns, rotation=45)

x2022_bar = list(annual_decom_data.columns).index(2022)
ax.axvline(x=x2022_bar, color='red', linestyle='--', linewidth=2)

ax.legend(title='Country', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig('stacked_decom.png')
plt.show()
