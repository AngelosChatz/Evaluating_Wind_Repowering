import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

FILE_NEW = RESULTS_DIR / "Approach_2_Cf.xlsx"
FILE_OLD = RESULTS_DIR / "Cf_old_updated.xlsx"

df_new = pd.read_excel(FILE_NEW).reset_index(drop=True)
df_old = pd.read_excel(FILE_OLD).reset_index(drop=True)

# standardize column names
df_new = df_new.rename(columns={
    'Annual_Energy_MWh_new':       'Annual_Energy_MWh',
    'Recommended_WT_Capacity':     'Recommended_WT_Capacity',
    'Number of turbines':          'New_Turbine_Count'
})
df_old = df_old.rename(columns={
    'Annual_Energy_MWh':           'Annual_Energy_MWh',
    'Representative_New_Capacity': 'Recommended_WT_Capacity',
    'Number of turbines':          'New_Turbine_Count'
})

# drop duplicate‐labeled columns
df_new = df_new.loc[:, ~df_new.columns.duplicated()]
df_old = df_old.loc[:, ~df_old.columns.duplicated()]


def ensure_kw(s: pd.Series) -> pd.Series:
    return s * 1000 if s.median() < 50 else s

df_new['Recommended_WT_Capacity'] = ensure_kw(df_new['Recommended_WT_Capacity'])
df_old['Recommended_WT_Capacity'] = ensure_kw(df_old['Recommended_WT_Capacity'])

# drop rows with missing or zero in key columns
req = ['ID', 'Annual_Energy_MWh', 'New_Turbine_Count', 'Recommended_WT_Capacity']
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(subset=req).reset_index(drop=True)
    mask = (
        (df2['Annual_Energy_MWh'] > 0) &
        (df2['New_Turbine_Count'] > 0) &
        (df2['Recommended_WT_Capacity'] > 0)
    )
    return df2.loc[mask].reset_index(drop=True)

df_new = clean(df_new)
df_old = clean(df_old)


common = set(df_new['ID']) & set(df_old['ID'])
df_new = df_new[df_new['ID'].isin(common)].reset_index(drop=True)
df_old = df_old[df_old['ID'].isin(common)].reset_index(drop=True)


df = df_new.merge(df_old[['ID']], on='ID', how='left').copy()


BASE_CAPEX_PER_KW  = 953.34    # €/kW repowering baseline
DECOM_COST_PER_KW  = 1267.1    # €/kW decomm. baseline
OM_PER_KW_YEAR     = 30.0      # €/kW·yr
LIFETIME           = 20        # yrs
DISCOUNT_RATE      = 0.025
DEFAULT_ELEC_PRICE = 80        # €/MWh

learning_scenarios = {
    'Baseline (0%)':      0.00,
    'Conservative (10%)': 0.10,
    'Moderate (14%)':     0.14,
    'Advanced (20%)':     0.20,
}

if 'Electricity_Price' not in df.columns:
    df['Electricity_Price'] = DEFAULT_ELEC_PRICE


def calc_lcoe(row, capex_per_kw,
              discount_rate=DISCOUNT_RATE,
              lifetime=LIFETIME,
              om_per_kw=OM_PER_KW_YEAR):
    # capacity (kW) and yield (kWh)
    cap_kw     = row['Recommended_WT_Capacity'] * row['New_Turbine_Count']
    annual_kwh = row['Annual_Energy_MWh'] * 1000
    if annual_kwh <= 0 or pd.isna(annual_kwh):
        return np.nan
    capex = cap_kw * capex_per_kw
    om_ann = cap_kw * om_per_kw
    i, n  = discount_rate, lifetime
    crf   = i * (1 + i)**n / ((1 + i)**n - 1)
    lcoe  = (capex * crf + om_ann) / annual_kwh  # €/kWh
    return lcoe * 1000  # €/MWh


records = []
for scen, reduction in learning_scenarios.items():
    capex_rep = BASE_CAPEX_PER_KW * (1 - reduction)
    capex_dec = DECOM_COST_PER_KW * (1 - reduction)
    rep_vals = df.apply(lambda r: calc_lcoe(r, capex_rep), axis=1)
    dec_vals = df.apply(lambda r: calc_lcoe(r, capex_dec), axis=1)
    records.append(pd.DataFrame({
        'Scenario':             scen,
        'LCOE_Repowering':      rep_vals,
        'LCOE_Decommissioning': dec_vals,
    }))
lcoe_df = pd.concat(records, ignore_index=True)


avg_stats = lcoe_df.groupby('Scenario').mean().reset_index()

# BOXPLOT ACROSS SCENARIOS
plt.figure(figsize=(12, 6))
box_data, positions, colors = [], [], []
width, spacing, pos = 0.35, 1.0, 0
for scen in learning_scenarios:
    rv = lcoe_df.loc[lcoe_df.Scenario==scen, 'LCOE_Repowering'].dropna()
    dv = lcoe_df.loc[lcoe_df.Scenario==scen, 'LCOE_Decommissioning'].dropna()
    box_data.extend([rv, dv])
    positions.extend([pos+1, pos+1+width])
    colors.extend(['skyblue','orange'])
    pos += spacing
bp = plt.boxplot(box_data, positions=positions, widths=width,
                 patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
ticks = [i+width/2+0.5 for i in np.arange(0,len(learning_scenarios)*spacing,spacing)]
plt.xticks(ticks, learning_scenarios.keys(), rotation=15, ha='right')
plt.ylabel('LCOE (€/MWh)')
plt.title('LCOE Across Learning-Curve Scenarios')
plt.legend([mpatches.Patch(fc='skyblue'), mpatches.Patch(fc='orange')],
           ['Repowering','Decommissioning'], title='Strategy')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#: AVG LCOE VS SCENARIO
plt.figure(figsize=(8,5))
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Repowering'],
         marker='o', label='Repowering')
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Decommissioning'],
         marker='s', linestyle='--', label='Decommissioning')
plt.xticks(rotation=15)
plt.ylabel('Average LCOE (€/MWh)')
plt.title('Average LCOE Under Learning-Curve Scenarios')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

#) PLOT 5: AVERAGE LCOE PER COUNTRY

reduc = learning_scenarios['Moderate (14%)']
cr_rep = BASE_CAPEX_PER_KW * (1 - reduc)
cr_dec = DECOM_COST_PER_KW * (1 - reduc)
df_mod = df.copy()
df_mod['LCOE_Repowering']      = df_mod.apply(lambda r: calc_lcoe(r, cr_rep), axis=1)
df_mod['LCOE_Decommissioning'] = df_mod.apply(lambda r: calc_lcoe(r, cr_dec), axis=1)

# avg per country, drop if both NaN
cs = (df_mod.groupby('Country')[['LCOE_Repowering','LCOE_Decommissioning']]
      .mean()
      .dropna(how='all')
      .sort_values('LCOE_Repowering', ascending=False)
      .reset_index())

plt.figure(figsize=(14,6))
x = np.arange(len(cs))
w = 0.35
plt.bar(x - w/2, cs['LCOE_Repowering'], w, label='Repowering', color='skyblue')
plt.bar(x + w/2, cs['LCOE_Decommissioning'], w, label='Decommissioning', color='orange')
plt.xticks(x, cs['Country'], rotation=45, ha='right')
plt.ylabel('Average LCOE (€/MWh)')
plt.title('Average LCOE per Country (Moderate 14% Learning)')
plt.legend(title='Strategy')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
