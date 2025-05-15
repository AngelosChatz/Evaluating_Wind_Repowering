import os
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
BASE_CAPEX_PER_KW      = 953.34   # €/kW repowering baseline
DECOM_COST_PER_KW      = 1267.1   # €/kW decommissioning baseline
OM_PER_KW_YEAR         = 30.0     # €/kW-yr
LIFETIME               = 20       # yrs
DISCOUNT_RATE          = 0.025
DEFAULT_ELEC_PRICE     = 80       # €/MWh

learning_scenarios = {
    'Baseline (0%)':      0.00,
    'Conservative (10%)': 0.10,
    'Moderate (14%)':     0.14,
    'Advanced (20%)':     0.20,
}

# ─── IMPORT DATA ───────────────────────────────────────────────────────────────
df = pd.read_excel(
    r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Energy_Yield_Parks.xlsx"
)
if 'Electricity_Price' not in df.columns:
    df['Electricity_Price'] = DEFAULT_ELEC_PRICE

# ─── FINANCIAL FUNCTION ────────────────────────────────────────────────────────
def calc_financials(row, capex_per_kw,
                    discount_rate=DISCOUNT_RATE,
                    lifetime=LIFETIME,
                    om_per_kw=OM_PER_KW_YEAR):
    cap_kw = row['Recommended_WT_Capacity'] * 1000 * row['New_Turbine_Count']
    annual_e_kwh = row['Annual_Energy_MWh'] * 1000
    if annual_e_kwh <= 0 or pd.isna(annual_e_kwh):
        return np.nan
    capex = cap_kw * capex_per_kw
    om_ann = cap_kw * om_per_kw
    i, n = discount_rate, lifetime
    crf = i * (1 + i) ** n / ((1 + i) ** n - 1)
    lcoe_kwh = (capex * crf + om_ann) / annual_e_kwh
    return lcoe_kwh * 1000  # €/MWh

# ─── COMPUTE LCOE FOR ALL SCENARIOS ────────────────────────────────────────────
records = []
for scen, reduction in learning_scenarios.items():
    capex_rep = BASE_CAPEX_PER_KW * (1 - reduction)
    capex_dec = DECOM_COST_PER_KW * (1 - reduction)

    lcoe_rep = df.apply(lambda r: calc_financials(r, capex_rep), axis=1)
    lcoe_dec = df.apply(lambda r: calc_financials(r, capex_dec), axis=1)

    tmp = pd.DataFrame({
        'Scenario': scen,
        'LCOE_Repowering':        lcoe_rep,
        'LCOE_Decommissioning':   lcoe_dec,
    })
    records.append(tmp)

lcoe_df = pd.concat(records, ignore_index=True)

# ─── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("Per-park LCOE (first 10 rows):")
print(lcoe_df.head(10).to_string(index=False))

avg_stats = lcoe_df.groupby('Scenario').mean().reset_index()
print("\nAverage LCOE by Scenario (€/MWh):")
print(avg_stats.to_string(index=False,
    formatters={
        'LCOE_Repowering':      '{:,.1f}'.format,
        'LCOE_Decommissioning': '{:,.1f}'.format,
    }
))

# ─── PLOT 1: BOXPLOT ───────────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
box_data, positions, colors = [], [], []
width, spacing, pos = 0.35, 1.0, 0

for scen in learning_scenarios:
    rep_vals = lcoe_df.loc[lcoe_df.Scenario == scen, 'LCOE_Repowering'].dropna()
    dec_vals = lcoe_df.loc[lcoe_df.Scenario == scen, 'LCOE_Decommissioning'].dropna()
    box_data.extend([rep_vals, dec_vals])
    positions.extend([pos + 1, pos + 1 + width])
    colors.extend(['skyblue', 'orange'])
    pos += spacing

bp = plt.boxplot(box_data,
                 positions=positions,
                 widths=width,
                 patch_artist=True,
                 showfliers=False)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)

ticks = [i + width/2 + 0.5 for i in np.arange(0, len(learning_scenarios)*spacing, spacing)]
plt.xticks(ticks, learning_scenarios.keys(), rotation=15)
plt.ylabel('LCOE (€/MWh)')
plt.title('LCOE Comparison Across Learning-Curve Scenarios')
plt.legend([mpatches.Patch(fc='skyblue'), mpatches.Patch(fc='orange')],
           ['Repowering','Decommissioning'], title='Strategy')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ─── PLOT 2: AVG LCOE VS SCENARIO ──────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Repowering'],
         marker='o', label='Repowering')
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Decommissioning'],
         marker='s', linestyle='--', label='Decommissioning')
plt.xticks(rotation=15)
plt.ylabel('Average LCOE (€/MWh)')
plt.title('Average LCOE Under Learning-Curve Scenarios')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── PLOT 3: BUBBLE PLOT FOR MODERATE (14%) ────────────────────────────────────
moderate = df.copy()
reduction = learning_scenarios['Moderate (14%)']
capex_rep = BASE_CAPEX_PER_KW * (1 - reduction)
capex_dec = DECOM_COST_PER_KW * (1 - reduction)

# compute per-park metrics
moderate['Capacity_kW'] = moderate['Recommended_WT_Capacity'] * 1000 * moderate['New_Turbine_Count']
moderate['LCOE_Repowering']      = moderate.apply(lambda r: calc_financials(r, capex_rep), axis=1)
moderate['LCOE_Decommissioning'] = moderate.apply(lambda r: calc_financials(r, capex_dec), axis=1)
moderate['Price_Repowering_M€']  = moderate['Capacity_kW'] * capex_rep / 1e6
moderate['Price_Decommissioning_M€'] = moderate['Capacity_kW'] * capex_dec / 1e6

# size ∝ √(LCOE) to tame extremes
max_lcoe = max(moderate['LCOE_Repowering'].max(),
               moderate['LCOE_Decommissioning'].max())
max_size = 2000
sizes_rep = np.sqrt(moderate['LCOE_Repowering'] / max_lcoe) * max_size
sizes_dec = np.sqrt(moderate['LCOE_Decommissioning'] / max_lcoe) * max_size

fig, ax = plt.subplots(figsize=(10, 6))

# green = repowering
ax.scatter(
    moderate['Capacity_kW'],
    moderate['Price_Repowering_M€'],
    s=sizes_rep, color='green', alpha=0.6,
    edgecolors='w', linewidth=0.5,
    label='Repowering'
)
# orange = decommissioning
ax.scatter(
    moderate['Capacity_kW'],
    moderate['Price_Decommissioning_M€'],
    s=sizes_dec, color='orange', alpha=0.6,
    edgecolors='w', linewidth=0.5,
    label='Decommissioning'
)

ax.set_xlabel('Park Capacity (kW)')
ax.set_ylabel('Total Park Price (M€)')
ax.set_title('Moderate Scenario (14% Learning)\nY = Price (M€), Bubble ∝ √(LCOE)')
ax.set_xlim(0, moderate['Capacity_kW'].max() * 1.05)
ax.set_ylim(0, moderate[['Price_Repowering_M€','Price_Decommissioning_M€']].max().max() * 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# strategy legend
leg1 = ax.legend(loc='upper left', title='Strategy')

# LCOE size legend
lcoe_qs = [0.5, 0.9]
all_lcoes = pd.concat([
    moderate['LCOE_Repowering'],
    moderate['LCOE_Decommissioning']
])
lcoe_vals = all_lcoes.quantile(lcoe_qs)

handles, labels = [], []
for val in lcoe_vals:
    sz = np.sqrt(val / max_lcoe) * max_size
    handles.append(plt.scatter([], [], s=sz,
                               color='grey', alpha=0.6,
                               edgecolors='w'))
    labels.append(f"{val:.0f} €/MWh")

leg2 = ax.legend(handles, labels, title='LCOE (€/MWh)', loc='lower right')
ax.add_artist(leg1)

plt.tight_layout()
plt.show()


# ─── PLOT 4: AVG LCOE PER COUNTRY (MODERATE 14%) ────────────────────────────
# Group by country and compute mean LCOEs
country_stats = (
    moderate
    .groupby('Country')[['LCOE_Repowering','LCOE_Decommissioning']]
    .mean()
    .reset_index()
    .sort_values('LCOE_Repowering', ascending=False)  # optional: order by repowering
)

# Bar chart
plt.figure(figsize=(12, 6))
x = np.arange(len(country_stats))
width = 0.35

plt.bar(x - width/2, country_stats['LCOE_Repowering'], width, label='Repowering')
plt.bar(x + width/2, country_stats['LCOE_Decommissioning'], width, label='Decommissioning')

plt.xticks(x, country_stats['Country'], rotation=45, ha='right')
plt.ylabel('Average LCOE (€/MWh)')
plt.title('Average LCOE per Country (Moderate 14% Learning Scenario)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# ─── PLOT 5: BOX PLOT PER COUNTRY (SORTED) ───────────────────────────────────
# 1) Determine sort order by mean repowering LCOE
country_order = (
    moderate
    .groupby('Country')['LCOE_Repowering']
    .mean()
    .sort_values(ascending=False)  # change to True for lowest→highest
    .index
    .tolist()
)

# 2) Prepare boxplot data in that order
width = 0.35
box_data = []
positions = []
colors = []

for i, country in enumerate(country_order):
    rep = moderate.loc[moderate['Country'] == country, 'LCOE_Repowering'].dropna()
    dec = moderate.loc[moderate['Country'] == country, 'LCOE_Decommissioning'].dropna()
    box_data.extend([rep, dec])
    positions.extend([i - width/2, i + width/2])
    colors.extend(['skyblue', 'orange'])

# 3) Plot
fig, ax = plt.subplots(figsize=(15, 7))
bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=width,
    patch_artist=True,
    showfliers=False
)

# 4) Color the boxes
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)

# 5) Labels and legend
ax.set_xticks(np.arange(len(country_order)))
ax.set_xticklabels(country_order, rotation=90, ha='center')
ax.set_ylabel('LCOE (€/MWh)')
ax.set_title('LCOE Distribution per Country (Moderate 14% Learning Scenario)\n(sorted by mean Repowering LCOE)')

legend_patches = [
    mpatches.Patch(facecolor='skyblue', label='Repowering'),
    mpatches.Patch(facecolor='orange', label='Decommissioning')
]
ax.legend(handles=legend_patches, title='Strategy', loc='upper right')

ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()