import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── PATHS & I/O ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FILE_NEW    = RESULTS_DIR / "Approach_2_Cf.xlsx"
FILE_OLD    = RESULTS_DIR / "Cf_old_updated.xlsx"

# ─── READ & INSPECT & RENAME NEW ───────────────────────────────────────────────
df_new = pd.read_excel(FILE_NEW).reset_index(drop=True)

new_cols = df_new.columns.tolist()
if 'Total_Recommended_WT_Capacity' in new_cols:
    rename_new = {
        'ID':                            'id',
        'Total_Recommended_WT_Capacity': 'total_capacity_mw',
        'Annual_Energy_MWh_new':         'annual_energy_mwh_new',
        'Country':                       'country'
    }
elif 'Total_New_Capacity' in new_cols:
    rename_new = {
        'ID':                    'id',
        'Total_New_Capacity':    'total_capacity_mw',
        'Annual_Energy_MWh_new': 'annual_energy_mwh_new',
        'Country':               'country'
    }
else:
    print(">>> df_new columns:", new_cols)
    raise KeyError(
        "Could not find 'Total_Recommended_WT_Capacity' or 'Total_New_Capacity' in df_new columns."
    )

df_new = df_new.rename(columns=rename_new)
df_new['capacity_kw'] = df_new['total_capacity_mw'] * 1000
df_new['energy_mwh']  = df_new['annual_energy_mwh_new']

# ─── READ & INSPECT & RENAME OLD ───────────────────────────────────────────────
df_old = pd.read_excel(FILE_OLD).reset_index(drop=True)

old_cols = df_old.columns.tolist()
if 'SingleWT_Capacity' in old_cols:
    rename_old = {
        'ID':                   'id',
        'SingleWT_Capacity':    'singlewt_capacity_kw',
        'Number of turbines':   'turbine_count_old',
        'Annual_Energy_MWh':    'annual_energy_mwh_old',
        'Country':              'country_old'
    }
elif 'Representative_New_Capacity' in old_cols:
    rename_old = {
        'ID':                         'id',
        'Representative_New_Capacity':'singlewt_capacity_kw',
        'Number of turbines':         'turbine_count_old',
        'Annual_Energy_MWh':          'annual_energy_mwh_old',
        'Country':                    'country_old'
    }
else:
    print(">>> df_old columns:", old_cols)
    raise KeyError(
        "Could not find 'SingleWT_Capacity' or 'Representative_New_Capacity' in df_old columns."
    )

df_old = df_old.rename(columns=rename_old)
df_old['capacity_kw_old'] = df_old['singlewt_capacity_kw'] * df_old['turbine_count_old']
df_old['energy_mwh_old']  = df_old['annual_energy_mwh_old'] * df_old['turbine_count_old']

# ─── FILTER OUT BAD DATA ─────────────────────────────────────────────────────
df_new = (
    df_new
    .dropna(subset=['capacity_kw','energy_mwh'])
    .query('capacity_kw > 0 and energy_mwh > 0')
    .reset_index(drop=True)
)
df_old = (
    df_old
    .dropna(subset=['capacity_kw_old','energy_mwh_old'])
    .query('capacity_kw_old > 0 and energy_mwh_old > 0')
    .reset_index(drop=True)
)

# ─── ALIGN & MERGE ─────────────────────────────────────────────────────────────
common = set(df_new['id']) & set(df_old['id'])
df_new = df_new[df_new['id'].isin(common)].reset_index(drop=True)
df_old = df_old[df_old['id'].isin(common)].reset_index(drop=True)

df = (
    df_new[['id','capacity_kw','energy_mwh','country','total_capacity_mw','annual_energy_mwh_new']]
    .merge(
        df_old[['id','capacity_kw_old','energy_mwh_old','turbine_count_old','annual_energy_mwh_old']],
        on='id', how='inner'
    )
)

# ─── FINANCIAL PARAMETERS ────────────────────────────────────────────────────
BASE_CAPEX_PER_KW  = 1077.4
DECOM_COST_PER_KW  = 1267.1
OM_PER_KW_YEAR     = 30.0     # €/kW·yr
LIFETIME           = 20       # years
DISCOUNT_RATE      = 0.025

learning_scenarios = {
    'Baseline (0%)':      0.00,
    'Conservative (10%)': 0.10,
    'Moderate (14%)':     0.14,
    'Advanced (20%)':     0.20,
}

def calc_lcoe(energy_mwh, cap_kW, capex_per_kW,
              discount_rate=DISCOUNT_RATE, lifetime=LIFETIME,
              om_per_kw=OM_PER_KW_YEAR):
    """Return LCOE in €/MWh."""
    if energy_mwh <= 0 or cap_kW <= 0:
        return np.nan
    annual_kwh = energy_mwh * 1000
    capex      = cap_kW * capex_per_kW
    om_ann     = cap_kW * om_per_kw
    i, n       = discount_rate, lifetime
    crf        = i * (1 + i)**n / ((1 + i)**n - 1)
    return (capex * crf + om_ann) / annual_kwh * 1000

# ─── BASELINE LCOE COLUMNS (0% learning) ────────────────────────────────────
df['LCOE_Repowering_baseline']      = df.apply(
    lambda r: calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     BASE_CAPEX_PER_KW), axis=1
)
df['LCOE_Decommissioning_baseline'] = df.apply(
    lambda r: calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], DECOM_COST_PER_KW), axis=1
)

# ─── PRINT EXTREME LCOE VALUES PER COUNTRY ───────────────────────────────────
country_extremes = (
    df
    .groupby('country')
    .agg(
        Rep_LCOE_min=('LCOE_Repowering_baseline','min'),
        Rep_LCOE_max=('LCOE_Repowering_baseline','max'),
        Dec_LCOE_min=('LCOE_Decommissioning_baseline','min'),
        Dec_LCOE_max=('LCOE_Decommissioning_baseline','max'),
    )
    .reset_index()
)

print("\n=== EXTREME BASELINE LCOE VALUES BY COUNTRY ===")
for _, row in country_extremes.iterrows():
    c = row['country']
    print(f"{c}: Repowering  →  min = {row['Rep_LCOE_min']:.1f} €/MWh,  max = {row['Rep_LCOE_max']:.1f} €/MWh")
    print(f"    Decommissioning →  min = {row['Dec_LCOE_min']:.1f} €/MWh,  max = {row['Dec_LCOE_max']:.1f} €/MWh\n")

# ─── BUILD LCOE TABLE FOR ALL SCENARIOS ─────────────────────────────────────
records = []
for scen, rate in learning_scenarios.items():
    rep_cost = BASE_CAPEX_PER_KW * (1 - rate)
    dec_cost = DECOM_COST_PER_KW * (1 - rate)

    rep = df.apply(lambda r: calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     rep_cost), axis=1)
    dec = df.apply(lambda r: calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], dec_cost), axis=1)

    records.append(pd.DataFrame({
        'Scenario':             scen,
        'LCOE_Repowering':      rep,
        'LCOE_Decommissioning': dec,
    }))

lcoe_df   = pd.concat(records, ignore_index=True)
avg_stats = lcoe_df.groupby('Scenario').mean().reset_index()

# ─── PLOT 1: BOXPLOT ─────────────────────────────────────────────────────────
plt.figure(figsize=(12,6))
box_data, colors, positions = [], [], []
for i, scen in enumerate(learning_scenarios):
    rv = lcoe_df.loc[lcoe_df['Scenario'] == scen, 'LCOE_Repowering']
    dv = lcoe_df.loc[lcoe_df['Scenario'] == scen, 'LCOE_Decommissioning']
    box_data += [rv, dv]
    colors    += ['skyblue', 'orange']
    positions += [i*2+1, i*2+1+0.35]

bp = plt.boxplot(
    box_data,
    positions=positions,
    widths=0.35,
    patch_artist=True,
    showfliers=False
)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)

tick_pos = [i*2+1 + 0.35/2 for i in range(len(learning_scenarios))]
plt.xticks(tick_pos, learning_scenarios.keys(), rotation=15, ha='right')
plt.ylabel('LCOE (€/MWh)')
plt.legend(
    [mpatches.Patch(color='skyblue'), mpatches.Patch(color='orange')],
    ['Repowering', 'Decommissioning'],
    title='Strategy'
)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ─── PLOT 2: AVG LCOE vs SCENARIO ────────────────────────────────────────────
plt.figure(figsize=(8,5))
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Repowering'], marker='o', label='Repowering')
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Decommissioning'], marker='s', linestyle='--', label='Decommissioning')
plt.xticks(rotation=15)
plt.ylabel('Avg LCOE (€/MWh)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ─── PLOT 3: COUNTRY COMPARISON WITH HYBRID (Moderate 14%) ─────────────────
moderate_rate = learning_scenarios['Moderate (14%)']

mod = df.copy()
mod['LCOE_Repowering']      = mod.apply(
    lambda r: calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     BASE_CAPEX_PER_KW * (1 - moderate_rate)), axis=1
)
mod['LCOE_Decommissioning'] = mod.apply(
    lambda r: calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], DECOM_COST_PER_KW * (1 - moderate_rate)), axis=1
)
mod['LCOE_Hybrid'] = np.where(
    mod['energy_mwh'] >= mod['energy_mwh_old'],
    mod['LCOE_Repowering'],
    mod['LCOE_Decommissioning']
)

cs_hybrid = (
    mod
    .groupby('country')[['LCOE_Repowering','LCOE_Decommissioning','LCOE_Hybrid']]
    .mean()
    .dropna()
    .reset_index()
    .sort_values('LCOE_Repowering', ascending=False)
)

plt.figure(figsize=(14,6))
x = np.arange(len(cs_hybrid))
w = 0.25

plt.bar(x - w,             cs_hybrid['LCOE_Repowering'],      w, label='Repowering',      color='skyblue')
plt.bar(x,                 cs_hybrid['LCOE_Decommissioning'], w, label='Decommissioning', color='orange')
plt.bar(x + w,             cs_hybrid['LCOE_Hybrid'],          w, label='No-Loss Hybrid',  color='seagreen')

plt.xticks(x, cs_hybrid['country'], rotation=45, ha='right')
plt.ylabel('Avg LCOE (€/MWh)')
plt.legend(title='Strategy')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ─── PLOT 4: REP vs NLH % OF DECOMMISSIONING ────────────────────────────────
cs_hybrid['Rep_%_of_Dec'] = (
    (cs_hybrid['LCOE_Repowering'] - cs_hybrid['LCOE_Decommissioning'])
    / cs_hybrid['LCOE_Decommissioning']
) * 100

cs_hybrid['NLH_%_of_Dec'] = (
    (cs_hybrid['LCOE_Hybrid'] - cs_hybrid['LCOE_Decommissioning'])
    / cs_hybrid['LCOE_Decommissioning']
) * 100

plt.figure(figsize=(14, 5))
x = np.arange(len(cs_hybrid))
w = 0.35

plt.bar(x - w/2, cs_hybrid['Rep_%_of_Dec'], width=w, color='skyblue', label='Repowering % of Dec')
plt.bar(x + w/2, cs_hybrid['NLH_%_of_Dec'], width=w, color='seagreen', label='No-Loss Hybrid % of Dec')

plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(x, cs_hybrid['country'], rotation=45, ha='right')
plt.ylabel('% Difference vs Decommissioning')
plt.title('Repowering vs No-Loss Hybrid LCOE (Moderate 14%)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ─── CAPACITY FACTOR COMPARISON FOR ALL PARKS ────────────────────────────────
cf_new_all = df['energy_mwh']     / (df['capacity_kw']     / 1000 * 8760)
cf_old_all = df['energy_mwh_old'] / (df['capacity_kw_old'] / 1000 * 8760)

num_higher = (cf_new_all > cf_old_all).sum()
total_parks = len(df)
print(f"\nOut of {total_parks} parks, {num_higher} have a higher capacity factor after repowering than before.")

# ─── SUMMARY FOR MODERATE (14%) SCENARIO ──────────────────────────────────────
mod_avg = avg_stats.loc[avg_stats['Scenario'] == 'Moderate (14%)']
rep_avg = mod_avg['LCOE_Repowering'].values[0]
dec_avg = mod_avg['LCOE_Decommissioning'].values[0]
print(f"\nOverall average LCOE at 14% learning rate:")
print(f"  • Repowering:      {rep_avg:.1f} €/MWh")
print(f"  • Decommissioning: {dec_avg:.1f} €/MWh\n")

print("Average LCOE by country (Moderate 14%):")
print(
    cs_hybrid[['country', 'LCOE_Repowering', 'LCOE_Decommissioning']]
      .rename(columns={
          'country': 'Country',
          'LCOE_Repowering': 'Repowering (€/MWh)',
          'LCOE_Decommissioning': 'Replacement (€/MWh)'
      })
      .to_string(index=False, float_format="%.1f")
)

# ─── ENERGY YIELD COMPARISON ──────────────────────────────────────────────────
num_higher_yield = (df['energy_mwh'] > df['energy_mwh_old']).sum()
yield_pct        = num_higher_yield / total_parks * 100
print(f"\nOut of {total_parks} parks, {num_higher_yield} ({yield_pct:.1f}%) have a higher annual energy yield after repowering than before.\n")

# ─── PLOT 5: LCOE vs ENERGY YIELD (Baseline, Continuous LINES) ─────────────
df_sorted = df.sort_values('energy_mwh').copy()

x_vals   = df_sorted['energy_mwh']
lcoe_rep = df_sorted['LCOE_Repowering_baseline']
lcoe_dec = df_sorted['LCOE_Decommissioning_baseline']

plt.figure(figsize=(10, 6))
plt.plot(x_vals, lcoe_rep,   linestyle='-',  linewidth=2, label='LCOE: Repowering (Baseline)')
plt.plot(x_vals, lcoe_dec,   linestyle='--', linewidth=2, label='LCOE: Decommissioning (Baseline)')
plt.xlabel('Annual Energy Generation (MWh)')
plt.ylabel('LCOE (€/MWh)')
plt.title('LCOE vs. Energy Yield (Parks Sorted by Yield, Baseline)')
plt.legend()
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
