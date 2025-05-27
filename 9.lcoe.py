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

# ─── READ & RENAME NEW ────────────────────────────────────────────────────────
df_new = pd.read_excel(FILE_NEW).reset_index(drop=True)
df_new = df_new.rename(columns={
    'ID':                            'id',
    'Total_Recommended_WT_Capacity': 'total_capacity_mw',
    'Annual_Energy_MWh_new':         'annual_energy_mwh_new',
    'Country':                       'country'
})
df_new['capacity_kw'] = df_new['total_capacity_mw'] * 1000
df_new['energy_mwh']  = df_new['annual_energy_mwh_new']

# ─── READ & RENAME OLD ────────────────────────────────────────────────────────
df_old = pd.read_excel(FILE_OLD).reset_index(drop=True)
df_old = df_old.rename(columns={
    'ID':                          'id',
    'SingleWT_Capacity':          'singlewt_capacity_kw',   # already in kW
    'Number of turbines':         'turbine_count_old',
    'Annual_Energy_MWh':          'annual_energy_mwh_old',
    'Country':                    'country_old'
})
df_old['capacity_kw_old'] = df_old['singlewt_capacity_kw'] * df_old['turbine_count_old']
df_old['energy_mwh_old']  = df_old['annual_energy_mwh_old'] * df_old['turbine_count_old']

# ─── FILTER OUT BAD DATA ─────────────────────────────────────────────────────
df_new = df_new.dropna(subset=['capacity_kw','energy_mwh']).query('capacity_kw>0 and energy_mwh>0')
df_old = df_old.dropna(subset=['capacity_kw_old','energy_mwh_old']).query('capacity_kw_old>0 and energy_mwh_old>0')

# ─── ALIGN & MERGE (include old annual and count) ─────────────────────────────
common = set(df_new['id']) & set(df_old['id'])
df_new = df_new[df_new['id'].isin(common)].reset_index(drop=True)
df_old = df_old[df_old['id'].isin(common)].reset_index(drop=True)

df = (
    df_new[['id','capacity_kw','energy_mwh','country','total_capacity_mw','annual_energy_mwh_new']]
    .merge(
        df_old[['id',
                'capacity_kw_old',
                'energy_mwh_old',
                'turbine_count_old',
                'annual_energy_mwh_old']],
        on='id', how='inner'
    )
)

# ─── FINANCIAL PARAMETERS ────────────────────────────────────────────────────
BASE_CAPEX_PER_KW  = 1010.0   # €/kW repowering
DECOM_COST_PER_KW  = 1267.1   # €/kW decommissioning
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

# ─── BUILD LCOE TABLE ────────────────────────────────────────────────────────
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
    rv = lcoe_df.loc[lcoe_df['Scenario']==scen, 'LCOE_Repowering']
    dv = lcoe_df.loc[lcoe_df['Scenario']==scen, 'LCOE_Decommissioning']
    box_data += [rv, dv]
    colors    += ['skyblue','orange']
    positions += [i*2+1, i*2+1+0.35]

bp = plt.boxplot(box_data, positions=positions, widths=0.35,
                 patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
tick_pos = [i*2+1+0.35/2 for i in range(len(learning_scenarios))]
plt.xticks(tick_pos, learning_scenarios.keys(), rotation=15, ha='right')
plt.ylabel('LCOE (€/MWh)')
plt.legend([mpatches.Patch(color='skyblue'), mpatches.Patch(color='orange')],
           ['Repowering','Decommissioning'], title='Strategy')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ─── PLOT 2: AVG LCOE vs SCENARIO ────────────────────────────────────────────
plt.figure(figsize=(8,5))
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Repowering'], marker='o', label='Repowering')
plt.plot(avg_stats['Scenario'], avg_stats['LCOE_Decommissioning'], marker='s', linestyle='--', label='Decommissioning')
plt.xticks(rotation=15)
plt.ylabel('Avg LCOE (€/MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── PLOT 3: COUNTRY COMPARISON (Moderate 14%) ───────────────────────────────
moderate_rate = learning_scenarios['Moderate (14%)']
mod = df.copy()
mod['LCOE_Repowering']      = mod.apply(lambda r: calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     BASE_CAPEX_PER_KW*(1-moderate_rate)), axis=1)
mod['LCOE_Decommissioning'] = mod.apply(lambda r: calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], BASE_CAPEX_PER_KW*(1-moderate_rate)), axis=1)

cs = (mod.groupby('country')[['LCOE_Repowering','LCOE_Decommissioning']]
         .mean()
         .dropna()
         .reset_index()
         .sort_values('LCOE_Repowering', ascending=False))

plt.figure(figsize=(14,6))
x = np.arange(len(cs)); w = 0.35
plt.bar(x - w/2, cs['LCOE_Repowering'],      w, label='Repowering',      color='skyblue')
plt.bar(x + w/2, cs['LCOE_Decommissioning'], w, label='Decommissioning', color='orange')
plt.xticks(x, cs['country'], rotation=45, ha='right')
plt.ylabel('Avg LCOE (€/MWh)'); plt.legend(title='Strategy')
plt.grid(axis='y', linestyle='--', alpha=0.5); plt.tight_layout(); plt.show()

# ─── DEBUG OUTPUT: first 5 parks ─────────────────────────────────────────────
print("\n*** DEBUG OUTPUT: first 5 parks ***\n")
cf_new_list = []
cf_old_list = []
for _, r in df.head(5).iterrows():
    rep_lcoe = calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     BASE_CAPEX_PER_KW)
    dec_lcoe = calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], DECOM_COST_PER_KW)
    per_old  = r['energy_mwh_old'] / r['turbine_count_old']
    cf_new   = r['energy_mwh']     / (r['capacity_kw']    / 1000 * 8760)
    cf_old   = r['energy_mwh_old'] / (r['capacity_kw_old']/ 1000 * 8760)
    cf_new_list.append(cf_new)
    cf_old_list.append(cf_old)
    print(
        f"ID {r['id']}: New→ cap {r['total_capacity_mw']:.2f} MW, prod {r['annual_energy_mwh_new']:.2f} MWh, "
        f"CF={cf_new:.2%}, LCOE {rep_lcoe:.2f} €/MWh; "
        f"Old→ per‐turbine prod {per_old:.1f} MWh, total prod {r['energy_mwh_old']:.1f} MWh, "
        f"CF={cf_old:.2%}, LCOE {dec_lcoe:.2f} €/MWh"
    )

# ─── CAPACITY FACTOR COMPARISON FOR ALL PARKS ────────────────────────────────
cf_new_all = df['energy_mwh']     / (df['capacity_kw']     / 1000 * 8760)
cf_old_all = df['energy_mwh_old'] / (df['capacity_kw_old'] / 1000 * 8760)

# How many parks where CF_repowered > CF_old?
num_higher = (cf_new_all > cf_old_all).sum()
total_parks = len(df)

print(f"\nOut of {total_parks} parks, {num_higher} have a higher capacity factor after repowering than before.")
