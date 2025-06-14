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
FILE_OLD    = RESULTS_DIR / "CF_old_updated.xlsx"

# ─── READ & RENAME NEW ────────────────────────────────────────────────────────
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
    raise KeyError("Couldn't find expected new‐capacity column.")
df_new = df_new.rename(columns=rename_new)
df_new['capacity_kw'] = df_new['total_capacity_mw'] * 1000
df_new['energy_mwh']  = df_new['annual_energy_mwh_new']

# ─── READ & RENAME OLD ────────────────────────────────────────────────────────
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
    raise KeyError("Couldn't find expected old‐capacity column.")
df_old = df_old.rename(columns=rename_old)
df_old['capacity_kw_old'] = df_old['singlewt_capacity_kw'] * df_old['turbine_count_old']
df_old['energy_mwh_old']  = df_old['annual_energy_mwh_old'] * df_old['turbine_count_old']

# ─── FILTER OUT BAD DATA ─────────────────────────────────────────────────────
df_new = df_new.dropna(subset=['capacity_kw','energy_mwh']) \
               .query('capacity_kw>0 and energy_mwh>0') \
               .reset_index(drop=True)
df_old = df_old.dropna(subset=['capacity_kw_old','energy_mwh_old']) \
               .query('capacity_kw_old>0 and energy_mwh_old>0') \
               .reset_index(drop=True)

# ─── MERGE NEW & OLD ─────────────────────────────────────────────────────────
common = set(df_new['id']) & set(df_old['id'])
df_new = df_new[df_new['id'].isin(common)].reset_index(drop=True)
df_old = df_old[df_old['id'].isin(common)].reset_index(drop=True)

df = (
    df_new[['id','capacity_kw','energy_mwh','country']]
      .merge(
          df_old[['id','capacity_kw_old','energy_mwh_old','turbine_count_old']],
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

# ─── BASELINE LCOE ────────────────────────────────────────────────────────────
df['LCOE_Repowering_baseline']      = df.apply(
    lambda r: calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     BASE_CAPEX_PER_KW), axis=1)
df['LCOE_Decommissioning_baseline'] = df.apply(
    lambda r: calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], DECOM_COST_PER_KW), axis=1)

# ─── EXTREME BASELINE LCOE BY COUNTRY ────────────────────────────────────────
ext = (
    df.groupby('country')
      .agg(
        Rep_LCOE_min=('LCOE_Repowering_baseline','min'),
        Rep_LCOE_max=('LCOE_Repowering_baseline','max'),
        Dec_LCOE_min=('LCOE_Decommissioning_baseline','min'),
        Dec_LCOE_max=('LCOE_Decommissioning_baseline','max'),
      )
      .reset_index()
)
print("\n=== EXTREME BASELINE LCOE VALUES BY COUNTRY ===")
for _, r in ext.iterrows():
    print(f"{r['country']}: Repowering → min {r['Rep_LCOE_min']:.1f}, max {r['Rep_LCOE_max']:.1f} €/MWh")
    print(f"             Decommissioning → min {r['Dec_LCOE_min']:.1f}, max {r['Dec_LCOE_max']:.1f} €/MWh\n")

# ─── BUILD LCOE TABLE FOR ALL SCENARIOS (with Hybrid) ────────────────────────
records = []
for scen, rate in learning_scenarios.items():
    rep_cost = BASE_CAPEX_PER_KW * (1 - rate)
    dec_cost = DECOM_COST_PER_KW * (1 - rate)

    rep    = df.apply(lambda r: calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     rep_cost), axis=1)
    dec    = df.apply(lambda r: calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], dec_cost), axis=1)
    hybrid = np.where(df['energy_mwh'] >= df['energy_mwh_old'], rep, dec)

    records.append(pd.DataFrame({
        'Scenario':        scen,
        'country':         df['country'],
        'Repowering':      rep,
        'Decommissioning': dec,
        'No_Loss_Hybrid':  hybrid,
    }))

lcoe_df = pd.concat(records, ignore_index=True)

# ─── PLOT 1: BOXPLOT OF ALL SCENARIOS ────────────────────────────────────────
plt.figure(figsize=(12,6))
box_data, colors, positions = [], [], []
for i, scen in enumerate(learning_scenarios):
    rv = lcoe_df.loc[lcoe_df['Scenario']==scen, 'Repowering']
    dv = lcoe_df.loc[lcoe_df['Scenario']==scen, 'Decommissioning']
    box_data += [rv, dv]
    colors    += ['skyblue','orange']
    positions += [i*2+1, i*2+1+0.35]

bp = plt.boxplot(box_data, positions=positions, widths=0.35,
                 patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)

tick_pos = [i*2+1 + 0.35/2 for i in range(len(learning_scenarios))]
plt.xticks(tick_pos, learning_scenarios.keys(), rotation=15, ha='right')
plt.ylabel('LCOE (€/MWh)')
plt.legend([mpatches.Patch(color='skyblue'), mpatches.Patch(color='orange')],
           ['Approach 2-Capacity Maximization', 'Approach 0-Old (Decomm+Repl)'], title='Strategy')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("BOXPLOT OF ALL SCENARIOS .png", dpi=300, bbox_inches="tight")
plt.show()

# ─── PLOT 2: AVG LCOE vs SCENARIO ────────────────────────────────────────────
avg_stats = (
    lcoe_df
    .groupby('Scenario')[['Repowering','Decommissioning']]
    .mean()
    .reset_index()
)
plt.figure(figsize=(8,5))
plt.plot(avg_stats['Scenario'], avg_stats['Repowering'], marker='o', label='Approach 2-Capacity Maximization')
plt.plot(avg_stats['Scenario'], avg_stats['Decommissioning'], marker='s', linestyle='--', label='Approach 0-Old (Decomm+Repl)')
plt.xticks(rotation=15)
plt.ylabel('Avg LCOE (€/MWh)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("AVG LCOE vs SCENARIO.png", dpi=300, bbox_inches="tight")
plt.show()

# ─── PLOT 3: COUNTRY COMPARISON (Moderate, 14%) ─────────────────────────────
mod_rate = learning_scenarios['Moderate (14%)']
mod = df.copy()
mod['Repowering']      = mod.apply(lambda r: calc_lcoe(r['energy_mwh'],     r['capacity_kw'],     BASE_CAPEX_PER_KW*(1-mod_rate)), axis=1)
mod['Decommissioning'] = mod.apply(lambda r: calc_lcoe(r['energy_mwh_old'], r['capacity_kw_old'], DECOM_COST_PER_KW*(1-mod_rate)), axis=1)
mod['No_Loss_Hybrid']  = np.where(mod['energy_mwh']>=mod['energy_mwh_old'], mod['Repowering'], mod['Decommissioning'])

cs_hybrid = (
    mod.groupby('country')[['Repowering','Decommissioning','No_Loss_Hybrid']]
       .mean()
       .dropna()
       .reset_index()
       .sort_values('Repowering', ascending=False)
)

plt.figure(figsize=(14,6))
x = np.arange(len(cs_hybrid)); w = 0.25
plt.bar(x - w, cs_hybrid['Decommissioning'], w,
        label='Approach 0-Old (Decomm+Repl)', color='black')
plt.bar(x,     cs_hybrid['Repowering'],      w,
        label='Approach 2-Capacity Maximization', color='orange')
plt.bar(x + w, cs_hybrid['No_Loss_Hybrid'],  w,
        label='Approach 6-No-Loss Hybrid (Yield-based)', color='purple')

plt.xticks(x, cs_hybrid['country'], rotation=45, ha='right')
plt.ylabel('Avg LCOE (€/MWh)')
plt.title('LCOE by Country – Moderate 14% Learning Scenario')
plt.legend(title='Strategy')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("LCOE by Country – Moderate 14% Learning Scenario.png", dpi=300, bbox_inches="tight")
plt.show()


# ─── PLOT 4: % DIFFERENCE vs Decommissioning (14%) ───────────────────────────
cs_hybrid['Rep_%_of_Dec'] = (cs_hybrid['Repowering'] - cs_hybrid['Decommissioning']) \
                            / cs_hybrid['Decommissioning'] * 100
cs_hybrid['NLH_%_of_Dec'] = (cs_hybrid['No_Loss_Hybrid'] - cs_hybrid['Decommissioning']) \
                            / cs_hybrid['Decommissioning'] * 100

plt.figure(figsize=(14,5))
x = np.arange(len(cs_hybrid)); w = 0.35

plt.bar(x - w/2, cs_hybrid['Rep_%_of_Dec'], width=w,
        color='orange', label='Approach 2-Capacity Maximization')
plt.bar(x + w/2, cs_hybrid['NLH_%_of_Dec'], width=w,
        color='purple', label='Approach 6-No-Loss Hybrid (Yield-based)')

plt.axhline(0, color='black', linewidth=1.0)
plt.xticks(x, cs_hybrid['country'], rotation=45, ha='right')
plt.ylabel('% Difference vs Decommissioning')
plt.title('Relative LCOE Difference vs Approach 0 – Moderate 14% Scenario')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Relative LCOE Difference vs Approach 0 – Moderate 14% Scenario.png", dpi=300, bbox_inches="tight")
plt.show()

# ─── PLOT 5: LCOE vs ENERGY YIELD (Baseline) ─────────────────────────────────
df_sorted = df.sort_values('energy_mwh').copy()
plt.figure(figsize=(10,6))
plt.plot(df_sorted['energy_mwh'], df_sorted['LCOE_Repowering_baseline'],
         linestyle='-',  linewidth=2, label='Approach 2-Capacity Maximization')
plt.plot(df_sorted['energy_mwh'], df_sorted['LCOE_Decommissioning_baseline'],
         linestyle='--', linewidth=2, label='Approach 0-Old (Decomm+Repl)')
plt.xlabel('Annual Energy Generation (MWh)')
plt.ylabel('LCOE (€/MWh)')
plt.title('LCOE vs. Energy Yield (Sorted by Yield, Baseline)')
plt.legend()
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("LCOE vs. Energy Yield (Sorted by Yield, Baseline).png", dpi=300, bbox_inches="tight")
plt.show()
