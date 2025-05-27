import os
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
CAPEX_PER_KW       = 1010.4     # €/kW for repowering
DECOM_COST_PER_KW  = 1267.1     # €/kW for decommissioning
OM_PER_KW_YEAR     = 30.0       # €/kW·year
LIFETIME           = 20         # years
DISCOUNT_RATE      = 0.025      # per year
DEFAULT_ELEC_PRICE = 80         # €/MWh

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FILE_NEW    = RESULTS_DIR / "Approach_2_Cf.xlsx"
FILE_OLD    = RESULTS_DIR / "Cf_old_updated.xlsx"

# ─── LOAD & RENAME ─────────────────────────────────────────────────────────────
df_new = pd.read_excel(FILE_NEW).reset_index(drop=True).rename(columns={
    'ID':                            'ID',
    'Total_Recommended_WT_Capacity': 'Total_Capacity_MW',      # park total
    'Annual_Energy_MWh_new':         'Annual_Energy_MWh',      # park total
    'Number of turbines':            'New_Turbine_Count'       # just for info
})

df_old = pd.read_excel(FILE_OLD).reset_index(drop=True).rename(columns={
    'ID':                          'ID',
    'SingleWT_Capacity':          'Per_Turbine_Capacity_kW', # kW each
    'Annual_Energy_MWh':           'Annual_Energy_MWh_old',    # per turbine
    'Number of turbines':          'Turbine_Count_old'
})

# ─── DROP DUPLICATES ────────────────────────────────────────────────────────────
df_new = df_new.loc[:, ~df_new.columns.duplicated()]
df_old = df_old.loc[:, ~df_old.columns.duplicated()]

# ─── CLEAN ──────────────────────────────────────────────────────────────────────
def clean(df, cols):
    df2 = df.dropna(subset=cols).reset_index(drop=True)
    mask = np.logical_and.reduce([df2[c] != 0 for c in cols])
    return df2.loc[mask].reset_index(drop=True)

df_new = clean(df_new, ['ID','Total_Capacity_MW','Annual_Energy_MWh'])
df_old = clean(df_old, ['ID','Per_Turbine_Capacity_kW','Annual_Energy_MWh_old','Turbine_Count_old'])

# ─── ALIGN & MERGE ──────────────────────────────────────────────────────────────
common = set(df_new['ID']) & set(df_old['ID'])
df_new = df_new[df_new['ID'].isin(common)]
df_old = df_old[df_old['ID'].isin(common)]
df     = pd.merge(
    df_new,
    df_old[['ID','Per_Turbine_Capacity_kW','Annual_Energy_MWh_old','Turbine_Count_old']],
    on='ID', how='inner'
)

# ─── DERIVE TOTAL OLD CAPACITY & ENERGY ────────────────────────────────────────
df['Capacity_kW_old']  = df['Per_Turbine_Capacity_kW'] * df['Turbine_Count_old']
df['Energy_MWh_old']   = df['Annual_Energy_MWh_old'] * df['Turbine_Count_old']

# ─── DERIVE NEW CAPACITY & ENERGY ──────────────────────────────────────────────
df['Capacity_kW']      = df['Total_Capacity_MW'] * 1000  # park total in kW
df['Energy_MWh']       = df['Annual_Energy_MWh']         # park total

# ─── FINANCIAL CALCULATION ─────────────────────────────────────────────────────
def calc_financials(cap_kw, energy_mwh, capex_per_kw,
                    discount_rate=DISCOUNT_RATE, lifetime=LIFETIME,
                    om_per_kw=OM_PER_KW_YEAR, price=DEFAULT_ELEC_PRICE):
    energy_kwh = energy_mwh * 1000
    if cap_kw<=0 or energy_kwh<=0:
        return dict(NPV=np.nan, IRR=np.nan, CAPEX=np.nan, OM=np.nan,
                    Revenue=np.nan, LCOE=np.nan, Cap_kW=cap_kw)
    capex     = cap_kw * capex_per_kw
    om_ann    = cap_kw * om_per_kw
    revenue   = (energy_kwh/1000) * price
    cfs       = [-capex] + [(revenue - om_ann)] * lifetime
    npv       = sum(cf/((1+discount_rate)**t) for t,cf in enumerate(cfs))
    try:
        irr = npf.irr(cfs)
    except:
        irr = np.nan
    crf       = discount_rate*(1+discount_rate)**lifetime/(((1+discount_rate)**lifetime)-1)
    lcoe      = ((capex*crf) + om_ann)/energy_kwh * 1000
    return dict(NPV=npv, IRR=irr, CAPEX=capex, OM=om_ann,
                Revenue=revenue, LCOE=lcoe, Cap_kW=cap_kw)

# ─── WRAPPERS ──────────────────────────────────────────────────────────────────
def repowering_metrics(row, price=DEFAULT_ELEC_PRICE):
    return calc_financials(row['Capacity_kW'], row['Energy_MWh'], CAPEX_PER_KW,
                           price=price)

def decommissioning_metrics(row, price=DEFAULT_ELEC_PRICE):
    return calc_financials(row['Capacity_kW_old'], row['Energy_MWh_old'], DECOM_COST_PER_KW,
                           price=price)

# ─── APPLY METRICS ─────────────────────────────────────────────────────────────
rep = df.apply(repowering_metrics, axis=1, result_type='expand')
dec = df.apply(decommissioning_metrics, axis=1, result_type='expand')

for prefix, metrics in [('rep', rep), ('dec', dec)]:
    for col in metrics.columns:
        df[f"{col}_{prefix}"] = metrics[col]

# ─── PAYBACK ───────────────────────────────────────────────────────────────────
def payback(cost, benefit, life=LIFETIME):
    cfs = [-cost] + [benefit]*life
    cum = np.cumsum(cfs)
    years = np.arange(len(cfs))
    valid = years[cum>=0]
    return valid[0] if valid.size else np.nan

df['Payback_rep'] = df.apply(lambda r: payback(r['CAPEX_rep'], r['Revenue_rep'] - r['OM_rep']), axis=1)
df['Payback_dec'] = df.apply(lambda r: payback(r['CAPEX_dec'], r['Revenue_dec'] - r['OM_dec']), axis=1)

# ─── PLOTTING ──────────────────────────────────────────────────────────────────

# A) NPV bar
plt.figure(figsize=(10,6))
npv_sorted = df.sort_values('NPV_rep', ascending=False)
plt.bar(npv_sorted.index, npv_sorted['NPV_rep'], color='skyblue')
plt.title("NPV Across Parks (Repowering)")
plt.ylabel("NPV (€)"); plt.xlabel("Parks (sorted)")
plt.grid(axis='y'); plt.tight_layout(); plt.show()

# B) IRR bar
plt.figure(figsize=(10,6))
irr_sorted = df.sort_values('IRR_rep', ascending=False)
plt.bar(irr_sorted.index, irr_sorted['IRR_rep'], color='lightgreen')
plt.title("IRR Across Parks (Repowering)")
plt.ylabel("IRR"); plt.xlabel("Parks (sorted)")
plt.grid(axis='y'); plt.tight_layout(); plt.show()

# C) Boxplots NPV, CAPEX, Revenue
metrics = ['NPV','CAPEX','Revenue']
labels  = ['NPV','CAPEX','Revenue']
plt.figure(figsize=(14,6))
data = []
positions = []
colors = []
for i, m in enumerate(metrics):
    data += [df[f"{m}_rep"], df[f"{m}_dec"]]
    positions += [i*3+1, i*3+2]
    colors += ['skyblue','orange']
bp = plt.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
plt.xticks([np.mean(positions[i*2:i*2+2]) for i in range(len(metrics))], labels)
plt.legend([mpatches.Patch(color='skyblue'), mpatches.Patch(color='orange')],
           ['Repowering','Decommissioning'], title='Scenario')
plt.title("Financial Metrics Comparison"); plt.grid(True); plt.tight_layout(); plt.show()

# D) LCOE boxplot
plt.figure(figsize=(8,6))
data = [df['LCOE_rep'], df['LCOE_dec']]
bp = plt.boxplot(data, positions=[1,2], widths=0.8, patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], ['skyblue','orange']):
    patch.set_facecolor(c)
plt.xticks([1,2], ['Repowering','Decommissioning'])
plt.title("LCOE Comparison"); plt.ylabel("€/MWh")
plt.grid(True); plt.tight_layout(); plt.show()

# E) Payback comparison
plt.figure(figsize=(10,6))
sorted_pb = df.sort_values('Payback_rep')
plt.plot(sorted_pb.index, sorted_pb['Payback_rep'], marker='o', label='Repowering')
plt.plot(sorted_pb.index, sorted_pb['Payback_dec'], marker='x', label='Decommissioning')
plt.title("Payback Period Comparison")
plt.ylabel("Years"); plt.xlabel("Parks (sorted)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ─── F–H) Sensitivity Analyses vs Electricity Price ──────────────────────────
prices = np.arange(50, 121, 10)
records = []

for price in prices:
    # Recompute all metrics for this price
    rep_metrics = df.apply(lambda r: repowering_metrics(r, price=price), axis=1, result_type='expand')
    dec_metrics = df.apply(lambda r: decommissioning_metrics(r, price=price), axis=1, result_type='expand')

    cap_mw_rep = df['Capacity_kW'] / 1000
    cap_mw_dec = df['Capacity_kW_old'] / 1000

    records.append({
        'Price': price,
        'NPV_per_MW_Rep':    (rep_metrics['NPV'].sum()    / cap_mw_rep.sum()),
        'NPV_per_MW_Dec':    (dec_metrics['NPV'].sum()    / cap_mw_dec.sum()),
        'IRR_wgt_Rep':       ((rep_metrics['IRR']*cap_mw_rep).sum() / cap_mw_rep.sum()),
        'IRR_wgt_Dec':       ((dec_metrics['IRR']*cap_mw_dec).sum() / cap_mw_dec.sum()),
        '%CapPos_Rep':       (cap_mw_rep[rep_metrics['NPV'] > 0].sum() / cap_mw_rep.sum() * 100),
        '%CapPos_Dec':       (cap_mw_dec[dec_metrics['NPV'] > 0].sum() / cap_mw_dec.sum() * 100),
    })

sens_df = pd.DataFrame(records)

# F) NPV per MW vs Price
plt.figure(figsize=(8,5))
plt.plot(sens_df['Price'], sens_df['NPV_per_MW_Rep'], label='Repowering')
plt.plot(sens_df['Price'], sens_df['NPV_per_MW_Dec'], linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('NPV per MW (€)')
plt.title('NPV per MW vs Electricity Price')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# G) Capacity-Weighted IRR vs Price
plt.figure(figsize=(8,5))
plt.plot(sens_df['Price'], sens_df['IRR_wgt_Rep'], marker='o', label='Repowering')
plt.plot(sens_df['Price'], sens_df['IRR_wgt_Dec'], marker='x', linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('Capacity-Weighted IRR')
plt.title('Capacity-Weighted IRR vs Electricity Price')
plt.legend(loc='best'); plt.grid(True); plt.tight_layout(); plt.show()

# H) % Capacity In-The-Money vs Price
plt.figure(figsize=(8,5))
plt.plot(sens_df['Price'], sens_df['%CapPos_Rep'], marker='o', label='% Repowering NPV>0')
plt.plot(sens_df['Price'], sens_df['%CapPos_Dec'], marker='x', linestyle='--', label='% Decommissioning NPV>0')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('% of Capacity')
plt.title('% Capacity In-the-Money vs Electricity Price')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# I) Two sample parks metrics vs price
samples = df[df['NPV_rep']>0].sample(2, random_state=42)
for _, park in samples.iterrows():
    prices = np.arange(50,121,10)
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    axes = axes.flatten()
    rep_lcoe = repowering_metrics(park)['LCOE']
    dec_lcoe = decommissioning_metrics(park)['LCOE']
    rep_npv = []; dec_npv = []
    rep_irr = []; dec_irr = []
    rep_pb  = []; dec_pb  = []
    for p in prices:
        rp = repowering_metrics(park, price=p)
        dp = decommissioning_metrics(park, price=p)
        rep_npv.append(rp['NPV']); dec_npv.append(dp['NPV'])
        rep_irr.append(rp['IRR']); dec_irr.append(dp['IRR'])
        rep_pb.append(payback(rp['CAPEX'], rp['Revenue']-rp['OM']))
        dec_pb.append(payback(dp['CAPEX'], dp['Revenue']-dp['OM']))
    # NPV
    axes[0].plot(prices, rep_npv, '-o', label='Rep')
    axes[0].plot(prices, dec_npv, '-x', label='Dec')
    axes[0].set_title("NPV vs Price")
    axes[0].legend(); axes[0].grid(True)
    # IRR
    axes[1].plot(prices, rep_irr, '-o', label='Rep')
    axes[1].plot(prices, dec_irr, '-x', label='Dec')
    axes[1].set_title("IRR vs Price")
    axes[1].legend(); axes[1].grid(True)
    # LCOE bar
    axes[2].bar(['Rep','Dec'], [rep_lcoe, dec_lcoe], color=['skyblue','orange'])
    axes[2].set_title("LCOE")
    # Payback
    axes[3].plot(prices, rep_pb, '-o', label='Rep')
    axes[3].plot(prices, dec_pb, '-x', label='Dec')
    axes[3].set_title("Payback vs Price")
    axes[3].legend(); axes[3].grid(True)
    plt.tight_layout(); plt.show()

# J) Scatter LCOE vs CF
df['CF_pct'] = df['Annual_Energy_MWh'] / (df['Capacity_kW']/1000 * 8760) * 100
plt.figure(figsize=(8,6))
plt.scatter(df['CF_pct'], df['LCOE_rep'], alpha=0.7, edgecolor='k')
plt.title("LCOE vs Capacity Factor (Repowering)")
plt.xlabel("CF (%)"); plt.ylabel("LCOE (€/MWh)"); plt.grid(True); plt.tight_layout(); plt.show()

# K) Pie CAPEX vs PV(O&M)
sample = samples.iloc[0]
rp = repowering_metrics(sample)
pv_om = rp['OM'] * (1 - (1+DISCOUNT_RATE)**(-LIFETIME)) / DISCOUNT_RATE
plt.figure(figsize=(6,6))
plt.pie([rp['CAPEX'], pv_om], labels=['CAPEX','PV O&M'], explode=(0.1,0),
        autopct='%1.1f%%', startangle=90, shadow=True)
plt.title(f"CAPEX vs PV(O&M) Park ID {sample['ID']}"); plt.tight_layout(); plt.show()

# ─── F–H) Sensitivity Analyses vs Electricity Price ──────────────────────────
prices = np.arange(50, 121, 10)
records = []

for price in prices:
    # Recompute repowering and decommissioning metrics at this price
    rep_metrics = df.apply(lambda r: repowering_metrics(r, price=price), axis=1, result_type='expand')
    dec_metrics = df.apply(lambda r: decommissioning_metrics(r, price=price), axis=1, result_type='expand')

    cap_mw_rep = df['Capacity_kW'] / 1000
    cap_mw_dec = df['Capacity_kW_old'] / 1000

    records.append({
        'Price':           price,
        'NPV_per_MW_Rep':  rep_metrics['NPV'].sum()    / cap_mw_rep.sum(),
        'NPV_per_MW_Dec':  dec_metrics['NPV'].sum()    / cap_mw_dec.sum(),
        'IRR_wgt_Rep':     (rep_metrics['IRR'] * cap_mw_rep).sum() / cap_mw_rep.sum(),
        'IRR_wgt_Dec':     (dec_metrics['IRR'] * cap_mw_dec).sum() / cap_mw_dec.sum(),
        '%CapPos_Rep':     cap_mw_rep[rep_metrics['NPV'] > 0].sum() / cap_mw_rep.sum() * 100,
        '%CapPos_Dec':     cap_mw_dec[dec_metrics['NPV'] > 0].sum() / cap_mw_dec.sum() * 100,
    })

sens_df = pd.DataFrame(records)

# F) NPV per MW vs Electricity Price
plt.figure(figsize=(8,5))
plt.plot(sens_df['Price'], sens_df['NPV_per_MW_Rep'], label='Repowering')
plt.plot(sens_df['Price'], sens_df['NPV_per_MW_Dec'], linestyle='--', label='Decommissioning')
plt.title("NPV per MW vs Electricity Price")
plt.xlabel("€/MWh"); plt.ylabel("NPV per MW (€)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# G) Capacity-Weighted IRR vs Electricity Price
plt.figure(figsize=(8,5))
plt.plot(sens_df['Price'], sens_df['IRR_wgt_Rep'], marker='o', label='Repowering')
plt.plot(sens_df['Price'], sens_df['IRR_wgt_Dec'], marker='x', linestyle='--', label='Decommissioning')
plt.title("Capacity-Weighted IRR vs Electricity Price")
plt.xlabel("€/MWh"); plt.ylabel("IRR")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# H) % Capacity In-the-Money vs Electricity Price
plt.figure(figsize=(8,5))
plt.plot(sens_df['Price'], sens_df['%CapPos_Rep'], marker='o', label='% Repowering NPV>0')
plt.plot(sens_df['Price'], sens_df['%CapPos_Dec'], marker='x', linestyle='--', label='% Decommissioning NPV>0')
plt.title("% Capacity In-The-Money vs Electricity Price")
plt.xlabel("€/MWh"); plt.ylabel("% of Capacity")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# ─── L) Combined Boxplots: NPV & CAPEX and IRR ────────────────────────────────
width=0.3; space=3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

# Panel L1: NPV & CAPEX
box_data, positions, colors = [], [], []
pos = 0
for metric in ['NPV', 'CAPEX']:
    for scenario, col in [('rep','skyblue'), ('dec','orange')]:
        data = df[f"{metric}_{scenario}"]
        q1, med, q3 = np.percentile(data.dropna(), [25,50,75])
        iqr = q3 - q1
        box_data.append({
            'label': metric,
            'q1': q1, 'med': med, 'q3': q3,
            'whislo': q1 - 1.5*iqr, 'whishi': q3 + 1.5*iqr, 'fliers': []
        })
        positions.append(pos + (1 if scenario=='rep' else 1+width))
        colors.append(col)
    pos += space

bp1 = ax1.bxp(box_data, positions=positions, widths=width, showfliers=False, patch_artist=True)
for patch, col in zip(bp1['boxes'], colors):
    patch.set_facecolor(col)
ax1.set_xticks([1+width/2, 1+space+width/2])
ax1.set_xticklabels(['NPV','CAPEX'])
ax1.set_title("NPV & CAPEX Comparison")
ax1.legend([mpatches.Patch(facecolor='skyblue'), mpatches.Patch(facecolor='orange')],
           ['Repowering','Decommissioning'], title='Scenario')
ax1.grid(True)

# Panel L2: IRR
box_data, positions, colors = [], [], []
pos = 0
for scenario, col in [('rep','skyblue'), ('dec','orange')]:
    data = df[f"IRR_{scenario}"]
    q1, med, q3 = np.percentile(data.dropna(), [25,50,75])
    iqr = q3 - q1
    box_data.append({
        'label': 'IRR',
        'q1': q1, 'med': med, 'q3': q3,
        'whislo': q1 - 1.5*iqr, 'whishi': q3 + 1.5*iqr, 'fliers': []
    })
    positions.append(pos + (1 if scenario=='rep' else 1+width))
    colors.append(col)
    pos += space

bp2 = ax2.bxp(box_data, positions=positions, widths=width, showfliers=False, patch_artist=True)
for patch, col in zip(bp2['boxes'], colors):
    patch.set_facecolor(col)
ax2.set_xticks([1+width/2])
ax2.set_xticklabels(['IRR'])
ax2.set_title("IRR Comparison")
ax2.legend([mpatches.Patch(facecolor='skyblue'), mpatches.Patch(facecolor='orange')],
           ['Repowering','Decommissioning'], title='Scenario')
ax2.grid(True)

plt.tight_layout()
plt.show()


# ─── M) Three-Panel Boxplots: NPV, CAPEX, and IRR ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18,6))
width = 0.3
for ax, (metric, label) in zip(axes, [('NPV','NPV'), ('CAPEX','CAPEX'), ('IRR','IRR')]):
    bd = []
    for scen, col in [('rep','skyblue'), ('dec','orange')]:
        data = df[f"{metric}_{scen}"].dropna()
        q1, med, q3 = np.percentile(data, [25,50,75])
        iqr = q3 - q1
        bd.append({
            'label': scen,
            'q1': q1, 'med': med, 'q3': q3,
            'whislo': q1 - 1.5*iqr, 'whishi': q3 + 1.5*iqr, 'fliers': []
        })
    bp = ax.bxp(bd, positions=[1,1+width], widths=width, showfliers=False, patch_artist=True)
    for patch, col in zip(bp['boxes'], ['skyblue','orange']):
        patch.set_facecolor(col)
    ax.set_xticks([1+width/2]); ax.set_xticklabels([label])
    ax.set_title(f"{label} Comparison")
    ax.legend([mpatches.Patch(facecolor='skyblue'), mpatches.Patch(facecolor='orange')],
              ['Repowering','Decommissioning'], title='Scenario')
    ax.grid(True)

plt.tight_layout()
plt.show()

# ─── DEBUG SANITY CHECK FOR FIRST 5 PARKS ─────────────────────────────────────
print("\n\n=== SANITY CHECK: FIRST 5 PARKS ===\n")
for i, r in df.head(5).iterrows():
    # basics
    cap_new   = r['Capacity_kW']
    eng_new   = r['Energy_MWh']
    cap_old   = r['Capacity_kW_old']
    eng_old   = r['Energy_MWh_old']
    cf_new    = eng_new / (cap_new/1000 * 8760)
    cf_old    = eng_old / (cap_old/1000 * 8760)

    # financials at default price
    rep = repowering_metrics(r)
    dec = decommissioning_metrics(r)

    print(f"Park ID {r['ID']}:")
    print(f"  → NEW  cap = {cap_new:,.0f} kW, energy = {eng_new:,.0f} MWh, CF = {cf_new:.2%}")
    print(f"           CAPEX = €{rep['CAPEX']:,.0f},  O&M = €{rep['OM']:,.0f}/yr,")
    print(f"           Rev@€{DEFAULT_ELEC_PRICE}/MWh = €{rep['Revenue']:,.0f}/yr")
    print(f"           NPV = €{rep['NPV']:,.0f},  IRR = {rep['IRR']:.1%},  LCOE = €{rep['LCOE']:.1f}/MWh")
    print(f"  → OLD  cap = {cap_old:,.0f} kW, energy = {eng_old:,.0f} MWh, CF = {cf_old:.2%}")
    print(f"           CAPEX = €{dec['CAPEX']:,.0f},  O&M = €{dec['OM']:,.0f}/yr,")
    print(f"           Rev@€{DEFAULT_ELEC_PRICE}/MWh = €{dec['Revenue']:,.0f}/yr")
    print(f"           NPV = €{dec['NPV']:,.0f},  IRR = {dec['IRR']:.1%},  LCOE = €{dec['LCOE']:.1f}/MWh\n")


