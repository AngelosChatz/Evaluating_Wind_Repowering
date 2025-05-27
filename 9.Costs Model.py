import os
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
CAPEX_PER_KW       = 953.34    # €/kW for repowering
DECOM_COST_PER_KW  = 1267.1    # €/kW for decommissioning
OM_PER_KW_YEAR     = 30.0      # €/kW·year
LIFETIME           = 20        # years
DISCOUNT_RATE      = 0.025     # per year
DEFAULT_ELEC_PRICE = 80        # €/MWh

# ─── PATHS ─────────────────────────────────────────────────────────────────────
this_dir    = Path(__file__).resolve().parent
results_dir = this_dir / "results"
file_new    = results_dir / "Approach_2_Cf.xlsx"
file_old    = results_dir / "Cf_old_updated.xlsx"

# ─── LOAD & RESET INDEX ────────────────────────────────────────────────────────
df_new = pd.read_excel(file_new).reset_index(drop=True)
df_old = pd.read_excel(file_old).reset_index(drop=True)

# ─── STANDARDIZE COLUMN NAMES ─────────────────────────────────────────────────
df_new = df_new.rename(columns={
    'Annual_Energy_MWh_new': 'Annual_Energy_MWh',
    'Recommended_WT_Capacity': 'Recommended_WT_Capacity',
    'Number of turbines': 'New_Turbine_Count'
})
df_old = df_old.rename(columns={
    'Annual_Energy_MWh': 'Annual_Energy_MWh',
    'Representative_New_Capacity': 'Recommended_WT_Capacity',
    'Number of turbines': 'New_Turbine_Count'
})

# ─── DROP DUPLICATE‐LABELED COLUMNS ─────────────────────────────────────────────
df_new = df_new.loc[:, ~df_new.columns.duplicated()]
df_old = df_old.loc[:, ~df_old.columns.duplicated()]

# ─── CONVERT MW→kW IF NEEDED ───────────────────────────────────────────────────
def ensure_kw(series):
    return series * 1000 if series.median() < 50 else series

df_new['Recommended_WT_Capacity'] = ensure_kw(df_new['Recommended_WT_Capacity'])
df_old['Recommended_WT_Capacity'] = ensure_kw(df_old['Recommended_WT_Capacity'])

# ─── CLEANING: DROP NaN & ZERO ─────────────────────────────────────────────────
req = ['ID', 'Annual_Energy_MWh', 'New_Turbine_Count', 'Recommended_WT_Capacity']
def clean(df):
    df2 = df.dropna(subset=req).reset_index(drop=True)
    mask = (
        (df2['Annual_Energy_MWh'] != 0) &
        (df2['New_Turbine_Count'] != 0) &
        (df2['Recommended_WT_Capacity'] != 0)
    )
    return df2.loc[mask].reset_index(drop=True)

df_new = clean(df_new)
df_old = clean(df_old)

# ─── KEEP ONLY COMMON PARKS ────────────────────────────────────────────────────
common = set(df_new['ID']) & set(df_old['ID'])
df_new = df_new[df_new['ID'].isin(common)].reset_index(drop=True)
df_old = df_old[df_old['ID'].isin(common)].reset_index(drop=True)

# ─── MERGE INPUT DATA ──────────────────────────────────────────────────────────
df = df_new.merge(df_old[['ID']], on='ID', how='left')

# ─── FINANCIAL CALCULATION FUNCTION ───────────────────────────────────────────
def calc_financials(row, cost_per_kw,
                    discount_rate=DISCOUNT_RATE,
                    lifetime=LIFETIME,
                    default_price=DEFAULT_ELEC_PRICE):
    cap_kw     = row['Recommended_WT_Capacity'] * row['New_Turbine_Count']
    energy_kwh = row['Annual_Energy_MWh'] * row['New_Turbine_Count'] * 1000
    if cap_kw <= 0 or energy_kwh <= 0:
        return pd.Series({k: np.nan for k in (
            'Total_Capacity_kW','CAPEX_Eur','Annual_OM_Eur',
            'NPV_Eur','IRR','Annual_Revenue_Eur','LCOE_Eur_per_MWh'
        )})
    price = row.get('Electricity_Price', default_price)
    capex = cap_kw * cost_per_kw
    om    = cap_kw * OM_PER_KW_YEAR
    rev   = (energy_kwh / 1000) * price
    cfs   = [-capex] + [(rev - om)] * lifetime
    npv   = sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cfs))
    try:
        irr = npf.irr(cfs)
    except:
        irr = np.nan
    crf  = discount_rate * (1 + discount_rate)**lifetime / ((1 + discount_rate)**lifetime - 1)
    lcoe = ((capex * crf) + om) / energy_kwh * 1000
    return pd.Series({
        'Total_Capacity_kW':   cap_kw,
        'CAPEX_Eur':           capex,
        'Annual_OM_Eur':       om,
        'NPV_Eur':             npv,
        'IRR':                 irr,
        'Annual_Revenue_Eur':  rev,
        'LCOE_Eur_per_MWh':    lcoe
    })

# ─── APPLY METRICS ──────────────────────────────────────────────────────────────
df_rep   = df.apply(lambda r: calc_financials(r, CAPEX_PER_KW), axis=1)
df_decom = df.apply(lambda r: calc_financials(r, DECOM_COST_PER_KW), axis=1)
for col in df_rep.columns:
    df[f"{col}_rep"]   = df_rep[col]
    df[f"{col}_decom"] = df_decom[col]

# Working DF
df_final = df.copy()

# ─── PAYBACK FUNCTION ─────────────────────────────────────────────────────────
def calc_payback(cost, benefit, life):
    cf    = np.concatenate(([-cost], np.repeat(benefit, life)))
    cum   = np.cumsum(cf)
    years = np.arange(0, life+1)
    return next((y for y, v in zip(years, cum) if v >= 0), np.nan)

# ─── ORIGINAL PLOTS ────────────────────────────────────────────────────────────
# A) NPV Across Parks (Repowering)
sorted_npv = df_final.sort_values('NPV_Eur_rep', ascending=False)
plt.figure(figsize=(10,6))
plt.bar(range(len(sorted_npv)), sorted_npv['NPV_Eur_rep'], color='skyblue')
plt.xlabel('Wind Parks (sorted by NPV)')
plt.ylabel('NPV (EUR)')
plt.title('NPV Across Wind Parks (Repowering)')
plt.ylim(-0.25e9, 0.75e9)
plt.grid(axis='y')
plt.show()

# B) IRR Across Parks (Repowering)
sorted_irr = df_final.sort_values('IRR_rep', ascending=False)
plt.figure(figsize=(10,6))
plt.bar(range(len(sorted_irr)), sorted_irr['IRR_rep'], color='lightgreen')
plt.xlabel('Wind Parks (sorted by IRR)')
plt.ylabel('IRR')
plt.title('IRR Across Wind Parks (Repowering)')
plt.grid(axis='y')
plt.show()

# C) Financial Metrics Comparison (Boxplots)
metrics = ['NPV_Eur','CAPEX_Eur','Annual_Revenue_Eur']
pos, width, space = 0, 0.3, 3
box_data, positions, colors = [], [], []
for m in metrics:
    for scenario, color in [('rep','skyblue'), ('decom','orange')]:
        data = df_final[f"{m}_{scenario}"].dropna().to_numpy()
        q1, med, q3 = np.percentile(data, [25,50,75])
        iqr = q3 - q1
        box_data.append({
            'label': f"{m}_{scenario}",
            'q1': q1, 'med': med, 'q3': q3,
            'whislo': q1-1.5*iqr, 'whishi': q3+1.5*iqr, 'fliers': []
        })
        positions.append(pos + (1 if scenario=='rep' else 1+width))
        colors.append(color)
    pos += space
plt.figure(figsize=(14,6))
ax = plt.gca()
bp = ax.bxp(box_data, positions=positions, widths=width,
            showfliers=False, patch_artist=True)
for patch, col in zip(bp['boxes'], colors):
    patch.set_facecolor(col)
ax.legend([mpatches.Patch(facecolor='skyblue'),
           mpatches.Patch(facecolor='orange')],
          ['Repowering','Decommissioning'],
          title='Scenario')
ticks = [i + width for i in np.arange(1, len(metrics)*space, space)]
ax.set_xticks(ticks)
ax.set_xticklabels(['NPV','CAPEX','Revenue'])
ax.set_title('Financial Metrics Comparison')
ax.grid(True)
plt.show()

# D) LCOE Comparison (Boxplot)
stats = []
for scenario, color in [('rep','skyblue'), ('decom','orange')]:
    data = df_final[f"LCOE_Eur_per_MWh_{scenario}"].dropna().to_numpy()
    q1, med, q3 = np.percentile(data, [25,50,75])
    iqr = q3 - q1
    stats.append({
        'label': scenario,
        'q1': q1, 'med': med, 'q3': q3,
        'whislo': q1-1.5*iqr, 'whishi': q3+1.5*iqr, 'fliers': []
    })
plt.figure(figsize=(8,6))
ax2 = plt.gca()
bp2 = ax2.bxp(stats, positions=[1,1+width], widths=width,
              showfliers=False, patch_artist=True)
for patch, col in zip(bp2['boxes'], ['skyblue','orange']):
    patch.set_facecolor(col)
ax2.legend([mpatches.Patch(facecolor='skyblue'),
            mpatches.Patch(facecolor='orange')],
           ['Repowering','Decommissioning'],
           title='Scenario')
ax2.set_xticks([1 + width/2])
ax2.set_xticklabels(['LCOE (€/MWh)'])
ax2.set_title('LCOE Comparison')
ax2.grid(True)
plt.show()

# ─── ADDITIONAL PLOTS ──────────────────────────────────────────────────────────
# 1) Sorted Payback Period Comparison
df_final['Payback_rep'] = df_final.apply(
    lambda r: calc_payback(
        r['Total_Capacity_kW_rep'] * CAPEX_PER_KW,
        r['Annual_Revenue_Eur_rep'] - r['Total_Capacity_kW_rep'] * OM_PER_KW_YEAR,
        LIFETIME
    ), axis=1
)
df_final['Payback_decom'] = df_final.apply(
    lambda r: calc_payback(
        r['Total_Capacity_kW_decom'] * DECOM_COST_PER_KW,
        r['Annual_Revenue_Eur_decom'] - r['Total_Capacity_kW_decom'] * OM_PER_KW_YEAR,
        LIFETIME
    ), axis=1
)
sorted_pb = df_final.sort_values('Payback_rep')
plt.figure(figsize=(10,6))
idx = np.arange(len(sorted_pb))
plt.plot(idx, sorted_pb['Payback_rep'], marker='o', label='Repowering')
plt.plot(idx, sorted_pb['Payback_decom'], marker='x', linestyle='--', label='Decommissioning')
plt.xlabel('Parks sorted by Repowering Payback')
plt.ylabel('Payback Period (years)')
plt.title('Sorted Payback Period Comparison')
plt.legend(); plt.grid(True)
plt.show()

# 2–4) Sensitivity Analyses vs Electricity Price
prices = np.arange(50, 121, 10)
records = []
for price in prices:
    df_temp = df_final.copy()
    df_temp['Electricity_Price'] = price

    rep = df_temp.apply(
        lambda r: calc_financials(r, CAPEX_PER_KW, default_price=price),
        axis=1
    )
    cap_mw_rep     = rep['Total_Capacity_kW'] / 1000
    total_npv_rep  = rep['NPV_Eur'].sum()
    total_cap_rep  = cap_mw_rep.sum()
    npv_per_mw_rep = total_npv_rep / total_cap_rep
    irr_wgt_rep    = (rep['IRR'] * cap_mw_rep).sum() / total_cap_rep
    pctpos_rep     = cap_mw_rep[rep['NPV_Eur'] > 0].sum() / total_cap_rep * 100

    dec = df_temp.apply(
        lambda r: calc_financials(r, DECOM_COST_PER_KW, default_price=price),
        axis=1
    )
    cap_mw_dec     = dec['Total_Capacity_kW'] / 1000
    total_npv_dec  = dec['NPV_Eur'].sum()
    total_cap_dec  = cap_mw_dec.sum()
    npv_per_mw_dec = total_npv_dec / total_cap_dec
    irr_wgt_dec    = (dec['IRR'] * cap_mw_dec).sum() / total_cap_dec
    pctpos_dec     = cap_mw_dec[dec['NPV_Eur'] > 0].sum() / total_cap_dec * 100

    records.append({
        'Price': price,
        'NPV_per_MW_Rep': npv_per_mw_rep,
        'NPV_per_MW_Dec': npv_per_mw_dec,
        'IRR_wgt_Rep':    irr_wgt_rep,
        'IRR_wgt_Dec':    irr_wgt_dec,
        '%CapPos_Rep':    pctpos_rep,
        '%CapPos_Dec':    pctpos_dec
    })

sens = pd.DataFrame(records)

# a) NPV per MW vs Price
plt.figure(figsize=(8,5))
plt.plot(sens['Price'], sens['NPV_per_MW_Rep'], label='Repowering')
plt.plot(sens['Price'], sens['NPV_per_MW_Dec'], linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('NPV per MW (€)')
plt.title('NPV per MW vs Electricity Price')
plt.legend(); plt.grid(True)
plt.show()

# b) Capacity-Weighted IRR vs Price
plt.figure(figsize=(8,5))
plt.plot(sens['Price'], sens['IRR_wgt_Rep'], label='Repowering')
plt.plot(sens['Price'], sens['IRR_wgt_Dec'], linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('Capacity-Weighted IRR')
plt.title('IRR vs Electricity Price')
plt.legend(); plt.grid(True)
plt.show()

# c) % of Capacity In-The-Money vs Price
plt.figure(figsize=(8,5))
plt.plot(sens['Price'], sens['%CapPos_Rep'], marker='o', label='Repowering')
plt.plot(sens['Price'], sens['%CapPos_Dec'], marker='x', linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('% of Capacity with NPV > 0')
plt.title('% Capacity In-The-Money vs Price')
plt.legend(); plt.grid(True)
plt.show()

# 5) Two Random Parks: 2×2 Grid of Metrics vs Price
positive = df_final[df_final['NPV_Eur_rep'] > 0]
parks   = positive.sample(2, random_state=42)

for _, park in parks.iterrows():
    prices = np.arange(50, 121, 10)
    results = {'NPV': {'rep': [], 'dec': []},
               'IRR': {'rep': [], 'dec': []},
               'Payback': {'rep': [], 'dec': []}}
    lcoe_rep = calc_financials(park, CAPEX_PER_KW)['LCOE_Eur_per_MWh']
    lcoe_dec = calc_financials(park, DECOM_COST_PER_KW)['LCOE_Eur_per_MWh']

    for price in prices:
        row            = park.copy()
        row['Electricity_Price'] = price
        rep_metric     = calc_financials(row, CAPEX_PER_KW, default_price=price)
        dec_metric     = calc_financials(row, DECOM_COST_PER_KW, default_price=price)
        results['NPV']['rep'].append(rep_metric['NPV_Eur'])
        results['NPV']['dec'].append(dec_metric['NPV_Eur'])
        results['IRR']['rep'].append(rep_metric['IRR'])
        results['IRR']['dec'].append(dec_metric['IRR'])
        pay_rep = calc_payback(
            rep_metric['Total_Capacity_kW'] * CAPEX_PER_KW,
            rep_metric['Annual_Revenue_Eur'] - rep_metric['Total_Capacity_kW'] * OM_PER_KW_YEAR,
            LIFETIME
        )
        pay_dec = calc_payback(
            dec_metric['Total_Capacity_kW'] * DECOM_COST_PER_KW,
            dec_metric['Annual_Revenue_Eur'] - dec_metric['Total_Capacity_kW'] * OM_PER_KW_YEAR,
            LIFETIME
        )
        results['Payback']['rep'].append(pay_rep)
        results['Payback']['dec'].append(pay_dec)

    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    axes = axes.flatten()
    for ax, metric in zip(axes, ['NPV', 'IRR', 'LCOE', 'Payback']):
        if metric == 'LCOE':
            ax.bar(['Repowering','Decommissioning'], [lcoe_rep, lcoe_dec],
                   color=['skyblue','orange'])
        else:
            ax.plot(prices, results[metric]['rep'], marker='o', label='Repowering')
            ax.plot(prices, results[metric]['dec'], marker='x', linestyle='--', label='Decommissioning')
        ax.set_title(f"{metric} vs Price\n{park['Country']} ({park['Longitude']:.2f},{park['Latitude']:.2f})")
        ax.set_xlabel('€/MWh' if metric != 'LCOE' else '')
        ax.set_ylabel(metric if metric != 'LCOE' else '€/MWh')
        ax.grid(True)
        if metric != 'LCOE':
            ax.legend()
    plt.tight_layout()
    plt.show()

# 6) Scatter: LCOE vs Capacity Factor (Repowering)
df_final['Cap_MW']       = df_final['Total_Capacity_kW_rep'] / 1000
df_final['Cap_Factor_%'] = df_final['Annual_Energy_MWh'] / (df_final['Cap_MW'] * 8760) * 100

plt.figure(figsize=(8,6))
plt.scatter(df_final['Cap_Factor_%'], df_final['LCOE_Eur_per_MWh_rep'],
            alpha=0.7, edgecolor='k')
plt.xlabel('Capacity Factor (%)')
plt.ylabel('LCOE (€/MWh)')
plt.title('LCOE vs Capacity Factor (Repowering)')
plt.grid(True)
plt.show()

# 7) Pie Chart: CAPEX vs PV(O&M) for a Sample Park
sample = parks.iloc[0]
rep1   = calc_financials(sample, CAPEX_PER_KW)
capex1 = rep1['CAPEX_Eur']
om1    = rep1['Annual_OM_Eur']
pv_om1 = om1 * (1 - (1 + DISCOUNT_RATE)**(-LIFETIME)) / DISCOUNT_RATE

plt.figure(figsize=(6,6))
plt.pie([capex1, pv_om1],
        labels=['CAPEX','PV of O&M'],
        explode=(0.1,0),
        autopct='%1.1f%%',
        startangle=90,
        shadow=True)
plt.title(f'CAPEX vs PV(O&M) (Repowering)\n'
          f'{sample["Country"]} ({sample["Longitude"]:.2f},{sample["Latitude"]:.2f})')
plt.show()

# 8) Two-Panel Boxplots: NPV & CAPEX and IRR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
width, space = 0.3, 3

# Panel 1: NPV & CAPEX
box_data, positions, colors = [], [], []
pos = 0
for m in ['NPV_Eur','CAPEX_Eur']:
    for scenario, col in [('rep','skyblue'), ('decom','orange')]:
        data = df_final[f"{m}_{scenario}"].dropna().to_numpy()
        q1, med, q3 = np.percentile(data, [25,50,75]); iqr = q3 - q1
        box_data.append({
            'label': m,
            'q1': q1, 'med': med, 'q3': q3,
            'whislo': q1-1.5*iqr, 'whishi': q3+1.5*iqr, 'fliers': []
        })
        positions.append(pos + (1 if scenario=='rep' else 1+width))
        colors.append(col)
    pos += space

bp1 = ax1.bxp(box_data, positions=positions, widths=width,
              showfliers=False, patch_artist=True)
for patch, col in zip(bp1['boxes'], colors):
    patch.set_facecolor(col)
ax1.set_xticks([1 + width/2, 1 + space + width/2])
ax1.set_xticklabels(['NPV','CAPEX'])
ax1.set_title('NPV & CAPEX Comparison')
ax1.legend([mpatches.Patch(facecolor='skyblue'),
            mpatches.Patch(facecolor='orange')],
           ['Repowering','Decommissioning'],
           title='Scenario')
ax1.grid(True)

# Panel 2: IRR
box_data, positions, colors = [], [], []
pos = 0
for m in ['IRR']:
    for scenario, col in [('rep','skyblue'), ('decom','orange')]:
        data = df_final[f"{m}_{scenario}"].dropna().to_numpy()
        q1, med, q3 = np.percentile(data, [25,50,75]); iqr = q3 - q1
        box_data.append({
            'label': m,
            'q1': q1, 'med': med, 'q3': q3,
            'whislo': q1-1.5*iqr, 'whishi': q3+1.5*iqr, 'fliers': []
        })
        positions.append(pos + (1 if scenario=='rep' else 1+width))
        colors.append(col)
    pos += space

bp2 = ax2.bxp(box_data, positions=positions, widths=width,
              showfliers=False, patch_artist=True)
for patch, col in zip(bp2['boxes'], colors):
    patch.set_facecolor(col)
ax2.set_xticks([1 + width/2])
ax2.set_xticklabels(['IRR'])
ax2.set_title('IRR Comparison')
ax2.legend([mpatches.Patch(facecolor='skyblue'),
            mpatches.Patch(facecolor='orange')],
           ['Repowering','Decommissioning'],
           title='Scenario')
ax2.grid(True)

plt.tight_layout()
plt.show()

# 8) Three-Panel Boxplots: NPV, CAPEX, and IRR Comparison
fig, axes = plt.subplots(1, 3, figsize=(18,6))
width = 0.3
colors = ['skyblue','orange']

# define the metrics and their labels
metrics = ['NPV_Eur', 'CAPEX_Eur', 'IRR']
labels  = ['NPV', 'CAPEX', 'IRR']

for ax, m, lbl in zip(axes, metrics, labels):
    # build the two box entries (repowering, decommissioning)
    box_data = []
    for scenario, col in [('rep', colors[0]), ('decom', colors[1])]:
        data = df_final[f"{m}_{scenario}"].dropna().to_numpy()
        q1, med, q3 = np.percentile(data, [25,50,75])
        iqr = q3 - q1
        box_data.append({
            'label': scenario,
            'q1':      q1,
            'med':     med,
            'q3':      q3,
            'whislo':  q1 - 1.5*iqr,
            'whishi':  q3 + 1.5*iqr,
            'fliers':  []
        })

    # draw them side by side at x=1 and x=1+width
    bp = ax.bxp(box_data,
                positions=[1, 1+width],
                widths=width,
                showfliers=False,
                patch_artist=True)
    # color the boxes
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)

    # center the single tick under the pair
    ax.set_xticks([1 + width/2])
    ax.set_xticklabels([lbl])
    ax.set_title(f"{lbl} Comparison")
    ax.legend([mpatches.Patch(facecolor=colors[0]),
               mpatches.Patch(facecolor=colors[1])],
              ['Repowering','Decommissioning'],
              title='Scenario')
    ax.grid(True)

plt.tight_layout()
plt.show()


# ─── SUMMARY STATISTICS ─────────────────────────────────────────────────────────
metrics = ['NPV_Eur', 'CAPEX_Eur', 'IRR']
rows = []
for m in metrics:
    for scen,label in [('rep','Repowering'), ('decom','Decommissioning')]:
        col = f"{m}_{scen}"
        s   = df_final[col].dropna()
        rows.append({
            'Metric':    m.replace('_Eur','').replace('_',' '),
            'Scenario':  label,
            'Mean':      s.mean(),
            'Median':    s.median(),
            'Minimum':   s.min(),
            'Maximum':   s.max()
        })

stats_df = pd.DataFrame(rows)
# round for readability
stats_df[['Mean','Median','Minimum','Maximum']] = stats_df[['Mean','Median','Minimum','Maximum']].round(2)

print(stats_df.to_string(index=False))
