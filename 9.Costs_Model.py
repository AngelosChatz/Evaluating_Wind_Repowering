import os
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
CAPEX_PER_KW       = 926.5     # €/kW for repowering
DECOM_COST_PER_KW  = 1089.9     # €/kW for decommissioning
OM_PER_KW_YEAR     = 30.0       # €/kW·year
LIFETIME           = 20         # years
DISCOUNT_RATE      = 0.025      # per year
DEFAULT_ELEC_PRICE = 80         # €/MWh

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FILE_NEW    = RESULTS_DIR / "Approach_2_Cf.xlsx"
FILE_OLD    = RESULTS_DIR / "CF_old_updated.xlsx"

# ─── LOAD & RENAME ─────────────────────────────────────────────────────────────
df_new = (
    pd.read_excel(FILE_NEW)
      .rename(columns={
          'ID':                     'ID',
          'Total_New_Capacity': 'Total_Capacity_MW',
          'Annual_Energy_MWh_new':  'Annual_Energy_MWh',
          'New_Turbine_Counts':     'New_Turbine_Count',
          'CapacityFactor':         'CF_new_pct'
      })
      .reset_index(drop=True)
)

df_old = (
    pd.read_excel(FILE_OLD)
      .rename(columns={
          'ID':                     'ID',
          'Representative_New_Capacity':'Per_Turbine_Capacity_kW',
          'Annual_Energy_MWh':      'Annual_Energy_MWh_old',
          'Number of turbines':     'Turbine_Count_old',
          'CapacityFactor':         'CF_old_pct'
      })
      .reset_index(drop=True)
)


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

import matplotlib.ticker as mtick

# F) NPV per MW vs Price
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(sens_df['Price'], sens_df['NPV_per_MW_Rep'], label='Repowering')
ax.plot(sens_df['Price'], sens_df['NPV_per_MW_Dec'], linestyle='--', label='Decommissioning')
ax.set_xlabel('Electricity Price (€/MWh)')
ax.set_ylabel('NPV per MW (€)')
ax.set_title('NPV per MW vs Electricity Price')
ax.legend()
ax.grid(True)

# Turn off scientific notation and use comma separators
ax.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"{int(x):,}"))

plt.tight_layout()
plt.show()


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

# I) Two sample parks metrics vs price (deterministic selection)

# 1) Park where decommissioning > repowering by the largest amount
diff_dec = df['NPV_dec'] - df['NPV_rep']
park_dec_idx = diff_dec.idxmax()

# 2) Park where repowering > decommissioning by the largest amount
diff_rep = df['NPV_rep'] - df['NPV_dec']
park_rep_idx = diff_rep.idxmax()

for idx in [park_dec_idx, park_rep_idx]:
    park = df.loc[idx]  # Series for the chosen park
    prices = np.arange(50, 121, 10)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Base-case LCOE
    rp_base = repowering_metrics(park)
    dp_base = decommissioning_metrics(park)
    rep_lcoe = rp_base['LCOE']
    dec_lcoe = dp_base['LCOE']

    # Prepare lists
    rep_npv, dec_npv = [], []
    rep_irr, dec_irr = [], []
    rep_pb,  dec_pb  = [], []

    for p in prices:
        rp = repowering_metrics(park, price=p)
        dp = decommissioning_metrics(park, price=p)
        rep_npv.append(rp['NPV'])
        dec_npv.append(dp['NPV'])
        rep_irr.append(rp['IRR'])
        dec_irr.append(dp['IRR'])
        rep_pb.append(payback(rp['CAPEX'], rp['Revenue'] - rp['OM']))
        dec_pb.append(payback(dp['CAPEX'], dp['Revenue'] - dp['OM']))

    # NPV vs Price
    axes[0].plot(prices, rep_npv, '-o', label='Repowering')
    axes[0].plot(prices, dec_npv, '-x', label='Decommissioning')
    axes[0].set_title(f"NPV vs Price (Park ID {park['ID']})")
    axes[0].set_xlabel("Price (€/MWh)")
    axes[0].set_ylabel("NPV (€)")
    axes[0].legend()
    axes[0].grid(True)

    # IRR vs Price
    axes[1].plot(prices, rep_irr, '-o', label='Repowering')
    axes[1].plot(prices, dec_irr, '-x', label='Decommissioning')
    axes[1].set_title("IRR vs Price")
    axes[1].set_xlabel("Price (€/MWh)")
    axes[1].set_ylabel("IRR")
    axes[1].legend()
    axes[1].grid(True)

    # LCOE bar
    axes[2].bar(['Repowering', 'Decommissioning'], [rep_lcoe, dec_lcoe],
                color=['skyblue','orange'])
    axes[2].set_title("LCOE at Base Price (€)")
    axes[2].set_ylabel("€/MWh")

    # Payback vs Price
    axes[3].plot(prices, rep_pb, '-o', label='Repowering')
    axes[3].plot(prices, dec_pb, '-x', label='Decommissioning')
    axes[3].set_title("Payback vs Price")
    axes[3].set_xlabel("Price (€/MWh)")
    axes[3].set_ylabel("Payback (years)")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


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

# ─── EXPORT PER-PARK RESULTS TO NEW EXCEL ──────────────────────────────────────
# Ensure CFs are computed
df['CF_old_pct'] = df['Energy_MWh_old'] / (df['Capacity_kW_old']/1000 * 8760) * 100
df['CF_new_pct'] = df['Energy_MWh']     / (df['Capacity_kW']    /1000 * 8760) * 100

# Select the columns you want per park
out_cols = [
    'ID',
    # capacities & yields
    'Capacity_kW', 'Capacity_kW_old',
    'Energy_MWh',  'Energy_MWh_old',
    'CF_new_pct',  'CF_old_pct',
    # repowering metrics
    'CAPEX_rep', 'NPV_rep', 'IRR_rep', 'Payback_rep', 'LCOE_rep',
    # replacement metrics
    'CAPEX_dec', 'NPV_dec', 'IRR_dec', 'Payback_dec', 'LCOE_dec',
]

# Write to a new file
park_results_file = RESULTS_DIR / "per_park_results.xlsx"
df.to_excel(park_results_file, columns=out_cols, index=False)
print(f"Wrote per-park results for both strategies to {park_results_file}")
# ─── (A) SINGLE PANEL: CAPEX BOXPLOT WITH CLAMPED WHISKERS ───────────────────
width = 0.3

capex_rep = df['CAPEX_rep'].dropna().values
capex_dec = df['CAPEX_dec'].dropna().values

def make_box_stat(data):
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo = max(data.min(), q1 - 1.5 * iqr)
    hi = min(data.max(), q3 + 1.5 * iqr)
    return {
        'label':  None,
        'q1':     q1,
        'med':    med,
        'q3':     q3,
        'whislo': lo,
        'whishi': hi,
        'fliers': []  # we’re not plotting individual outliers here
    }

bd_capex = [make_box_stat(capex_rep), make_box_stat(capex_dec)]

fig_capex, ax_capex = plt.subplots(figsize=(8, 6))
bp_capex = ax_capex.bxp(
    bd_capex,
    positions=[1, 1 + width],
    widths=width,
    showfliers=False,
    patch_artist=True
)
for patch, color in zip(bp_capex['boxes'], ['skyblue', 'orange']):
    patch.set_facecolor(color)

ax_capex.set_xticks([1 + width/2])
ax_capex.set_xticklabels(['CAPEX'], fontsize=12)
ax_capex.set_ylabel('CAPEX (€)', fontsize=12)
ax_capex.set_title('CAPEX: Repowering vs. Decommissioning', fontsize=14)

leg_handles = [
    mpatches.Patch(facecolor='skyblue', label='Repowering'),
    mpatches.Patch(facecolor='orange',   label='Decommissioning')
]
ax_capex.legend(handles=leg_handles, title='Scenario', loc='upper left')
ax_capex.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# ─── (B) TWO-PANEL: NPV & IRR (also with clamped whiskers) ───────────────────
npv_rep = df['NPV_rep'].dropna().values
npv_dec = df['NPV_dec'].dropna().values
irr_rep = df['IRR_rep'].dropna().values
irr_dec = df['IRR_dec'].dropna().values

bd_npv = [make_box_stat(npv_rep), make_box_stat(npv_dec)]
bd_irr = [make_box_stat(irr_rep), make_box_stat(irr_dec)]

fig, (ax_npv, ax_irr) = plt.subplots(1, 2, figsize=(16, 6))
width = 0.3

# NPV subplot
bp_npv = ax_npv.bxp(
    bd_npv,
    positions=[1, 1 + width],
    widths=width,
    showfliers=False,
    patch_artist=True
)
for patch, color in zip(bp_npv['boxes'], ['skyblue', 'orange']):
    patch.set_facecolor(color)

ax_npv.set_xticks([1 + width/2])
ax_npv.set_xticklabels(['NPV'], fontsize=12)
ax_npv.set_ylabel('NPV (€)', fontsize=12)
ax_npv.set_title('NPV: Repowering vs. Decommissioning', fontsize=14)
ax_npv.grid(axis='y', linestyle='--', alpha=0.4)
ax_npv.legend(handles=leg_handles, title='Scenario', loc='upper left')

# IRR subplot
bp_irr = ax_irr.bxp(
    bd_irr,
    positions=[1, 1 + width],
    widths=width,
    showfliers=False,
    patch_artist=True
)
for patch, color in zip(bp_irr['boxes'], ['skyblue', 'orange']):
    patch.set_facecolor(color)

ax_irr.set_xticks([1 + width/2])
ax_irr.set_xticklabels(['IRR'], fontsize=12)
ax_irr.set_ylabel('IRR', fontsize=12)
ax_irr.set_title('IRR: Repowering vs. Decommissioning', fontsize=14)
ax_irr.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# Convert to pandas Series (if still as numpy arrays)
capex_rep_series = pd.Series(capex_rep)
capex_dec_series = pd.Series(capex_dec)

def capex_summary(series):
    data = series.dropna()
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    whisker_low  = max(data.min(),       q1 - 1.5 * iqr)
    whisker_high = min(data.max(),       q3 + 1.5 * iqr)
    return {
        'Mean':         data.mean(),
        'Median':       median,
        'Min':          data.min(),
        'Max':          data.max(),
        'Whisker Low':  whisker_low,
        'Whisker High': whisker_high
    }

# Build and display a summary table
summary = pd.DataFrame({
    'Repowering':      capex_summary(capex_rep_series),
    'Decommissioning': capex_summary(capex_dec_series)
}).T

print("\nCAPEX Summary Statistics:\n", summary)