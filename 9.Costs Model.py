import os
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#PARAMETERS
CAPEX_PER_KW = 953.34  # Repowering cost (€/kW)
DECOM_COST_PER_KW = 1267.1  # Decommissioning cost (€/kW)
OM_PER_KW_YEAR = 30.0
LIFETIME = 20
DISCOUNT_RATE = 0.025
DEFAULT_ELEC_PRICE = 80

#DATA IMPORT
excel_file = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\Energy_Yield_Parks.xlsx"
df = pd.read_excel(excel_file)

#CALCULATION FUNCTION
def calc_financials(row, cost_per_kw, discount_rate=DISCOUNT_RATE, lifetime=LIFETIME,
                    default_elec_price=DEFAULT_ELEC_PRICE):
    capacity_kw = row['Recommended_WT_Capacity'] * 1000 * row['New_Turbine_Count']
    annual_energy_kwh = row['Annual_Energy_MWh'] * 1000
    if annual_energy_kwh == 0 or pd.isna(annual_energy_kwh):
        return pd.Series({
            'Total_Capacity_kW': np.nan,
            'CAPEX_Eur': np.nan,
            'Annual_OM_Eur': np.nan,
            'NPV_Eur': np.nan,
            'IRR': np.nan,
            'Annual_Revenue_Eur': np.nan,
            'LCOE_Eur_per_MWh': np.nan
        })
    elec_price = row.get('Electricity_Price', default_elec_price)
    capex = capacity_kw * cost_per_kw
    annual_om = capacity_kw * OM_PER_KW_YEAR
    annual_revenue = (annual_energy_kwh / 1000) * elec_price
    cashflows = [-capex] + [annual_revenue - annual_om] * lifetime
    npv_val = sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cashflows))
    try:
        irr_val = npf.irr(cashflows)
    except:
        irr_val = np.nan
    crf = discount_rate * (1 + discount_rate) ** lifetime / ((1 + discount_rate) ** lifetime - 1)
    lcoe_per_kwh = (capex * crf + annual_om) / annual_energy_kwh
    lcoe_per_mwh = lcoe_per_kwh * 1000
    return pd.Series({
        'Total_Capacity_kW': capacity_kw,
        'CAPEX_Eur': capex,
        'Annual_OM_Eur': annual_om,
        'NPV_Eur': npv_val,
        'IRR': irr_val,
        'Annual_Revenue_Eur': annual_revenue,
        'LCOE_Eur_per_MWh': lcoe_per_mwh
    })

#COMPUTE METRICS
df_rep = df.apply(lambda r: calc_financials(r, CAPEX_PER_KW), axis=1)
df_decom = df.apply(lambda r: calc_financials(r, DECOM_COST_PER_KW), axis=1)
for col in df_rep.columns:
    df[f"{col}_rep"] = df_rep[col]
    df[f"{col}_decom"] = df_decom[col]
if 'Electricity_Price' not in df.columns:
    df['Electricity_Price'] = DEFAULT_ELEC_PRICE

# PLOT A: Sorted NPV Across Parks
sorted_npv = df.sort_values(by='NPV_Eur_rep', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_npv)), sorted_npv['NPV_Eur_rep'], color='skyblue')
plt.xlabel('Wind Parks (sorted by NPV)')
plt.ylabel('NPV (EUR)')
plt.title('NPV Across Wind Parks (Repowering)')
plt.ylim(-0.25e9, 0.75e9)
plt.grid(axis='y')
plt.show()

# PLOT B: Sorted IRR Across Parks
sorted_irr = df.sort_values(by='IRR_rep', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_irr)), sorted_irr['IRR_rep'], color='lightgreen')
plt.xlabel('Wind Parks (sorted by IRR)')
plt.ylabel('IRR')
plt.title('IRR Across Wind Parks (Repowering)')
plt.grid(axis='y')
plt.show()

#PLOT C: Financial Metrics Comparison
def custom_stats(vals):
    arr = np.array(vals)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    return {'q1': q1, 'med': med, 'q3': q3,
            'whislo': q1 - 1.5 * iqr, 'whishi': q3 + 1.5 * iqr,
            'fliers': []}
metrics = ['NPV_Eur', 'CAPEX_Eur', 'Annual_Revenue_Eur']
pos, width, space = 0, 0.3, 3
box_data, positions, colors = [], [], []
for m in metrics:
    rep = custom_stats(df[f"{m}_rep"].dropna()); rep['label'] = f"{m} Repowering"
    dec = custom_stats(df[f"{m}_decom"].dropna()); dec['label'] = f"{m} Decommissioning"
    box_data.extend([rep, dec])
    positions.extend([pos + 1, pos + 1 + width])
    colors.extend(['skyblue', 'orange'])
    pos += space
plt.figure(figsize=(14, 6)); ax = plt.gca()
bp = ax.bxp(box_data, positions=positions, widths=width, showfliers=False, patch_artist=True)
for patch, col in zip(bp['boxes'], colors): patch.set_facecolor(col)
ax.legend([mpatches.Patch(facecolor='skyblue'), mpatches.Patch(facecolor='orange')], ['Repowering', 'Decommissioning'], title='Scenario')
ticks = [i + width for i in np.arange(1, len(metrics) * space, space)]
ax.set_xticks(ticks); ax.set_xticklabels(['NPV', 'CAPEX', 'Revenue'])
ax.set_title('Financial Metrics Comparison'); ax.grid(True); plt.show()

#PLOT D: LCOE Comparison
stats_r = custom_stats(df['LCOE_Eur_per_MWh_rep'].dropna()); stats_r['label']='LCOE Rep'
stats_d = custom_stats(df['LCOE_Eur_per_MWh_decom'].dropna()); stats_d['label']='LCOE Dec'
plt.figure(figsize=(8,6)); ax2=plt.gca()
bp2=ax2.bxp([stats_r,stats_d],positions=[1,1+width],widths=width,showfliers=False,patch_artist=True)
for p,c in zip(bp2['boxes'],['skyblue','orange']): p.set_facecolor(c)
ax2.legend([mpatches.Patch(facecolor='skyblue'),mpatches.Patch(facecolor='orange')],['Repowering','Decommissioning'],title='Scenario')
ax2.set_xticks([1+width/2]); ax2.set_xticklabels(['LCOE (€/MWh)']); ax2.set_title('LCOE Comparison'); ax2.grid(True); plt.show()

# PAYBACK PERIOD AND SORTED PLOT
def calc_payback(cost, benefit, life):
    cf = np.concatenate(([-cost], np.repeat(benefit, life)))
    cum = np.cumsum(cf)
    years = np.arange(0, life+1)
    return next((y for y,v in zip(years,cum) if v>=0), np.nan)
df['Payback_rep'] = df.apply(lambda r: calc_payback(r['Total_Capacity_kW_rep']*CAPEX_PER_KW,
                                                      r['Annual_Revenue_Eur_rep']-r['Total_Capacity_kW_rep']*OM_PER_KW_YEAR,
                                                      LIFETIME), axis=1)
df['Payback_decom'] = df.apply(lambda r: calc_payback(r['Total_Capacity_kW_decom']*DECOM_COST_PER_KW,
                                                        r['Annual_Revenue_Eur_decom']-r['Total_Capacity_kW_decom']*OM_PER_KW_YEAR,
                                                        LIFETIME), axis=1)
# Sort by repowering payback ascending
df_sorted_pb = df.sort_values(by='Payback_rep')
plt.figure(figsize=(10,6))
idx = np.arange(len(df_sorted_pb))
plt.plot(idx, df_sorted_pb['Payback_rep'], marker='o', linestyle='-', label='Repowering')
plt.plot(idx, df_sorted_pb['Payback_decom'], marker='x', linestyle='--', label='Decommissioning')
plt.xlabel('Parks sorted by Repowering Payback')
plt.ylabel('Payback Period (years)')
plt.title('Sorted Payback Period Comparison')
plt.legend(); plt.grid(True); plt.show()


#SENSITIVITY ANALYSIS PER MW
prices = np.arange(50, 121, 10)  # €50 to €120 by €10
records = []

for price in prices:
    df_temp = df.copy()
    df_temp['Electricity_Price'] = price

    # repowering metrics per park
    rep = df_temp.apply(lambda r: calc_financials(r, CAPEX_PER_KW, default_elec_price=price), axis=1)
    # extract per‐park capacity in MW
    cap_mw_rep = rep['Total_Capacity_kW'] / 1000.0

    # aggregated NPV per MW (€/MW)
    total_npv_rep = rep['NPV_Eur'].sum()
    total_cap_rep = cap_mw_rep.sum()
    npv_per_mw_rep = total_npv_rep / total_cap_rep

    # capacity-weighted IRR
    irr_wgt_rep = (rep['IRR'] * cap_mw_rep).sum() / total_cap_rep

    # share of capacity with positive NPV
    pctpos_mw_rep = cap_mw_rep[rep['NPV_Eur'] > 0].sum() / total_cap_rep * 100

    # same for decommissioning
    dec = df_temp.apply(lambda r: calc_financials(r, DECOM_COST_PER_KW, default_elec_price=price), axis=1)
    cap_mw_dec = dec['Total_Capacity_kW'] / 1000.0

    total_npv_dec = dec['NPV_Eur'].sum()
    total_cap_dec = cap_mw_dec.sum()
    npv_per_mw_dec = total_npv_dec / total_cap_dec

    irr_wgt_dec = (dec['IRR'] * cap_mw_dec).sum() / total_cap_dec
    pctpos_mw_dec = cap_mw_dec[dec['NPV_Eur'] > 0].sum() / total_cap_dec * 100

    records.append({
        'Price_€/MWh': price,
        'NPV_per_MW_Rep': npv_per_mw_rep,
        'IRR_wgt_Rep': irr_wgt_rep,
        '%CapPosNPV_Rep': pctpos_mw_rep,
        'NPV_per_MW_Dec': npv_per_mw_dec,
        'IRR_wgt_Dec': irr_wgt_dec,
        '%CapPosNPV_Dec': pctpos_mw_dec,
    })

sens_mw_df = pd.DataFrame(records)


plt.plot(sens_mw_df['Price_€/MWh'], sens_mw_df['NPV_per_MW_Rep'], label='Repowering')
plt.plot(sens_mw_df['Price_€/MWh'], sens_mw_df['NPV_per_MW_Dec'], linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('NPV per MW (€)')
plt.title('NPV per MW vs Electricity Price')
plt.legend(); plt.grid(True)
plt.show()

#PLOT: Weighted IRR vs Pric
plt.figure(figsize=(8,5))
plt.plot(sens_mw_df['Price_€/MWh'], sens_mw_df['IRR_wgt_Rep'], label='Repowering')
plt.plot(sens_mw_df['Price_€/MWh'], sens_mw_df['IRR_wgt_Dec'], linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('Capacity-Weighted IRR')
plt.title('IRR (weighted by capacity) vs Electricity Price')
plt.legend(); plt.grid(True)
plt.show()

#PLOT: % of Capacity with NPV>0 vs Price
plt.figure(figsize=(8,5))
plt.plot(sens_mw_df['Price_€/MWh'], sens_mw_df['%CapPosNPV_Rep'], marker='o', label='Repowering')
plt.plot(sens_mw_df['Price_€/MWh'], sens_mw_df['%CapPosNPV_Dec'], marker='x', linestyle='--', label='Decommissioning')
plt.xlabel('Electricity Price (€/MWh)')
plt.ylabel('% of Total Capacity with NPV > 0')
plt.title('% of Capacity In‐The‐Money vs Electricity Price')
plt.legend(); plt.grid(True)
plt.show()

#SHOW TABLE
print(sens_mw_df.to_string(index=False))

#SUMMARY STATISTICS

# 1. Global percentages (repowering)
pct_neg_npv = (df['NPV_Eur_rep'] < 0).mean() * 100
pct_neg_irr = (df['IRR_rep'] < 0).mean() * 100
print(f"Global % of parks with negative NPV (repowering): {pct_neg_npv:.1f}%")
print(f"Global % of parks with negative IRR (repowering): {pct_neg_irr:.1f}%\n")

# 2. Per‐country aggregation
country_stats = df.groupby('Country').agg(
    n_parks               = ('NPV_Eur_rep',    'count'),
    pct_neg_npv           = ('NPV_Eur_rep',    lambda x: (x < 0).mean() * 100),
    pct_neg_irr           = ('IRR_rep',        lambda x: (x < 0).mean() * 100),
    avg_npv               = ('NPV_Eur_rep',    'mean'),
    avg_irr               = ('IRR_rep',        'mean'),
    avg_capex             = ('CAPEX_Eur_rep',  'mean'),
    avg_om                = ('Annual_OM_Eur_rep',  'mean'),
    avg_revenue           = ('Annual_Revenue_Eur_rep', 'mean'),
    avg_lcoe              = ('LCOE_Eur_per_MWh_rep',   'mean'),
    avg_payback           = ('Payback_rep',    'mean')
).reset_index()

print("Per-country financial summary (repowering):")
print(country_stats.to_string(index=False,
    formatters={
        'pct_neg_npv': "{:.1f}%".format,
        'pct_neg_irr': "{:.1f}%".format,
        'avg_npv': "{:,.0f} €".format,
        'avg_irr': "{:.2%}".format,
        'avg_capex': "{:,.0f} €".format,
        'avg_om': "{:,.0f} €".format,
        'avg_revenue': "{:,.0f} €".format,
        'avg_lcoe': "{:.1f} €/MWh".format,
        'avg_payback': "{:.1f} yrs".format
    }
))
print()

# 3. Overall averages across all countries
metrics = ['avg_npv','avg_irr','avg_capex','avg_om','avg_revenue','avg_lcoe','avg_payback']
overall = country_stats[metrics].mean()
print("Overall average metrics (across countries):")
print(f"  NPV:         {overall['avg_npv']:,.0f} €")
print(f"  IRR:         {overall['avg_irr']:.2%}")
print(f"  CAPEX:       {overall['avg_capex']:,.0f} €")
print(f"  O&M:         {overall['avg_om']:,.0f} €/yr")
print(f"  Revenue:     {overall['avg_revenue']:,.0f} €/yr")
print(f"  LCOE:        {overall['avg_lcoe']:.1f} €/MWh")
print(f"  Payback:     {overall['avg_payback']:.1f} yrs")

# TWO RANDOM PARKS: 2×2 LINE & BAR PLOTS VS PRICE

import numpy as np
import matplotlib.pyplot as plt

# 1. pick two random parks with positive repowering NPV
positive_df = df[df['NPV_Eur_rep'] > 0]
parks = positive_df.sample(2, random_state=42)

prices = np.arange(50, 121, 10)

for _, park in parks.iterrows():
    country = park['Country']
    lon, lat = park['Longitude'], park['Latitude']

    # compute constant LCOE for rep & dec
    lcoe_rep = calc_financials(park, CAPEX_PER_KW)['LCOE_Eur_per_MWh']
    lcoe_dec = calc_financials(park, DECOM_COST_PER_KW)['LCOE_Eur_per_MWh']

    # prepare container for metrics
    results = {
        'NPV (EUR)': {'rep': [], 'dec': []},
        'IRR': {'rep': [], 'dec': []},
        'LCOE (€/MWh)': {'rep': [lcoe_rep] * len(prices), 'dec': [lcoe_dec] * len(prices)},
        'Payback (yrs)': {'rep': [], 'dec': []},
    }

    # sweep over prices
    for price in prices:
        row = park.copy()
        row['Electricity_Price'] = price

        rep = calc_financials(row, CAPEX_PER_KW, default_elec_price=price)
        dec = calc_financials(row, DECOM_COST_PER_KW, default_elec_price=price)

        # NPV & IRR
        results['NPV (EUR)']['rep'].append(rep['NPV_Eur'])
        results['NPV (EUR)']['dec'].append(dec['NPV_Eur'])
        results['IRR']['rep'].append(rep['IRR'])
        results['IRR']['dec'].append(dec['IRR'])

        # Payback
        pay_rep = calc_payback(
            rep['Total_Capacity_kW'] * CAPEX_PER_KW,
            rep['Annual_Revenue_Eur'] - rep['Total_Capacity_kW'] * OM_PER_KW_YEAR,
            LIFETIME
        )
        pay_dec = calc_payback(
            dec['Total_Capacity_kW'] * DECOM_COST_PER_KW,
            dec['Annual_Revenue_Eur'] - dec['Total_Capacity_kW'] * OM_PER_KW_YEAR,
            LIFETIME
        )
        results['Payback (yrs)']['rep'].append(pay_rep)
        results['Payback (yrs)']['dec'].append(pay_dec)

    # plot 2×2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (metric, data) in zip(axes, results.items()):
        if metric == 'LCOE (€/MWh)':
            # bar chart for constant LCOE
            ax.bar(['Repowering', 'Decommissioning'], [lcoe_rep, lcoe_dec],
                   color=['skyblue', 'orange'])
        else:
            # line plots vs price
            ax.plot(prices, data['rep'], marker='o', label='Repowering')
            ax.plot(prices, data['dec'], marker='x', linestyle='--', label='Decommissioning')
        ax.set_title(f"{metric}\n{country} (Lon {lon:.2f}, Lat {lat:.2f})")
        ax.set_xlabel('Electricity Price (€/MWh)' if metric != 'LCOE (€/MWh)' else '')
        if 'IRR' in metric:
            ax.set_ylabel('IRR')
        elif 'NPV' in metric:
            ax.set_ylabel('€')
        elif 'LCOE' in metric:
            ax.set_ylabel('€/MWh')
        else:
            ax.set_ylabel('Years')
        ax.grid(True)
        ax.legend() if metric != 'LCOE (€/MWh)' else None

    fig.suptitle(f"Financial Metrics vs Price\nfor Park in {country} ({lon:.2f}, {lat:.2f})",
                 fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#ADDITIONAL “TYPICAL” PLOTS

# 1. Scatter: LCOE vs Capacity Factor (Repowering)
df['Capacity_MW_rep'] = df['Total_Capacity_kW_rep'] / 1000
df['Capacity_Factor_%'] = df['Annual_Energy_MWh'] / (df['Capacity_MW_rep'] * 8760) * 100

plt.figure(figsize=(8, 6))
plt.scatter(df['Capacity_Factor_%'], df['LCOE_Eur_per_MWh_rep'],
            alpha=0.7, edgecolor='k')
plt.xlabel('Capacity Factor (%)')
plt.ylabel('LCOE (€/MWh)')
plt.title('LCOE vs Capacity Factor (Repowering)')
plt.grid(True)
plt.show()

# 2. Pie: CAPEX vs. PV of O&M for the first random park
park1 = parks.iloc[0]
rep1 = calc_financials(park1, CAPEX_PER_KW)
capex1 = rep1['CAPEX_Eur']
annual_om1 = rep1['Annual_OM_Eur']
# PV of O&M over lifetime
pv_om1 = annual_om1 * (1 - (1 + DISCOUNT_RATE) ** (-LIFETIME)) / DISCOUNT_RATE

labels = ['CAPEX', 'PV of O&M']
sizes = [capex1, pv_om1]
explode = (0.1, 0.0)

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%',
        startangle=90, shadow=True)
plt.title(f'CAPEX vs. PV(O&M) (Repowering)\n'
          f'{park1["Country"]} ({park1["Longitude"]:.2f}, {park1["Latitude"]:.2f})')
plt.show()

import plotly.express as px

# Ensure you already have df with Longitude, Latitude, IRR_rep, Payback_rep
df['Fast_Payback'] = df['Payback_rep'] < 20

fig = px.scatter_geo(
    df,
    lon='Longitude',
    lat='Latitude',
    color='IRR_rep',
    color_continuous_scale='RdYlGn',
    hover_name='Country',
    hover_data={'IRR_rep': True, 'Payback_rep': True},
    scope='europe',
    projection='natural earth',
    opacity=0.6,
    title='Repowering IRR across Europe<br>(Outlined = Payback < 20 yrs)',
)

# Outline fast-payback parks
fig.update_traces(
    marker=dict(
        size=8,
        line=dict(width=1, color=['black' if fp else 'rgba(0,0,0,0)' for fp in df['Fast_Payback']])
    )
)

fig.update_layout(
    coloraxis_colorbar=dict(title="Repowering IRR"),
    margin=dict(l=0, r=0, t=50, b=0)
)

# Save the interactive HTML
output_path = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results\irr_map.html"
fig.write_html(output_path)
print(f"Interactive map exported to {output_path}")


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# -- prepare the two panels --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# common layout parameters
width = 0.3
space = 3

# PANEL 1: NPV & CAPEX
metrics1 = ['NPV_Eur', 'CAPEX_Eur']
box_data, positions, colors = [], [], []
pos = 0
for m in metrics1:
    rep = custom_stats(df[f"{m}_rep"].dropna());  rep['label'] = f"{m} Repowering"
    dec = custom_stats(df[f"{m}_decom"].dropna()); dec['label'] = f"{m} Decommissioning"
    box_data.extend([rep, dec])
    positions.extend([pos + 1, pos + 1 + width])
    colors.extend(['skyblue', 'orange'])
    pos += space

bp1 = ax1.bxp(box_data,
              positions=positions,
              widths=width,
              showfliers=False,
              patch_artist=True)
for patch, col in zip(bp1['boxes'], colors):
    patch.set_facecolor(col)

ax1.set_xticks([i + width for i in np.arange(1, len(metrics1)*space, space)])
ax1.set_xticklabels(['NPV', 'CAPEX'])
ax1.set_title('NPV & CAPEX Comparison')
ax1.legend([mpatches.Patch(facecolor='skyblue'),
            mpatches.Patch(facecolor='orange')],
           ['Repowering', 'Decommissioning'],
           title='Scenario')
ax1.grid(True)


# PANEL 2: IRR
metrics2 = ['IRR']
box_data, positions, colors = [], [], []
pos = 0
for m in metrics2:
    rep = custom_stats(df[f"{m}_rep"].dropna());  rep['label'] = f"{m} Repowering"
    dec = custom_stats(df[f"{m}_decom"].dropna()); dec['label'] = f"{m} Decommissioning"
    box_data.extend([rep, dec])
    positions.extend([pos + 1, pos + 1 + width])
    colors.extend(['skyblue', 'orange'])
    pos += space

bp2 = ax2.bxp(box_data,
              positions=positions,
              widths=width,
              showfliers=False,
              patch_artist=True)
for patch, col in zip(bp2['boxes'], colors):
    patch.set_facecolor(col)

# single tick in the centre
ax2.set_xticks([1 + width/2])
ax2.set_xticklabels(['IRR'])
ax2.set_title('IRR Comparison')
ax2.legend([mpatches.Patch(facecolor='skyblue'),
            mpatches.Patch(facecolor='orange')],
           ['Repowering', 'Decommissioning'],
           title='Scenario')
ax2.grid(True)


plt.tight_layout()
plt.show()
# --- UPDATED SUMMARY STATISTICS FOR LAST PLOT METRICS (INCLUDING LCOE) ---
stats = {
    'NPV Repowering':           df['NPV_Eur_rep'],
    'NPV Decommissioning':      df['NPV_Eur_decom'],
    'CAPEX Repowering':         df['CAPEX_Eur_rep'],
    'CAPEX Decommissioning':    df['CAPEX_Eur_decom'],
    'IRR Repowering':           df['IRR_rep'],
    'IRR Decommissioning':      df['IRR_decom'],
    'LCOE Repowering (€/MWh)':  df['LCOE_Eur_per_MWh_rep'],
    'LCOE Decommissioning (€/MWh)': df['LCOE_Eur_per_MWh_decom']
}

print("\nSummary statistics for final plot metrics:")
for name, series in stats.items():
    s = series.dropna()
    print(
        f"{name:35s}"
        f" max = {s.max():,.2f}"
        f" min = {s.min():,.2f}"
        f" mean = {s.mean():,.2f}"
        f" median = {s.median():,.2f}"
    )

# --- OVERALL PERCENTAGE INCREASE FROM DECOMMISSIONING TO REPOWERING ---
metrics = [
    ('NPV',            'NPV_Eur'),
    ('CAPEX',          'CAPEX_Eur'),
    ('IRR',            'IRR'),
    ('LCOE (€/MWh)',   'LCOE_Eur_per_MWh')
]

print("\nOverall % increase (Repowering vs Decommissioning):")
for label, col in metrics:
    mean_rep = df[f"{col}_rep"].mean()
    mean_dec = df[f"{col}_decom"].mean()
    # avoid division by zero
    if mean_dec != 0:
        pct_inc = (mean_rep - mean_dec) / abs(mean_dec) * 100
        print(f"{label:15s}  {pct_inc:6.2f}%")
    else:
        print(f"{label:15s}  cannot compute (mean_decommissioning = 0)")
