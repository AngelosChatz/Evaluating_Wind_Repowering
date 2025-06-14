import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# --- Global plotting styles ---
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'

# --- Approach labels and colors ---
APP_LABELS = {
    0: 'Approach 0-Old (Decomm+Repl)',
    1: 'Approach 1-Power Density',
    2: 'Approach 2-Capacity Maximization',
    3: 'Approach 3-Rounding Up',
    4: 'Approach 4-Single Turbine Flex',
    5: 'Approach 5-No-Loss Hybrid (Cap-based)',
    6: 'Approach 6-No-Loss Hybrid (Yield-based)',
}

APP_COLORS = {
    APP_LABELS[0]: 'black',
    APP_LABELS[1]: 'blue',
    APP_LABELS[2]: 'orange',
    APP_LABELS[3]: 'green',
    APP_LABELS[4]: 'red',
    APP_LABELS[5]: 'brown',
    APP_LABELS[6]: 'purple',
}

COLOR_MARKER = 'red'

# --- Helpers ---
def normalize_cols(df):
    df.columns = df.columns.str.strip().str.replace('²', '2', regex=False)
    return df

def detect_col(df, keyword):
    key = keyword.lower()
    for c in df.columns:
        norm = c.lower().replace('_', ' ')
        norm = pd.Series([norm]).str.replace(r"\s*\(.*\)", '', regex=True)[0].strip()
        if key in norm:
            return c
    raise KeyError(f"No column containing '{keyword}'")

def load_and_clean(path):
    df = pd.read_excel(path).replace({'#ND': np.nan, '': np.nan})
    df = normalize_cols(df)
    for kw in ["total power", "park area", "total new capacity", "new total park area"]:
        try:
            c = detect_col(df, kw)
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except KeyError:
            pass
    for kw in ["commissioning date", "decommissioning date"]:
        try:
            c = detect_col(df, kw)
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except KeyError:
            pass
    return df

# --- Load data for Approaches 1–6 ---
base = Path(__file__).resolve().parent / "results"
filenames = {
    1: "Approach_1.xlsx",
    2: "Approach_2.xlsx",
    3: "Approach_3.xlsx",
    4: "Approach_4.xlsx",
    5: "Approach_5.xlsx",
    6: "Approach_6_No_Loss_Hybrid_Yield_based.xlsx",
}
dfs = {i: load_and_clean(base / fn) for i, fn in filenames.items()}

# Pre-detect key columns
turb_col    = detect_col(dfs[1], "number of turbines")
pow_col     = detect_col(dfs[1], "total power")
area_col    = detect_col(dfs[1], "park area")
newcap_col  = detect_col(dfs[2], "total new capacity")
newarea_col = detect_col(dfs[2], "new total park area")
comm_col    = detect_col(dfs[1], "commissioning date")
decomm_col  = detect_col(dfs[1], "decommissioning date")

# --- SECTION 3: Power Density – Approach 1 vs Approaches 2–6 ---
df1 = dfs[1]
base_df = (
    df1[df1[pow_col] > 0]
       .groupby("Country")
       .agg(tp=(pow_col, 'sum'), ta=(area_col, 'sum'))
       .assign(density=lambda d: d['tp']/(d['ta']/1e6))
)
all_d = base_df[['density']].rename(columns={'density': APP_LABELS[1]}).copy()
for i in range(2, 7):
    rep = dfs[i][dfs[i][newcap_col] > 0]
    tmp = (
        rep.groupby("Country")
           .agg(nc=(newcap_col,'sum'), na=(newarea_col,'sum'))
           .assign(**{APP_LABELS[i]: lambda d, _i=i: d['nc']/(d['na']/1e6)})
    )
    all_d = all_d.join(tmp[[APP_LABELS[i]]], how='outer')
all_d.fillna(0, inplace=True)
all_d.sort_values(APP_LABELS[1], ascending=False, inplace=True)

plt.figure(figsize=(14, 8))
idx = np.arange(len(all_d))
labels = [APP_LABELS[1]] + [APP_LABELS[i] for i in range(2, 7)]
w = 0.12
for j, lbl in enumerate(labels):
    plt.bar(
        idx + (j - (len(labels)-1)/2)*w,
        all_d[lbl],
        w,
        color=APP_COLORS[lbl],
        label=lbl
    )
plt.xticks(idx, all_d.index, rotation=90, ha='center')
plt.title("Power Density: Approach 1 vs Approaches 2–6", fontsize=16)
plt.xlabel("Country")
plt.ylabel("MW/km²")
plt.legend(ncol=1, loc='upper right')
plt.tight_layout()
plt.savefig("Power Density: Approach 1 vs Approaches 2–6", dpi=300, bbox_inches="tight")

plt.show()

# --- SECTION 4: Land Area Comparison (Approaches 1–6) ---
def compute_country_capacities(df):
    bc, rc = {}, {}
    for _, r in df.iterrows():
        c = r['Country']
        if pd.notna(r[comm_col]) and pd.notna(r[pow_col]):
            start = r[comm_col].year
            end = r[decomm_col].year if pd.notna(r[decomm_col]) else start + 30
            bc.setdefault(c, []).extend([(start, r[pow_col]), (end, -r[pow_col])])
        if pd.notna(r[comm_col]) and pd.notna(r[newcap_col]):
            start2 = (r[decomm_col].year if pd.notna(r[decomm_col]) else r[comm_col].year + 20)
            cap2 = r[newcap_col]
            y = start2
            while y <= 2050:
                rc.setdefault(c, []).extend([(y, cap2), (y+20, -cap2)])
                y += 20
    records = []
    for c in set(df['Country']):
        hist, run = {}, 0
        for y in range(1980, 2023):
            for yr, ch in bc.get(c, []):
                if yr == y: run += ch
            hist[y] = run
        trend = ((hist[2022] - hist.get(2000, 0)) / 22) if 2000 in hist else 0
        for y in range(2023, 2051):
            hist[y] = hist[2022] + trend * (y - 2022)
        base2050 = hist[2050]
        run2, hist2 = 0, {}
        for y in range(1980, 2051):
            for yr, ch in rc.get(c, []):
                if yr == y: run2 += ch
            hist2[y] = run2
        records.append({
            'Country': c,
            'Baseline_2050': base2050,
            'Combined_2050': base2050 + hist2[2050]
        })
    return pd.DataFrame(records)

def compute_land_area(df):
    caps = compute_country_capacities(df)
    caps['Repowered'] = caps['Combined_2050'] - caps['Baseline_2050']
    orig = (
        df[df[pow_col] > 0]
          .groupby('Country')
          .agg(tp=(pow_col,'sum'), ta=(area_col,'sum'))
          .assign(d=lambda d: d['tp']/(d['ta']/1e6))
          .reset_index()
    )
    merged = caps.merge(orig[['Country','d']], on='Country', how='left')
    merged['BaseArea'] = merged['Baseline_2050'] / merged['d']
    merged['ReArea']   = merged['Repowered'] / merged['d']
    return merged[['Country','BaseArea','ReArea']]

land_tables = [compute_land_area(dfs[i]).rename(columns={'BaseArea': f'B{i}', 'ReArea': f'R{i}'}) for i in range(1,7)]
ld = land_tables[0]
for t in land_tables[1:]:
    ld = ld.merge(t, on='Country', how='outer')
ld.fillna(0, inplace=True)
ld['TotalReq'] = ld[[c for c in ld.columns if c.startswith(('B','R'))]].sum(axis=1)
ld.sort_values('TotalReq', ascending=False, inplace=True)

plt.figure(figsize=(14, 8))
idx = np.arange(len(ld))
w = 0.15
for i in range(1, 7):
    plt.bar(idx + (i-4)*w, ld[f'B{i}'], w, color='lightgray')
    plt.bar(
        idx + (i-4)*w,
        ld[f'R{i}'],
        w,
        bottom=ld[f'B{i}'],
        color=APP_COLORS[APP_LABELS[i]],
        label=APP_LABELS[i]
    )
plt.xticks(idx, ld['Country'], rotation=90, ha='center')
plt.title("Required Land Area Comparison (Approaches 1–6)", fontsize=16)
plt.xlabel("Country")
plt.ylabel("Area km²")
plt.legend(ncol=1, loc='upper right')
plt.tight_layout()
plt.savefig("Required Land Area Comparison (Approaches 1–6)", dpi=300, bbox_inches="tight")
plt.show()

# --- SECTION 5: Saved Land Area (table) ---
saved = {i: ld[f'R{i}'].sum() for i in range(1,7)}
saved_df = pd.Series(saved, name='Saved_km²').rename_axis('Approach').reset_index()
print("\n--- Section 5: Total km² Saved by Repowering ---")
print(saved_df.to_string(index=False, float_format="{:.2f}".format))

# --- SECTION 6: Average Density – table ---
stats = []
for i, df in dfs.items():
    p0 = df[pow_col].sum()
    a0 = df[area_col].sum() / 1e6
    d0 = p0/a0 if a0 else np.nan
    if newcap_col in df.columns and newarea_col in df.columns:
        p1 = df[newcap_col].sum()
        a1 = df[newarea_col].sum() / 1e6
        d1 = p1/a1 if a1 else np.nan
    else:
        d1 = np.nan
    stats.append({
        'Approach': APP_LABELS.get(i, f'Approach {i}'),
        'Baseline_MW/km²': d0,
        'Repowered_MW/km²': d1
    })
avg_df = pd.DataFrame(stats)
print("\n--- Section 6: Average Power Density per Approach ---")
print(avg_df.to_string(index=False, float_format="{:.2f}".format))

# --- common setup for both plots ---
approaches = [2, 6]
df_ref = dfs[2]
total  = df_ref.groupby('Country').size()
single = df_ref[df_ref[turb_col]==1].groupby('Country').size()
share  = (single/total*100).fillna(0)

df_ref2 = df_ref.dropna(subset=[comm_col]).copy()
df_ref2['Age'] = datetime.date.today().year - df_ref2[comm_col].dt.year
age    = df_ref2.groupby('Country')['Age'].mean().fillna(0)

BAR_COLORS = {2: APP_COLORS[APP_LABELS[2]], 6: APP_COLORS[APP_LABELS[6]]}

def compute_delta(a, order):
    df = dfs[a]
    orig = (df[df[pow_col]>0]
            .groupby('Country')
            .agg(tp=(pow_col,'sum'), ta=(area_col,'sum'))
            .assign(orig_density=lambda d: d['tp']/(d['ta']/1e6)))
    rep  = (df[df[newcap_col]>0]
            .groupby('Country')
            .agg(cap=(newcap_col,'sum'), area=(newarea_col,'sum'))
            .assign(rep_density=lambda d: d['cap']/(d['area']/1e6)))
    cmp  = orig[['orig_density']].join(rep['rep_density'], how='outer').fillna(0)
    return (cmp['rep_density'] - cmp['orig_density']).reindex(order)

# --- SECTION 7: ΔDensity vs Single‐Turbine Share ---
order7 = share.sort_values().index.tolist()
x7 = np.arange(len(order7))
width = 0.3

fig, ax1 = plt.subplots(figsize=(12,8))
for i in approaches:
    delta = compute_delta(i, order7)
    ax1.bar(
        x7 + (approaches.index(i) - 0.5)*width,
        delta,
        width,
        color=BAR_COLORS[i],
        label=f'ΔDensity {APP_LABELS[i]}'
    )
ax1.set_xticks(x7)
ax1.set_xticklabels(order7, rotation=90, ha='center')
ax1.set_xlabel("Country")
ax1.set_ylabel("Δ Density (MW/km²)")
ax1.set_title("Approaches 2 & 6: ΔDensity vs Single‐Turbine Share")

ax2 = ax1.twinx()
share_sorted = share.reindex(order7)
ax2.plot(x7, share_sorted, 'o-', color=COLOR_MARKER, label='Single‐Turbine Share')
ax2.set_ylabel("Single‐Turbine Parks (%)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=1, loc='upper left')
fig.tight_layout()
plt.savefig("Approaches 2 & 6: ΔDensity vs Single‐Turbine Share.png", dpi=300, bbox_inches="tight")
plt.show()

# --- SECTION 11: ΔDensity vs Avg Turbine Age ---
order11 = age.sort_values().index.tolist()
x11 = np.arange(len(order11))

fig, ax1 = plt.subplots(figsize=(12,8))
for i in approaches:
    delta = compute_delta(i, order11)
    ax1.bar(
        x11 + (approaches.index(i) - 0.5)*width,
        delta,
        width,
        color=BAR_COLORS[i],
        label=f'ΔDensity {APP_LABELS[i]}'
    )
ax1.set_xticks(x11)
ax1.set_xticklabels(order11, rotation=90, ha='center')
ax1.set_xlabel("Country")
ax1.set_ylabel("Δ Density (MW/km²)")
ax1.set_title("Approaches 2 & 6: ΔDensity vs Avg Turbine Age")

ax2 = ax1.twinx()
age_sorted = age.reindex(order11)
ax2.plot(x11, age_sorted, 'o-', color=COLOR_MARKER, label='Avg Turbine Age')
ax2.set_ylabel("Avg Turbine Age (years)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=1, loc='upper left')
fig.tight_layout()
plt.savefig("DDensity vs. age .png", dpi=300, bbox_inches="tight")
plt.show()


# --- SECTION 12: Global land‐use summary for Approaches 1–6 ---

# 1. Compute global baseline total (Approach 1)
baseline_total = (ld['B1'] + ld['R1']).sum()

# 2. Build summary for each approach
summary = []
for i in range(1, 7):
    total_i = (ld[f'B{i}'] + ld[f'R{i}']).sum()
    diff_i  = total_i - baseline_total
    pct_i   = (diff_i / baseline_total * 100) if baseline_total else np.nan
    summary.append({
        'Approach': APP_LABELS[i],
        'Total_Land_km²': total_i,
        'Δ_vs_Baseline_km²': diff_i,
        '% Expansion vs Baseline': pct_i
    })

# 3. Create and print the DataFrame
summary_df = pd.DataFrame(summary)
print("\n--- Global Land Use Summary by Approach (km²) ---")
print(summary_df.to_string(index=False, float_format="{:.1f}".format))
