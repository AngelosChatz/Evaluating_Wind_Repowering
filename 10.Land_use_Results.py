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

# Color definitions
COLOR_ORIG   = 'tab:blue'
COLOR_REP    = 'tab:orange'
COLOR_INC    = 'tab:green'
COLOR_MARKER = 'red'
COLOR_AGE    = 'tab:purple'
COLOR_HYB    = 'tab:red'  # new scenario color for Approach 6
APP_COLORS   = {
    'Original': COLOR_ORIG,
    'Approach 2': COLOR_REP,
    'Approach 3': COLOR_AGE,
    'Approach 4': COLOR_INC,
    'Approach 5': COLOR_AGE,
    'Approach 6': COLOR_HYB,
}

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
dfs = {i: load_and_clean(base / fname) for i, fname in filenames.items()}

# Pre-detect key columns from Approach 1
turb_col    = detect_col(dfs[1], "number of turbines")
pow_col     = detect_col(dfs[1], "total power")
area_col    = detect_col(dfs[1], "park area")
newcap_col  = detect_col(dfs[2], "total new capacity")
newarea_col = detect_col(dfs[2], "new total park area")
comm_col    = detect_col(dfs[1], "commissioning date")
decomm_col  = detect_col(dfs[1], "decommissioning date")

# --- SECTION 3: Power Density – Original vs Approaches 2–6 ---
df1 = dfs[1]
base_df = (
    df1[df1[pow_col] > 0]
       .groupby("Country")
       .agg(tp=(pow_col, 'sum'), ta=(area_col, 'sum'))
       .assign(orig_density=lambda d: d['tp']/(d['ta']/1e6))
)
all_d = base_df[['orig_density']].copy()
for i in range(2, 7):
    rep = dfs[i][dfs[i][newcap_col] > 0]
    tmp = (
        rep.groupby("Country")
           .agg(nc=(newcap_col,'sum'), na=(newarea_col,'sum'))
           .assign(**{f'rep_density_{i}': lambda d: d['nc']/(d['na']/1e6)})
    )
    all_d = all_d.join(tmp[f'rep_density_{i}'], how='outer')
all_d.fillna(0, inplace=True)
all_d.sort_values('orig_density', ascending=False, inplace=True)

plt.figure(figsize=(14, 8))
idx = np.arange(len(all_d))
labels = ['Original'] + [f'Approach {i}' for i in range(2, 7)]
w = 0.12
for j, lbl in enumerate(labels):
    col = 'orig_density' if j == 0 else f'rep_density_{j+1}'
    plt.bar(idx + (j-(len(labels)-1)/2)*w, all_d[col], w, color=APP_COLORS[lbl], label=lbl)
plt.xticks(idx, all_d.index, rotation=45, ha='right')
plt.title("Power Density: Original vs Approaches 2–6", fontsize=16)
plt.xlabel("Country"); plt.ylabel("MW/km²"); plt.legend(ncol=3); plt.tight_layout(); plt.show()

# --- SECTION 4: Land Area Comparison (Approaches 1–6) ---
def compute_country_capacities(df):
    rec_list, bc, rc = [], {}, {}
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
        run2 = 0
        hist2 = {}
        for y in range(1980, 2051):
            for yr, ch in rc.get(c, []):
                if yr == y: run2 += ch
            hist2[y] = run2
        records.append({'Country': c, 'Baseline_2050': base2050, 'Combined_2050': base2050 + hist2[2050]})
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
idx = np.arange(len(ld)); w = 0.15
for i in range(1, 7):
    plt.bar(idx + (i-4)*w, ld[f'B{i}'], w, color='lightgray')
    plt.bar(idx + (i-4)*w, ld[f'R{i}'], w, bottom=ld[f'B{i}'], label=f'Approach {i}')
plt.xticks(idx, ld['Country'], rotation=45, ha='right')
plt.title("Required Land Area Comparison (Approaches 1–6)", fontsize=16)
plt.xlabel("Country"); plt.ylabel("Area km²"); plt.legend(ncol=3); plt.tight_layout(); plt.show()

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
    stats.append({'Approach': i, 'Baseline_MW/km²': d0, 'Repowered_MW/km²': d1})
avg_df = pd.DataFrame(stats)
print("\n--- Section 6: Average Power Density per Approach ---")
print(avg_df.to_string(index=False, float_format="{:.2f}".format))

# --- common setup for both plots ---
approaches = [2, 6]
# compute single‐turbine share & avg age once (they’re the same)
df_ref = dfs[2]
total  = df_ref.groupby('Country').size()
single = df_ref[df_ref[turb_col]==1].groupby('Country').size()
share  = (single/total*100).fillna(0)

df_ref2 = df_ref.dropna(subset=[comm_col]).copy()
df_ref2['Age'] = datetime.date.today().year - df_ref2[comm_col].dt.year
age    = df_ref2.groupby('Country')['Age'].mean().fillna(0)

# colors
COLOR_2 = 'darkgoldenrod'
COLOR_6 = 'purple'
BAR_COLORS = {2: COLOR_2, 6: COLOR_6}

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
    delta = (cmp['rep_density'] - cmp['orig_density']).reindex(order)
    return delta

# --- SECTION 7: ΔDensity vs Single‐Turbine Share ---
# sort countries by share
order7 = share.sort_values().index.tolist()
x7 = np.arange(len(order7))
width = 0.3

fig, ax1 = plt.subplots(figsize=(12,8))
for i, a in enumerate(approaches):
    delta = compute_delta(a, order7)
    ax1.bar(x7 + (i-0.5)*width,
            delta,
            width,
            color=BAR_COLORS[a],
            label=f'ΔDensity A{a}')

ax1.set_xticks(x7)
ax1.set_xticklabels(order7, rotation=45, ha='right')
ax1.set_xlabel("Country")
ax1.set_ylabel("Δ Density (MW/km²)")
ax1.set_title("Approaches 2 & 6: ΔDensity vs Single‐Turbine Share")

ax2 = ax1.twinx()
share_sorted = share.reindex(order7)
ax2.plot(x7, share_sorted, 'o-', color=COLOR_MARKER, label='Single‐Turbine Share')
ax2.set_ylabel("Single‐Turbine Parks (%)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=2, loc='upper left')
fig.tight_layout()
plt.show()


# --- SECTION 11: ΔDensity vs Avg Turbine Age ---
# sort countries by age
order11 = age.sort_values().index.tolist()
x11 = np.arange(len(order11))

fig, ax1 = plt.subplots(figsize=(12,8))
for i, a in enumerate(approaches):
    delta = compute_delta(a, order11)
    ax1.bar(x11 + (i-0.5)*width,
            delta,
            width,
            color=BAR_COLORS[a],
            label=f'ΔDensity A{a}')

ax1.set_xticks(x11)
ax1.set_xticklabels(order11, rotation=45, ha='right')
ax1.set_xlabel("Country")
ax1.set_ylabel("Δ Density (MW/km²)")
ax1.set_title("Approaches 2 & 6: ΔDensity vs Avg Turbine Age")

ax2 = ax1.twinx()
age_sorted = age.reindex(order11)
ax2.plot(x11, age_sorted, 'o-', color=COLOR_MARKER, label='Avg Turbine Age')
ax2.set_ylabel("Avg Turbine Age (years)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=2, loc='upper left')
fig.tight_layout()
plt.show()


# --- SECTION 8: Average Turbine Age (Approach 5 & 6) ---
for approach in [5,6]:
    df = dfs[approach]
    single = df[df[turb_col]==1].dropna(subset=[comm_col]).copy()
    single['Age'] = datetime.date.today().year - single[comm_col].dt.year
    avg_age = single.groupby('Country')['Age'].mean().sort_values(ascending=False)
    age_df = avg_age.reset_index().rename(columns={'Age':'Avg_Turbine_Age'})
    print(f"\n--- Section 8: Average Turbine Age per Country (Approach {approach}) ---")
    print(age_df.to_string(index=False, float_format="{:.1f}".format))

# --- SECTION 10: Total Park Area per Approach (table) ---
ta = {i: dfs[i][area_col].sum() for i in dfs}
ta_df = pd.Series(ta, name='Total_Park_Area_m²').rename_axis('Approach').reset_index()
print("\n--- Section 10: Total Park Area per Approach ---")
print(ta_df.to_string(index=False, float_format="{:.2f}".format))


# --- SECTION 12: Baseline vs. Repowered Park Area per Approach (table) ---
rows = []
for i, df in dfs.items():
    base_area = df[area_col].sum()
    rep_area = df[newarea_col].sum() if newarea_col in df.columns else np.nan
    rows.append({
        "Approach": i,
        "Baseline_Park_Area (m²)": base_area,
        "Repowered_New_Area (m²)": rep_area
    })
area_comparison_df = pd.DataFrame(rows)
print("\n--- Section 12: Park Area per Approach (Baseline vs Repowered) ---")
print(area_comparison_df.to_string(index=False, float_format="{:.2f}".format))


# --- Print Power Density Table (Original + Approaches 2–6) ---
pd.set_option('display.float_format', lambda x: f"{x:.2f}")
density_table = all_d.rename(columns={
    'orig_density':   'Original',
    'rep_density_2':  'Approach 2',
    'rep_density_3':  'Approach 3',
    'rep_density_4':  'Approach 4',
    'rep_density_5':  'Approach 5',
    'rep_density_6':  'Approach 6',
})
print("\nPower density by Country and Approach (MW/km²):\n")
print(density_table.to_string())
