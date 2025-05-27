import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
this_dir = Path(__file__).resolve().parent
results_dir = this_dir / "results"
files = {
    "Approach 1": results_dir / "Approach_1.xlsx",
    "Approach 2": results_dir / "Approach_2.xlsx",
    "Approach 3": results_dir / "Approach_3.xlsx",
    "Approach 4": results_dir / "Approach_4.xlsx",
    "Approach 5": results_dir / "Approach_5.xlsx",
}

# Helper: safe column conversion
def to_numeric_col(df, col):
    df[col] = pd.to_numeric(df.get(col), errors='coerce')
    return df

# SECTION 1: REPOWERED POWER DENSITY – Approach 5
rep5 = pd.read_excel(files['Approach 5']).replace({'#ND': np.nan, '': np.nan})
rep5 = to_numeric_col(rep5, 'Total_New_Capacity')
rep5 = to_numeric_col(rep5, 'New_Total_Park_Area (m²)')
rep5 = rep5[rep5['Total_New_Capacity'] > 0]

g5 = rep5.groupby('Country').agg(
    Total_New_Capacity=('Total_New_Capacity', 'sum'),
    New_Area=('New_Total_Park_Area (m²)', 'sum')
).reset_index()
g5['Repowered_Power_Density'] = g5['Total_New_Capacity'] / (g5['New_Area'] / 1e6)
g5 = g5.sort_values('Repowered_Power_Density', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(g5['Country'], g5['Repowered_Power_Density'], color=plt.cm.viridis(np.linspace(0, 1, len(g5))))
plt.title("Repowered Power Density per Country (Approach 5)", fontsize=16, fontweight='bold')
plt.xlabel("MW/km²", fontsize=14)
plt.ylabel("Country", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# SECTION 2: ORIGINAL vs. REPOWERED – Approach 5
orig5 = pd.read_excel(files['Approach 5']).replace({'#ND': np.nan, '': np.nan})
orig5 = to_numeric_col(orig5, 'Total power')
orig5 = to_numeric_col(orig5, 'Total Park Area (m²)')
orig5 = orig5[(orig5['Total power'] > 0) & (orig5['Total Park Area (m²)'] > 0)]
go5 = orig5.groupby('Country').agg(
    Total_power=('Total power', 'sum'),
    Total_area=('Total Park Area (m²)', 'sum')
).reset_index()
go5['Original_Power_Density'] = go5['Total_power'] / (go5['Total_area'] / 1e6)

cmp5 = pd.merge(
    go5[['Country', 'Original_Power_Density']],
    g5[['Country', 'Repowered_Power_Density']],
    on='Country', how='outer'
).fillna(0)
cmp5 = cmp5.sort_values('Repowered_Power_Density', ascending=False)

plt.figure(figsize=(12, 8))
y = np.arange(len(cmp5))
plt.barh(y - 0.2, cmp5['Original_Power_Density'], height=0.4, label='Original')
plt.barh(y + 0.2, cmp5['Repowered_Power_Density'], height=0.4, label='Repowered')
plt.yticks(y, cmp5['Country'])
plt.xlabel("MW/km²", fontsize=14)
plt.title("Original vs. Approach 5 Power Density per Country", fontsize=16, fontweight='bold')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# SECTION 3: POWER DENSITY COMPARISON – Original & Approaches 1–5
# compute original density
def compute_orig_density():
    df = pd.read_excel(files['Approach 1']).replace({'#ND': np.nan, '': np.nan})
    df = to_numeric_col(df, 'Total power')
    df = to_numeric_col(df, 'Total Park Area (m²)')
    df = df[(df['Total power'] > 0) & (df['Total Park Area (m²)'] > 0)]
    grp = df.groupby('Country').agg(
        Total_power=('Total power', 'sum'),
        Total_area=('Total Park Area (m²)', 'sum')
    ).reset_index()
    grp['Orig_Power_Density'] = grp['Total_power'] / (grp['Total_area'] / 1e6)
    return grp[['Country', 'Orig_Power_Density']]

def compute_rep_density(a):
    df = pd.read_excel(files[f'Approach {a}']).replace({'#ND': np.nan, '': np.nan})
    df = to_numeric_col(df, 'Total_New_Capacity')
    df = to_numeric_col(df, 'New_Total_Park_Area (m²)')
    df = df[df['Total_New_Capacity'] > 0]
    grp = df.groupby('Country').agg(
        Total_New_Capacity=('Total_New_Capacity', 'sum'),
        New_Area=('New_Total_Park_Area (m²)', 'sum')
    ).reset_index()
    grp[f'Rep_Power_Density_{a}'] = grp['Total_New_Capacity'] / (grp['New_Area'] / 1e6)
    return grp[['Country', f'Rep_Power_Density_{a}']]

orig_dens = compute_orig_density()
all_dens = orig_dens.copy()
for a in [1, 2, 3, 4, 5]:
    rep = compute_rep_density(a)
    all_dens = all_dens.merge(rep, on='Country', how='outer')
all_dens.fillna(0, inplace=True)
all_dens.sort_values('Orig_Power_Density', ascending=False, inplace=True)

plt.figure(figsize=(14, 8))
idx = np.arange(len(all_dens))
labels = ['Original'] + [f'Approach {i}' for i in [1, 2, 3, 4, 5]]
width = 0.12
for i, label in enumerate(labels):
    col = 'Orig_Power_Density' if label == 'Original' else f'Rep_Power_Density_{label.split()[-1]}'
    offset = (i - (len(labels) - 1) / 2) * width
    plt.bar(idx + offset, all_dens[col], width, label=label)
plt.xticks(idx, all_dens['Country'], rotation=45, ha='right', fontsize=12)
plt.xlabel("MW/km²", fontsize=14)
plt.ylabel("Power Density", fontsize=14)
plt.title("Power Density per Country: Original vs. Approaches 1–5", fontsize=16, fontweight='bold')
plt.legend(ncol=3)
plt.tight_layout()
plt.show()

# SECTION 4: LAND AREA COMPARISON (Approaches 1–5)
def compute_country_capacities(file_path):
    df = pd.read_excel(file_path).replace({'#ND': np.nan, '': np.nan})
    df['Commissioning date'] = pd.to_datetime(df.get('Commissioning date'), errors='coerce')
    df['Decommissioning date'] = pd.to_datetime(df.get('Decommissioning date'), errors='coerce')
    df['Total power'] = pd.to_numeric(df.get('Total power'), errors='coerce')
    df['Total_New_Capacity'] = pd.to_numeric(df.get('Total_New_Capacity'), errors='coerce')
    rec = []
    def add(dic, yr, val): dic[yr] = dic.get(yr, 0) + val
    for c in df['Country'].dropna().unique():
        sub = df[df['Country'] == c]
        bc, rc = {}, {}
        for _, r in sub.iterrows():
            if pd.notna(r['Commissioning date']) and pd.notna(r['Total power']):
                s = r['Commissioning date'].year
                e = r['Decommissioning date'].year if pd.notna(r['Decommissioning date']) else s + 30
                add(bc, s, r['Total power']); add(bc, e, -r['Total power'])
            if pd.notna(r['Commissioning date']) and pd.notna(r['Total_New_Capacity']):
                s2 = r['Decommissioning date'].year if pd.notna(r['Decommissioning date']) else r['Commissioning date'].year + 20
                cap2 = r['Total_New_Capacity']; yr = s2
                while yr <= 2050:
                    add(rc, yr, cap2); add(rc, yr+20, -cap2); yr += 20
        h, run = {}, 0.0
        for y in range(1980, 2023): run += bc.get(y,0); h[y] = run
        trend = ((h.get(2022,0) - h.get(2000,0)) / 22) if 2000 in h else 0
        for y in range(2023, 2051): h[y] = h.get(2022,0) + trend * (y - 2022)
        base2050 = h[2050]
        run, hr = 0.0, {}
        for y in range(1980, 2051): run += rc.get(y,0); hr[y] = run
        rec.append({'Country': c, 'Baseline_2050': base2050, 'Combined_2050': base2050 + hr[2050]})
    return pd.DataFrame(rec)

def compute_land_area(a):
    caps = compute_country_capacities(files[f'Approach {a}'])
    caps['Repowered'] = caps['Combined_2050'] - caps['Baseline_2050']
    o = pd.read_excel(files[f'Approach {a}']).replace({'#ND': np.nan, '': np.nan})
    o = to_numeric_col(o, 'Total power')
    o = to_numeric_col(o, 'Total Park Area (m²)')
    g = o[(o['Total power'] > 0) & (o['Total Park Area (m²)'] > 0)]
    g = g.groupby('Country').agg(
        Total_power=('Total power','sum'),
        Total_area=('Total Park Area (m²)','sum')
    ).reset_index()
    g['Dens'] = g['Total_power'] / (g['Total_area'] / 1e6)
    m = pd.merge(caps, g[['Country','Dens']], on='Country', how='left')
    m['Base Area'] = m['Baseline_2050'] / m['Dens']
    m['Re Area'] = m['Repowered'] / m['Dens']
    return m[['Country','Base Area','Re Area']]

lds = [compute_land_area(i).rename(columns={'Base Area': f'B{i}', 'Re Area': f'R{i}'}) for i in [1,2,3,4,5]]
ld = lds[0]
for df in lds[1:]:
    ld = ld.merge(df, on='Country', how='outer')
ld.fillna(0, inplace=True)
ld['T1'] = ld['B1'] + ld['R1']
ld.sort_values('T1', ascending=False, inplace=True)

plt.figure(figsize=(14, 8))
idx = np.arange(len(ld))
w = 0.15
for i, a in enumerate([1,2,3,4,5]):
    off = (i - 2) * w
    plt.bar(idx + off, ld[f'B{a}'], w, color='lightgray')
    plt.bar(idx + off, ld[f'R{a}'], w, bottom=ld[f'B{a}'], label=f'Approach {a}')
plt.xticks(idx, ld['Country'], rotation=45, ha='right')
plt.xlabel("Country")
plt.ylabel("Area km²")
plt.title("Required Land Area Comparison (Approaches 1–5)")
plt.legend(ncol=3)
plt.tight_layout()
plt.show()
