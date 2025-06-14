import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

# --- Helper: detect column by keyword in DataFrame ---
def detect_col(df, keyword):
    key = keyword.lower()
    for c in df.columns:
        norm = c.lower().replace('_', ' ')
        # strip any parenthetical
        norm = pd.Series([norm]).str.replace(r"\s*\(.*\)", '', regex=True)[0].strip()
        if key in norm:
            return c
    raise KeyError(f"No column containing '{keyword}'")

# --- Setup relative paths ---
this_dir = Path(__file__).resolve().parent
results_dir = this_dir / "results"

# --- Section 0: Load and process Approach 0 (old) ---
old_fp = results_dir / "Cf_old_updated.xlsx"
df_old = pd.read_excel(old_fp, index_col=0)

df_old["RepCap_kW"]   = pd.to_numeric(df_old["Representative_New_Capacity"], errors="coerce").fillna(0)
df_old["NumTurbines"] = pd.to_numeric(df_old["Number of turbines"],        errors="coerce").fillna(0)
df_old["CF_pct"]      = pd.to_numeric(df_old["CapacityFactor"],              errors="coerce").fillna(0)
df_old["TotalCap_kW"] = df_old["RepCap_kW"] * df_old["NumTurbines"]
df_old["TotalCap_MW"] = df_old["TotalCap_kW"] / 1e3
df_old["TWh_old"]     = df_old["TotalCap_MW"] * df_old["CF_pct"] * 8760 / 1e6  # TWh

agg = {
    "Approach 0-Old (Decomm+Repl)": df_old.groupby("Country")["TWh_old"].sum()
}
scenario_dfs = {"Approach 0-Old (Decomm+Repl)": df_old.copy()}

# --- Section 1–4: Load Approaches 1–4 ---
pattern = os.path.join(results_dir, "Approach_*_Cf.xlsx")
name_map = {
    "Approach_1_Cf.xlsx": "Approach 1-Power Density",
    "Approach_2_Cf.xlsx": "Approach 2-Capacity Maximization",
    "Approach_3_Cf.xlsx": "Approach 3-Rounding Up",
    "Approach_4_Cf.xlsx": "Approach 4-Single Turbine Flex",
}

for fp in glob.glob(pattern):
    fname = os.path.basename(fp)
    if fname.lower().endswith("_old.xlsx"):
        continue
    label = name_map.get(fname)
    if label is None:
        continue

    df = pd.read_excel(fp, index_col=0)
    df["CF"]     = pd.to_numeric(df["CapacityFactor"],     errors="coerce").fillna(0)
    df["Cap_MW"] = pd.to_numeric(df["Total_New_Capacity"], errors="coerce").fillna(0)
    df["TWh"]    = df["Cap_MW"] * df["CF"] * 8760 / 1e6

    agg[label] = df.groupby("Country")["TWh"].sum()
    scenario_dfs[label] = df.copy()

# --- Section 5: Build No-Loss Hybrids (Approaches 5 & 6) ---
fp2 = results_dir / "Approach_2_Cf.xlsx"
df_nlh = pd.read_excel(fp2, index_col=0)
df_nlh["CF"]      = pd.to_numeric(df_nlh["CapacityFactor"],     errors="coerce").fillna(0)
df_nlh["Cap_MW"]  = pd.to_numeric(df_nlh["Total_New_Capacity"], errors="coerce").fillna(0)
df_nlh["TWh_new"] = df_nlh["Cap_MW"] * df_nlh["CF"] * 8760 / 1e6

# 5a) Yield-based → Approach 6
df_yield = df_nlh.join(df_old["TWh_old"], how="left")
df_yield["TWh_yield"] = np.where(df_yield["TWh_old"] > df_yield["TWh_new"],
                                  df_yield["TWh_old"], df_yield["TWh_new"])
agg["Approach 6-No-Loss Hybrid (Yield-based)"] = df_yield.groupby("Country")["TWh_yield"].sum()
scenario_dfs["Approach 6-No-Loss Hybrid (Yield-based)"] = df_yield.copy()

# 5b) Cap-based → Approach 5
df_cap = df_nlh.join(df_old[["TotalCap_MW", "TWh_old"]], how="left")
df_cap["TWh_cap_based"] = np.where(df_cap["TotalCap_MW"] > df_cap["Cap_MW"],
                                   df_cap["TWh_old"], df_cap["TWh_new"])
agg["Approach 5-No-Loss Hybrid (Cap-based)"] = df_cap.groupby("Country")["TWh_cap_based"].sum()
scenario_dfs["Approach 5-No-Loss Hybrid (Cap-based)"] = df_cap.copy()

# --- Section 6: Aggregate for plotting ---
agg_df = pd.DataFrame(agg).fillna(0)
plot_cols = [
    "Approach 0-Old (Decomm+Repl)",
    "Approach 1-Power Density",
    "Approach 2-Capacity Maximization",
    "Approach 3-Rounding Up",
    "Approach 4-Single Turbine Flex",
    "Approach 5-No-Loss Hybrid (Cap-based)",
    "Approach 6-No-Loss Hybrid (Yield-based)"
]
agg_df = agg_df[plot_cols].sort_values("Approach 2-Capacity Maximization", ascending=False)

scenario_colors = {
    col: clr for col, clr in zip(plot_cols,
    ["black","blue","orange","green","red","brown","purple"])
}
COLOR_MARKER = 'red'

# --- Section 7: Absolute Annual Energy by Scenario ---
x     = np.arange(len(agg_df))
n     = len(plot_cols)
width = 0.8 / n
fig, ax = plt.subplots(figsize=(14, 6))
for i, col in enumerate(plot_cols):
    ax.bar(x + (i - (n-1)/2) * width,
           agg_df[col],
           width,
           label=col,
           color=scenario_colors[col],
           alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(agg_df.index, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Annual Energy Production (TWh)', fontsize=11)
ax.set_title('Energy by Scenario per Country (All Approaches)', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("Energy by Scenario per Country.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Section 8: Relative Percentage Increase over Old Baseline ---
baseline = agg_df["Approach 0-Old (Decomm+Repl)"]
rel_cols = plot_cols[1:]
fig, ax = plt.subplots(figsize=(14, 6))
for i, col in enumerate(rel_cols):
    pct_increase = (agg_df[col] - baseline) / baseline * 100
    ax.bar(x + (i - (len(rel_cols)-1)/2) * width,
           pct_increase,
           width,
           label=col,
           color=scenario_colors[col],
           alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(agg_df.index, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Percentage Increase over Old Baseline (%)', fontsize=11)
ax.set_title('Relative Percentage Increase by Scenario per Country', fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("elative Percentage Increase by Scenario per Country.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Section 9: Summary Statistics ---
stats = []
for name, df in scenario_dfs.items():
    if "NumTurbines" not in df.columns:
        df["NumTurbines"] = df_old["NumTurbines"]
    if "RepCap_kW" not in df.columns:
        df = df.join(df_old["RepCap_kW"], how="left")
    if "Cap_MW" in df.columns:
        total_cap_MW = df["Cap_MW"].sum()
        new_kW = df["Cap_MW"] * 1e3
    elif "TotalCap_MW" in df.columns:
        total_cap_MW = df["TotalCap_MW"].sum()
        new_kW = df["TotalCap_MW"] * 1e3
    else:
        total_cap_MW = 0
        new_kW = pd.Series(0, index=df.index)
    n_parks = df.shape[0]
    n_turbs = df["NumTurbines"].sum()
    avg_kW  = (total_cap_MW * 1e3 / n_turbs) if n_turbs > 0 else 0
    pct_rep = (n_turbs / df_old["NumTurbines"].sum() * 100) if df_old["NumTurbines"].sum() > 0 else 0
    df["NewRepCap_kW"] = new_kW / df["NumTurbines"].replace(0, np.nan)
    inc = df.loc[df["NewRepCap_kW"] >= df["RepCap_kW"], "NumTurbines"].sum()
    pct_inc = (inc / n_turbs * 100) if n_turbs > 0 else 0
    stats.append({
        "Approach": name,
        "TotalCap (MW)": total_cap_MW,
        "Parks": n_parks,
        "Turbines": n_turbs,
        "Avg Turb (kW)": avg_kW,
        "% Repowered": pct_rep,
        "% ↑ or = Cap": pct_inc
    })
stats_df = pd.DataFrame(stats).set_index("Approach")

energy = []
for name in stats_df.index:
    df = scenario_dfs[name]
    if name == "Approach 5-No-Loss Hybrid (Cap-based)":
        energy.append(df["TWh_cap_based"].sum())
    elif name == "Approach 6-No-Loss Hybrid (Yield-based)":
        energy.append(df["TWh_yield"].sum())
    else:
        energy.append(df["TWh"].sum() if "TWh" in df.columns else df["TWh_old"].sum())
stats_df["TotalEnergy_TWh"] = energy

print("\nTotal Annual Energy Production per Approach (All Countries Combined):")
print(stats_df[["TotalEnergy_TWh"]])
# --- Section 10–11: Demand Coverage Plot for Approaches 0, 5 & 6 ---
consumption_data = {
    "Country": [
        "Russia","Germany","France","Italy","UK","Turkey","Spain","Poland","Sweden","Norway",
        "Netherlands","Ukraine","Belgium","Finland","Austria","Czechia","Switzerland","Portugal",
        "Romania","Greece","Hungary","Bulgaria","Denmark","Belarus","Serbia","Ireland","Slovakia",
        "Iceland","Croatia","Slovenia","Bosnia & Herzegovina","Lithuania","Estonia","Albania",
        "Latvia","North Macedonia","Luxembourg","Moldova","Cyprus","Montenegro","Malta",
        "Faroe Islands","Gibraltar"
    ],
    "Electricity_Consumption_TWh": [
        1020.67,512.19,431.94,301.48,288.93,287.32,232.84,167.54,131.12,124.91,
        111.71,98.01,82.23,80.55,68.17,63.75,57.19,50.57,50.44,49.54,
        43.83,35.47,34.30,33.79,33.49,29.72,26.35,19.33,16.73,13.52,
        12.45,11.28,8.62,6.94,6.93,6.23,6.22,5.59,5.12,2.99,2.75,0.46,0.21
    ]
}
df_cons = pd.DataFrame(consumption_data)
wind_old   = agg_df["Approach 0-Old (Decomm+Repl)"].rename("Wind_Old_TWh")
wind_cap   = agg_df["Approach 5-No-Loss Hybrid (Cap-based)"].rename("Wind_Cap_TWh")
wind_yield = agg_df["Approach 6-No-Loss Hybrid (Yield-based)"].rename("Wind_Yield_TWh")

df_cmp = (
    df_cons
      .merge(wind_old,   left_on="Country", right_index=True, how="left")
      .merge(wind_cap,   left_on="Country", right_index=True, how="left")
      .merge(wind_yield, left_on="Country", right_index=True, how="left")
)
df_cmp["Cov_Old_%"]   = df_cmp["Wind_Old_TWh"]   / df_cmp["Electricity_Consumption_TWh"] * 100
df_cmp["Cov_Cap_%"]   = df_cmp["Wind_Cap_TWh"]    / df_cmp["Electricity_Consumption_TWh"] * 100
df_cmp["Cov_Yield_%"] = df_cmp["Wind_Yield_TWh"]  / df_cmp["Electricity_Consumption_TWh"] * 100

df_plot = df_cmp.sort_values("Cov_Old_%", ascending=False).reset_index(drop=True)
x = np.arange(len(df_plot))
w = 0.25
fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(x - w, df_plot["Cov_Old_%"],   width=w, label="Approach 0 (Old)",          color="black", edgecolor='black')
ax.bar(x,     df_plot["Cov_Cap_%"],   width=w, label="Approach 5 (Cap-based NLH)", color="#a0522d", edgecolor='black')
ax.bar(x + w, df_plot["Cov_Yield_%"], width=w, label="Approach 6 (Yield-based NLH)", color="#800080", edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(df_plot["Country"], rotation=45, ha='right', fontsize=10)
ax.set_ylabel("Demand Coverage (%)", fontsize=14)
ax.set_title("EU+ Countries: Wind Demand Coverage by Approach 0, 5 & 6", fontsize=18, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', fontsize=12, frameon=False)
plt.tight_layout()
plt.savefig("Countries: Wind Demand Coverage by Approach.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Section 12: ΔAnnual Energy vs Single-Turbine Share (A2 & A6) ---
baseline = agg_df["Approach 0-Old (Decomm+Repl)"]
y2 = agg_df["Approach 2-Capacity Maximization"]
y6 = agg_df["Approach 6-No-Loss Hybrid (Yield-based)"]
delta2 = y2 - baseline
delta6 = y6 - baseline

total  = df_old.groupby("Country").size()
single = df_old[df_old["NumTurbines"]==1].groupby("Country").size()
share  = (single/total*100).reindex(agg_df.index).fillna(0)

order12 = share.sort_values().index.tolist()
delta2_s = delta2.reindex(order12)
delta6_s = delta6.reindex(order12)
share_s  = share.reindex(order12)

x12 = np.arange(len(order12))
w12 = 0.3

fig, ax1 = plt.subplots(figsize=(14,6))
ax1.bar(x12 - w12/2, delta2_s, width=w12,
        label="ΔTWh A2",
        color=scenario_colors["Approach 2-Capacity Maximization"], alpha=0.8)
ax1.bar(x12 + w12/2, delta6_s, width=w12,
        label="ΔTWh A6",
        color=scenario_colors["Approach 6-No-Loss Hybrid (Yield-based)"], alpha=0.8)
ax1.set_xticks(x12)
ax1.set_xticklabels(order12, rotation=45, ha='right')
ax1.set_ylabel("Δ Annual Energy (TWh)")
ax1.set_title("Δ Annual Energy vs Single-Turbine Share (A2 & A6)")

ax2 = ax1.twinx()
ax2.plot(x12, share_s, 'o-', color=COLOR_MARKER, label="Single-Turbine Share (%)")
ax2.set_ylabel("Single-Turbine Share (%)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=2, loc="upper left")
plt.tight_layout()
plt.savefig("Annual Energy vs Single-Turbine Share .png", dpi=300, bbox_inches="tight")
plt.show()

# --- Section 13: ΔAnnual Energy vs Avg Turbine Age (A2 & A6) ---
df_age = df_old.dropna(subset=["Commissioning date"]).copy()
df_age["Age"] = datetime.date.today().year - pd.to_datetime(df_age["Commissioning date"]).dt.year
age = df_age.groupby("Country")["Age"].mean().reindex(agg_df.index).fillna(0)

order13 = age.sort_values().index.tolist()
delta2_s2 = delta2.reindex(order13)
delta6_s2 = delta6.reindex(order13)
age_s     = age.reindex(order13)

x13 = np.arange(len(order13))

fig, ax1 = plt.subplots(figsize=(14,6))
ax1.bar(x13 - w12/2, delta2_s2, width=w12,
        label="ΔTWh A2",
        color=scenario_colors["Approach 2-Capacity Maximization"], alpha=0.8)
ax1.bar(x13 + w12/2, delta6_s2, width=w12,
        label="ΔTWh A6",
        color=scenario_colors["Approach 6-No-Loss Hybrid (Yield-based)"], alpha=0.8)
ax1.set_xticks(x13)
ax1.set_xticklabels(order13, rotation=45, ha='right')
ax1.set_ylabel("Δ Annual Energy (TWh)")
ax1.set_title("Δ Annual Energy vs Avg Turbine Age (A2 & A6)")

ax2 = ax1.twinx()
ax2.plot(x13, age_s, 'o-', color=COLOR_MARKER, label="Avg Turbine Age (years)")
ax2.set_ylabel("Avg Turbine Age (years)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=2, loc="upper left")
plt.tight_layout()
plt.savefig(" Annual Energy vs Avg Turbine Age .png", dpi=300, bbox_inches="tight")
plt.show()

# --- Section 14: Energy Density by Scenario (TWh/km²), all approaches sorted by A2 ---
orig_area_col = detect_col(scenario_dfs["Approach 0-Old (Decomm+Repl)"], "park area")
try:
    new_area_col = detect_col(scenario_dfs["Approach 2-Capacity Maximization"], "new park area")
except KeyError:
    new_area_col = detect_col(scenario_dfs["Approach 2-Capacity Maximization"], "park area")

ed = pd.DataFrame(index=agg_df.index)
for i, label in enumerate(plot_cols):
    if i == 0:
        area = scenario_dfs[label].groupby("Country")[orig_area_col].sum() / 1e6
    else:
        area = scenario_dfs[label].groupby("Country")[new_area_col].sum() / 1e6
    energy = agg_df[label]
    ed[f"Density_{i}"] = energy / area.replace(0, np.nan)

ed.sort_values("Density_2", ascending=False, inplace=True)

fig, ax = plt.subplots(figsize=(14,6))
x = np.arange(len(ed))
w = 0.12
for i, label in enumerate(plot_cols):
    ax.bar(x + (i-3)*w,
           ed[f"Density_{i}"],
           width=w,
           label=label,
           color=scenario_colors[label],
           alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(ed.index, rotation=45, ha='right')
ax.set_ylabel("Energy Density (TWh/km²)")
ax.set_title("Annual Energy per Land Area by Scenario and Country", fontsize=14)
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("Annual Energy per Land Area by Scenario and Country.png", dpi=300, bbox_inches="tight")
plt.show()


# make a copy and rename columns to the approach names
density_table = ed.copy()
density_table.columns = plot_cols

# round to 3 decimal places and print
print("\n--- Section 15: Energy Density (TWh/km²) by Country & Approach ---")
print(density_table.round(3).to_string())


# --- Section 16+17: Combined Capacity Factor Distribution and Difference Plot ---

# Extract and sort CF values
cf_vals_old = scenario_dfs["Approach 0-Old (Decomm+Repl)"]["CF_pct"].dropna().sort_values(ascending=False).reset_index(drop=True)
cf_vals_a2  = scenario_dfs["Approach 2-Capacity Maximization"]["CF"].dropna().sort_values(ascending=False).reset_index(drop=True)

# Get matched CF data
df_old_cf = scenario_dfs["Approach 0-Old (Decomm+Repl)"][["CF_pct"]].copy()
df_a2_cf  = scenario_dfs["Approach 2-Capacity Maximization"][["CF"]].copy()
cf_diff = df_old_cf.join(df_a2_cf, how="inner").dropna()
cf_diff = cf_diff[cf_diff["CF"] > 0]
cf_diff["ΔCF_pct_points"] = (cf_diff["CF"] - cf_diff["CF_pct"]) * 100
cf_diff_sorted = cf_diff.sort_values("ΔCF_pct_points", ascending=False).reset_index(drop=True)

# ─── Plot Combined Subplots ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 10), sharex=False)

# Top plot: Sorted capacity factors
ax1.plot(cf_vals_old.index, cf_vals_old * 100, label='Baseline Approach 0', color='blue', linewidth=2)
ax1.plot(cf_vals_a2.index,  cf_vals_a2  * 100, label='Repowering Capacity Maximization (Approach 2)', color='orange', linewidth=2)
ax1.set_ylabel("Capacity Factor (%)", fontsize=12)
ax1.set_title("Sorted Capacity Factor Distributions: Baseline vs. Repowering (Approach 2)", fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=11)

# Bottom plot: ΔCF per park
ax2.bar(cf_diff_sorted.index, cf_diff_sorted["ΔCF_pct_points"], color='green', alpha=0.7)
ax2.axhline(0, color='black', linewidth=1)
ax2.set_xlabel("Park Index", fontsize=12)
ax2.set_ylabel("Δ Capacity Factor (percentage points)", fontsize=12)
ax2.set_title("Difference in Capacity Factor per Park (Approach 2 − Approach 0)", fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("Difference in Capacity Factor per Park.png", dpi=300, bbox_inches="tight")
plt.show()
