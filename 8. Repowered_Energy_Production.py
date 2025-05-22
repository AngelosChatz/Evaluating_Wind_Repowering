import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Point to your results folder
results_dir = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results"

# 2) Load the old decommission+replacement baseline and compute GWh_old
old_fp = os.path.join(results_dir, "Cf_old_updated.xlsx")
df_old = pd.read_excel(old_fp, index_col=0)

# Ensure numeric
df_old["RepCap_kW"]   = pd.to_numeric(df_old["Representative_New_Capacity"], errors="coerce").fillna(0)
df_old["NumTurbines"] = pd.to_numeric(df_old["Number of turbines"],       errors="coerce").fillna(0)
df_old["CF_pct"]      = pd.to_numeric(df_old["CapacityFactor"],           errors="coerce").fillna(0)

# Compute total capacity per park in kW → MW
df_old["TotalCap_kW"] = df_old["RepCap_kW"] * df_old["NumTurbines"]
df_old["TotalCap_MW"] = df_old["TotalCap_kW"] / 1e3

# CF to decimal, then annual energy in GWh
df_old["CF_dec"]  = df_old["CF_pct"] / 100.0
# GWh = MW * CF * 8760 h / 1e3
df_old["GWh_old"] = df_old["TotalCap_MW"] * df_old["CF_dec"] * 8760 / 1e3

# 3) Initialize aggregation dict with the old baseline
agg = {
    "Approach 0-Old (Decomm+Repl)": df_old.groupby("Country")["GWh_old"].sum()
}

# 4) Load each Approach_1–4 repowering scenario, convert to GWh
pattern = os.path.join(results_dir, "Approach_*_Cf.xlsx")
name_map = {
    "Approach_1_Cf.xlsx": "Approach 1-Power Density",
    "Approach_2_Cf.xlsx": "Approach 2-Capacity Maximization",
    "Approach_3_Cf.xlsx": "Approach 3-Rounding Up",
    "Approach_4_Cf.xlsx": "Approach 4-Single Turbine Flex",
}

for fp in glob.glob(pattern):
    fname = os.path.basename(fp)
    # skip the old baseline file
    if fname.lower().endswith("_old.xlsx"):
        continue

    label = name_map.get(fname)
    # only load Approaches 1–4 here
    if label is None:
        continue

    df = pd.read_excel(fp, index_col=0)
    df["CF"]     = pd.to_numeric(df["CapacityFactor"],     errors="coerce").fillna(0) / 100.0
    df["Cap_MW"] = pd.to_numeric(df["Total_New_Capacity"], errors="coerce").fillna(0)
    df["GWh"]    = df["Cap_MW"] * df["CF"] * 8760 / 1e3

    agg[label] = df.groupby("Country")["GWh"].sum()

# 5) Prepare Approach_2 frame for the two hybrids
fp2 = os.path.join(results_dir, "Approach_2_Cf.xlsx")
df_nlh = pd.read_excel(fp2, index_col=0)
df_nlh["CF"]      = pd.to_numeric(df_nlh["CapacityFactor"],     errors="coerce").fillna(0) / 100.0
df_nlh["Cap_MW"]  = pd.to_numeric(df_nlh["Total_New_Capacity"], errors="coerce").fillna(0)
df_nlh["GWh_new"] = df_nlh["Cap_MW"] * df_nlh["CF"] * 8760 / 1e3

# 5a) No-Loss Hybrid (Yield-based)
df_yield = df_nlh.join(df_old["GWh_old"], how="left")
df_yield["GWh_yield"] = np.where(
    df_yield["GWh_old"] > df_yield["GWh_new"],
    df_yield["GWh_old"],
    df_yield["GWh_new"]
)
agg["Approach 6-No-Loss Hybrid (Yield-based)"] = df_yield.groupby("Country")["GWh_yield"].sum()

# 5b) No-Loss Hybrid (Capacity-based)
df_cap = df_nlh.join(
    df_old[["TotalCap_MW", "GWh_old"]],
    how="left"
)
df_cap["GWh_cap_based"] = np.where(
    df_cap["TotalCap_MW"] > df_cap["Cap_MW"],
    df_cap["GWh_old"],
    df_cap["GWh_new"]
)
agg["Approach 5-No-Loss Hybrid (Cap-based)"] = df_cap.groupby("Country")["GWh_cap_based"].sum()

# 6) Combine into a single DataFrame
agg_df = pd.DataFrame(agg).fillna(0)

# 6b) Ensure the baseline is first, then order the rest
cols = [
    "Approach 0-Old (Decomm+Repl)",
    "Approach 1-Power Density",
    "Approach 2-Capacity Maximization",
    "Approach 3-Rounding Up",
    "Approach 4-Single Turbine Flex",
    "Approach 5-No-Loss Hybrid (Cap-based)",
    "Approach 6-No-Loss Hybrid (Yield-based)"
]
agg_df = agg_df[cols]

# Optional: sort countries by one scenario
agg_df = agg_df.sort_values("Approach 2-Capacity Maximization", ascending=False)

# --- Styling for plotting ---
scenario_colors = {
    "Approach 0-Old (Decomm+Repl)":      "black",
    "Approach 1-Power Density":          "blue",
    "Approach 2-Capacity Maximization":  "orange",
    "Approach 3-Rounding Up":            "green",
    "Approach 4-Single Turbine Flex":    "red",
    "Approach 5-No-Loss Hybrid (Cap-based)": "brown",
    "Approach 6-No-Loss Hybrid (Yield-based)": "purple"
}
legend_fs = 9
label_fs  = 11
title_fs  = 14
tick_rot  = 45
tick_ha   = 'right'
alpha_val = 0.8

# 7) Plot absolute annual energy (GWh) by country
x     = np.arange(len(agg_df))
n     = len(cols)
width = 0.8 / n

fig, ax = plt.subplots(figsize=(14, 6))
for i, col in enumerate(cols):
    ax.bar(
        x + (i - (n-1)/2) * width,
        agg_df[col],
        width,
        label=col,
        color=scenario_colors[col],
        alpha=alpha_val
    )

ax.set_xticks(x)
ax.set_xticklabels(agg_df.index, rotation=tick_rot, ha=tick_ha, fontsize=label_fs)
ax.set_ylabel('Annual Energy Production (GWh)', fontsize=label_fs)
ax.set_title('Energy by Scenario per Country (All Approaches)', fontsize=title_fs)
ax.legend(loc='upper right', fontsize=legend_fs)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 8) Plot relative percentage increase over old baseline
baseline = agg_df["Approach 0-Old (Decomm+Repl)"]
rel_cols = cols[1:]

fig, ax = plt.subplots(figsize=(14, 6))
for i, col in enumerate(rel_cols):
    pct_increase = (agg_df[col] - baseline) / baseline * 100
    ax.bar(
        x + (i - (len(rel_cols)-1)/2) * width,
        pct_increase,
        width,
        label=col,
        color=scenario_colors[col],
        alpha=alpha_val
    )

ax.set_xticks(x)
ax.set_xticklabels(agg_df.index, rotation=tick_rot, ha=tick_ha, fontsize=label_fs)
ax.set_ylabel('Percentage Increase over Old Baseline (%)', fontsize=label_fs)
ax.set_title('Relative Percentage Increase by Scenario per Country', fontsize=title_fs)

# legend on the left:
ax.legend(loc='upper left', fontsize=legend_fs)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
