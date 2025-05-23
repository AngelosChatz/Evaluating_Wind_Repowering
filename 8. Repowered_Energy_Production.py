import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Point to your results folder
results_dir = r"D:\SET 2023\Thesis Delft\Model\Evaluating_Wind_Repowering\results"

# 2) Load the old decommission+replacement baseline and compute TWh_old
old_fp = os.path.join(results_dir, "Cf_old_updated.xlsx")
df_old = pd.read_excel(old_fp, index_col=0)

# Ensure numeric
df_old["RepCap_kW"]   = pd.to_numeric(df_old["Representative_New_Capacity"], errors="coerce").fillna(0)
df_old["NumTurbines"] = pd.to_numeric(df_old["Number of turbines"],       errors="coerce").fillna(0)
df_old["CF_pct"]      = pd.to_numeric(df_old["CapacityFactor"],           errors="coerce").fillna(0)

# Compute total capacity per park
df_old["TotalCap_kW"] = df_old["RepCap_kW"] * df_old["NumTurbines"]
df_old["TotalCap_MW"] = df_old["TotalCap_kW"] / 1e3

# Capacity factor is already in decimal, so no need for conversion
df_old["TWh_old"] = df_old["TotalCap_MW"] * df_old["CF_pct"] * 8760 / 1e6  # Divide by 1e6 to convert to TWh

# 3) Initialize aggregation dict and scenario_dfs dict
agg = {
    "Approach 0-Old (Decomm+Repl)": df_old.groupby("Country")["TWh_old"].sum()
}
scenario_dfs = {
    "Approach 0-Old (Decomm+Repl)": df_old.copy()
}

# 4) Load each Approach_1–4 repowering scenario, convert to TWh
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
    df["CF"]     = pd.to_numeric(df["CapacityFactor"],     errors="coerce").fillna(0)  # CF is already decimal
    df["Cap_MW"] = pd.to_numeric(df["Total_New_Capacity"], errors="coerce").fillna(0)
    df["TWh"]    = df["Cap_MW"] * df["CF"] * 8760 / 1e6  # Divide by 1e6 to convert to TWh

    agg[label] = df.groupby("Country")["TWh"].sum()
    scenario_dfs[label] = df.copy()

# 5) Prepare Approach_2 frame for the two hybrids
fp2 = os.path.join(results_dir, "Approach_2_Cf.xlsx")
df_nlh = pd.read_excel(fp2, index_col=0)
df_nlh["CF"]      = pd.to_numeric(df_nlh["CapacityFactor"],     errors="coerce").fillna(0)  # CF is already decimal
df_nlh["Cap_MW"]  = pd.to_numeric(df_nlh["Total_New_Capacity"], errors="coerce").fillna(0)
df_nlh["TWh_new"] = df_nlh["Cap_MW"] * df_nlh["CF"] * 8760 / 1e6  # Divide by 1e6 to convert to TWh

# 5a) No-Loss Hybrid (Yield-based)
df_yield = df_nlh.join(df_old["TWh_old"], how="left")
df_yield["TWh_yield"] = np.where(
    df_yield["TWh_old"] > df_yield["TWh_new"],
    df_yield["TWh_old"],
    df_yield["TWh_new"]
)
agg["Approach 6-No-Loss Hybrid (Yield-based)"] = df_yield.groupby("Country")["TWh_yield"].sum()
scenario_dfs["Approach 6-No-Loss Hybrid (Yield-based)"] = df_yield.copy()

# Create a new DataFrame for the Approach 6 results and save it to a new Excel file
df_yield["Approach"] = np.where(
    df_yield["TWh_yield"] > df_yield["TWh_old"],  # If yield is greater than the old value, it's considered 'repowered'
    "repowered",
    "DnR"  # Else, it's decommissioned and replaced
)

# Save to a new Excel file
output_fp = os.path.join(results_dir, "Approach_6_No_Loss_Hybrid_Yield_based.xlsx")
df_yield.to_excel(output_fp)

print(f"Approach 6-No-Loss Hybrid (Yield-based) results saved to {output_fp}")

# 5b) No-Loss Hybrid (Capacity-based)
df_cap = df_nlh.join(
    df_old[["TotalCap_MW", "TWh_old"]],
    how="left"
)
df_cap["TWh_cap_based"] = np.where(
    df_cap["TotalCap_MW"] > df_cap["Cap_MW"],
    df_cap["TWh_old"],
    df_cap["TWh_new"]
)
agg["Approach 5-No-Loss Hybrid (Cap-based)"] = df_cap.groupby("Country")["TWh_cap_based"].sum()
scenario_dfs["Approach 5-No-Loss Hybrid (Cap-based)"] = df_cap.copy()

# 6) Combine into a single DataFrame for energy plots
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

# 7) Plot absolute annual energy (TWh) by country
x     = np.arange(len(agg_df))
n     = len(plot_cols)
width = 0.8 / n
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

fig, ax = plt.subplots(figsize=(14, 6))
for i, col in enumerate(plot_cols):
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
ax.set_ylabel('Annual Energy Production (TWh)', fontsize=label_fs)
ax.set_title('Energy by Scenario per Country (All Approaches)', fontsize=title_fs)
ax.legend(loc='upper right', fontsize=legend_fs)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 8) Plot relative percentage increase over old baseline
baseline = agg_df["Approach 0-Old (Decomm+Repl)"]
rel_cols = plot_cols[1:]

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
ax.legend(loc='upper left', fontsize=legend_fs)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 9) Compute summary stats into a list of dicts
stats = []
for name, df in scenario_dfs.items():
    # ensure NumTurbines
    if "NumTurbines" not in df.columns:
        if "Number of turbines" in df.columns:
            df["NumTurbines"] = pd.to_numeric(df["Number of turbines"], errors="coerce").fillna(0)
        else:
            df["NumTurbines"] = df_old["NumTurbines"]

    # ensure old rep cap present
    if "RepCap_kW" not in df.columns:
        df = df.join(df_old["RepCap_kW"], how="left")

    # total capacity and new capacity in kW
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

# 10) Build and print a summary table without extra deps
stats_df = pd.DataFrame(stats).set_index("Approach")

# 11) Add annual energy so the hybrids differ
energy = []
for name in stats_df.index:
    df = scenario_dfs[name]
    if name == "Approach 5-No-Loss Hybrid (Cap-based)":
        energy.append(df["TWh_cap_based"].sum())
    elif name == "Approach 6-No-Loss Hybrid (Yield-based)":
        energy.append(df["TWh_yield"].sum())
    else:
        energy.append(df["TWh"].sum() if "TWh" in df.columns else df["TWh_old"].sum())

stats_df["AnnualEnergy (TWh)"] = energy
