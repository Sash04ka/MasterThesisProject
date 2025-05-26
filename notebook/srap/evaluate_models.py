import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

paths = {
    "structured": os.path.join(project_root, "results", "benchmark_all_structured"),
    "random": os.path.join(project_root, "results", "benchmark_all_random")
}

dfs = []

for mode, folder in paths.items():
    for opt_type in ["asian", "barrier", "lookback"]:
        path = os.path.join(folder, f"sample_comparison_{opt_type}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["option_type"] = opt_type
            df["source"] = mode
            dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Add relative error column if missing
if "rel_error" not in df_all.columns:
    df_all["rel_error"] = (df_all["MLP"] - df_all["MC"]).abs() / (df_all["MC"].abs() + 1e-6)

# === 1. True vs Predicted Scatter Plot ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_all, x="MC", y="MLP", hue="option_type", style="source", alpha=0.6)
max_val = df_all[["MC", "MLP"]].max().max()
plt.plot([0, max_val], [0, max_val], 'k--')
plt.title("True (MC) vs Predicted (MLP) Option Prices")
plt.xlabel("Monte Carlo Price")
plt.ylabel("MLP Predicted Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "true_vs_predicted_all.png"))

# === 2. Absolute Error Distribution ===
plt.figure(figsize=(10, 6))
sns.histplot(data=df_all, x="abs_error", hue="option_type", multiple="stack", kde=True, bins=40)
plt.title("Distribution of Absolute Errors")
plt.xlabel("Absolute Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "error_distribution_all.png"))

# === 3. Boxplot of Absolute Errors ===
plt.figure(figsize=(9, 6))
sns.boxplot(data=df_all, x="option_type", y="abs_error", hue="source")
plt.title("Absolute Error by Option Type and Source")
plt.xlabel("Option Type")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "boxplot_all.png"))

# === 4. Mean Relative Error (MRE) vs MC Price Cutoff (percentage) ===
cutoffs = np.linspace(0, df_all["MC"].max(), 100)
mre_values = []
for cutoff in cutoffs:
    subset = df_all[df_all["MC"] > cutoff]
    if len(subset) == 0:
        mre_values.append(np.nan)
    else:
        mre_values.append(subset["rel_error"].mean())

plt.figure(figsize=(8, 5))
plt.plot(cutoffs, np.array(mre_values)*100, marker='o', markersize=4, linewidth=1)
plt.xlabel("Monte Carlo Price Cutoff")
plt.ylabel("Mean Relative Error (MRE) [%]")
plt.title("Mean Relative Error vs Monte Carlo Price Cutoff\n(This is percentage of MC price)")
plt.grid(True)
plt.ylim(0, 30)  # Cap Y axis at 30%
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "mre_vs_cutoff_percentage.png"))
plt.show()

print("✅ All plots saved in the results folder.")
