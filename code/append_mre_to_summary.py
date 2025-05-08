import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_path = os.path.join(project_root, "results", "benchmark_structured")
csv_path = os.path.join(results_path, "sample_comparison_structured.csv")
summary_path = os.path.join(results_path, "summary_metrics_structured.txt")
chart_path = os.path.join(results_path, "mre_vs_cutoff.png")

# === Load results ===
df = pd.read_csv(csv_path)

# === Compute MRE for multiple MC cutoffs ===
cutoffs = np.linspace(0.01, 10, 100)
mre_values = []

for c in cutoffs:
    valid = df["MC"] > c
    if valid.sum() == 0:
        mre_values.append(np.nan)
    else:
        mre = np.mean(np.abs((df.loc[valid, "MLP"] - df.loc[valid, "MC"]) / df.loc[valid, "MC"])) * 100
        mre_values.append(mre)

# === Plot and save the chart ===
plt.figure(figsize=(8, 5))
plt.plot(cutoffs, mre_values, marker='o', markersize=3, linewidth=1)
plt.xlabel("MC Price Cutoff (exclude MC < x)")
plt.ylabel("Mean Relative Error (MRE) [%]")
plt.title("MRE vs. Monte Carlo Price Cutoff")
plt.grid(True)
plt.tight_layout()
plt.savefig(chart_path)
plt.close()

# === Compute final selected MRE for cutoff = 10 ===
cutoff_chosen = 10
valid = df["MC"] > cutoff_chosen
if valid.sum() == 0:
    mre_text = "Mean Relative Error (MRE): undefined (all MC values too small)"
else:
    mre = np.mean(np.abs((df.loc[valid, "MLP"] - df.loc[valid, "MC"]) / df.loc[valid, "MC"])) * 100
    mre_text = f"Mean Relative Error (MRE) for MC > {cutoff_chosen}: {mre:.2f}%"

# === Append to summary file ===
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(mre_text + "\n")

print("✅", mre_text)
print(f"📈 Chart saved to: {chart_path}")
