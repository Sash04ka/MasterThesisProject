import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")

# Load all filtered datasets and assign option type
dfs = []
for opt_type in ["asian", "barrier", "lookback"]:
    path = os.path.join(data_dir, f"option_dataset_{opt_type}_filtered.csv")
    df = pd.read_csv(path)
    df["type"] = opt_type
    dfs.append(df)

# Concatenate into a single DataFrame
df_all = pd.concat(dfs, ignore_index=True)

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(data=df_all, x="price", hue="type", element="step", stat="count", common_norm=False, bins=60, kde=True)
plt.xlabel("Option Price")
plt.ylabel("Frequency")
plt.title("Distribution of Option Prices by Type")
plt.grid(True)
plt.tight_layout()
plt.show()
