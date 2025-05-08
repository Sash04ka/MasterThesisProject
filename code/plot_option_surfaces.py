import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

# === Load dataset ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = pd.read_csv(os.path.join(project_root, "data", "raw", "option_dataset.csv"))

# === Output directory ===
results_dir = os.path.join(project_root, "results", "dataset_overview")
os.makedirs(results_dir, exist_ok=True)

# === 3D price surfaces (T, K, Price) ===
option_types = ["asian", "barrier", "lookback"]

for opt_type in option_types:
    df_type = df[df["type"] == opt_type].copy()
    X = df_type["T"]
    Y = df_type["K"]
    Z = df_type["price"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_title(f"{opt_type.capitalize()} Option Price Surface")
    ax.set_xlabel("Time to Maturity T")
    ax.set_ylabel("Strike K")
    ax.set_zlabel("Option Price")
    ax.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{opt_type}_3d_surface.png"), dpi=300)
    plt.close()

# === Distribution of option prices by type ===
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="price", hue="type", kde=True, bins=40, multiple="stack")
plt.title("Distribution of Option Prices by Type")
plt.xlabel("Option Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "price_distributions.png"), dpi=300)
plt.close()

# === Correlation heatmap (excluding moneyness if present) ===
plt.figure(figsize=(10, 8))
corr = df.drop(columns=["type"], errors="ignore")
if "moneyness" in corr.columns:
    corr = corr.drop(columns=["moneyness"])
corr = corr.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

print("✅ Clean visualizations saved to:", results_dir)
