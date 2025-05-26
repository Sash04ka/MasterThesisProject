import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from utils.bates_model import simulate_bates_paths

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(project_root, "results", "visualizations")
os.makedirs(results_dir, exist_ok=True)

# === 1. Bates Model Price Paths ===
S, v = simulate_bates_paths(M=10, T=1.0, N=252, S0=100, seed=42)

plt.figure(figsize=(12, 6))
for i in range(S.shape[0]):
    plt.plot(S[i], label=f"Path {i+1}")
plt.title("Sample Bates Model Price Paths (10 Paths, 1 Year)")
plt.xlabel("Time Steps")
plt.ylabel("Asset Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "bates_price_paths.png"))
plt.close()

# === Helper to load and plot datasets ===
def dataset_overview(dataset_name):
    df = pd.read_csv(os.path.join(project_root, "data", f"option_dataset_{dataset_name}.csv"))
    smile_cols = [c for c in df.columns if c.startswith("sigma_")]

    # Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["price"], bins=40, kde=True)
    plt.title(f"{dataset_name.capitalize()} Option Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_price_distribution.png"))
    plt.close()

    # 3D Price surface by K and T
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs = df["K"]
    ys = df["T"]
    zs = df["price"]
    surf = ax.plot_trisurf(xs, ys, zs, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Time to Maturity (T)")
    ax.set_zlabel("Option Price")
    ax.set_title(f"{dataset_name.capitalize()} Option Price Surface")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_price_surface_3d.png"))
    plt.close()

    # Volatility smile surface (moneyness vs maturity)
    # Collect smile data for selected moneyness and maturities
    moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
    maturities = [1, 3, 6, 12]  # in months
    smile_data = []
    for m in moneyness:
        for t in maturities:
            col_name = f"sigma_{int(m*100):03d}_{t}m"
            if col_name in df.columns:
                smile_data.append((m, t, df[col_name].mean()))

    smile_df = pd.DataFrame(smile_data, columns=["Moneyness", "MaturityMonths", "ImpliedVol"])
    smile_pivot = smile_df.pivot(index="MaturityMonths", columns="Moneyness", values="ImpliedVol")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X = smile_pivot.columns.values
    Y = smile_pivot.index.values
    X, Y = np.meshgrid(X, Y)
    Z = smile_pivot.values
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k', alpha=0.8)
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Maturity (Months)")
    ax.set_zlabel("Average Implied Volatility")
    ax.set_title(f"{dataset_name.capitalize()} Average Volatility Smile Surface")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_volatility_smile_surface.png"))
    plt.close()

# Generate overview for all datasets
for opt_type in ["asian", "barrier", "lookback"]:
    print(f"Generating dataset overview for {opt_type}...")
    dataset_overview(opt_type)

print("✅ Visualization generation complete. Check the results/visualizations folder.")
