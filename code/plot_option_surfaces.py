import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from config import S0

# === Define project root ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# === Load dataset ===
df = pd.read_csv(os.path.join(project_root, "data", "raw", "option_dataset.csv"))

# === Create output folder ===
results_dir = os.path.join(project_root, "results", "evaluation")
os.makedirs(results_dir, exist_ok=True)

# === Plot 3D surfaces by option type ===
option_types = ["asian", "barrier", "lookback"]

for opt_type in option_types:
    df_type = df[df["type"] == opt_type].copy()
    df_type["moneyness"] = df_type["K"] / S0

    X = df_type["moneyness"]
    Y = df_type["T"]
    Z = df_type["price"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(X, Y, Z, cmap="viridis", edgecolor="none")

    ax.set_title(f"{opt_type.capitalize()} Option Price Surface")
    ax.set_xlabel("Moneyness K/S₀")
    ax.set_ylabel("Time to Maturity T")
    ax.set_zlabel("Option Price")
    ax.view_init(elev=30, azim=120)  # Updated viewing angle for clarity

    plt.tight_layout()
    surface_path = os.path.join(results_dir, f"{opt_type}_option_surface.png")
    plt.savefig(surface_path, dpi=300)
    plt.close()

print("✅ All option price surfaces saved to results/evaluation/")
