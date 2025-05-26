import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
output_dir = os.path.join(project_root, "results", "vol_surfaces")
os.makedirs(output_dir, exist_ok=True)

option_types = ["asian", "barrier", "lookback"]

def plot_iv_surface(option_type):
    # Load dataset (filtered or random)
    file_random = os.path.join(data_dir, f"option_dataset_{option_type}_random.csv")
    file_filtered = os.path.join(data_dir, f"option_dataset_{option_type}_filtered.csv")

    if os.path.exists(file_random):
        df = pd.read_csv(file_random)
    elif os.path.exists(file_filtered):
        df = pd.read_csv(file_filtered)
    else:
        raise FileNotFoundError(f"No dataset found for {option_type}")

    # Extract implied volatility columns
    iv_columns = [col for col in df.columns if col.startswith("sigma_")]
    if not iv_columns:
        raise ValueError(f"No implied volatility features found in {option_type} dataset.")

    # Parse moneyness and maturity from column names
    iv_data = []
    for col in iv_columns:
        parts = col.split("_")
        m = int(parts[1]) / 100  # e.g. 120 → 1.20 moneyness
        t = int(parts[2].replace("m", "")) / 12  # e.g. 6m → 0.5 years
        vols = df[col].values
        for v in vols:
            if np.isfinite(v):
                iv_data.append((m, t, v))

    iv_df = pd.DataFrame(iv_data, columns=["M", "T", "IV"])

    # Pivot into grid for plotting
    M_grid = np.linspace(iv_df["M"].min(), iv_df["M"].max(), 50)
    T_grid = np.linspace(iv_df["T"].min(), iv_df["T"].max(), 50)
    M_mesh, T_mesh = np.meshgrid(M_grid, T_grid)

    from scipy.interpolate import griddata
    IV_grid = griddata(
        (iv_df["M"], iv_df["T"]),
        iv_df["IV"],
        (M_mesh, T_mesh),
        method='cubic'
    )

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M_mesh, T_mesh, IV_grid, cmap="jet", edgecolor="none")
    ax.set_xlabel("Moneyness $M = S/K$")
    ax.set_ylabel("Time to Maturity $T$")
    ax.set_zlabel("Implied Volatility $\\sigma(T, M)$")
    ax.set_title(f"Implied Volatility Surface – {option_type.capitalize()}")
    # 👇 Invert T-axis direction (like in image)
    ax.invert_xaxis()
    # 👇 Invert T-axis direction (like in image)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{option_type}_vol_surface.png"))
    plt.close()

# Run for all option types
if __name__ == "__main__":
    for opt in option_types:
        plot_iv_surface(opt)
