import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import joblib
from mlp_model import MLP  # Make sure train_model.py with MLP is accessible

# === Setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
model_dir = os.path.join(project_root, "models")
results_dir = os.path.join(project_root, "results")
visual_dir = os.path.join(results_dir, "visualizations")
os.makedirs(visual_dir, exist_ok=True)

option_types = ["asian", "barrier", "lookback"]
S0 = 100

for option_type in option_types:
    print(f"Processing {option_type}...")

    # Load data
    X_test = pd.read_csv(os.path.join(data_dir, f"X_test_{option_type}.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, f"y_test_{option_type}.csv"))

    # Load scaler and models
    scaler = joblib.load(os.path.join(model_dir, f"scaler_{option_type}.pkl"))
    model = MLP(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load(os.path.join(model_dir, f"mlp_{option_type}.pth")))
    model.eval()

    # Scale and predict
    X_scaled = scaler.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze().numpy()

    # Combine MC and ML prices with features
    df_prices = X_test[['K', 'T']].copy()
    df_prices['price_MC'] = y_test.values.flatten()
    df_prices['price_ML'] = y_pred

    # Create interpolation grid
    K_grid = np.linspace(df_prices['K'].min(), df_prices['K'].max(), 100)
    T_grid = np.linspace(df_prices['T'].min(), df_prices['T'].max(), 100)
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    # Interpolate MC and ML prices
    MC_values = griddata((df_prices['K'], df_prices['T']), df_prices['price_MC'], (K_mesh, T_mesh), method='cubic')
    ML_values = griddata((df_prices['K'], df_prices['T']), df_prices['price_ML'], (K_mesh, T_mesh), method='cubic')

    # Plot surfaces side-by-side
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(K_mesh, T_mesh * 12, MC_values, cmap='viridis', edgecolor='none')
    ax1.set_title(f'{option_type.capitalize()} Monte Carlo Price Surface')
    ax1.set_xlabel('Strike Price (K)')
    ax1.set_ylabel('Time to Maturity (Months)')
    ax1.set_zlabel('Option Price')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(K_mesh, T_mesh * 12, ML_values, cmap='plasma', edgecolor='none')
    ax2.set_title(f'{option_type.capitalize()} ML Predicted Price Surface')
    ax2.set_xlabel('Strike Price (K)')
    ax2.set_ylabel('Time to Maturity (Months)')
    ax2.set_zlabel('Option Price')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    output_path = os.path.join(visual_dir, f"{option_type}_mc_vs_ml_price_surface.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {option_type} MC vs ML price surface to {output_path}")

print("✅ All visualizations generated and saved.")
