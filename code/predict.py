import torch
import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.preprocessing import StandardScaler

# === Input Parameters ===
print("--- Enter Option Parameters ---")
K = float(input("Strike price K (e.g. 100): "))
T = float(input("Time to maturity T in years (e.g. 1.0 = 1 year): "))

valid_types = ["asian", "barrier", "lookback"]
option_type = ""
while option_type not in valid_types:
    option_type = input("Option type (asian / barrier / lookback): ").strip().lower()

barrier = float(input("Barrier level (only if applicable): ")) if option_type == "barrier" else 0.0

print("--- Volatility Characteristics ---")
vol_mean_pct = float(input("Average volatility over path (in %, e.g. 20 for 20%): "))
vol_T_pct = float(input("Volatility at maturity (in %): "))
vol_std_pct = float(input("Volatility std deviation (in %): "))

# Convert % to variance
vol_mean = (vol_mean_pct / 100)**2
vol_T = (vol_T_pct / 100)**2
vol_std = (vol_std_pct / 100)**2

# === One-hot encode type ===
type_encoded = [1.0 if t == option_type else 0.0 for t in valid_types]

# === Prepare input as DataFrame ===
x_input = pd.DataFrame([[K, T, barrier, vol_mean, vol_T, vol_std] + type_encoded],
                       columns=["K", "T", "barrier", "vol_mean", "vol_T", "vol_std", "type_asian", "type_barrier", "type_lookback"])

# === Load scaler ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scaler_path = os.path.join(project_root, "models", "scaler.pkl")
scaler = load(scaler_path)
X_scaled = scaler.transform(x_input)

# === Define model ===
class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Load model ===
model_path = os.path.join(project_root, "models", "mlp_option_pricing.pth")
model = MLP(input_dim=9)
model.load_state_dict(torch.load(model_path))
model.eval()

# === Predict ===
with torch.no_grad():
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_pred = model(x_tensor).item()

print(f"\n💰 Predicted option price: {y_pred:.4f}")
