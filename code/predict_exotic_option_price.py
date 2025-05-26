import torch
import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.preprocessing import StandardScaler
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff
from mlp_model import MLP

# === Setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# === Input ===
print("--- Enter Option Parameters ---")
K_levels = [80, 90, 100, 110, 120]
T_levels = [1/12, 3/12, 6/12, 9/12, 1.0]

print("Choose a strike price K:")
for i, val in enumerate(K_levels):
    print(f"{i + 1}: {val}")
K = K_levels[int(input("Select option (1–5): ")) - 1]

print("Choose time to maturity T:")
for i, val in enumerate(T_levels):
    print(f"{i + 1}: {int(val * 12)} months")
T = T_levels[int(input(f"Select option (1–{len(T_levels)}): ")) - 1]

r = 0.02

valid_types = ["asian", "barrier", "lookback"]
option_type = ""
while option_type not in valid_types:
    option_type = input("Option type (asian / barrier / lookback): ").strip().lower()

barrier = None
if option_type == "barrier":
    barrier = float(input("Barrier level (suggested: 100–200): "))

# === Smile Surface ===
print("\n--- Volatility Smile Surface ---")
use_default = input("Use default 20% flat smile? (y/n): ").strip().lower() == 'y'

moneyness_levels = [0.8, 0.9, 1.0, 1.1, 1.2]
maturity_levels = [1/12, 3/12, 6/12, 9/12, 1.0]

smile_features = {}
for m in moneyness_levels:
    for t in maturity_levels:
        key = f"sigma_{int(m*100):03d}_{int(t*12)}m"
        smile_features[key] = 0.2 if use_default else float(input(f"{key}: "))

# === Load scalers and model ===
model_tag = option_type
model_path = os.path.join(project_root, "models", f"mlp_{model_tag}.pth")
scaler_path = os.path.join(project_root, "models", f"scaler_{model_tag}.pkl")

feature_scaler: StandardScaler = load(scaler_path)

# === Build input row ===
row = {**smile_features, "K": K, "T": T, "r": r}
if option_type == "barrier":
    row["barrier"] = barrier

x_input_full = pd.DataFrame(columns=feature_scaler.feature_names_in_)
for col in feature_scaler.feature_names_in_:
    x_input_full[col] = [row.get(col, 0.0)]

X_scaled = feature_scaler.transform(x_input_full)

# === Load and run model ===
model = MLP(input_dim=X_scaled.shape[1])
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_pred = model(x_tensor).item()
    y_pred = max(0.0, y_pred)

print(f"\n💰 MLP Predicted {option_type} option price: {y_pred:.4f}")

# === Monte Carlo Comparison ===
print("\n⏳ Simulating Monte Carlo price...")
N = int(252 * T)

# === Sample realistic Bates parameters
params = {
    "v0": np.random.uniform(0.01, 0.09),
    "theta": np.random.uniform(0.01, 0.09),
    "sigma_v": np.random.uniform(0.1, 0.6),
    "kappa": np.random.uniform(0.5, 4.0),
    "rho": np.random.uniform(-0.9, -0.1),
    "lambda_jump": np.random.uniform(0.01, 0.2),
    "mu_jump": np.random.uniform(-0.1, 0.0),
    "sigma_jump": np.random.uniform(0.1, 0.4),
}

S, _ = simulate_bates_paths(
    M=1000, N=N, T=T, S0=100, r=r,
    v0=params["v0"], theta=params["theta"], sigma_v=params["sigma_v"],
    kappa=params["kappa"], rho=params["rho"],
    lambda_jump=params["lambda_jump"], mu_jump=params["mu_jump"], sigma_jump=params["sigma_jump"]
)

if option_type == "asian":
    mc_price = np.mean(asian_call_payoff(S, strike=K, r=r, T=T))
elif option_type == "barrier":
    mc_price = np.mean(barrier_call_payoff(S, strike=K, barrier=barrier, r=r, T=T))
else:
    mc_price = np.mean(lookback_call_payoff(S, strike=K, r=r, T=T))

print(f"🎲 Monte Carlo estimated price: {mc_price:.4f}")
