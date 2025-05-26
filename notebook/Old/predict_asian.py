import torch
import pandas as pd
import numpy as np
import os
from joblib import load
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff

# === Input Parameters ===
print("\n--- Asian Option Input ---")
K_values = [80, 90, 100, 110, 120]
T_values = [1/12, 3/12, 6/12, 1.0]

print("Choose a strike price K:")
for i, val in enumerate(K_values):
    print(f"{i + 1}: {val}")
K = K_values[int(input("Select option (1–5): ")) - 1]

print("Choose time to maturity T:")
for i, val in enumerate(T_values):
    print(f"{i + 1}: {int(val * 12)} months")
T = T_values[int(input("Select option (1–4): ")) - 1]

r = 0.02

# === Simulate paths for smile ===
N = int(252 * T)
v0 = np.random.uniform(0.01, 0.09)
theta = np.random.uniform(0.01, 0.09)
sigma_v = np.random.uniform(0.1, 0.6)
kappa = np.random.uniform(0.5, 4.0)
rho = np.random.uniform(-0.9, -0.1)
lambda_jump = np.random.uniform(0.01, 0.2)
mu_jump = np.random.uniform(-0.1, 0.0)
sigma_jump = np.random.uniform(0.1, 0.4)

S, v = simulate_bates_paths(
    M=1000, N=N, T=T, S0=100, r=r,
    v0=v0, theta=theta, sigma_v=sigma_v,
    kappa=kappa, rho=rho,
    lambda_jump=lambda_jump,
    mu_jump=mu_jump, sigma_jump=sigma_jump
)

# === Build smile features ===
T_smiles = [1/12, 3/12, 6/12, 1.0]
M_smiles = [0.8, 0.9, 1.0, 1.1, 1.2]
smile_features = {}
for m in M_smiles:
    for t in T_smiles:
        col = f"sigma_{int(m*100):03d}_{int(t*12)}m"
        idx = int(t * N)
        idx = min(idx, N - 1)
        smile_features[col] = np.mean(v[:, idx]) / 0.2

# === Prepare models input ===
row = {**smile_features, "K": K, "T": T, "r": r}
x_df = pd.DataFrame([row])

# === Load models and scaler ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scaler = load(os.path.join(project_root, "models", "scaler_asian.pkl"))
model_path = os.path.join(project_root, "models", "mlp_asian.pth")

X_scaled = scaler.transform(x_df)

class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

model = MLP(input_dim=X_scaled.shape[1])
model.load_state_dict(torch.load(model_path))
model.eval()

# === Predict ===
x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    mlp_pred = model(x_tensor).item()

print(f"\n💰 MLP Predicted Asian Option Price: {mlp_pred:.4f}")

# === Compare with MC ===
print("\n⏳ Simulating Monte Carlo...")
mc_price = np.mean(asian_call_payoff(S, strike=K, r=r, T=T))
print(f"🎲 Monte Carlo Price: {mc_price:.4f}")
