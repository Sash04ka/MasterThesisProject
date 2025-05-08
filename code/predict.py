import torch
import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.preprocessing import StandardScaler
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff

# === Input Parameters ===
print("--- Enter Option Parameters ---")
K = float(input("Strike price K (suggested range: 50–150): "))
if not (50 <= K <= 150):
    print("⚠️ Warning: K is outside the model's training range [50–150]")

T = float(input("Time to maturity T in years (suggested range: 0.1–5.0): "))
if not (0.1 <= T <= 5.0):
    print("⚠️ Warning: T is outside the model's training range [0.1–5.0]")

valid_types = ["asian", "barrier", "lookback"]
option_type = ""
while option_type not in valid_types:
    option_type = input("Option type (asian / barrier / lookback): ").strip().lower()

barrier = 0.0
if option_type == "barrier":
    barrier = float(input("Barrier level (suggested: 100–200): "))
    if not (100 <= barrier <= 200):
        print("⚠️ Warning: Barrier level is outside training range [100–200]")

# === Volatility Inputs ===
print("\n--- Volatility Characteristics ---")
print("Please enter volatility values as percentages (e.g. for 20% type 20)")
vol_mean_pct = float(input("Average volatility over path (suggested: 15%–30%): "))
vol_T_pct = float(input("Volatility at maturity (suggested: 15%–30%): "))
vol_std_pct = float(input("Volatility standard deviation (suggested: 7%–17%): "))

if not (15 <= vol_mean_pct <= 30):
    print("⚠️ Warning: Average volatility is outside recommended [15–30]%")
if not (15 <= vol_T_pct <= 30):
    print("⚠️ Warning: Volatility at maturity is outside recommended [15–30]%")
if not (7 <= vol_std_pct <= 17):
    print("⚠️ Warning: Volatility std is outside recommended [7–17]%")

# === Convert % to variance ===
vol_mean = (vol_mean_pct / 100) ** 2
vol_T = (vol_T_pct / 100) ** 2
vol_std = (vol_std_pct / 100) ** 2

# === One-hot encode type ===
type_encoded = [1.0 if t == option_type else 0.0 for t in valid_types]

# === Prepare input DataFrame ===
x_input = pd.DataFrame([[K, T, barrier, vol_mean, vol_T, vol_std] + type_encoded],
    columns=["K", "T", "barrier", "vol_mean", "vol_T", "vol_std", "type_asian", "type_barrier", "type_lookback"])

# === Load scaler and model ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scaler = load(os.path.join(project_root, "model", "scaler.pkl"))
X_scaled = scaler.transform(x_input)

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

model = MLP(input_dim=9)
model.load_state_dict(torch.load(os.path.join(project_root, "model", "mlp_option_pricing.pth")))
model.eval()

# === Predict with MLP ===
with torch.no_grad():
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_pred = model(x_tensor).item()

print(f"\n💰 MLP Predicted option price: {y_pred:.4f}")

# === Monte Carlo Comparison ===
print("\n⏳ Simulating Monte Carlo price...")
N = int(252 * T)
S, _ = simulate_bates_paths(M=1000, N=N, T=T, seed=123)

if option_type == "asian":
    mc_price = np.mean(asian_call_payoff(S, strike=K, r=0.0, T=T))
elif option_type == "barrier":
    mc_price = np.mean(barrier_call_payoff(S, strike=K, barrier=barrier, r=0.0, T=T))
elif option_type == "lookback":
    mc_price = np.mean(lookback_call_payoff(S, strike=K, r=0.0, T=T))

print(f"🎲 Monte Carlo estimated price: {mc_price:.4f}")
