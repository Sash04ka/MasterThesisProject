import numpy as np
import pandas as pd
import torch
import os
import time
from tqdm import tqdm
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff

# === Paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scaler = load(os.path.join(project_root, "models", "scaler_asian.pkl"))

# === Load trained models ===
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

model = MLP(input_dim=23)
model.load_state_dict(torch.load(os.path.join(project_root, "models", "mlp_asian.pth")))
model.eval()

# === Setup ===
K_values = [80, 90, 100, 110, 120]
T_values = [1/12, 3/12, 6/12, 1.0]
N_SAMPLES = 200
records = []

start = time.time()
for _ in tqdm(range(N_SAMPLES), desc="Random comparison"):
    # Random contract
    K = np.random.choice(K_values)
    T = np.random.choice(T_values)
    r = 0.02
    N = int(252 * T)

    # Bates params
    v0 = np.random.uniform(0.01, 0.09)
    theta = np.random.uniform(0.01, 0.09)
    sigma_v = np.random.uniform(0.1, 0.6)
    kappa = np.random.uniform(0.5, 4.0)
    rho = np.random.uniform(-0.9, -0.1)
    lambda_jump = np.random.uniform(0.01, 0.2)
    mu_jump = np.random.uniform(-0.1, 0.0)
    sigma_jump = np.random.uniform(0.1, 0.4)

    S, v = simulate_bates_paths(M=1000, N=N, T=T, S0=100, r=r,
                                v0=v0, theta=theta, sigma_v=sigma_v,
                                kappa=kappa, rho=rho,
                                lambda_jump=lambda_jump, mu_jump=mu_jump,
                                sigma_jump=sigma_jump)

    # Smile features
    smile = {}
    for m in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for t in T_values:
            col = f"sigma_{int(m*100):03d}_{int(t*12)}m"
            idx = int(t * N)
            idx = min(idx, N - 1)
            smile[col] = np.mean(v[:, idx]) / 0.2

    # Prepare input
    row = {**smile, "K": K, "T": T, "r": r}
    x_df = pd.DataFrame([row])
    x_scaled = scaler.transform(x_df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        mlp_pred = model(x_tensor).item()

    mc_price = np.mean(asian_call_payoff(S, strike=K, r=r, T=T))

    records.append({
        "K": K, "T": T, "r": r,
        "MC": mc_price,
        "MLP": max(0, mlp_pred),
        "abs_error": abs(mc_price - mlp_pred),
        "rel_error": abs(mc_price - mlp_pred) / (mc_price + 1e-6),
        "diff": mlp_pred - mc_price
    })

end = time.time()

# Save results
df = pd.DataFrame(records)
rmse = np.sqrt(mean_squared_error(df["MC"], df["MLP"]))
r2 = r2_score(df["MC"], df["MLP"])

results_dir = os.path.join(project_root, "results", "benchmark_random_asian")
os.makedirs(results_dir, exist_ok=True)
df.to_csv(os.path.join(results_dir, "sample_comparison_random.csv"), index=False)

with open(os.path.join(results_dir, "summary_metrics_random.txt"), "w") as f:
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"Time Elapsed: {end - start:.2f} s\n")

print("✅ Random comparison complete. Results saved.")
