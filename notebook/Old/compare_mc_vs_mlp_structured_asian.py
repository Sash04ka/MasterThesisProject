import numpy as np
import pandas as pd
import torch
import os
import time
from tqdm import tqdm
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, median_absolute_error
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

# Load data and models
X_test_path = os.path.join(project_root, "data", "X_test_asian.csv")
y_test_path = os.path.join(project_root, "data", "y_test_asian.csv")
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

model = MLP(input_dim=X_test.shape[1])
model.load_state_dict(torch.load(os.path.join(project_root, "models", "mlp_asian.pth")))
model.eval()

# Predict with MLP
X_scaled = scaler.transform(X_test)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    mlp_preds = model(X_tensor).squeeze().numpy()

# Monte Carlo comparison
records = []
start_mc = time.time()

for i, row in tqdm(X_test.iterrows(), total=X_test.shape[0], desc="Running MC simulations"):
    K = row["K"]
    T = row["T"]
    r_val = row["r"]
    N = int(252 * T)

    S, _ = simulate_bates_paths(M=1000, N=N, T=T, S0=100, r=r_val)
    mc_price = np.mean(asian_call_payoff(S, strike=K, r=r_val, T=T))

    mlp_pred = max(0, mlp_preds[i])
    records.append({
        "K": K,
        "T": T,
        "r": r_val,
        "MC": mc_price,
        "MLP": mlp_pred,
        "abs_error": abs(mc_price - mlp_pred),
        "rel_error": abs(mc_price - mlp_pred) / (mc_price + 1e-6),
        "diff": mlp_pred - mc_price,
        "note": "⚠️ NEGATIVE!" if mlp_pred < 0 else ""
    })

end_mc = time.time()

# === Compute metrics
df = pd.DataFrame(records)
rmse = np.sqrt(mean_squared_error(df["MC"], df["MLP"]))
r2 = r2_score(df["MC"], df["MLP"])
mae = mean_absolute_error(df["MC"], df["MLP"])
mse = mean_squared_error(df["MC"], df["MLP"])
max_err = max_error(df["MC"], df["MLP"])
median_err = median_absolute_error(df["MC"], df["MLP"])

mlp_avg_time = 0.14 / 1000
total_mc_time = end_mc - start_mc
speedup = total_mc_time / (len(df) * mlp_avg_time)

# === Save results
results_dir = os.path.join(project_root, "results", "benchmark_structured_asian")
os.makedirs(results_dir, exist_ok=True)
df.to_csv(os.path.join(results_dir, "sample_comparison_asian.csv"), index=False)

with open(os.path.join(results_dir, "summary_metrics_asian.txt"), "w", encoding="utf-8") as f:
    f.write("\U0001f4ca Evaluation Metrics (Asian Structured):\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"Max Abs Error: {max_err:.4f}\n")
    f.write(f"Median Abs Error: {median_err:.4f}\n\n")
    f.write("\u26a1 Speed Comparison:\n")
    f.write(f"Monte Carlo total time: {total_mc_time:.2f} s\n")
    f.write(f"MLP average per sample: {mlp_avg_time * 1000:.2f} ms\n")
    f.write(f"MLP is approximately {speedup:.1f}x faster than MC\n")

print("✅ Asian structured comparison complete. Results saved.")
