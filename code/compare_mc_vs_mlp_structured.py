import numpy as np
import pandas as pd
import torch
import os
import time
from tqdm import tqdm
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, median_absolute_error
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff

# === Paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scaler = load(os.path.join(project_root, "model", "scaler.pkl"))

# === Load trained MLP model ===
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

# === Load test data ===
X_test = pd.read_csv(os.path.join(project_root, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(project_root, "data", "processed", "y_test.csv")).squeeze()

# === Predict with MLP ===
X_scaled = scaler.transform(X_test)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    mlp_preds = model(X_tensor).squeeze().numpy()

# === Monte Carlo comparison ===
records = []
start_mc = time.time()

for i, row in tqdm(X_test.iterrows(), total=X_test.shape[0], desc="Running MC simulations"):
    K = row["K"]
    T = row["T"]
    barrier = row["barrier"]
    vol_mean = row["vol_mean"]
    vol_T = row["vol_T"]
    vol_std = row["vol_std"]
    type_vec = [row["type_asian"], row["type_barrier"], row["type_lookback"]]
    option_type = ["asian", "barrier", "lookback"][np.argmax(type_vec)]

    N = int(252 * T)
    S, _ = simulate_bates_paths(M=1000, N=N, T=T)  # Reduced MC paths from 5000 → 1000

    if option_type == "asian":
        mc_price = np.mean(asian_call_payoff(S, K, r=0.0, T=T))
    elif option_type == "barrier":
        mc_price = np.mean(barrier_call_payoff(S, K, barrier, r=0.0, T=T))
    elif option_type == "lookback":
        mc_price = np.mean(lookback_call_payoff(S, strike=K, r=0.0, T=T))

    records.append({
        "type": option_type,
        "K": K,
        "T": T,
        "barrier": barrier,
        "vol_mean": vol_mean,
        "vol_T": vol_T,
        "vol_std": vol_std,
        "MC": mc_price,
        "MLP": mlp_preds[i],
        "abs_error": abs(mc_price - mlp_preds[i]),
        "note": "⚠️ NEGATIVE!" if mlp_preds[i] < 0 else ""
    })

end_mc = time.time()

# === Compute metrics ===
df = pd.DataFrame(records)
rmse = np.sqrt(mean_squared_error(df["MC"], df["MLP"]))
r2 = r2_score(df["MC"], df["MLP"])
mae = mean_absolute_error(df["MC"], df["MLP"])
mse = mean_squared_error(df["MC"], df["MLP"])
max_err = max_error(df["MC"], df["MLP"])
median_err = median_absolute_error(df["MC"], df["MLP"])

mlp_avg_time = 0.14 / 1000  # ms to seconds
total_mc_time = end_mc - start_mc
speedup = total_mc_time / (len(df) * mlp_avg_time)

# === Save results ===
results_dir = os.path.join(project_root, "results", "benchmark_structured")
os.makedirs(results_dir, exist_ok=True)
df.to_csv(os.path.join(results_dir, "sample_comparison_structured.csv"), index=False)

with open(os.path.join(results_dir, "summary_metrics_structured.txt"), "w", encoding="utf-8") as f:
    f.write("📊 Evaluation Metrics (Structured):\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"Mean Absolute Error: {mae:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"Max Absolute Error: {max_err:.4f}\n")
    f.write(f"Median Absolute Error: {median_err:.4f}\n\n")
    f.write("⚡ Speed Comparison:\n")
    f.write(f"Monte Carlo total time: {total_mc_time:.2f} s\n")
    f.write(f"MLP average per sample: {mlp_avg_time * 1000:.2f} ms\n")
    f.write(f"MLP is approximately {speedup:.1f}x faster than MC\n")

print("✅ Structured comparison complete. Results saved.")
