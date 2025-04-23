import numpy as np
import pandas as pd
import torch
import os
import time
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, max_error
from utils.bates_model import simulate_bates_paths
from utils.payoffs import asian_call_payoff, barrier_call_payoff, lookback_call_payoff
from config import N_SAMPLES, M_MC, S0, v0, K_MIN, K_MAX, T_MIN, T_MAX, BARRIER_MIN, BARRIER_MAX

# === Load scaler and model ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scaler = load(os.path.join(project_root, "models", "scaler.pkl"))

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
model.load_state_dict(torch.load(os.path.join(project_root, "models", "mlp_option_pricing.pth")))
model.eval()

# === Run comparison ===
records = []
option_types = ["asian", "barrier", "lookback"]

start_mc = time.time()

for _ in range(N_SAMPLES):
    K = np.random.uniform(K_MIN, K_MAX)
    T = np.random.uniform(T_MIN, T_MAX)
    barrier = np.random.uniform(BARRIER_MIN, BARRIER_MAX)
    opt_type = np.random.choice(option_types)

    S, v = simulate_bates_paths(M=M_MC, T=T)
    vol_path = np.mean(v, axis=1)
    vol_mean = np.mean(vol_path)
    vol_T = np.mean(v[:, -1])
    vol_std = np.std(vol_path)

    if opt_type == "asian":
        mc_price = np.mean(asian_call_payoff(S, K, r=0.0, T=T))
        barrier_val = 0.0
        type_enc = [1, 0, 0]
    elif opt_type == "barrier":
        mc_price = np.mean(barrier_call_payoff(S, K, barrier, r=0.0, T=T))
        barrier_val = barrier
        type_enc = [0, 1, 0]
    elif opt_type == "lookback":
        mc_price = np.mean(lookback_call_payoff(S, r=0.0, T=T))
        barrier_val = 0.0
        type_enc = [0, 0, 1]

    x = pd.DataFrame([[K, T, barrier_val, vol_mean, vol_T, vol_std] + type_enc],
                     columns=["K", "T", "barrier", "vol_mean", "vol_T", "vol_std", "type_asian", "type_barrier", "type_lookback"])
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    start_mlp = time.time()
    with torch.no_grad():
        mlp_price = model(x_tensor).item()
    mlp_duration = time.time() - start_mlp

    warning = "⚠️ NEGATIVE!" if mlp_price < 0 else ""

    records.append({
        "type": opt_type,
        "K": K,
        "T": T,
        "barrier": barrier_val,
        "vol_mean": vol_mean,
        "vol_T": vol_T,
        "vol_std": vol_std,
        "MC": mc_price,
        "MLP": mlp_price,
        "abs_error": abs(mc_price - mlp_price),
        "note": warning,
        "mlp_time": mlp_duration
    })

end_mc = time.time()

# === Evaluation ===
df = pd.DataFrame(records)
rmse = np.sqrt(mean_squared_error(df["MC"], df["MLP"]))
r2 = r2_score(df["MC"], df["MLP"])
mae = np.mean(df["abs_error"])
mse = mean_squared_error(df["MC"], df["MLP"])
max_err = max_error(df["MC"], df["MLP"])
median_err = median_absolute_error(df["MC"], df["MLP"])

mlp_avg_time = df["mlp_time"].mean()
total_mc_time = end_mc - start_mc
speedup = (total_mc_time / (N_SAMPLES * mlp_avg_time)) if mlp_avg_time > 0 else float("inf")

# === Save results ===
results_dir = os.path.join(project_root, "results", "benchmark_comparison")
os.makedirs(results_dir, exist_ok=True)

df.to_csv(os.path.join(results_dir, "sample_comparison.csv"), index=False)

with open(os.path.join(results_dir, "summary_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("📊 Evaluation Metrics:\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"Mean Absolute Error: {mae:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"Max Absolute Error: {max_err:.4f}\n")
    f.write(f"Median Absolute Error: {median_err:.4f}\n\n")
    f.write("⚡ Speed Comparison:\n")
    f.write(f"Monte Carlo total time: {total_mc_time:.2f} s\n")
    f.write(f"MLP average per sample: {mlp_avg_time*1000:.2f} ms\n")
    f.write(f"MLP is approximately {speedup:.1f}x faster than MC\n")

# === Output to console ===
print("\n🔍 Sample Comparison Table (first 10 rows):")
print(df.head(10))
print("\n📊 Evaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Max Absolute Error: {max_err:.4f}")
print(f"Median Absolute Error: {median_err:.4f}")
print("\n⚡ Speed Comparison:")
print(f"Monte Carlo total time: {total_mc_time:.2f} s")
print(f"MLP average per sample: {mlp_avg_time*1000:.2f} ms")
print(f"⚡ MLP is approximately {speedup:.1f}x faster than MC")
