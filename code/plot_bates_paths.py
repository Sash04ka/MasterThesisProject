import numpy as np
import matplotlib.pyplot as plt
import os
from utils.bates_model import simulate_bates_paths
from config import S0

# === Output directory ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(project_root, "results", "evaluation")
os.makedirs(output_dir, exist_ok=True)

# === Simulate paths ===
S, _ = simulate_bates_paths(M=10, T=1.0, N=252, S0=S0, seed=42)

# === Plot ===
plt.figure(figsize=(10, 6))
for i in range(S.shape[0]):
    plt.plot(S[i], label=f"Path {i+1}")

plt.title("Bates Model Price Paths")
plt.xlabel("Time Step")
plt.ylabel("Asset Price")
plt.grid(True)
plt.tight_layout()

# === Save ===
plot_path = os.path.join(output_dir, "bates_price_paths.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"✅ Bates paths plot saved to {plot_path}")
