import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load

# === Paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
X_test_path = os.path.join(project_root, "data", "X_test_asian.csv")
y_test_path = os.path.join(project_root, "data", "y_test_asian.csv")
scaler_path = os.path.join(project_root, "models", "scaler_asian.pkl")
model_path = os.path.join(project_root, "models", "mlp_asian.pth")

# === Load data ===
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()
scaler = load(scaler_path)

# === Normalize features ===
X_scaled = scaler.transform(X_test)

# === Define MLP ===
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

# === Load models ===
model = MLP(input_dim=X_scaled.shape[1])
model.load_state_dict(torch.load(model_path))
model.eval()

# === Predict ===
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()

# === Metrics ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Evaluation Metrics for Asian Option MLP:")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R2 Score: {r2:.4f}")

# === Plot: predicted vs actual ===
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray')
plt.xlabel("True Price (MC)")
plt.ylabel("Predicted Price (MLP)")
plt.title("MLP vs Monte Carlo: Asian Option Pricing")
plt.grid(True)
plt.tight_layout()

# Save plot
results_dir = os.path.join(project_root, "results", "evaluation_asian")
os.makedirs(results_dir, exist_ok=True)
plot_path = os.path.join(results_dir, "evaluation_plot.png")
plt.savefig(plot_path)
print(f"✅ Plot saved to {plot_path}")
