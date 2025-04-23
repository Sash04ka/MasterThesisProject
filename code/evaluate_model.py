import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import load

# === Paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data", "processed")
model_path = os.path.join(project_root, "models", "mlp_option_pricing.pth")
scaler_path = os.path.join(project_root, "models", "scaler.pkl")
results_dir = os.path.join(project_root, "results", "evaluation")
os.makedirs(results_dir, exist_ok=True)

# === Load data ===
X = pd.read_csv(os.path.join(data_dir, "X.csv"))
y = pd.read_csv(os.path.join(data_dir, "y.csv"))
y = y.iloc[:, 0]  # ensure Series

# === Load scaler and transform ===
scaler = load(scaler_path)
X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

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

model = MLP(input_dim=X.shape[1])
model.load_state_dict(torch.load(model_path))
model.eval()

# === Predict ===
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()

y_true = y.values
residuals = y_pred - y_true
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# === Visualization ===
sns.set(style="whitegrid")
plt.figure(figsize=(12, 5))

# True vs Predicted
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_true, y=y_pred, s=20, alpha=0.4, color="blue", edgecolor=None)
plt.plot([0, 20], [0, 20], '--r')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("True vs Predicted Option Prices")
plt.xlim(0, 20)
plt.ylim(0, 20)

# Residuals
plt.subplot(1, 2, 2)
sns.histplot(residuals, bins=40, kde=True, color="gray")
plt.title("Residuals Distribution (Predicted - True)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.xlim(-1, 1)

# Save figure
fig_path = os.path.join(results_dir, "evaluation_results.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.show()

# Save metrics
with open(os.path.join(results_dir, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write("\U0001f4ca Evaluation Metrics:\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

# Print to console
print("\n\U0001f4ca Evaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"✅ Evaluation plot saved to {fig_path}")
