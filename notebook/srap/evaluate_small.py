import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

# Paths and constants
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "option_dataset_asian_filtered.csv")
model_path = os.path.join(project_root, "models", "mlp_asian_small.pth")
scaler_path = os.path.join(project_root, "models", "scaler_asian_small.pkl")

# Load dataset
df = pd.read_csv(data_path)

# Load scaler
import joblib
scaler = joblib.load(scaler_path)

# Define models class (small MLP)
class SmallMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.model(x)

# Prepare features and target
smile_cols = [col for col in df.columns if col.startswith('sigma_') or col.startswith('iv_')]
features = smile_cols + ['K', 'T', 'r']
X = df[features].values
y_true = df['price'].values

# Scale features
X_scaled = scaler.transform(X)

# Convert to torch tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Load models
model = SmallMLP(input_dim=X.shape[1])
model.load_state_dict(torch.load(model_path))
model.eval()

# Predict
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()

# Metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
rel_error = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"Mean Relative Error (%): {rel_error:.2f}")

# Plot True vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
plt.xlabel("Monte Carlo Price (True)")
plt.ylabel("MLP Predicted Price")
plt.title("True vs Predicted Option Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "true_vs_predicted_asian_small.png"))
plt.show()

# Plot Error Histogram
errors = np.abs(y_pred - y_true)
plt.figure(figsize=(8,6))
sns.histplot(errors, bins=40, kde=True)
plt.xlabel("Absolute Error")
plt.title("Distribution of Absolute Errors")
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "abs_error_distribution_asian_small.png"))
plt.show()

# --- New: MRE vs Price Cutoff plot ---
cutoffs = np.linspace(0, np.max(y_true), 50)
mres = []

for c in cutoffs:
    mask = y_true >= c
    if np.sum(mask) == 0:
        continue
    rel_err = np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]
    mre = np.mean(rel_err) * 100
    mres.append((c, mre))

cutoff_vals, mre_vals = zip(*mres)

plt.figure(figsize=(8,6))
plt.plot(cutoff_vals, mre_vals, marker='o')
plt.xlabel("Price Cutoff")
plt.ylabel("Mean Relative Error (%)")
plt.title("MRE vs Price Cutoff")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "mre_vs_price_cutoff_asian_small.png"))
plt.show()
