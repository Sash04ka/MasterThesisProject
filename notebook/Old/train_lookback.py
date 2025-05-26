import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from config import EPOCHS, BATCH_SIZE

# === Setup ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = pd.read_csv(os.path.join(project_root, "data", "option_dataset_lookback.csv"))

# === Preprocess ===
smile_cols = [col for col in df.columns if col.startswith("sigma_")]
df[smile_cols] = df[smile_cols] / 0.2  # normalize relative to 20%

features = smile_cols + ['K', 'T', 'r']  # no barrier input
X = df[features]
y = df['price']

# === Stratified split by discrete buckets of K and T
df['K_bucket'] = df['K'].astype(str)
df['T_bucket'] = df['T'].round(4).astype(str)
stratify_col = df['K_bucket'] + "_" + df['T_bucket']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_col
)

# === Standardize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Convert to tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# === Define models ===
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

model = MLP(input_dim=X_train.shape[1])

# === Training ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze().numpy()
    y_true = y_test_tensor.squeeze().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n✅ Test RMSE: {rmse:.4f}")
    print(f"✅ Test R2 Score: {r2:.4f}")

# === Save artifacts ===
model_path = os.path.join(project_root, "models", "mlp_lookback.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")

scaler_path = os.path.join(project_root, "models", "scaler_lookback.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to {scaler_path}")

X_test.to_csv(os.path.join(project_root, "data", "X_test_lookback.csv"), index=False)
y_test.to_csv(os.path.join(project_root, "data", "y_test_lookback.csv"), index=False)
print("✅ Test data saved")

print("👋 Finished train_lookback.py")
