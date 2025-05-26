import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
import sys
from config import EPOCHS, BATCH_SIZE

# === Logging ===
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(script_dir, ".."))
log_path = os.path.join(project_root, "results", "train_log.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

class DualWriter:
    def __init__(self, file):
        self.terminal = sys.__stdout__
        self.file = file
        self.closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self.closed:
            self.file.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.closed:
            self.file.flush()

    def close(self):
        if not self.closed:
            self.file.close()
            self.closed = True

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = DualWriter(log_file)

# === Load data ===
df = pd.read_csv(os.path.join(project_root, "data", "option_dataset.csv"))

# Drop rows with missing smile values
smile_cols = [col for col in df.columns if col.startswith("sigma_")]
df = df.dropna(subset=smile_cols)

# Add indicator for stratified sampling
df['type_raw'] = df['type']

# One-hot encode 'type'
ohe = OneHotEncoder(sparse_output=False, drop=None)
type_ohe = ohe.fit_transform(df[['type']])
type_cols = ohe.get_feature_names_out(['type'])
type_df = pd.DataFrame(type_ohe, columns=type_cols, index=df.index)

# Add barrier_input column
df['barrier_input'] = df['barrier']
df.loc[df['type'] != 'barrier', 'barrier_input'] = -1

# Normalize smile features relative to 20% baseline
for col in smile_cols:
    df[col] = df[col] / 0.2

# Apply log transformation to target
df['price_log'] = np.log(df['price'] + 1e-6)

# Final dataset for training
df = pd.concat([df.drop(columns=['type']), type_df], axis=1)
features = smile_cols + ['K', 'T', 'r', 'barrier_input'] + list(type_df.columns)
X = df[features]
y_log = df['price_log']

# === Stratified split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42, stratify=df['type_raw']
)

# === Standardize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# === Define MLP ===
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

# === Training setup ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

# === Train ===
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
    y_pred_log = model(X_test_tensor).squeeze().numpy()
    y_pred = np.exp(y_pred_log) - 1e-6
    y_true = np.exp(y_test_tensor.numpy()) - 1e-6
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n✅ Test RMSE: {rmse:.4f}")
    print(f"✅ Test R2 Score: {r2:.4f}")

# === Save ===
model_path = os.path.join(project_root, "models", "mlp_option_pricing.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")

scaler_path = os.path.join(project_root, "models", "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to {scaler_path}")

X_test.to_csv(os.path.join(project_root, "data", "X_test.csv"), index=False)
np.exp(y_test) - 1e-6  # revert log for saving
y_test_raw = pd.Series(np.exp(y_test) - 1e-6)
y_test_raw.to_csv(os.path.join(project_root, "data", "y_test.csv"), index=False)
print("✅ Test data saved")

print("👋 Finished train_model.py")
sys.stdout.close()
