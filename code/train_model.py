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
import sys
from config import EPOCHS, BATCH_SIZE

# === Redirect stdout to log file ===
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
data_dir = os.path.join(project_root, "data", "processed")
X = pd.read_csv(os.path.join(data_dir, "X.csv"))
y = pd.read_csv(os.path.join(data_dir, "y.csv"))

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Normalize features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# === Define MLP model ===
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
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

# === Training loop ===
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.squeeze())
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}", flush=True)

# === Evaluation ===
try:
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().numpy()
        y_true = y_test.squeeze().values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n✅ Test RMSE: {rmse:.4f}", flush=True)
        print(f"✅ Test R2 Score: {r2:.4f}", flush=True)
except Exception as e:
    print(f"❌ Evaluation failed: {e}", flush=True)

# === Save trained model ===
try:
    model_path = os.path.join(project_root, "models", "mlp_option_pricing.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving model to: {model_path}", flush=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}", flush=True)
except Exception as e:
    print(f"❌ Failed to save model: {e}", flush=True)

# === Save scaler ===
scaler_path = os.path.join(project_root, "models", "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to {scaler_path}", flush=True)
# === Save test set for evaluation ===
X_test_path = os.path.join(data_dir, "X_test.csv")
y_test_path = os.path.join(data_dir, "y_test.csv")
X_test.to_csv(X_test_path, index=False)
y_test.to_csv(y_test_path, index=False)
print(f"✅ Test data saved to:\n{X_test_path}\n{y_test_path}", flush=True)


print("👋 Finished train_model.py", flush=True)

# === Close log ===
sys.stdout.close()
