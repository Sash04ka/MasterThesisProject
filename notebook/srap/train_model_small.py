import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === Paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "option_dataset_asian_filtered.csv")
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, "mlp_asian_small.pth")
scaler_save_path = os.path.join(model_dir, "scaler_asian_small.pkl")

# === Load Data ===
df = pd.read_csv(data_path)
smile_cols = [col for col in df.columns if col.startswith('sigma_') or col.startswith('iv_')]
features = smile_cols + ['K', 'T', 'r']
X = df[features].values
y = df['price'].values

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for evaluation use
import joblib

joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved to {scaler_save_path}")

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Convert to torch tensors ===
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# === Define Small MLP ===
class SmallMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)


model = SmallMLP(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === Training Loop ===
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# === Save trained models ===
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
